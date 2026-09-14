import logging
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sacrebleu.metrics import BLEU
from src.data.kgdataset import KGDataset
from src.models.language_model import LanguageModel
from src.pipelines.base_pipeline import BasePipeline, _canon_triple, graph_key
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_bleu = BLEU(effective_order=True)
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def numbers_in(text: str) -> set[str]:
    return set(_NUM_RE.findall(text or ""))


def gate_naturalization(
    facts: List[str],
    naturalized: str,
    *,
    bleu_min: float = 12.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (accept, diagnostics).
    """
    ref = " ".join(facts).strip()
    hyp = (naturalized or "").strip()

    # Hard fail: empty or too short
    if not hyp or len(hyp.split()) < 6:
        return False, {"reason": "empty/too_short"}

    # Hard fail: numeric drift (very useful for KG facts)
    ref_nums = numbers_in(ref)
    hyp_nums = numbers_in(hyp)
    if not ref_nums.issubset(hyp_nums):
        return False, {
            "reason": "missing_numbers",
            "ref_nums": sorted(ref_nums),
            "hyp_nums": sorted(hyp_nums),
        }

    bleu = _bleu.sentence_score(hyp, [ref]).score
    return (bleu >= bleu_min), {"bleu": bleu}


@dataclass
class FactByFactConfig:
    generation_batch_size: int = 1

    # ---- stage 1 (triple realization) batching ----
    triple_batch_size: int = 8
    triple_max_new_tokens: int = 64
    triple_do_sample: bool = True
    triple_temperature: float = 0.7
    triple_top_p: float = 0.9
    triple_stop_strings: Optional[List[str]] = None

    triple_system_template: str = (
        "You convert knowledge graph triples into simple English factual sentences.\n"
        "Rules:\n"
        "- Use exactly ONE sentence per triple.\n"
        "- Do NOT add facts.\n"
        "- Do NOT remove facts.\n"
        "- Do NOT infer anything.\n"
        "- Preserve all names, numbers, and dates exactly.\n"
        "- Output only the sentence.\n"
    )

    triple_user_template: str = (
        "Triple:\n"
        "Subject: {subject}\n"
        "Predicate: {predicate}\n"
        "Object: {object}\n\n"
        "Sentence:"
    )

    # ---- stage 2 (optional naturalization) ----
    naturalize: bool = False
    naturalize_batch_size: int = 4
    naturalize_max_new_tokens: int = 256
    naturalize_do_sample: bool = True
    naturalize_temperature: float = 0.7
    naturalize_top_p: float = 0.9
    naturalize_stop_strings: Optional[List[str]] = None
    naturalize_repetition_penalty: float = 1.2

    naturalize_system_template: str = (
        "You are a careful editor. Rewrite the input factual sentences into a single fluent paragraph. "
        "Rules:\n"
        "- Do NOT add facts.\n"
        "- Do NOT remove facts.\n"
        "- Do NOT infer anything.\n"
        "- Preserve all names, numbers, and dates exactly.\n"
        "- Avoid repetition by using pronouns and conjunctions when safe.\n"
        "- Output only the final paragraph.\n"
    )

    naturalize_user_template: str = (
        "Input:\n{facts}\n\n"
        "Final:"
    )

    # ---- gating ----
    naturalize_bleu_min: float = 12.0

    # ---- BasePipeline cache knob (keep prompt_key stable across config changes) ----
    cache_use_prompt_key: bool = True


class FactByFactPipeline(BasePipeline):

    def __init__(
        self,
        lm: LanguageModel,
        dataset: KGDataset,
        config: Optional[FactByFactConfig] = None,
        *,
        dedup_by_graph: bool = True,
    ):
        self.config: FactByFactConfig = config or FactByFactConfig()
        super().__init__(lm=lm, dataset=dataset, config=self.config, dedup_by_graph=dedup_by_graph)

    def build_prompt(self, triples: List[Dict[str, str]]) -> str:
        return ""

    def extract_final(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]

        for line in (text or "").splitlines():
            if line.strip().lower().startswith("final:"):
                return line.split(":", 1)[1].strip()
        return (text or "").strip()

    def prompt_fingerprint(self) -> Dict[str, Any]:
        c = self.config
        return {
            "pipeline": self.__class__.__name__,
            "naturalize": bool(c.naturalize),
            "triple_system_template": c.triple_system_template,
            "triple_user_template": c.triple_user_template,
            "naturalize_system_template": c.naturalize_system_template,
            "naturalize_user_template": c.naturalize_user_template,
            "naturalize_bleu_min": float(c.naturalize_bleu_min),
            # include decoding knobs so cache is conservative
            "triple_decoding": {
                "max_new_tokens": int(c.triple_max_new_tokens),
                "do_sample": bool(c.triple_do_sample),
                "temperature": float(c.triple_temperature),
                "top_p": float(c.triple_top_p),
                "stop_strings": list(c.triple_stop_strings) if c.triple_stop_strings else None,
            },
            "naturalize_decoding": {
                "max_new_tokens": int(c.naturalize_max_new_tokens),
                "do_sample": bool(c.naturalize_do_sample),
                "temperature": float(c.naturalize_temperature),
                "top_p": float(c.naturalize_top_p),
                "stop_strings": list(c.naturalize_stop_strings) if c.naturalize_stop_strings else None,
                "repetition_penalty": float(c.naturalize_repetition_penalty),
            },
        }

    def _build_triple_prompt(self, triple: Dict[str, str]) -> str:
        user = self.config.triple_user_template.format(
            subject=(triple.get("subject") or ""),
            predicate=(triple.get("predicate") or ""),
            object=(triple.get("object") or ""),
        )
        return f"{self.config.triple_system_template}\n<|USER_PROMPT|>\n{user} "

    def _build_naturalize_prompt(self, facts: List[str]) -> str:
        facts_block = " ".join(f.strip() for f in facts if f.strip())
        user = self.config.naturalize_user_template.format(facts=facts_block)
        return f"{self.config.naturalize_system_template}\n<|USER_PROMPT|>\n{user} "

    def generate_for_split(
            self,
            split: str = "test",
            limit: Optional[int] = None,
            cache_path: Optional[str] = None,
            resume: bool = True,
            *args,
    ) -> List[Dict[str, Any]]:
        """
        Gap-fill resume semantics.

        If resume=True and cache_path contains records, this method keeps all cached
        records whose idx belongs to the current split/limit and generates only the
        missing idxs.

        This intentionally treats idx as the completion marker. Existing cached
        records are trusted as completed work, even if the config/prompt_key changed.
        """
        assert split in self.data, f"Unknown split: {split}"

        if split == "test":
            samples = self.dataset.load_test_500()
        elif split == "train":
            samples = self.dataset.load_train()
        else:
            samples = self.data[split]

        if limit is not None and limit > 0:
            samples = samples[:limit]

        n_samples = len(samples)

        cache_file: Optional[Path] = None
        cache_by_idx: Dict[int, Dict[str, Any]] = {}

        if cache_path is not None:
            cache_file = Path(cache_path)
            if resume:
                cache_by_idx = self._load_cache(cache_file)

        # Normalize and keep only cache records that are valid for this split/limit.
        normalized_cache_by_idx: Dict[int, Dict[str, Any]] = {}

        if resume:
            for idx, rec in cache_by_idx.items():
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    continue

                if 0 <= idx_int < n_samples:
                    normalized_cache_by_idx[idx_int] = rec

            cache_by_idx = normalized_cache_by_idx
        else:
            cache_by_idx = {}

        if resume:
            cached_idxs = set(cache_by_idx.keys())
            wanted_idxs = set(range(n_samples))
            missing_idxs = sorted(wanted_idxs - cached_idxs)
        else:
            missing_idxs = list(range(n_samples))

        logger.info(
            "Gap-fill resume: loaded %d cached records; missing %d/%d records",
            len(cache_by_idx),
            len(missing_idxs),
            n_samples,
        )

        results: List[Dict[str, Any]] = []

        # Return cached records as-is.
        if resume:
            for idx in sorted(cache_by_idx):
                results.append(cache_by_idx[idx])

        # Nothing left to generate.
        if not missing_idxs:
            results.sort(key=lambda r: r["idx"])
            return results

        # Build units only from missing indices.
        indexed_samples = [(idx, samples[idx]) for idx in missing_idxs]

        units: List[Dict[str, Any]] = []

        if self.dedup_by_graph:
            groups: Dict[str, Dict[str, Any]] = {}

            for idx, sample in indexed_samples:
                triples = sample.get("triples_parsed") or []
                gk = graph_key(triples)

                if gk not in groups:
                    groups[gk] = {
                        "graph_key": gk,
                        "triples": triples,
                        "members": [],
                    }

                groups[gk]["members"].append(
                    {
                        "idx": idx,
                        "reference": sample.get("text"),
                        "sample": sample,
                    }
                )

            units = list(groups.values())

        else:
            for idx, sample in indexed_samples:
                triples = sample.get("triples_parsed") or []

                units.append(
                    {
                        "graph_key": graph_key(triples),
                        "triples": triples,
                        "members": [
                            {
                                "idx": idx,
                                "reference": sample.get("text"),
                                "sample": sample,
                            }
                        ],
                    }
                )

        for u in units:
            u["prompt_key"] = self.prompt_key(u["triples"])

        # Approx length sort to reduce padding.
        if units:
            approx_prompts: List[str] = []

            for u in units:
                triple_prompts = [
                    self._build_triple_prompt(t)
                    for t in (u.get("triples") or [])
                ]
                approx_prompts.append(
                    "\n".join(triple_prompts) if triple_prompts else ""
                )

            tok = self.lm.tokenizer(
                approx_prompts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )

            lens = [len(ids) for ids in tok["input_ids"]]

            for u, L in zip(units, lens):
                u["prompt_len"] = L

            units.sort(key=lambda uu: uu.get("prompt_len") or 0)

        unit_bsz = max(1, int(getattr(self.config, "generation_batch_size", 1)))
        triple_bsz = max(1, int(getattr(self.config, "triple_batch_size", unit_bsz)))
        nat_bsz = max(1, int(getattr(self.config, "naturalize_batch_size", unit_bsz)))

        gen_kwargs_triple: Dict[str, Any] = dict(
            max_new_tokens=int(self.config.triple_max_new_tokens),
            do_sample=bool(self.config.triple_do_sample),
            temperature=float(self.config.triple_temperature),
            top_p=float(self.config.triple_top_p),
            tokenizer=getattr(self.lm, "tokenizer", None),
        )

        if self.config.triple_stop_strings:
            gen_kwargs_triple["stop_strings"] = list(self.config.triple_stop_strings)

        eos = getattr(self.lm.tokenizer, "eos_token_id", None)

        gen_kwargs_nat: Dict[str, Any] = dict(
            max_new_tokens=int(self.config.naturalize_max_new_tokens),
            do_sample=bool(self.config.naturalize_do_sample),
            temperature=float(self.config.naturalize_temperature),
            top_p=float(self.config.naturalize_top_p),
            repetition_penalty=float(self.config.naturalize_repetition_penalty),
            eos_token_id=eos,
            pad_token_id=eos,
            tokenizer=getattr(self.lm, "tokenizer", None),
        )

        if self.config.naturalize_stop_strings:
            gen_kwargs_nat["stop_strings"] = list(self.config.naturalize_stop_strings)

        num_batches = ceil(len(units) / unit_bsz) if units else 0

        progress = tqdm(
            range(num_batches),
            desc=(
                f"Predicting via FBF{' + naturalize' if self.config.naturalize else ''} "
                f"({self.lm.name}, missing={len(missing_idxs)}, "
                f"units={len(units)}, samples={n_samples})"
            ),
            total=num_batches,
        )

        for batch_idx in progress:
            start = batch_idx * unit_bsz
            end = min(len(units), (batch_idx + 1) * unit_bsz)
            batch_units = units[start:end]

            # With gap-fill resume, every unit here corresponds to missing idxs.
            to_generate = batch_units

            triple_prompts_flat: List[str] = []
            unit_ranges: List[Tuple[Dict[str, Any], int, int]] = []

            cursor = 0

            for u in to_generate:
                triples = u.get("triples") or []
                prompts = [self._build_triple_prompt(t) for t in triples]

                triple_prompts_flat.extend(prompts)
                unit_ranges.append((u, cursor, cursor + len(prompts)))
                cursor += len(prompts)

            triple_outputs_flat: List[str] = []
            triple_usages_flat: List[Dict[str, int]] = []

            if triple_prompts_flat:
                for i in range(0, len(triple_prompts_flat), triple_bsz):
                    chunk = triple_prompts_flat[i: i + triple_bsz]

                    chunk_outputs = self.lm.generate(
                        chunk,
                        generation_kwargs=gen_kwargs_triple,
                    )
                    chunk_usages = self.lm.last_generation_usages

                    triple_outputs_flat.extend(chunk_outputs)

                    if chunk_usages:
                        if len(chunk_usages) != len(chunk_outputs):
                            logger.warning(
                                "Stage-1 usage/output length mismatch: got %s usage records "
                                "for %s outputs. Dropping usage for this chunk.",
                                len(chunk_usages),
                                len(chunk_outputs),
                            )
                            chunk_usages = []

                        triple_usages_flat.extend(chunk_usages)

                if len(triple_outputs_flat) != len(triple_prompts_flat):
                    raise AssertionError(
                        f"Stage-1 expected {len(triple_prompts_flat)} generations, "
                        f"got {len(triple_outputs_flat)}"
                    )

                if triple_usages_flat and len(triple_usages_flat) != len(triple_outputs_flat):
                    logger.warning(
                        "Stage-1 usage/output length mismatch after batching: got %s usage records "
                        "for %s outputs. Dropping Stage-1 usage for this batch.",
                        len(triple_usages_flat),
                        len(triple_outputs_flat),
                    )
                    triple_usages_flat = []

            # Regroup into per-unit facts.
            facts_by_pk: Dict[str, List[str]] = {}
            triple_usage_by_pk: Dict[str, List[Dict[str, int]]] = {}

            for u, a, b2 in unit_ranges:
                pk = u["prompt_key"]

                outs = triple_outputs_flat[a:b2] if triple_outputs_flat else []
                usage_slice = triple_usages_flat[a:b2] if triple_usages_flat else []

                facts: List[str] = []
                kept_usages: List[Dict[str, int]] = []

                for j, out in enumerate(outs):
                    out = out.strip() or ""

                    if out:
                        sent = out.strip().splitlines()[0].strip('"').strip()

                        if sent:
                            facts.append(sent)

                            if j < len(usage_slice):
                                kept_usages.append(usage_slice[j])

                facts_by_pk[pk] = facts
                triple_usage_by_pk[pk] = kept_usages

            pack_by_pk: Dict[str, Dict[str, Any]] = {}

            if not self.config.naturalize:
                for u in to_generate:
                    pk = u["prompt_key"]
                    facts = facts_by_pk.get(pk, [])
                    raw_usage = triple_usage_by_pk.get(pk, [])

                    prediction = " ".join(facts).strip()
                    usage = self._sum_usages(raw_usage)

                    pack_by_pk[pk] = {
                        "prediction": prediction,
                        "all_prediction": prediction,
                        "raw_samples": facts,
                        "usage": usage,
                        "all_usage": usage,
                        "raw_usage": raw_usage,
                    }

            else:
                # Units with <= 1 triple do not need naturalization.
                for u in to_generate:
                    if len(u.get("triples") or []) <= 1:
                        pk = u["prompt_key"]
                        facts = facts_by_pk.get(pk, [])
                        raw_usage = triple_usage_by_pk.get(pk, [])

                        prediction = facts[0].strip() if facts else ""
                        usage = self._sum_usages(raw_usage)

                        pack_by_pk[pk] = {
                            "prediction": prediction,
                            "all_prediction": prediction,
                            "raw_samples": facts,
                            "usage": usage,
                            "all_usage": usage,
                            "raw_usage": raw_usage,
                        }

                prompts_nat: List[str] = []
                nat_units: List[Dict[str, Any]] = []

                for u in to_generate:
                    if len(u.get("triples") or []) <= 1:
                        continue

                    pk = u["prompt_key"]
                    facts = facts_by_pk.get(pk, [])

                    prompts_nat.append(self._build_naturalize_prompt(facts))
                    nat_units.append(u)

                nat_outputs: List[str] = []
                nat_usages: List[Dict[str, int]] = []

                for i in range(0, len(prompts_nat), nat_bsz):
                    chunk = prompts_nat[i: i + nat_bsz]

                    chunk_outputs = self.lm.generate(
                        chunk,
                        generation_kwargs=gen_kwargs_nat,
                    )
                    chunk_usages = self.lm.last_generation_usages

                    nat_outputs.extend(chunk_outputs)

                    if chunk_usages:
                        if len(chunk_usages) != len(chunk_outputs):
                            logger.warning(
                                "Stage-2 usage/output length mismatch: got %s usage records "
                                "for %s outputs. Dropping usage for this chunk.",
                                len(chunk_usages),
                                len(chunk_outputs),
                            )
                            chunk_usages = []

                        nat_usages.extend(chunk_usages)

                if len(nat_outputs) != len(prompts_nat):
                    raise AssertionError(
                        f"Stage-2 expected {len(prompts_nat)} generations, "
                        f"got {len(nat_outputs)}"
                    )

                if nat_usages and len(nat_usages) != len(nat_outputs):
                    logger.warning(
                        "Stage-2 usage/output length mismatch after batching: got %s usage records "
                        "for %s outputs. Dropping Stage-2 usage for this batch.",
                        len(nat_usages),
                        len(nat_outputs),
                    )
                    nat_usages = []

                for j, (u, out) in enumerate(zip(nat_units, nat_outputs)):
                    pk = u["prompt_key"]

                    facts = facts_by_pk.get(pk, [])
                    raw_usage = triple_usage_by_pk.get(pk, [])

                    naturalize_usage = (
                        nat_usages[j]
                        if j < len(nat_usages)
                        else self._empty_usage_totals()
                    )

                    pred = self.extract_final(out)

                    ok, diag = gate_naturalization(
                        facts,
                        pred,
                        bleu_min=float(self.config.naturalize_bleu_min),
                    )

                    if not ok:
                        pred = " ".join(facts).strip()
                        out = pred

                    usage = self._sum_usages(raw_usage + [naturalize_usage])

                    pack_by_pk[pk] = {
                        "prediction": pred,
                        "all_prediction": out,
                        "raw_samples": facts,
                        "usage": usage,
                        "all_usage": naturalize_usage,
                        "raw_usage": raw_usage,
                        "bleu": diag,
                    }

            for u in to_generate:
                pk = u["prompt_key"]

                pack = pack_by_pk.get(
                    pk,
                    {
                        "prediction": "",
                        "all_prediction": "",
                        "raw_samples": [],
                    },
                )

                for m in u["members"]:
                    idx = m["idx"]

                    rec = {
                        "idx": idx,
                        "graph_key": u["graph_key"],
                        "prompt_key": pk,
                        "triples": u["triples"],
                        "reference": m["reference"],
                        "prediction": pack["prediction"],
                        "all_prediction": pack["all_prediction"],
                        "raw_samples": pack["raw_samples"],
                        "usage": pack.get("usage", self._empty_usage_totals()),
                        "all_usage": pack.get("all_usage", self._empty_usage_totals()),
                        "raw_usage": pack.get("raw_usage", []),
                    }

                    results.append(rec)
                    cache_by_idx[idx] = rec

                    if cache_file is not None:
                        self._append_to_cache(cache_file, rec)

        results.sort(key=lambda r: r["idx"])
        return results
