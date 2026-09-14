import hashlib
import json
import logging
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from src.data.kgdataset import KGDataset
from src.models.language_model import LanguageModel
from src.rerankers.factspotter_reranker import FactSpotterReranker
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

PromptStyle = Literal["wikipedia", "paper_abstract"]


def _canon_triple(t: Dict[str, str]) -> Tuple[str, str, str]:
    s = (t.get("subject") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = (t.get("object") or "").strip()
    return (s, p, o)


def format_triples_as_text(triples: List[Dict[str, str]]) -> str:
    bits: List[str] = []
    for t in triples or []:
        s, p, o = _canon_triple(t)
        bits.append(f"({s}) --[{p}]--> ({o})")
    return " ; ".join(bits)


def graph_key(triples: List[Dict[str, str]]) -> str:
    canon = sorted(_canon_triple(t) for t in (triples or []))
    payload = json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PipelineConfig:
    generation_batch_size: int = 1  # batches over "units" (unique graphs if dedup, else per-example)
    num_samples_per_example: int = 1
    max_new_tokens: int = 128
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    stop_strings: Optional[List[str]] = None

    prompt_style: PromptStyle = None
    system_template_wikipedia: str = None
    user_template_wikipedia: str = None
    system_template_paper_abstract: str = None
    user_template_paper_abstract: str = None

    aggregation: str = "first"  # "first", "majority", "factspotter"
    factspotter_model_name: str = "Inria-CEDAR/FactSpotter-DeBERTaV3-Base"
    factspotter_device: str = "cuda"
    factspotter_batch_size: int = 32
    factspotter_entailment_threshold: float = 0.5

    num_shots: int = 0
    shot_source_split: str = "train"
    shot_selection: str = "last"  # "first", "last", "random"
    few_shot_examples: Optional[List[Dict[str, Any]]] = None  # [{"triples": [...], "answer": "..."}]
    shot_header_template: str = "Example {i}:"
    shot_final_prefix: str = "Final:"

    cache_use_prompt_key: bool = True  # recommended: True (robust when dedup/few-shot changes)


class BasePipeline(ABC):
    """
    Superclass for KG->text pipelines.

    Responsibilities:
      - Few-shot selection + prefix construction
      - Prompt-keying (graph+templates+fewshot) for cache correctness
      - Optional dedup-by-graph batching and fan-out to members
      - Sampling k times per unit and aggregation (first/majority/factspotter)
      - JSONL cache load/append

    Subclasses implement:
      - build_prompt(triples) -> str
      - extract_final(text) -> str
      - prompt_fingerprint() -> Dict[str,Any]  (templates/style knobs that affect prompt_key)
    """

    def __init__(
            self,
            lm: LanguageModel,
            dataset: KGDataset,
            config: Optional[PipelineConfig] = None,
            *,
            dedup_by_graph: bool = True,
    ):
        self.lm = lm
        self.dataset = dataset
        self.data = dataset.data
        self.config = config or PipelineConfig()
        self.dedup_by_graph = bool(dedup_by_graph)

        self._factspotter: Optional[FactSpotterReranker] = None
        try:
            logger.info(
                f"Initializing {self.__class__.__name__} with config: "
                f"{json.dumps(asdict(self.config), ensure_ascii=False, indent=2, default=str)}"
            )
        except Exception:
            logger.info(f"Initializing {self.__class__.__name__} with config (non-serializable)")

    @abstractmethod
    def build_prompt(self, triples: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_final(self, text: str) -> str:
        raise NotImplementedError

    def prompt_fingerprint(self) -> Dict[str, Any]:
        """
        Anything that changes the prompt semantics should be returned here so prompt_key changes.
        Subclasses should override to include templates, prompt_style, etc.
        """
        return {"pipeline": self.__class__.__name__}

    def _get_factspotter(self) -> Optional[FactSpotterReranker]:
        if self._factspotter is not None:
            return self._factspotter
        try:
            self._factspotter = FactSpotterReranker(
                model_name=self.config.factspotter_model_name,
                device=self.config.factspotter_device,
                batch_size=self.config.factspotter_batch_size,
                entailment_threshold=self.config.factspotter_entailment_threshold,
            )
            return self._factspotter
        except Exception as e:
            logger.warning(f"FactSpotter not available ({e}); falling back to 'first'.")
            self._factspotter = None
            return None

    def _select_few_shot_examples(self) -> List[Dict[str, Any]]:
        k = int(getattr(self.config, "num_shots", 0) or 0)
        if k <= 0:
            return []

        if self.config.few_shot_examples:
            exs = self.config.few_shot_examples[:k]
            normalized: List[Dict[str, Any]] = []
            for ex in exs:
                triples = ex.get("triples") or ex.get("triples_parsed") or []
                answer = (ex.get("answer") or ex.get("text") or "").strip()
                normalized.append({"triples": triples, "answer": answer})
            return normalized

        pool = self.dataset.load_train()
        if not pool:
            logger.warning(
                f"No data found for train split "
                f"to build few-shot examples; falling back to zero-shot."
            )
            return []

        if len(pool) <= k or self.config.shot_selection == "first":
            chosen = pool[:k]
        elif self.config.shot_selection == "last":
            chosen = pool[-k:]
        elif self.config.shot_selection == "random":
            chosen = random.sample(pool, k)
        else:
            logger.warning(f"Unknown shot_selection '{self.config.shot_selection}', falling back to 'first'.")
            chosen = pool[:k]

        normalized: List[Dict[str, Any]] = []
        for ex in chosen:
            triples = ex.get("triples_parsed") or []
            answer = (ex.get("text") or "").strip()
            normalized.append({"triples": triples, "answer": answer})
        return normalized

    def build_few_shot_prefix(self) -> str:
        examples = self._select_few_shot_examples()
        if not examples:
            return ""

        blocks: List[str] = []
        for i, ex in enumerate(examples, 1):
            triples_text = format_triples_as_text(ex["triples"])
            answer = (ex["answer"] or "").strip()
            header = self.config.shot_header_template.format(i=i)
            final_prefix = self.config.shot_final_prefix
            blocks.append(
                f"{header}\n"
                f"Triples:\n{triples_text}\n"
                f"{final_prefix} {answer}\n"
            )
        return "\n".join(blocks) + "\n"

    def prompt_key(self, triples: List[Dict[str, str]]) -> str:
        """
        Cache key for *prompt semantics* (graph + templates/style + few-shot prefix).
        """
        gk = graph_key(triples)
        payload = {
            "graph_key": gk,
            "fingerprint": self.prompt_fingerprint(),
            "few_shot_prefix": self.build_few_shot_prefix(),
        }
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _load_cache(self, cache_path: Path) -> Dict[int, Dict[str, Any]]:
        if not cache_path.exists():
            return {}
        cache: Dict[int, Dict[str, Any]] = {}
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[int(rec["idx"])] = rec
                except Exception as e:
                    logger.warning(f"Failed to parse cached line: {e}")
        logger.info(f"Loaded {len(cache)} cached examples from {cache_path}")
        return cache

    def _append_to_cache(self, cache_path: Path, record: Dict[str, Any]) -> None:
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _empty_usage_totals() -> Dict[str, int]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    @classmethod
    def _sum_usages(cls, usages: List[Dict[str, Any]]) -> Dict[str, int]:
        total = cls._empty_usage_totals()

        for usage in usages or []:
            if not isinstance(usage, dict):
                continue

            total["requests"] += int(usage.get("requests", 1) or 1)
            total["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            total["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            total["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            total["cached_tokens"] += int(usage.get("cached_tokens", 0) or 0)
            total["reasoning_tokens"] += int(usage.get("reasoning_tokens", 0) or 0)

        return total

    def aggregate(self, answers: List[str], triples: Optional[List[Dict[str, str]]] = None) -> str:
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]

        agg = (self.config.aggregation or "first").lower()

        if agg == "majority":
            counter = Counter(answers)
            return counter.most_common(1)[0][0]

        if agg == "factspotter":
            if not triples:
                return answers[0]
            fs = self._get_factspotter()
            if fs is None:
                return answers[0]
            best_i = 0
            best_key = (-1.0, -1.0)  # (recall, mean_p)
            for i, a in enumerate(answers):
                recall, mean_p = fs.score_text(triples, a)
                key = (recall, mean_p)
                if key > best_key:
                    best_key = key
                    best_i = i
            return answers[best_i]

        return answers[0]  # "first"

    def _make_generation_kwargs(self) -> Dict[str, Any]:
        kw = dict(
            max_new_tokens=int(self.config.max_new_tokens),
            do_sample=bool(self.config.do_sample),
            temperature=float(self.config.temperature),
            top_p=float(self.config.top_p),
            tokenizer=getattr(self.lm, "tokenizer", None),
        )
        if self.config.stop_strings:
            kw["stop_strings"] = list(self.config.stop_strings)
        return kw

    def generate_for_split(
            self,
            split: str = "test",
            limit: Optional[int] = None,
            cache_path: Optional[str] = None,
            resume: bool = True,
            *args,
    ) -> List[Dict[str, Any]]:
        assert split in self.data, f"Unknown split: {split}"

        if split == "test":
            samples = self.dataset.load_test_500()
        elif split == "train":
            samples = self.dataset.load_train()
        else:
            samples = self.data[split]

        if limit is not None and limit > 0:
            samples = samples[:limit]

        cache_file: Optional[Path] = None
        cache_by_idx: Dict[int, Dict[str, Any]] = {}
        if cache_path is not None:
            cache_file = Path(cache_path)
            if resume:
                cache_by_idx = self._load_cache(cache_file)

        # Build units: either per-example or per-unique-graph
        units: List[Dict[str, Any]] = []
        if self.dedup_by_graph:
            groups: Dict[str, Dict[str, Any]] = {}
            for idx, sample in enumerate(samples):
                triples = sample.get("triples_parsed") or []
                gk = graph_key(triples)
                if gk not in groups:
                    groups[gk] = {"graph_key": gk, "triples": triples, "members": []}
                groups[gk]["members"].append({"idx": idx, "reference": sample.get("text"), "sample": sample})
            units = list(groups.values())
        else:
            for idx, sample in enumerate(samples):
                triples = sample.get("triples_parsed") or []
                units.append(
                    {
                        "graph_key": graph_key(triples),
                        "triples": triples,
                        "members": [{"idx": idx, "reference": sample.get("text"), "sample": sample}],
                    }
                )

        # Build prompt + prompt_key per unit
        for u in units:
            u["prompt"] = self.build_prompt(u["triples"])
            u["prompt_key"] = self.prompt_key(u["triples"])

        # Sort by prompt length to reduce padding
        tok = self.lm.tokenizer(
            [u["prompt"] for u in units],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        lens = [len(ids) for ids in tok["input_ids"]]
        for u, L in zip(units, lens):
            u["prompt_len"] = L
        units.sort(key=lambda u: u["prompt_len"])

        # Build prompt-key cache view from idx-cache
        cached_by_idx_graph: Dict[int, Dict[str, Any]] = {}

        if cache_by_idx:
            for idx, rec in cache_by_idx.items():
                cached_gk = rec.get("graph_key")

                # Backward compatibility for older cache records that may not have graph_key.
                if not cached_gk:
                    try:
                        cached_gk = graph_key(rec.get("triples") or [])
                    except Exception:
                        cached_gk = None

                if cached_gk:
                    rec["graph_key"] = cached_gk
                    cached_by_idx_graph[int(idx)] = rec

        gen_kwargs = self._make_generation_kwargs()
        bsz = max(1, int(self.config.generation_batch_size))
        k = max(1, int(self.config.num_samples_per_example))

        num_batches = ceil(len(units) / bsz)
        progress = tqdm(
            range(num_batches),
            desc=f"Predicting via {self.lm.name} ({self.__class__.__name__}, units={len(units)}/{len(samples)})",
            total=num_batches,
        )

        results: List[Dict[str, Any]] = []

        for b in progress:
            start = b * bsz
            end = min(len(units), (b + 1) * bsz)
            batch_units = units[start:end]

            to_generate: List[Dict[str, Any]] = []

            for u in batch_units:
                uncached_members: List[Dict[str, Any]] = []

                for m in u["members"]:
                    idx = int(m["idx"])
                    cached = cached_by_idx_graph.get(idx)

                    if cached is not None and cached.get("graph_key") == u["graph_key"]:
                        # Cache hit: same dataset index and same triples.
                        results.append(cached)
                    else:
                        # Cache miss: either idx is absent or triples changed.
                        uncached_members.append(m)

                if uncached_members:
                    u2 = dict(u)
                    u2["members"] = uncached_members
                    to_generate.append(u2)

            if not to_generate:
                continue

            prompts_flat: List[str] = []
            for u in to_generate:
                prompts_flat.extend([u["prompt"]] * k)

            outputs_flat: List[str] = self.lm.generate(
                prompts_flat,
                generation_kwargs=gen_kwargs,
            )
            usages_flat: List[Dict[str, int]] = self.lm.last_generation_usages

            if usages_flat and len(usages_flat) != len(outputs_flat):
                logger.warning(
                    "Usage/output length mismatch: got %s usage records for %s outputs. "
                    "Dropping usage for this batch.",
                    len(usages_flat),
                    len(outputs_flat),
                )
                usages_flat = []

            expected = len(to_generate) * k
            assert len(outputs_flat) == expected, f"Expected {expected} generations, got {len(outputs_flat)}"

            for i, u in enumerate(to_generate):
                pk = u["prompt_key"]
                group_out = outputs_flat[i * k: (i + 1) * k]
                group_usage = usages_flat[i * k: (i + 1) * k] if usages_flat else []

                usage = self._sum_usages(group_usage)
                all_usage = group_usage[0] if group_usage else self._empty_usage_totals()

                answers = [self.extract_final(o) for o in group_out]
                aggregated = self.aggregate(answers, triples=u["triples"])

                for m in u["members"]:
                    idx = int(m["idx"])

                    rec = {
                        "idx": idx,
                        "graph_key": u["graph_key"],
                        "prompt_key": pk,  # kept only as metadata/debugging
                        "triples": u["triples"],
                        "reference": m["reference"],
                        "prediction": aggregated,
                        "all_prediction": group_out[0] if group_out else aggregated,
                        "raw_samples": answers,
                        "usage": usage,
                        "all_usage": all_usage,
                        "raw_usage": group_usage,
                    }

                    results.append(rec)
                    cached_by_idx_graph[idx] = rec

                    if cache_file is not None:
                        self._append_to_cache(cache_file, rec)

        results.sort(key=lambda r: r["idx"])
        return results
