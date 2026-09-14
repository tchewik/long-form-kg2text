#!/usr/bin/env python
"""
Evaluate KG→text predictions stored in JSON / JSONL.

Expected input format (per example, as produced by CoTBaseline.generate_for_split):

{
    "idx": int,
    "triples": [{"subject": "...", "predicate": "...", "object": "..."}, ...],
    "reference": "gold text",     # may be None
    "prediction": "model output", # required
    "raw_samples": [...],         # optional, ignored here
    ...
}

Supported metrics (if dependencies are installed):

- BLEU          (sacrebleu)
- BERTScore     (bert_score)
- AlignScore    (alignscore + checkpoint)
- FactSpotter   (Inria-CEDAR/FactSpotter-DeBERTaV3-* HF models)
"""

import os
os.environ["PYTORCH_JIT"] = "0"  # FactSpotter fix (Python 3.13)

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import gc
import torch

import numpy as np
import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(module)s.py: %(message)s",
)


def fmt_compact(x: float, sigfigs: int = 4) -> str:
    if not math.isfinite(x):
        return str(x)
    ax = abs(x)
    # Use scientific for very small or very large values
    if (ax != 0 and (ax < 1e-3 or ax >= 1e4)):
        return f"{x:.{sigfigs}e}"
    else:
        return f"{x:.{sigfigs}f}"



def free_torch_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """
    Load predictions from JSON or JSONL.

    Accepts:
    - JSONL: one JSON object per line
    - JSON: either a list[dict] or {"predictions": list[dict]}
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        logger.info("Loaded %d records from JSONL %s", len(records), path)
        return records

    # JSON
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        logger.info("Loaded %d records from JSON %s", len(data), path)
        return data
    if isinstance(data, dict) and "predictions" in data:
        preds = data["predictions"]
        logger.info("Loaded %d records from JSON %s (key='predictions')", len(preds), path)
        return preds

    raise ValueError(f"Unrecognized JSON structure in {path}")


def format_triples_as_text(triples: List[Dict[str, str]]) -> str:
    """
    Turn a list of KG triples into a compact string representation
    (same format as in your CoT baseline).
    """
    bits: List[str] = []
    for t in triples:
        s = (t.get("subject") or "").strip()
        p = (t.get("predicate") or "").strip()
        o = (t.get("object") or "").strip()
        bits.append(f"({s}) --[{p}]--> ({o})")
    return " ; ".join(bits)


@dataclass
class EvalConfig:
    input_path: Path
    output_path: Optional[Path] = None
    save_per_example: Optional[Path] = None

    compute_bleu: bool = True
    compute_bertscore: bool = True
    compute_alignscore: bool = True
    compute_factspotter: bool = True
    compute_bleurt: bool = True
    compute_infolm: bool = True

    # BERTScore
    bertscore_lang: str = "en"

    # BLEURT
    bleurt_checkpoint: Optional[str] = None

    # InfoLM
    infolm_model_name: str = "bert-base-uncased"
    infolm_information_measure: str = "kl_divergence"
    infolm_idf: bool = True
    infolm_temperature: float = 0.25
    infolm_alpha: Optional[float] = None
    infolm_beta: Optional[float] = None
    infolm_batch_size: int = 64
    infolm_max_length: Optional[int] = None
    infolm_device: Optional[str] = None

    # AlignScore
    align_model: str = "roberta-base"
    align_ckpt_path: Optional[Path] = None
    align_batch_size: int = 8
    align_device: str = "cuda:0"
    align_eval_mode: str = "nli_sp"

    # FactSpotter settings (HF DeBERTaV3 implementation)
    factspotter_model_name: str = "Inria-CEDAR/FactSpotter-DeBERTaV3-Base"
    factspotter_device: str = "cuda"
    factspotter_batch_size: int = 32
    factspotter_entailment_threshold: float = 0.499  # prob(entailment) ≥ threshold


def compute_bleu(references: List[str], predictions: List[str]) -> Dict[str, Any]:
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu is not installed; skipping BLEU.")
        return {}

    corpus_bleu = sacrebleu.corpus_bleu(predictions, [references])

    sent_scores = [
        float(sacrebleu.sentence_bleu(pred, [ref]).score)
        for pred, ref in zip(predictions, references)
    ]

    return {
        "BLEU": {
            "micro-avg": float(corpus_bleu.score),
            "sentence": {
                "mean": float(statistics.mean(sent_scores)) if sent_scores else 0.0,
                "std": float(statistics.stdev(sent_scores)) if len(sent_scores) > 1 else 0.0,
                "min": float(min(sent_scores)) if sent_scores else 0.0,
                "max": float(max(sent_scores)) if sent_scores else 0.0,
                "per_sample": sent_scores,
            },
        }
    }


def compute_bertscore(
    references: List[str],
    predictions: List[str],
    lang: str = "en",
) -> Dict[str, Any]:
    try:
        from bert_score import score as bert_score  # type: ignore
        import torch
    except ImportError:
        logger.warning("bert-score is not installed; skipping BERTScore.")
        return {}

    with torch.no_grad():
        P, R, F1 = bert_score(predictions, references, lang=lang)

    # Convert to plain Python lists
    p_list = P.tolist()
    r_list = R.tolist()
    f1_list = F1.tolist()

    def stats(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {"mean": 0.0, "std": 0.0, "per_sample": []}
        return {
            "mean": float(statistics.mean(xs)),
            "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
            "per_sample": [float(x) for x in xs],
        }

    return {
        "BERTScore": {
            "precision": stats(p_list),
            "recall": stats(r_list),
            "f1": stats(f1_list),
        }
    }


def compute_bleurt(
    references: List[str],
    predictions: List[str],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    BLEURT: supervised, learned evaluation metric.

    Requires:
        pip install bleurt
        # and a compatible checkpoint, e.g. downloaded from:
        # https://github.com/google-research/bleurt
    """
    if not cfg.bleurt_checkpoint:
        logger.warning("No BLEURT checkpoint provided; skipping BLEURT.")
        return {}

    try:
        from bleurt import score as bleurt_score  # type: ignore
    except ImportError:
        logger.warning("bleurt is not installed; skipping BLEURT.")
        return {}

    logger.info("Loading BLEURT checkpoint from %s", cfg.bleurt_checkpoint)
    scorer = bleurt_score.BleurtScorer(cfg.bleurt_checkpoint)

    logger.info("Computing BLEURT on %d examples...", len(predictions))
    scores = scorer.score(
        references=references,
        candidates=predictions,
    )  # list[float]

    if not scores:
        return {}

    scores = [float(s) for s in scores]
    mean = float(statistics.mean(scores))
    std = float(statistics.stdev(scores)) if len(scores) > 1 else 0.0
    min_v = float(min(scores))
    max_v = float(max(scores))

    return {
        "BLEURT": {
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "per_sample": scores,
        }
    }


def compute_infolm(
    references: List[str],
    predictions: List[str],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    InfoLM: information-theoretic distance/divergence between
    prediction and reference distributions computed via a masked LM.

    Implemented via torchmetrics.text.infolm.InfoLM.
    """
    try:
        import torch
        from torchmetrics.text.infolm import InfoLM  # type: ignore
    except ImportError:
        logger.warning("torchmetrics is not installed; skipping InfoLM.")
        return {}

    logger.info(
        "Loading InfoLM with model=%s, measure=%s",
        cfg.infolm_model_name,
        cfg.infolm_information_measure,
    )

    metric = InfoLM(
        model_name_or_path=cfg.infolm_model_name,
        temperature=cfg.infolm_temperature,
        information_measure=cfg.infolm_information_measure,
        idf=cfg.infolm_idf,
        alpha=cfg.infolm_alpha,
        beta=cfg.infolm_beta,
        device=cfg.infolm_device,
        max_length=cfg.infolm_max_length,
        batch_size=cfg.infolm_batch_size,
        return_sentence_level_score=True,
    )

    logger.info("Computing InfoLM on %d examples...", len(predictions))
    corpus_score, sentence_scores = metric(preds=predictions, target=references)

    # Convert corpus score to float
    if hasattr(corpus_score, "item"):
        corpus_score_f = float(corpus_score.item())
    else:
        corpus_score_f = float(corpus_score)

    # Convert sentence-level scores to a Python list[float]
    per_sample: List[float] = []

    if sentence_scores is None:
        per_sample = []
    elif isinstance(sentence_scores, (list, tuple)):
        per_sample = [
            float(s.item()) if hasattr(s, "item") else float(s)
            for s in sentence_scores
        ]
    elif torch.is_tensor(sentence_scores):
        if sentence_scores.ndim == 0:
            per_sample = [float(sentence_scores.item())]
        else:
            per_sample = [float(x) for x in sentence_scores.view(-1).tolist()]
    else:
        per_sample = [float(sentence_scores)]

    if per_sample:
        mean = float(statistics.mean(per_sample))
        std = float(statistics.stdev(per_sample)) if len(per_sample) > 1 else 0.0
        min_v = float(min(per_sample))
        max_v = float(max(per_sample))
    else:
        mean = corpus_score_f
        std = 0.0
        min_v = corpus_score_f
        max_v = corpus_score_f

    return {
        "InfoLM": {
            "corpus": corpus_score_f,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "per_sample": per_sample,
        }
    }


def compute_alignscore(
    contexts: List[str],
    claims: List[str],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    if not cfg.align_ckpt_path:
        logger.warning("No AlignScore ckpt path provided; skipping AlignScore.")
        return {}

    try:
        from alignscore import AlignScore  # type: ignore
        import numpy as np
    except ImportError:
        logger.warning("alignscore is not installed; skipping AlignScore.")
        return {}

    logger.info("Loading AlignScore (model=%s, ckpt=%s)", cfg.align_model, cfg.align_ckpt_path)
    scorer = AlignScore(
        model=cfg.align_model,
        batch_size=cfg.align_batch_size,
        device=cfg.align_device,
        ckpt_path=str(cfg.align_ckpt_path),
        evaluation_mode=cfg.align_eval_mode,
    )

    logger.info("Computing AlignScore on %d examples...", len(contexts))
    scores = scorer.score(contexts=contexts, claims=claims)  # type: ignore

    # Ensure Python list[float]
    if not isinstance(scores, list):
        scores = list(map(float, scores))
    else:
        scores = [float(s) for s in scores]

    if not scores:
        return {}

    mean = float(statistics.mean(scores))
    std = float(statistics.stdev(scores)) if len(scores) > 1 else 0.0
    min_v = float(min(scores))
    max_v = float(max(scores))

    return {
        "AlignScore": {
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "per_sample": scores,
        }
    }


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_factspotter(
    triples_list: List[List[Dict[str, str]]],
    predictions: List[str],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    We approximate the FactSpotter metric as:

        For each example:
            recall = (# triples with entailment(text, triple) == True) / (# triples)

        Dataset score = mean(recall over examples with at least 1 triple).
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning("transformers/torch not installed; skipping FactSpotter.")
        return {}

    device = torch.device(
        cfg.factspotter_device if torch.cuda.is_available() or cfg.factspotter_device == "cpu" else "cpu"
    )

    logger.info(
        "Loading FactSpotter model %s on device %s",
        cfg.factspotter_model_name,
        device,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.factspotter_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.factspotter_model_name)
    model.to(device)
    model.eval()

    def sentence_cls_score(pairs: List[Tuple[str, str]]) -> torch.Tensor:
        enc = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1)
        return probs

    per_example_recalls: List[float] = []

    logger.info("Computing FactSpotter scores...")
    all_pairs: List[Tuple[str, str]] = []
    example_offsets: List[int] = [0]
    for triples, text in zip(triples_list, predictions):
        if not triples:
            example_offsets.append(example_offsets[-1])
            continue
        for t in triples:
            triple_str = f"{t.get('subject','')} | {t.get('predicate','')} | {t.get('object','')}"
            all_pairs.append((text, triple_str))
        example_offsets.append(example_offsets[-1] + len(triples))

    if not all_pairs:
        logger.warning("No triples found; skipping FactSpotter.")
        return {}

    entail_probs: List[float] = []
    for batch_pairs in tqdm.tqdm(
        list(batched(all_pairs, cfg.factspotter_batch_size)),
        desc="FactSpotter",
    ):
        probs = sentence_cls_score(batch_pairs)  # [batch_size, 3]
        entail_probs.extend(probs[:, 0].tolist())  # class 0 = entailment

    for i in range(len(triples_list)):
        start = example_offsets[i]
        end = example_offsets[i + 1]
        if end == start:
            continue
        example_probs = entail_probs[start:end]
        n_entailed = sum(p >= cfg.factspotter_entailment_threshold for p in example_probs)
        recall = n_entailed / float(len(example_probs))
        per_example_recalls.append(float(recall))

    if not per_example_recalls:
        logger.warning("No examples with triples for FactSpotter.")
        return {}

    mean = float(statistics.mean(per_example_recalls))
    std = float(statistics.stdev(per_example_recalls)) if len(per_example_recalls) > 1 else 0.0
    min_v = float(min(per_example_recalls))
    max_v = float(max(per_example_recalls))

    return {
        "FactSpotter": {
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "per_sample": per_example_recalls,
        }
    }


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    records = load_predictions(cfg.input_path)

    # Extract fields
    predictions: List[str] = []
    references: List[str] = []
    triples_list: List[List[Dict[str, str]]] = []
    idxs: List[int] = []

    for rec in records:
        ref = rec.get("reference")

        if not ref:
            # skip broken rows
            continue

        pred = rec.get("prediction")
        if not pred:
            pred = "Don't know"

        triples = rec.get("triples") or rec.get("triples_parsed") or []
        idx = rec.get("idx", len(idxs))

        predictions.append(str(pred))
        references.append(str(ref))
        triples_list.append(list(triples))
        idxs.append(idx)

    if not predictions:
        raise ValueError("No valid predictions found in input file.")

    logger.info(
        "Loaded %d examples with predictions; %d have non-empty references.",
        len(predictions),
        sum(bool(r.strip()) for r in references),
    )

    overall: Dict[str, Any] = {}

    # BLEU
    if cfg.compute_bleu and any(r.strip() for r in references):
        bleu_scores = compute_bleu(references, predictions)
        overall.update(bleu_scores)

    # BERTScore
    if cfg.compute_bertscore and any(r.strip() for r in references):
        bert_scores = compute_bertscore(references, predictions, lang=cfg.bertscore_lang)
        overall.update(bert_scores)
        free_torch_gpu()

    # BLEURT
    if cfg.compute_bleurt and any(r.strip() for r in references):
        bleurt_scores = compute_bleurt(references, predictions, cfg)
        overall.update(bleurt_scores)
        free_torch_gpu()

    # InfoLM
    if cfg.compute_infolm and any(r.strip() for r in references):
        infolm_stats = compute_infolm(references, predictions, cfg)
        overall.update(infolm_stats)
        free_torch_gpu()

    # AlignScore (context = KG triples as text, claim = prediction)
    if cfg.compute_alignscore:
        contexts = [format_triples_as_text(t) for t in triples_list]
        align_stats = compute_alignscore(contexts, predictions, cfg)
        overall.update(align_stats)
        free_torch_gpu()

    # FactSpotter (graph→text factual faithfulness)
    if cfg.compute_factspotter:
        fs_stats = compute_factspotter(triples_list, predictions, cfg)
        overall.update(fs_stats)

    logger.info("Overall metrics:")
    for k, v in overall.items():
        if k == 'BLEU':
            logger.info("  %s = %s", k, fmt_compact(v['micro-avg']))
        elif k == 'BERTScore':
            logger.info("  %s = %s", k, fmt_compact(v['f1']['mean'] * 100))
        elif k == 'BLEURT':
            logger.info("  %s = %s", k, fmt_compact(v['mean']))
        elif k == 'InfoLM':
            logger.info("  %s = %s", k, fmt_compact(v['mean']))
        elif k == 'AlignScore':
            logger.info("  %s = %s", k, fmt_compact(v['mean'] * 100))
        elif k == 'FactSpotter':
            logger.info("  %s = %s", k, fmt_compact(v['mean'] * 100))
        else:
            print(f'{k = }: {v = }')
            logger.info("  %s = %.2f", k, np.round(v, 2))

    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.output_path.open("w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)
        logger.info("Saved overall metrics to %s", cfg.output_path)

    return overall


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate KG→text predictions JSON.")
    p.add_argument("input_path", type=Path, help="Path to JSON or JSONL predictions file.")
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write overall metrics JSON (optional).",
    )
    p.add_argument(
        "--save-per-example",
        type=Path,
        default=None,
        help="Optional path to save per-example records as JSONL.",
    )

    # Metric toggles
    p.add_argument("--no-bleu", action="store_true", help="Disable BLEU.")
    p.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore.")
    p.add_argument("--no-alignscore", action="store_true", help="Disable AlignScore.")
    p.add_argument("--no-factspotter", action="store_true", help="Disable FactSpotter.")
    p.add_argument("--no-bleurt", action="store_true", help="Disable BLEURT.")
    p.add_argument("--no-infolm", action="store_true", help="Disable InfoLM.")

    # BERTScore
    p.add_argument("--bertscore-lang", type=str, default="en")

    # BLEURT
    p.add_argument(
        "--bleurt-checkpoint",
        type=str,
        default='models/BLEURT-20',
        help=(
            "Path/name of BLEURT checkpoint for bleurt.BleurtScorer "
            "(if not set, BLEURT is skipped)."
        ),
    )

    # InfoLM
    p.add_argument(
        "--infolm-model-name",
        type=str,
        default="bert-base-uncased",
        help="HF model name/path for InfoLM's masked language model.",
    )
    p.add_argument(
        "--infolm-information-measure",
        type=str,
        default="kl_divergence",
        choices=[
            "kl_divergence",
            "alpha_divergence",
            "beta_divergence",
            "ab_divergence",
            "renyi_divergence",
            "l1_distance",
            "l2_distance",
            "l_infinity_distance",
            "fisher_rao_distance",
        ],
        help="Information measure used by InfoLM.",
    )
    p.add_argument(
        "--infolm-no-idf",
        action="store_false",
        dest="infolm_idf",
        help="Disable IDF weighting in InfoLM (enabled by default).",
    )
    p.add_argument(
        "--infolm-temperature",
        type=float,
        default=0.25,
        help="Temperature parameter for InfoLM.",
    )
    p.add_argument(
        "--infolm-alpha",
        type=float,
        default=None,
        help="Alpha parameter for alpha/AB/Rényi divergences in InfoLM.",
    )
    p.add_argument(
        "--infolm-beta",
        type=float,
        default=None,
        help="Beta parameter for beta/AB divergences in InfoLM.",
    )
    p.add_argument(
        "--infolm-batch-size",
        type=int,
        default=64,
        help="Batch size for InfoLM.",
    )
    p.add_argument(
        "--infolm-max-length",
        type=int,
        default=None,
        help="Maximum sequence length for InfoLM (tokens).",
    )
    p.add_argument(
        "--infolm-device",
        type=str,
        default=None,
        help="Device for InfoLM, e.g. 'cuda:0' or 'cpu' (default: torchmetrics default).",
    )

    # AlignScore options
    p.add_argument("--align-ckpt-path", type=Path, default=Path("models/AlignScore-base.ckpt"),
                   help="Path to AlignScore checkpoint (.ckpt).")
    p.add_argument("--align-model", type=str, default="roberta-base")
    p.add_argument("--align-batch-size", type=int, default=8)
    p.add_argument("--align-device", type=str, default="cuda:0")
    p.add_argument(
        "--align-eval-mode",
        type=str,
        default="nli_sp",
        choices=["nli_sp", "nli", "bin_sp", "bin"],
        help="AlignScore evaluation mode (see AlignScore README).",
    )

    # FactSpotter options
    p.add_argument(
        "--factspotter-model-name",
        type=str,
        default="Inria-CEDAR/FactSpotter-DeBERTaV3-Base",
    )
    p.add_argument("--factspotter-device", type=str, default="cuda")
    p.add_argument("--factspotter-batch-size", type=int, default=16)
    p.add_argument("--factspotter-entailment-threshold", type=float, default=0.5)

    args = p.parse_args()

    cfg = EvalConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        save_per_example=args.save_per_example,
        compute_bleu=not args.no_bleu,
        compute_bleurt=not args.no_bleurt,
        compute_infolm=False,
        compute_bertscore=not args.no_bertscore,
        compute_alignscore=not args.no_alignscore,
        compute_factspotter=not args.no_factspotter,

        bertscore_lang=args.bertscore_lang,

        bleurt_checkpoint=args.bleurt_checkpoint,

        infolm_model_name=args.infolm_model_name,
        infolm_information_measure=args.infolm_information_measure,
        infolm_idf=getattr(args, "infolm_idf", True),
        infolm_temperature=args.infolm_temperature,
        infolm_alpha=args.infolm_alpha,
        infolm_beta=args.infolm_beta,
        infolm_batch_size=args.infolm_batch_size,
        infolm_max_length=args.infolm_max_length,
        infolm_device=args.infolm_device,

        align_model=args.align_model,
        align_ckpt_path=args.align_ckpt_path,
        align_batch_size=args.align_batch_size,
        align_device=args.align_device,
        align_eval_mode=args.align_eval_mode,

        factspotter_model_name=args.factspotter_model_name,
        factspotter_device=args.factspotter_device,
        factspotter_batch_size=args.factspotter_batch_size,
        factspotter_entailment_threshold=args.factspotter_entailment_threshold,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    evaluate(cfg)


if __name__ == "__main__":
    main()
