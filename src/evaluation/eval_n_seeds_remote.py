#!/usr/bin/env python
"""
Instead of spawning local GPU evaluators, it uploads each predictions file to the metrics API.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

NORM_METRIC_PATHS = [
    "BLEU.sentence.per_sample",
    "BERTScore.f1.per_sample",
    "BLEURT.per_sample",
    "AlignScore.per_sample",
    "FactSpotter.per_sample",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_output_dir", type=str, required=True)
    p.add_argument("--pred_relpath", type=str, required=True)
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--api-url", type=str, default="http://localhost:8125/v1/evaluate/file")
    p.add_argument("--metrics", type=str, default="bleu,bertscore,bleurt,alignscore,factspotter")
    p.add_argument("--per_seed_metrics_name", type=str, default="eval_metrics.json")
    p.add_argument("--aggregate_out", type=str, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--continue_on_fail", action="store_true")
    p.add_argument("--timeout-sec", type=int, default=7200)
    p.add_argument("--poll-sec", type=float, default=0.0)
    p.add_argument("--norm_ranges", type=str, default="data/metric_ranges.json")
    p.add_argument("--bertscore-lang", type=str, default="en")
    p.add_argument("--align-ckpt-path", type=str, default="/models/AlignScore-base.ckpt")
    p.add_argument("--bleurt-checkpoint", type=str, default="/models/BLEURT-20")
    p.add_argument("--align-model", type=str, default="roberta-base")
    p.add_argument("--align-batch-size", type=int, default=8)
    p.add_argument("--align-eval-mode", type=str, default="nli_sp")
    p.add_argument("--factspotter-model-name", type=str, default="Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
    p.add_argument("--factspotter-batch-size", type=int, default=16)
    p.add_argument("--factspotter-entailment-threshold", type=float, default=0.5)
    return p.parse_args()


def get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        cur = cur[part]
    return cur


def load_metric_ranges(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metric_ranges" in data:
        data = data["metric_ranges"]
    return {k: {"min": float(v["min"]), "max": float(v["max"])} for k, v in data.items()}


def add_normalized_avg_metric(metrics_json: Dict[str, Any], metric_ranges: Dict[str, Dict[str, float]]) -> None:
    per_metric_arrays: Dict[str, List[float]] = {}
    n_samples = None
    for path in NORM_METRIC_PATHS:
        arr = get_nested(metrics_json, path)
        if n_samples is None:
            n_samples = len(arr)
        elif len(arr) != n_samples:
            raise ValueError(f"Per-sample length mismatch for {path}")
        per_metric_arrays[path] = [float(x) for x in arr]

    normalized_avg_per_sample: List[float] = []
    for i in range(n_samples or 0):
        vals = []
        for path in NORM_METRIC_PATHS:
            min_v = metric_ranges[path]["min"]
            max_v = metric_ranges[path]["max"]
            vals.append((per_metric_arrays[path][i] - min_v) / (max_v - min_v))
        normalized_avg_per_sample.append(float(statistics.mean(vals)))

    metrics_json["NormalizedAvg"] = {
        "mean": float(statistics.mean(normalized_avg_per_sample)),
        "std": float(statistics.stdev(normalized_avg_per_sample)) if len(normalized_avg_per_sample) > 1 else 0.0,
        "min": float(min(normalized_avg_per_sample)),
        "max": float(max(normalized_avg_per_sample)),
        "per_sample": normalized_avg_per_sample,
        "n": len(normalized_avg_per_sample),
    }


def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, p))
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    return out


def should_aggregate(path: str) -> bool:
    if ".per_sample" in path:
        return False
    if path.endswith(".std") or path.endswith(".min") or path.endswith(".max"):
        return False
    return True


def aggregate_metrics(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    flat_by_seed = {seed: flatten_numeric(metrics) for seed, metrics in per_seed.items()}
    keys = sorted({k for fm in flat_by_seed.values() for k in fm if should_aggregate(k)})
    agg: Dict[str, Any] = {"n_seeds": len(per_seed), "metrics": {}}
    for k in keys:
        vals: List[Tuple[int, float]] = [(seed, fm[k]) for seed, fm in flat_by_seed.items() if k in fm]
        xs = [v for _, v in vals]
        agg["metrics"][k] = {
            "mean": float(statistics.mean(xs)),
            "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
            "n": len(xs),
            "by_seed": {str(seed): float(v) for seed, v in vals},
        }
    return agg


def post_file(args: argparse.Namespace, pred_path: Path) -> Dict[str, Any]:
    data = {
        "metrics": args.metrics,
        "bertscore_lang": args.bertscore_lang,
        "align_ckpt_path": args.align_ckpt_path,
        "bleurt_checkpoint": args.bleurt_checkpoint,
        "align_model": args.align_model,
        "align_batch_size": str(args.align_batch_size),
        "align_eval_mode": args.align_eval_mode,
        "factspotter_model_name": args.factspotter_model_name,
        "factspotter_batch_size": str(args.factspotter_batch_size),
        "factspotter_entailment_threshold": str(args.factspotter_entailment_threshold),
    }
    with pred_path.open("rb") as f:
        resp = requests.post(
            args.api_url,
            data=data,
            files={"file": (pred_path.name, f, "application/octet-stream")},
            timeout=args.timeout_sec,
        )
    resp.raise_for_status()
    payload = resp.json()
    return payload["metrics"]


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    base = Path(args.base_output_dir)
    per_seed_metrics: Dict[int, Dict[str, Any]] = {}
    failures: List[int] = []

    for seed in seeds:
        seed_dir = base / f"seed-{seed}"
        pred_path = seed_dir / args.pred_relpath
        out_path = pred_path.parent / args.per_seed_metrics_name
        if args.skip_existing and out_path.exists():
            per_seed_metrics[seed] = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[skip] seed={seed} already has {out_path}")
            continue

        try:
            print(f"[remote-eval] seed={seed} file={pred_path}")
            t0 = time.perf_counter()

            metrics = post_file(args, pred_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            per_seed_metrics[seed] = metrics

            elapsed = time.perf_counter() - t0
            print(f"[done] seed={seed} in {elapsed:.2f}s -> {out_path}")

            if args.poll_sec > 0:
                time.sleep(args.poll_sec)

        except Exception as exc:
            failures.append(seed)
            print(f"[fail] seed={seed}: {exc}")
            if not args.continue_on_fail:
                raise

    if args.norm_ranges is not None:
        metric_ranges = load_metric_ranges(Path(args.norm_ranges))
        for seed, metrics_json in per_seed_metrics.items():
            add_normalized_avg_metric(metrics_json, metric_ranges)

    agg = aggregate_metrics(per_seed_metrics)
    agg_out = Path(args.aggregate_out) if args.aggregate_out else (base / "aggregate_metrics.json")
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[all done] Wrote aggregated metrics to {agg_out}")

    if failures:
        raise SystemExit(f"Completed with failures for seeds: {failures}")


if __name__ == "__main__":
    main()
