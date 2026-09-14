import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


NORM_METRIC_PATHS = [
    "BLEU.sentence.per_sample",
    "BERTScore.f1.per_sample",
    "BLEURT.per_sample",
    "AlignScore.per_sample",
    "FactSpotter.per_sample",
]

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--seeds", type=str, required=True, help="Comma-separated: 42,43,44,...")
    p.add_argument("--base_output_dir", type=str, required=True,
                   help="Base dir containing seed-* subdirs produced by run_experiments.py")

    # Where to find predictions inside each seed dir (relative path).
    # Examples from your runner:
    # - CoT:      cot/cot_predictions.jsonl
    # - Finetune: finetune/test_predictions.jsonl
    p.add_argument("--pred_relpath", type=str, required=True,
                   help="Relative path from seed dir to predictions JSON/JSONL")

    # Multi-GPU queue control
    p.add_argument("--gpu_num", type=int, default=1, help="How many GPUs to schedule across.")
    p.add_argument("--gpus_per_run", type=int, default=1, help="How many GPUs each eval process sees.")

    # Evaluator entrypoint
    p.add_argument("--python", type=str, default="python")
    p.add_argument("--script", type=str, default="src/evaluation/evaluate_jsonl_lowmemory.py")

    # Output locations
    p.add_argument("--per_seed_metrics_name", type=str, default="eval_metrics.json",
                   help="Filename for per-seed metrics JSON (written under each seed dir, next to preds)")
    p.add_argument("--aggregate_out", type=str, default=None,
                   help="Path to write aggregated JSON (default: <base_output_dir>/aggregate_metrics.json)")

    # Behavior
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--continue_on_fail", action="store_true")
    p.add_argument("--poll_sec", type=float, default=2.0)

    p.add_argument(
        "--norm_ranges",
        type=str,
        default="data/metric_ranges.json",
        help="JSON file with global min/max for per-sample normalization.",
    )

    # Pass-through args to evaluator (e.g., --no-bleurt, --align-ckpt-path, etc.)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    return p.parse_args()


def get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def load_metric_ranges(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metric_ranges" in data:
        data = data["metric_ranges"]

    if not isinstance(data, dict):
        raise ValueError("Range file must be a dict or contain top-level key 'metric_ranges'")

    out: Dict[str, Dict[str, float]] = {}
    for k, v in data.items():
        if not isinstance(v, dict) or "min" not in v or "max" not in v:
            raise ValueError(f"Invalid range entry for {k}: expected {{'min': ..., 'max': ...}}")
        out[k] = {"min": float(v["min"]), "max": float(v["max"])}
    return out


def add_normalized_avg_metric(
    metrics_json: Dict[str, Any],
    metric_ranges: Dict[str, Dict[str, float]],
) -> None:
    per_metric_arrays: Dict[str, List[float]] = {}
    n_samples = None

    for path in NORM_METRIC_PATHS:
        arr = get_nested(metrics_json, path)
        if not isinstance(arr, list):
            raise ValueError(f"{path} must be a list")

        if n_samples is None:
            n_samples = len(arr)
        elif len(arr) != n_samples:
            raise ValueError(
                f"Per-sample length mismatch: {path} has {len(arr)} items, expected {n_samples}"
            )

        if path not in metric_ranges:
            raise KeyError(f"Missing normalization range for {path}")

        per_metric_arrays[path] = arr

    if n_samples is None:
        raise ValueError("No per-sample metrics found for normalized average")

    normalized_avg_per_sample: List[float] = []

    for i in range(n_samples):
        vals: List[float] = []
        for path in NORM_METRIC_PATHS:
            raw = per_metric_arrays[path][i]
            if raw is None:
                raise ValueError(f"Missing sample value for {path}[{i}]")

            raw_f = float(raw)
            min_v = metric_ranges[path]["min"]
            max_v = metric_ranges[path]["max"]

            if max_v == min_v:
                raise ValueError(f"Degenerate range for {path}: min == max == {min_v}")

            norm = (raw_f - min_v) / (max_v - min_v)
            vals.append(norm)

        normalized_avg_per_sample.append(float(statistics.mean(vals)))

    metrics_json["NormalizedAvg"] = {
        "mean": float(statistics.mean(normalized_avg_per_sample)),
        "std": float(statistics.stdev(normalized_avg_per_sample)) if len(normalized_avg_per_sample) > 1 else 0.0,
        "min": float(min(normalized_avg_per_sample)),
        "max": float(max(normalized_avg_per_sample)),
        "per_sample": normalized_avg_per_sample,
        "n": len(normalized_avg_per_sample),
    }


def chunk_gpus(gpu_ids: List[int], k: int) -> List[List[int]]:
    return [gpu_ids[i:i + k] for i in range(0, len(gpu_ids), k) if len(gpu_ids[i:i + k]) == k]


def print_log_tail(log_path: Path, n: int = 200):
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        print(f"\n----- log tail ({log_path}, last {len(tail)} lines) -----")
        print("\n".join(tail))
        print("----- end log tail -----\n")
    except Exception as e:
        print(f"[warn] Could not read log tail from {log_path}: {e}")


def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    """
    Flatten numeric scalars in a nested dict to {"A.B.C": value}.
    Skips lists entirely (notably per-sample arrays).
    """
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, p))
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    else:
        # skip lists/strings/etc.
        pass
    return out


def should_aggregate(path: str) -> bool:
    """
    Default rule: aggregate “main” scalar scores; skip per-sample arrays and
    per-run dispersion/range fields.
    """
    if ".per_sample" in path:
        return False
    if path.endswith(".std") or path.endswith(".min") or path.endswith(".max"):
        return False
    return True


def aggregate_metrics(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    # Convert each seed's nested JSON to flattened scalars
    flat_by_seed: Dict[int, Dict[str, float]] = {
        seed: flatten_numeric(metrics) for seed, metrics in per_seed.items()
    }

    # Collect all keys
    keys = sorted({k for fm in flat_by_seed.values() for k in fm.keys() if should_aggregate(k)})

    agg: Dict[str, Any] = {"n_seeds": len(per_seed), "metrics": {}}
    for k in keys:
        vals: List[Tuple[int, float]] = []
        for seed, fm in flat_by_seed.items():
            if k in fm:
                vals.append((seed, fm[k]))

        if len(vals) < 1:
            continue

        xs = [v for _, v in vals]
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0

        agg["metrics"][k] = {
            "mean": float(m),
            "std": float(s),
            "n": len(xs),
            "by_seed": {str(seed): float(v) for seed, v in vals},
        }

    return agg


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    base = Path(args.base_output_dir)

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    gpu_ids = list(range(max(1, args.gpu_num)))
    slots = chunk_gpus(gpu_ids, args.gpus_per_run)
    if not slots:
        raise ValueError("No GPU slots available with given --gpu_num/--gpus_per_run")

    pending = list(seeds)
    active: Dict[int, Dict[str, Any]] = {}
    failures: List[int] = []
    per_seed_metrics: Dict[int, Dict[str, Any]] = {}

    def seed_paths(seed: int) -> Tuple[Path, Path, Path]:
        seed_dir = base / f"seed-{seed}"
        pred_path = seed_dir / args.pred_relpath
        out_path = pred_path.parent / args.per_seed_metrics_name
        log_path = out_path.with_suffix(".log")
        return pred_path, out_path, log_path

    def launch(seed: int, slot_idx: int):
        pred_path, out_path, log_path = seed_paths(seed)

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for seed {seed}: {pred_path}")

        if args.skip_existing and out_path.exists():
            # Load and store immediately; do not schedule a process.
            per_seed_metrics[seed] = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[skip] seed={seed} already has {out_path}")
            return

        visible = ",".join(map(str, slots[slot_idx]))
        cmd = [
            args.python, args.script,
            str(pred_path),
            "--output-path", str(out_path),
            *rest,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w", encoding="utf-8")

        print(f"[launch] seed={seed} slot={slot_idx} gpus={visible}")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

        active[slot_idx] = {
            "proc": proc,
            "seed": seed,
            "cmd": cmd,
            "visible": visible,
            "out_path": out_path,
            "log_path": log_path,
            "log_f": log_f,
        }

    # initial fill
    slot_idx = 0
    while pending and slot_idx < len(slots):
        seed = pending.pop(0)
        launch(seed, slot_idx)
        # if skipped, slot is still free; otherwise occupied
        if slot_idx in active:
            slot_idx += 1

    # loop
    while active:
        time.sleep(args.poll_sec)

        finished: List[int] = []
        for slot_idx, info in active.items():
            proc = info["proc"]
            rc = proc.poll()
            if rc is None:
                continue

            seed = int(info["seed"])
            info["log_f"].close()
            finished.append(slot_idx)

            if rc != 0:
                failures.append(seed)
                print(f"[done] seed={seed} slot={slot_idx} FAIL(rc={rc})")
                print(f"[error] CUDA_VISIBLE_DEVICES={info['visible']}")
                print(f"[error] cmd: {' '.join(info['cmd'])}")
                print(f"[error] log: {info['log_path']}")
                print_log_tail(Path(info["log_path"]), n=200)

                if not args.continue_on_fail:
                    # terminate others
                    for s2, info2 in active.items():
                        if s2 != slot_idx:
                            info2["proc"].terminate()
                    raise SystemExit(f"Stopping on failure. Failed seed={seed}, returncode={rc}")
            else:
                print(f"[done] seed={seed} slot={slot_idx} ok")
                out_path: Path = info["out_path"]
                if not out_path.exists():
                    raise FileNotFoundError(f"Eval succeeded but metrics file missing: {out_path}")
                per_seed_metrics[seed] = json.loads(out_path.read_text(encoding="utf-8"))

        # free slots and refill
        for slot_idx in finished:
            del active[slot_idx]
            while pending:
                seed = pending.pop(0)
                launch(seed, slot_idx)
                if slot_idx in active:
                    break  # slot filled by a real process; move on
                # else it was skipped, keep filling same slot

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
