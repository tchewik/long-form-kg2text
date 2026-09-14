#!/usr/bin/env python
"""
Low-memory wrapper around evaluate_jsonl.py.

Runs the main evaluator multiple times, each time enabling only one metric
(using the --no-* flags), so that each heavy metric runs in its own process
and GPU memory is fully freed between runs.

Example:

    python evaluate_jsonl_lowmemory.py \
        data.jsonl \
        --output-path results_all_metrics.json

You can still pass any of the original evaluate_jsonl.py flags, e.g.:

    python evaluate_jsonl_lowmemory.py \
        data.jsonl \
        --output-path results_all_metrics.json \
        --align-ckpt-path models/AlignScore-base.ckpt \
        --align-device cuda:0 \
        --factspotter-device cuda \
        --bleurt-checkpoint models/BLEURT-20

This wrapper now also supports partial metric options, mirroring the main script:

    --no-bleu
    --no-bertscore
    --no-bleurt
    --no-infolm
    --no-alignscore
    --no-factspotter

These flags control which metric subprocesses are run.
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    from src.evaluation.evaluate_jsonl import fmt_compact
except ModuleNotFoundError:
    from evaluate_jsonl import fmt_compact

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [lowmemory wrapper]: %(message)s",
)


def filter_forward_args(args: List[str]) -> List[str]:
    """
    Remove flags that we manage ourselves before forwarding to evaluate_jsonl.py:

    - Any --no-* flags (we decide which metrics to run per subprocess).
    - --output-path and its value (we set per-run output paths).
    """
    filtered: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue

        if a == "--output-path":
            # Skip this flag AND its value
            skip_next = True
            continue

        if a.startswith("--no-"):
            # These are handled by the wrapper, not forwarded
            continue

        filtered.append(a)
    return filtered


def run_single_metric(
    metric_label: str,
    input_path: Path,
    evaluator_script: Path,
    base_args: List[str],
    metric_specific_flags: List[str],
    tmpdir: Path,
) -> Dict[str, Any]:
    """
    Run evaluate_jsonl.py once for a specific metric (using metric_specific_flags)
    and return the JSON it produces.
    """
    out_path = tmpdir / f"{metric_label}.json"

    cmd = [
        sys.executable,
        str(evaluator_script),
        str(input_path),
        "--output-path",
        str(out_path),
    ]
    cmd.extend(base_args)
    cmd.extend(metric_specific_flags)

    logger.info(f"Running {metric_label} with command:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not out_path.exists():
        logger.warning(
            f"Expected output {out_path} not found; "
            f"maybe this metric was skipped by the underlying script."
        )
        return {}

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        logger.warning("Output from {metric_label} is not a dict; ignoring.")
        return {}

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-memory wrapper for evaluate_jsonl.py (runs metrics in separate processes)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to JSON or JSONL predictions file (same as for evaluate_jsonl.py).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Final merged metrics JSON path.",
    )
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=Path(__file__).with_name("evaluate_jsonl.py"),
        help="Path to the main evaluator script (default: evaluate_jsonl.py in the same directory).",
    )

    # Wrapper-level metric toggles (mirror the main script)
    parser.add_argument("--no-bleu", action="store_true", help="Disable BLEU run.")
    parser.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore run.")
    parser.add_argument("--no-bleurt", action="store_true", help="Disable BLEURT run.")
    parser.add_argument("--no-infolm", action="store_true", help="Disable InfoLM run.")
    parser.add_argument("--no-alignscore", action="store_true", help="Disable AlignScore run.")
    parser.add_argument("--no-factspotter", action="store_true", help="Disable FactSpotter run.")

    # Everything else (model names, devices, etc.) gets passed through
    args, unknown = parser.parse_known_args()

    input_path: Path = args.input_path
    output_path: Path = args.output_path
    evaluator_script: Path = args.evaluator_script

    if not evaluator_script.exists():
        parser.error(f"Evaluator script not found: {evaluator_script}")

    # Remove wrapper-managed flags before forwarding
    base_args = filter_forward_args(unknown)

    # Define which metrics to run and which other metrics to disable in each run
    metric_runs: Dict[str, List[str]] = {}

    # Underlying script supports:
    #   --no-bleu --no-bertscore --no-alignscore --no-factspotter --no-bleurt --no-infolm

    if not args.no_bleu:
        metric_runs["BLEU"] = [
            "--no-bertscore",
            "--no-alignscore",
            "--no-factspotter",
            "--no-bleurt",
            "--no-infolm",
        ]

    if not args.no_bertscore:
        metric_runs["BERTScore"] = [
            "--no-bleu",
            "--no-alignscore",
            "--no-factspotter",
            "--no-bleurt",
            "--no-infolm",
        ]

    if not args.no_bleurt:
        metric_runs["BLEURT"] = [
            "--no-bleu",
            "--no-bertscore",
            "--no-alignscore",
            "--no-factspotter",
            "--no-infolm",
        ]

    if not args.no_infolm:
        metric_runs["InfoLM"] = [
            "--no-bleu",
            "--no-bertscore",
            "--no-alignscore",
            "--no-factspotter",
            "--no-bleurt",
        ]

    if not args.no_alignscore:
        metric_runs["AlignScore"] = [
            "--no-bleu",
            "--no-bertscore",
            "--no-factspotter",
            "--no-bleurt",
            "--no-infolm",
        ]

    if not args.no_factspotter:
        metric_runs["FactSpotter"] = [
            "--no-bleu",
            "--no-bertscore",
            "--no-alignscore",
            "--no-bleurt",
            "--no-infolm",
        ]

    if not metric_runs:
        parser.error("All metrics are disabled (all --no-* set). Nothing to run.")

    combined: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)

        for metric_label, flags in metric_runs.items():
            data = run_single_metric(
                metric_label=metric_label,
                input_path=input_path,
                evaluator_script=evaluator_script,
                base_args=base_args,
                metric_specific_flags=flags,
                tmpdir=tmpdir,
            )

            # Merge keys (BLEU, BERTScore, BLEURT, InfoLM, AlignScore, FactSpotter)
            # into one dict. If a key appears twice, the later one wins (shouldn't happen).
            for k, v in data.items():
                if k in combined:
                    logger.warning(
                        f"Key {k} already present in combined "
                        f"results; overwriting with {metric_label} run."
                    )
                combined[k] = v

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote merged metrics to {output_path}")

    logger.info("Overall metrics:")
    for k, v in combined.items():
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


if __name__ == "__main__":
    main()
