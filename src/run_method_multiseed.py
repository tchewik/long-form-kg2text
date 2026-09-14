import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List


METHOD_TO_FLAG = {
    "cot": "--run_cot",
    "finetune": "--run_finetune",
    "direct": "--run_direct",
}


def model_to_path(model: str) -> Path:
    return Path(*model.split("/"))


def parse_test_arg(test: str) -> tuple[str, int | None]:
    """
    Returns (tag, num_test_examples)
      - tag used in output path
      - num_test_examples passed to run_experiments
    """
    t = test.strip().lower()
    if t == "full":
        return "full", -1
    if t == "200":
        return "200", 200

    # Allow arbitrary integer limits, e.g., --test 500
    n = int(t)
    if n <= 0:
        return "full", -1
    return str(n), n


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--method", required=True, choices=sorted(METHOD_TO_FLAG.keys()))
    p.add_argument("--dataset", required=True, choices=["webnlg", "lagrange", "lagrange_doc", "wikidockg"])
    p.add_argument("--test", required=True, help='Either "200", "full", or an integer like 500')

    p.add_argument("--seeds", default='42,43,44', help="Comma-separated")
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-4B")

    p.add_argument("--gpu_num", type=int, default=1)
    p.add_argument("--gpus_per_run", type=int, default=1)

    p.add_argument("--results_root", default="results/runs")

    p.add_argument("--finetune_epochs", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=1)   # (!) 5 by default when aggregation != "first"
    p.add_argument("--aggregation", type=str, default="first")
    p.add_argument("--num_shots", type=int, default=0)

    p.add_argument("--train_extra", default="--load_in_4bit", help='Extra args for run_experiments')
    p.add_argument("--eval_extra", default="--no-alignscore", help='Extra args for evaluator')

    args = p.parse_args()

    test_tag, num_test_examples = parse_test_arg(args.test)

    method_flag = METHOD_TO_FLAG[args.method]
    out_dir = (
        Path(args.results_root)
        / f"{args.dataset}-{test_tag}"
        / args.method
        / model_to_path(args.model)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Multi-seed run (train / predict)
    train_cmd = [
        "python", "src/run_n_seeds.py",
        "--seeds", args.seeds,
        "--gpu_num", str(args.gpu_num),
        "--gpus_per_run", str(args.gpus_per_run),
        "--base_output_dir", str(out_dir),
        "--",
        "--dataset", args.dataset,
        method_flag,
        "--policy_model_name", args.model,
    ]

    if num_test_examples is not None:
        train_cmd += ["--num_test_examples", str(num_test_examples)]

    if args.method == "finetune":
        train_cmd += ["--finetune_epochs", str(args.finetune_epochs)]

    if args.method == "cot":
        train_cmd += [
            "--num_samples", str(args.num_samples),
            "--aggregation", args.aggregation,
            "--num_shots", str(args.num_shots),
        ]

    if args.train_extra.strip():
        train_cmd += shlex.split(args.train_extra.strip())

    run(train_cmd)

    # 2) Multi-seed evaluation + aggregation
    pred_relpath = f"{args.method}/test_predictions.jsonl"

    eval_cmd = [
        "python", "src/evaluation/eval_n_seeds.py",
        "--seeds", args.seeds,
        "--gpu_num", str(args.gpu_num),
        "--gpus_per_run", str(args.gpus_per_run),
        "--base_output_dir", str(out_dir),
        "--pred_relpath", pred_relpath,
        "--",
    ]

    if args.eval_extra.strip():
        eval_cmd += shlex.split(args.eval_extra.strip())

    run(eval_cmd)

    print(f"\nDone. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
