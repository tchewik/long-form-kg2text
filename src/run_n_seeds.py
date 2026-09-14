import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated.")
    p.add_argument("--base_output_dir", type=str, required=True)

    p.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to schedule across.")
    p.add_argument("--gpus_per_run", type=int, default=1)

    p.add_argument("--python", type=str, default="python")
    p.add_argument("--script", type=str, default="src/run_experiment.py")
    p.add_argument("--poll_sec", type=float, default=2.0)
    p.add_argument("--continue_on_fail", action="store_true")

    p.add_argument("rest", nargs=argparse.REMAINDER)
    return p.parse_args()


def chunk_gpus(gpu_ids: List[int], k: int) -> List[List[int]]:
    return [gpu_ids[i:i + k] for i in range(0, len(gpu_ids), k) if len(gpu_ids[i:i + k]) == k]


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.gpu_num <= 0:
        raise ValueError("--gpu_num must be >= 1")
    if args.gpus_per_run <= 0:
        raise ValueError("--gpus_per_run must be >= 1")
    if args.gpus_per_run > args.gpu_num:
        raise ValueError("--gpus_per_run cannot exceed --gpu_num")

    base_out = Path(args.base_output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    gpu_ids = list(range(args.gpu_num))
    slots = chunk_gpus(gpu_ids, args.gpus_per_run)
    if not slots:
        raise ValueError("No GPU slots available with the given --gpu_num/--gpus_per_run")

    # active processes: slot_idx -> dict(proc, seed)
    active: Dict[int, Dict[str, object]] = {}
    pending = list(seeds)

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    def launch(seed: int, slot_idx: int):
        slot_gpus = slots[slot_idx]
        visible = ",".join(map(str, slot_gpus))

        run_out = base_out / f"seed-{seed}"
        run_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python, args.script,
            "--seed", str(seed),
            "--output_dir", str(run_out),
            *rest,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        repo_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        log_path = run_out / "stdout.log"
        log_f = log_path.open("w", encoding="utf-8")

        print(f"[launch] seed={seed} slot={slot_idx} gpus={visible} out={run_out}")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        active[slot_idx] = {"proc": proc, "seed": seed, "log_f": log_f}

    # initial fill
    for slot_idx in range(len(slots)):
        if not pending:
            break
        launch(pending.pop(0), slot_idx)

    # main scheduling loop
    failures: List[int] = []
    while active:
        time.sleep(args.poll_sec)
        finished_slots: List[int] = []

        for slot_idx, info in active.items():
            proc: subprocess.Popen = info["proc"]
            rc = proc.poll()
            if rc is None:
                continue

            seed = int(info["seed"])
            log_f = info["log_f"]
            log_f.close()
            finished_slots.append(slot_idx)

            status = "ok" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[done] seed={seed} slot={slot_idx} {status}")

            if rc != 0:
                failures.append(seed)
                if not args.continue_on_fail:
                    # terminate the rest
                    for s2, info2 in active.items():
                        if s2 == slot_idx:
                            continue
                        p2: subprocess.Popen = info2["proc"]
                        p2.terminate()
                    raise SystemExit(f"Stopping on failure. Failed seed={seed}, returncode={rc}")

        # clear finished and refill
        for slot_idx in finished_slots:
            del active[slot_idx]
            if pending:
                launch(pending.pop(0), slot_idx)

    if failures:
        raise SystemExit(f"Completed with failures for seeds: {failures}")

    print("[all done] All seeds completed successfully.")


if __name__ == "__main__":
    main()
