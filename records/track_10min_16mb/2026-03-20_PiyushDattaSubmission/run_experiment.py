#!/usr/bin/env python3
"""
Experiment runner with early stopping.

Usage:
  # Step 1: Establish baseline (run once, saves to baseline.json)
  python run_experiment.py --baseline

  # Step 2: Run experiment with early stopping against baseline
  python run_experiment.py --name "my_experiment" --env "FOO=bar,BAZ=1"

  # Step 3: Run experiment without early stopping (just log checkpoints)
  python run_experiment.py --name "my_experiment" --env "FOO=bar" --no-early-stop

The script:
- Sets VAL_LOSS_EVERY=50 (~23s between val checks at ~461ms/step)
- Parses training output in real-time
- At time checkpoints (30s, 1m, 2m, ..., 10m), records val_bpb
- Compares against baseline and kills training if worse at 2+ consecutive checkpoints
- After training, runs eval and reports final val_bpb
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASELINE_FILE = SCRIPT_DIR / "baseline.json"
RESULTS_DIR = SCRIPT_DIR / "experiment_results"

# Time checkpoints in seconds
CHECKPOINTS = [30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]

BASE_ENV = {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "VAL_LOSS_EVERY": "25",  # frequent val checks; we gate on wall-clock time below
    "TRAIN_LOG_EVERY": "50",
}

def detect_gpu_count() -> int:
    """Auto-detect number of available CUDA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split("\n") if l.strip()])
    except Exception:
        pass
    return 1


def get_train_cmd(nproc: int) -> list[str]:
    return [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        str(SCRIPT_DIR / "train_gpt.py"),
    ]


def parse_val_line(line: str):
    """Extract val_bpb and train_time_ms from a val log line."""
    bpb_match = re.search(r'val_bpb:(\d+\.\d+)', line)
    time_match = re.search(r'train_time:(\d+)ms', line)
    step_match = re.search(r'step:(\d+)/', line)
    if bpb_match and time_match:
        return {
            "val_bpb": float(bpb_match.group(1)),
            "train_time_ms": int(time_match.group(1)),
            "step": int(step_match.group(1)) if step_match else -1,
        }
    return None


def find_checkpoint_value(records: list, target_ms: int):
    """Find the first val record at or after target_ms (closest match)."""
    candidates = [r for r in records if r["train_time_ms"] >= target_ms]
    return candidates[0] if candidates else None


def run_training(extra_env: dict = None, name: str = "baseline",
                 baseline: dict = None, early_stop: bool = True,
                 ttt_enabled: bool = False, nproc: int = None):
    """Run training, parse output, optionally early-stop."""
    if nproc is None:
        nproc = detect_gpu_count()

    env = os.environ.copy()
    env.update(BASE_ENV)
    env["RUN_ID"] = name
    if not ttt_enabled:
        env["TTT_ENABLED"] = "0"  # disable TTT during training phase
    if extra_env:
        env.update(extra_env)

    train_cmd = get_train_cmd(nproc)

    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  GPUs: {nproc}")
    print(f"  Extra env: {extra_env or {}}")
    print(f"  Early stop: {early_stop} (baseline: {'loaded' if baseline else 'none'})")
    print(f"{'='*70}\n")

    records = []
    consecutive_worse = 0
    process = subprocess.Popen(
        train_cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    last_checkpoint_idx = -1
    killed = False

    try:
        for line in process.stdout:
            line = line.rstrip()
            # Only print key lines to avoid flooding
            if any(k in line for k in ("val_bpb", "step:", "warmup", "stopping",
                                        "final_", "quant", "ttt:", "ema:", "qat:")):
                print(line)

            val = parse_val_line(line)
            if val and val["step"] > 0:
                records.append(val)
                train_s = val["train_time_ms"] / 1000.0

                # Check if we crossed any checkpoint boundary
                for ci, cp in enumerate(CHECKPOINTS):
                    if ci > last_checkpoint_idx and train_s >= cp:
                        last_checkpoint_idx = ci
                        cp_label = f"{cp}s" if cp < 60 else f"{cp//60}m"

                        if baseline and str(cp) in baseline:
                            bl_bpb = baseline[str(cp)]
                            diff = val["val_bpb"] - bl_bpb
                            status = "BETTER" if diff < 0 else "WORSE"
                            marker = "+" if diff < 0 else "x"
                            print(f"\n  [{cp_label:>4s}] val_bpb={val['val_bpb']:.4f} "
                                  f"baseline={bl_bpb:.4f} diff={diff:+.4f} {marker} {status}")

                            if diff > 0.002:  # worse by more than noise
                                consecutive_worse += 1
                            else:
                                consecutive_worse = 0

                            if early_stop and consecutive_worse >= 2 and train_s >= 120:
                                print(f"\n  EARLY STOP: worse than baseline at {consecutive_worse} "
                                      f"consecutive checkpoints. Killing training.")
                                process.send_signal(signal.SIGTERM)
                                killed = True
                                break
                        else:
                            print(f"\n  [{cp_label:>4s}] val_bpb={val['val_bpb']:.4f} step={val['step']}")
                        break

            if killed:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        process.send_signal(signal.SIGTERM)
        killed = True

    process.wait()

    # Build checkpoint map
    checkpoint_map = {}
    for cp in CHECKPOINTS:
        cv = find_checkpoint_value(records, cp * 1000)
        if cv:
            checkpoint_map[str(cp)] = cv["val_bpb"]

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Results: {name}")
    print(f"{'='*70}")
    print(f"  {'Time':>6s}  {'val_bpb':>10s}", end="")
    if baseline:
        print(f"  {'baseline':>10s}  {'diff':>10s}", end="")
    print()
    print(f"  {'-'*6}  {'-'*10}", end="")
    if baseline:
        print(f"  {'-'*10}  {'-'*10}", end="")
    print()

    for cp in CHECKPOINTS:
        cp_label = f"{cp}s" if cp < 60 else f"{cp//60}m"
        if str(cp) in checkpoint_map:
            bpb = checkpoint_map[str(cp)]
            print(f"  {cp_label:>6s}  {bpb:>10.4f}", end="")
            if baseline and str(cp) in baseline:
                bl = baseline[str(cp)]
                diff = bpb - bl
                print(f"  {bl:>10.4f}  {diff:>+10.4f}", end="")
            print()

    if killed:
        print(f"\n  (killed early)")

    # Parse final result if available
    final_bpb = None
    for r in reversed(records):
        final_bpb = r["val_bpb"]
        break

    return checkpoint_map, final_bpb, records


def main():
    parser = argparse.ArgumentParser(description="Run training experiments with early stopping")
    parser.add_argument("--baseline", action="store_true", help="Run baseline and save checkpoints")
    parser.add_argument("--name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--env", type=str, default="", help="Extra env vars, comma-separated KEY=VAL")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--show-baseline", action="store_true", help="Show saved baseline values")
    parser.add_argument("--nproc", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    args = parser.parse_args()

    nproc = args.nproc or detect_gpu_count()
    print(f"Using {nproc} GPU(s)")

    if args.show_baseline:
        if BASELINE_FILE.exists():
            bl = json.loads(BASELINE_FILE.read_text())
            print("Saved baseline:")
            for cp in CHECKPOINTS:
                if str(cp) in bl:
                    cp_label = f"{cp}s" if cp < 60 else f"{cp//60}m"
                    print(f"  {cp_label:>6s}: val_bpb={bl[str(cp)]:.4f}")
        else:
            print("No baseline saved. Run with --baseline first.")
        return

    # Parse extra env vars
    extra_env = {}
    if args.env:
        for pair in args.env.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                extra_env[k.strip()] = v.strip()

    if args.baseline:
        print("Running BASELINE training (no TTT during training, standard config)...")
        checkpoint_map, final_bpb, records = run_training(
            name="baseline", early_stop=False, nproc=nproc
        )
        # Save baseline
        BASELINE_FILE.write_text(json.dumps(checkpoint_map, indent=2))
        print(f"\nBaseline saved to {BASELINE_FILE}")
        print(f"Final pre-quant val_bpb: {final_bpb:.4f}" if final_bpb else "")
    else:
        # Load baseline
        baseline = None
        if BASELINE_FILE.exists():
            baseline = json.loads(BASELINE_FILE.read_text())
            print(f"Loaded baseline from {BASELINE_FILE}")
        else:
            print("WARNING: No baseline found. Run with --baseline first.")
            print("Running without early stopping.\n")

        checkpoint_map, final_bpb, records = run_training(
            extra_env=extra_env,
            name=args.name,
            baseline=baseline,
            early_stop=not args.no_early_stop,
            nproc=nproc,
        )

        # Save experiment results
        RESULTS_DIR.mkdir(exist_ok=True)
        result = {
            "name": args.name,
            "env": extra_env,
            "checkpoints": checkpoint_map,
            "final_bpb": final_bpb,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result_file = RESULTS_DIR / f"{args.name}_{int(time.time())}.json"
        result_file.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
