#!/usr/bin/env python3
"""
profiler_do_not_touch.py - Reusable GPU training profiler.
Wraps any PyTorch training script without modification.

Uses PyTorch's built-in profiler for comprehensive GPU analysis:
  - Per-kernel GPU timing (via CUPTI)
  - FLOPS counting and MFU estimation
  - Memory allocation tracking with snapshot export
  - Chrome trace for visual timeline inspection

NOTE: Do NOT wrap with nsys — it fights PyTorch profiler over the CUPTI
subscriber and you lose all GPU kernel times. PyTorch profiler alone gives
you everything nsys would (kernel times, FLOPS, memory, timeline).

Usage:
    torchrun --standalone --nproc_per_node=4 \
        profiler_do_not_touch.py train_gpt_do_not_touch.py

Environment variables:
    PROFILE_SKIP_STEPS=25       Steps to skip before profiling (warmup + compilation)
    PROFILE_ACTIVE_STEPS=5      Steps to actively profile
    PROFILE_GRAD_ACCUM=N        Backward calls per step (default: 8 // WORLD_SIZE)
    PROFILE_OUTPUT_DIR=./logs/profiling_outputs   Output directory for traces and snapshots
    PROFILE_MEMORY=1            Record memory allocations (for pytorch.org/memory_viz)
    PROFILE_STACKS=0            Capture Python call stacks (slower, more detail)
    MAX_WALLCLOCK_SECONDS=120   Override training time (set automatically if unset)

Output:
    trace.json              Chrome trace  -> open in chrome://tracing or ui.perfetto.dev
    memory_snapshot.pickle  Memory snapshot -> upload to pytorch.org/memory_viz
    Console tables          Top kernels, category breakdown, FLOPS/MFU, memory peaks
"""

from __future__ import annotations

import atexit
import os
import runpy
import signal
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, schedule

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORLD = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
IS_MASTER = RANK == 0

SKIP = int(os.environ.get("PROFILE_SKIP_STEPS", 25))
ACTIVE = int(os.environ.get("PROFILE_ACTIVE_STEPS", 5))
GRAD_ACCUM = int(os.environ.get("PROFILE_GRAD_ACCUM", 8 // max(WORLD, 1)))
OUTDIR = os.environ.get("PROFILE_OUTPUT_DIR", "./logs/profiling_outputs")
MEMORY = bool(int(os.environ.get("PROFILE_MEMORY", "1")))
STACKS = bool(int(os.environ.get("PROFILE_STACKS", "0")))

# ---------------------------------------------------------------------------
# Kernel categorization
# ---------------------------------------------------------------------------

_CATEGORIES = [
    ("matmul",              ("gemm", "matmul", "_mm", "cublas", "cutlass", "ampere_bf16", "sm80_xmma", "sm90")),
    ("attention",           ("attention", "sdpa", "flash", "fmha", "fused_attention", "efficient_attention")),
    ("communication",       ("nccl", "all_reduce", "allgather", "broadcast", "reduce_scatter")),
    ("normalization",       ("norm", "rms_norm", "layer_norm", "rmsnorm")),
    ("activation/pointwise",("relu", "tanh", "sigmoid", "gelu", "silu", "leaky", "square", "add_", "mul_")),
    ("embedding/indexing",  ("embedding", "index_select", "gather", "scatter", "one_hot")),
    ("optimizer",           ("adam", "sgd", "zero_", "fill_", "foreach")),
    ("copy/cast",           ("copy_", "to_", "contiguous", "_to_copy", "convert")),
    ("softmax/loss",        ("softmax", "log_softmax", "cross_entropy", "nll_loss")),
]

def _categorize(name: str) -> str:
    nl = name.lower()
    for cat, keywords in _CATEGORIES:
        if any(k in nl for k in keywords):
            return cat
    return "other"


def _peak_tflops(gpu_name: str) -> float:
    if "H100" in gpu_name:
        return 989.0
    if "A100" in gpu_name:
        return 312.0
    if "L40" in gpu_name:
        return 181.0
    if "4090" in gpu_name:
        return 165.0
    if "3090" in gpu_name:
        return 71.0
    return 312.0  # conservative default

# ---------------------------------------------------------------------------
# Analysis & reporting
# ---------------------------------------------------------------------------

def _safe_cuda_time(e, use_self: bool = False) -> float:
    """Get CUDA/device time, handling PyTorch version differences.
    use_self=True returns exclusive (self) time — no double-counting children.
    use_self=False returns inclusive (total) time."""
    if use_self:
        attrs = ("self_cuda_time_total", "self_device_time_total")
    else:
        attrs = ("cuda_time_total", "device_time_total")
    for attr in attrs:
        try:
            v = getattr(e, attr, None)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return 0.0


def _safe_cuda_mem(e) -> int:
    """Get CUDA/device memory usage, handling PyTorch version differences."""
    for attr in ("self_cuda_memory_usage", "self_device_memory_usage"):
        try:
            v = getattr(e, attr, None)
            if v is not None:
                return int(v)
        except Exception:
            pass
    return 0


def _report(prof: profile) -> None:
    events = prof.key_averages()

    # -- Detect if CUDA/device times are available --
    has_cuda_times = any(_safe_cuda_time(e) > 0 for e in events)

    # Figure out the correct sort_by key for this PyTorch version
    if has_cuda_times:
        # Try both attribute names to find which one the table method accepts
        time_label = "GPU"
        time_sort = "cuda_time_total"
        try:
            events.table(sort_by="cuda_time_total", row_limit=1)
        except Exception:
            time_sort = "device_time_total"
    else:
        time_label = "CPU (no CUDA times — likely CUPTI conflict with nsys)"
        time_sort = "cpu_time_total"
        print("\n" + "!" * 100)
        print("WARNING: No CUDA kernel times available (CUPTI may be blocked by nsys).")
        print("         Run WITHOUT nsys for GPU kernel times, FLOPS, and memory breakdown:")
        print("         torchrun --standalone --nproc_per_node=N profiler_do_not_touch.py <script>")
        print("!" * 100)

    # -- Top kernels --
    print(f"\n{'=' * 100}")
    print(f"TOP 25 OPERATIONS BY {time_label} TIME")
    print("=" * 100)
    try:
        print(events.table(sort_by=time_sort, row_limit=25))
    except Exception:
        print(events.table(sort_by="cpu_time_total", row_limit=25))

    # -- Category breakdown (using SELF/exclusive times to avoid double-counting) --
    def _get_self_time(e) -> float:
        return _safe_cuda_time(e, use_self=True) if has_cuda_times else e.self_cpu_time_total

    cats: dict[str, float] = defaultdict(float)
    total_self_us = 0.0
    for e in events:
        t = _get_self_time(e)
        if t > 0:
            cats[_categorize(e.key)] += t
            total_self_us += t

    print(f"\n{'=' * 100}")
    print(f"{time_label} TIME BY CATEGORY (exclusive/self time — no double-counting)")
    print("=" * 100)
    for cat, us in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "#" * int(40 * us / max(total_self_us, 1))
        print(f"  {cat:25s} {us/1e3:10.1f} ms  ({100*us/max(total_self_us,1):5.1f}%)  {bar}")
    print(f"  {'TOTAL':25s} {total_self_us/1e3:10.1f} ms")

    # -- Compute vs bandwidth --
    matmul_us = cats.get("matmul", 0) + cats.get("attention", 0)
    mem_us = total_self_us - matmul_us - cats.get("communication", 0)
    if total_self_us > 0:
        print(f"\n  Compute-heavy (matmul+attn): {100*matmul_us/total_self_us:.1f}%")
        print(f"  Memory-bound (everything else excl. comm): {100*mem_us/total_self_us:.1f}%")
        if has_cuda_times:
            if matmul_us < mem_us:
                print("  --> Model is MEMORY-BANDWIDTH BOUND (more time in pointwise/norm/copy than matmul)")
            else:
                print("  --> Model is COMPUTE BOUND (good — most time in matmul/attention)")
        else:
            print("  --> (CPU times only — run without nsys for accurate GPU bound analysis)")

    # -- FLOPS / MFU (use self CUDA time for accurate utilization) --
    total_flops = sum(getattr(e, 'flops', 0) for e in events if getattr(e, 'flops', 0) > 0)
    if total_flops > 0 and total_self_us > 0:
        gpu = torch.cuda.get_device_name(0)
        peak = _peak_tflops(gpu)
        if has_cuda_times:
            achieved = total_flops / (total_self_us / 1e6) / 1e12
            print(f"\n  Total FLOPS:     {total_flops:.3e}")
            print(f"  Achieved:        {achieved:.1f} TFLOPS")
            print(f"  Peak ({gpu}): {peak:.0f} TFLOPS")
            print(f"  MFU:             {100*achieved/peak:.1f}%")
        else:
            print(f"\n  Total FLOPS:     {total_flops:.3e}  (MFU requires CUDA times — run without nsys)")

    # -- Per-step timing (use wall-clock from profiler steps) --
    if ACTIVE > 0 and total_self_us > 0:
        print(f"\n  Avg self {time_label} time per step: {total_self_us/1e3/ACTIVE:.1f} ms  ({ACTIVE} steps)")

    # -- Memory --
    print(f"\n  Peak memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  Peak memory reserved:  {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

    # -- FLOPS by operator (top consumers) --
    flops_events = [e for e in events if getattr(e, 'flops', 0) > 0]
    if flops_events:
        print(f"\n{'=' * 100}")
        print("TOP FLOPS CONSUMERS")
        print("=" * 100)
        for e in sorted(flops_events, key=lambda x: -x.flops)[:15]:
            pct = 100 * e.flops / max(total_flops, 1)
            t = _safe_cuda_time(e, use_self=True) if has_cuda_times else e.self_cpu_time_total
            label = "cuda" if has_cuda_times else "cpu"
            print(f"  {e.key:60s} {e.flops:.2e} FLOPS ({pct:5.1f}%)  {label}:{t/1e3:.1f}ms")

    # -- Memory by operator --
    if MEMORY:
        mem_events = [(e, _safe_cuda_mem(e)) for e in events if _safe_cuda_mem(e) != 0]
        if mem_events:
            print(f"\n{'=' * 100}")
            print("MEMORY ALLOCATIONS BY OPERATOR (top allocators & deallocators)")
            print("=" * 100)
            for e, mem in sorted(mem_events, key=lambda x: -abs(x[1]))[:20]:
                sign = "+" if mem > 0 else ""
                print(f"  {e.key:60s} {sign}{mem/1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Cleanup — kill all GPU processes and distributed state on interrupt
# ---------------------------------------------------------------------------

_cleanup_done = False

def _cleanup(signame: str = "atexit") -> None:
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    if IS_MASTER:
        print(f"\n[profiler] Cleaning up ({signame})...", flush=True)

    # 1. Stop memory recording
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception:
        pass

    # 2. Destroy distributed process group
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            if IS_MASTER:
                print("[profiler] Destroyed distributed process group")
    except Exception:
        pass

    # 3. Release all GPU memory on all visible devices
    try:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        if IS_MASTER:
            print(f"[profiler] Released GPU memory on {torch.cuda.device_count()} device(s)")
    except Exception:
        pass

    # 4. Kill any leftover child processes in our process group
    import subprocess
    try:
        my_pid = os.getpid()
        my_pgid = os.getpgid(my_pid)
        # Find all PIDs in our process group (excluding ourselves)
        result = subprocess.run(
            ["ps", "-o", "pid=", "-g", str(my_pgid)],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if IS_MASTER:
            print("[profiler] Killed child processes")
    except Exception:
        pass

    # 5. Synchronize and reset CUDA to release driver-level resources
    try:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
    except Exception:
        pass

    if IS_MASTER:
        print("[profiler] Cleanup complete", flush=True)


def _signal_handler(signum: int, frame) -> None:
    signame = signal.Signals(signum).name
    _cleanup(signame)
    sys.exit(128 + signum)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    script = sys.argv[1]
    if not Path(script).exists():
        print(f"Error: {script} not found")
        sys.exit(1)

    # Register cleanup for all exit paths
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup, "atexit")

    os.makedirs(OUTDIR, exist_ok=True)

    # Auto-limit training time for profiling runs
    if "MAX_WALLCLOCK_SECONDS" not in os.environ:
        os.environ["MAX_WALLCLOCK_SECONDS"] = "120"

    # Detect nsys wrapper — warn user they'll lose GPU times
    nsys_detected = any(k.startswith("NSYS_") or k.startswith("__NSYS") for k in os.environ)
    if not nsys_detected:
        # Also check parent process name
        try:
            import subprocess
            ppid = os.getppid()
            result = subprocess.run(["ps", "-o", "comm=", "-p", str(ppid)],
                                    capture_output=True, text=True, timeout=2)
            if "nsys" in result.stdout.lower():
                nsys_detected = True
        except Exception:
            pass

    if nsys_detected and IS_MASTER:
        print("\n" + "!" * 80)
        print("WARNING: nsys detected! It will block CUPTI — you'll lose GPU kernel times.")
        print("Run WITHOUT nsys for full profiling:")
        print(f"  torchrun --standalone --nproc_per_node={WORLD} {Path(__file__).name} {script}")
        print("!" * 80 + "\n")

    if IS_MASTER:
        print(f"[profiler] target={script}")
        print(f"[profiler] skip_steps={SKIP}  active_steps={ACTIVE}  grad_accum={GRAD_ACCUM}")
        print(f"[profiler] memory={MEMORY}  stacks={STACKS}  output={OUTDIR}")
        print(f"[profiler] MAX_WALLCLOCK_SECONDS={os.environ.get('MAX_WALLCLOCK_SECONDS')}")
        needed = SKIP + 1 + ACTIVE
        est_s = needed * 0.5  # ~500ms/step rough estimate
        print(f"[profiler] Need ~{needed} steps (~{est_s:.0f}s) to complete profiling")

    # -- Patch backward() for step counting + NVTX markers --
    state = {"bwd": 0, "steps": 0, "done": False}
    prof_ref: dict = {"prof": None}
    got_results = {"v": False}

    _orig_backward = torch.Tensor.backward

    def _patched_backward(self, *args, **kwargs):
        torch.cuda.nvtx.range_push("backward")
        result = _orig_backward(self, *args, **kwargs)
        torch.cuda.nvtx.range_pop()

        state["bwd"] += 1
        if state["bwd"] % GRAD_ACCUM == 0:
            state["steps"] += 1
            p = prof_ref["prof"]
            if p is not None and not state["done"]:
                p.step()
                s = state["steps"]
                if IS_MASTER and s == SKIP + 2:
                    print(f"[profiler] >>> Active profiling started at step {s}")
                if s >= SKIP + 1 + ACTIVE + 1:
                    state["done"] = True
                    if IS_MASTER:
                        print(f"[profiler] >>> Active profiling ended at step {s}")
        return result

    torch.Tensor.backward = _patched_backward

    # -- Memory history recording --
    if MEMORY:
        try:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        except Exception:
            pass

    # -- Profiler setup --
    # IMPORTANT: on_trace_ready is called INSIDE prof.step() which is called from
    # _patched_backward, which runs mid-training-step. If we do heavy I/O here
    # (export trace, print tables), rank 0 blocks for seconds while other ranks
    # hit the next all_reduce and NCCL times out. So we just mark it done here
    # and defer all reporting to after training finishes.
    deferred_prof = {"ref": None}

    def on_trace_ready(p):
        got_results["v"] = True
        # Stash a reference — we'll export & report after training exits
        deferred_prof["ref"] = p
        if IS_MASTER:
            print(f"[profiler] Trace data captured — will report after training completes")

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=SKIP, warmup=1, active=ACTIVE, repeat=1),
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=MEMORY,
        with_stack=STACKS,
        with_flops=True,
    )
    prof_ref["prof"] = prof

    # -- Run the training script --
    sys.argv = [script] + sys.argv[2:]

    prof.start()
    try:
        runpy.run_path(script, run_name="__main__")
    except (KeyboardInterrupt, SystemExit):
        if IS_MASTER:
            print("\n[profiler] Interrupted — cleaning up...")
    finally:
        # Stop profiler first (before any reporting)
        try:
            prof.stop()
        except Exception:
            pass

        torch.Tensor.backward = _orig_backward

        # -- Deferred reporting (safe: training is done, no NCCL in flight) --
        if IS_MASTER:
            dp = deferred_prof.get("ref")
            if dp is not None:
                # Chrome trace (for manual inspection)
                try:
                    trace_path = os.path.join(OUTDIR, "trace.json")
                    dp.export_chrome_trace(trace_path)
                    print(f"\n[profiler] Chrome trace saved: {trace_path}")
                except Exception as e:
                    print(f"[profiler] Trace export failed: {e}")
                # TensorBoard trace (for tensorboard --logdir)
                try:
                    tb_dir = os.path.join(OUTDIR, "tensorboard")
                    os.makedirs(tb_dir, exist_ok=True)
                    tb_trace = os.path.join(tb_dir, f"worker{RANK}.pt.trace.json")
                    dp.export_chrome_trace(tb_trace)
                    print(f"[profiler] TensorBoard trace saved: {tb_dir}/")
                    print(f"[profiler] Run: tensorboard --logdir={tb_dir} --port=6006")
                    print(f"[profiler] Then SSH tunnel: ssh -L 6006:localhost:6006 <remote>")
                    print(f"[profiler] Open http://localhost:6006/#pytorch_profiler")
                except Exception as e:
                    print(f"[profiler] TensorBoard export failed: {e}")
                try:
                    _report(dp)
                except Exception as e:
                    print(f"[profiler] Report failed: {e}")

        # Save memory snapshot
        if MEMORY and IS_MASTER:
            try:
                snap = os.path.join(OUTDIR, "memory_snapshot.pickle")
                torch.cuda.memory._dump_snapshot(snap)
                print(f"\n[profiler] Memory snapshot saved: {snap}")
                print(f"[profiler] Upload to: https://pytorch.org/memory_viz")
            except Exception as e:
                print(f"[profiler] Memory snapshot failed: {e}")

        # Handle case where profiling never reached active phase
        if not got_results["v"] and IS_MASTER:
            print(f"\n[profiler] WARNING: Never reached active profiling phase!")
            print(f"[profiler] Completed {state['steps']} steps, needed {SKIP + 1 + ACTIVE}")
            suggestion = max(state["steps"] - ACTIVE - 2, 3)
            print(f"[profiler] Fix: PROFILE_SKIP_STEPS={suggestion} or increase MAX_WALLCLOCK_SECONDS")

            # Try to export whatever we have
            try:
                trace_path = os.path.join(OUTDIR, "trace_partial.json")
                prof.export_chrome_trace(trace_path)
                print(f"[profiler] Partial trace saved: {trace_path}")
            except Exception:
                pass

        if IS_MASTER:
            print(f"\n{'=' * 100}")
            print("HOW TO USE THE OUTPUT")
            print("=" * 100)
            print(f"  1. Chrome trace:    {OUTDIR}/trace.json")
            print(f"     -> Open in chrome://tracing  OR  https://ui.perfetto.dev")
            print(f"     -> Zoom into a training step to see GPU kernel timeline")
            print(f"     -> Look for gaps between kernels (= CPU overhead or sync stalls)")
            if MEMORY:
                print(f"  2. Memory snapshot: {OUTDIR}/memory_snapshot.pickle")
                print(f"     -> Upload to https://pytorch.org/memory_viz")
                print(f"     -> Shows allocation timeline, peak crossovers, tensor lifetimes")
                print(f"     -> Look for memory spikes during backward pass (activation memory)")
            print(f"  3. What to look for:")
            print(f"     -> High 'other' % in category breakdown = wasted time in small ops")
            print(f"     -> MFU < 30% on A100 = memory-bandwidth bound, need larger matmuls")
            print(f"     -> Big gap between CPU and GPU time = CPU bottleneck (data loading, Python)")
            print(f"  4. Do NOT wrap with nsys — it blocks CUPTI and you lose GPU kernel times")

        # Final cleanup (GPU memory, dist, child processes)
        _cleanup("finally")


if __name__ == "__main__":
    main()
