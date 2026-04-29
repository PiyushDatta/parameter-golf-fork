#!/usr/bin/env python3
"""Precompute teacher logits (top-K per position) from a quantized teacher model.

This script loads a trained + quantized teacher model (final_model.int6.ptz),
runs it over training shards, and for each position saves:
  - top-K token indices (uint16, since vocab_size=8192 < 65535)
  - top-K logit values  (float16)

Output is stored as numpy memory-mapped files alongside training shards so the
ShuffledSequenceLoader can load them efficiently during KD training.

Usage:
    # Single-GPU precomputation (recommended — runs on 1 GPU, writes to shared dir)
    python precompute_teacher_logits.py \
        --teacher-model /path/to/final_model.int6.ptz \
        --data-dir ./data/ \
        --output-dir ./data/teacher_logits_sp8192/ \
        --top-k 32 \
        --batch-seqs 64 \
        --vocab-size 8192

    # Multi-GPU (each GPU processes a shard subset):
    torchrun --nproc_per_node=4 precompute_teacher_logits.py \
        --teacher-model /path/to/final_model.int6.ptz \
        --data-dir ./data/ \
        --output-dir ./data/teacher_logits_sp8192/ \
        --top-k 32

The output directory will contain one .npy file per training shard:
    fineweb_train_000000_teacher_topk.npy   — shape (num_tokens-1, top_k), uint16 indices
    fineweb_train_000000_teacher_vals.npy   — shape (num_tokens-1, top_k), float16 logit values

These are memory-mapped at training time for zero-copy loading.
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="Precompute teacher logits for KD")
    p.add_argument("--teacher-model", type=str, required=True,
                    help="Path to the quantized teacher model (.int6.ptz)")
    p.add_argument("--data-dir", type=str, default="./data/",
                    help="Data directory containing datasets/ and tokenizers/")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Directory to write teacher logit files")
    p.add_argument("--top-k", type=int, default=32,
                    help="Number of top logits to save per position")
    p.add_argument("--batch-seqs", type=int, default=32,
                    help="Number of sequences per forward pass")
    p.add_argument("--seq-len", type=int, default=2048,
                    help="Sequence length for chunking")
    p.add_argument("--vocab-size", type=int, default=8192,
                    help="Vocabulary size (must match teacher model)")
    p.add_argument("--num-layers", type=int, default=11)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=float, default=4.0)
    p.add_argument("--logit-softcap", type=float, default=30.0)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--rope-dims", type=int, default=16)
    p.add_argument("--qk-gain-init", type=float, default=5.0)
    p.add_argument("--parallel-residual-start", type=int, default=7)
    p.add_argument("--xsa-last-n", type=int, default=11)
    p.add_argument("--compressor", type=str, default="brotli",
                    choices=["brotli", "lzma"])
    p.add_argument("--shards", type=str, default=None,
                    help="Comma-separated shard indices to process (default: all)")
    return p.parse_args()


def build_teacher_hparams(args):
    """Create a minimal Hyperparameters-like object for deserialize()."""

    class TeacherHparams:
        pass

    h = TeacherHparams()
    h.vocab_size = args.vocab_size
    h.num_layers = args.num_layers
    h.model_dim = args.model_dim
    h.embedding_dim = args.embedding_dim
    h.num_heads = args.num_heads
    h.num_kv_heads = args.num_kv_heads
    h.mlp_mult = args.mlp_mult
    h.logit_softcap = args.logit_softcap
    h.rope_base = args.rope_base
    h.rope_dims = args.rope_dims
    h.rope_train_seq_len = args.seq_len
    h.qk_gain_init = args.qk_gain_init
    h.parallel_residual_start = args.parallel_residual_start
    h.xsa_last_n = args.xsa_last_n
    h.train_seq_len = args.seq_len
    h.tie_embeddings = True
    h.tied_embed_init_std = 0.005
    h.skip_gates_enabled = True
    h.ln_scale = True
    h.compressor = args.compressor
    h.quantized_model_path = args.teacher_model
    h.data_dir = args.data_dir
    h.datasets_dir = os.path.join(args.data_dir, "datasets",
                                  f"fineweb10B_sp{args.vocab_size}")
    h.train_files = os.path.join(h.datasets_dir, "fineweb_train_*.bin")
    h.tokenizer_path = os.path.join(args.data_dir, "tokenizers",
                                    f"fineweb_{args.vocab_size}_bpe.model")
    # Fields needed by GPT constructor but not critical for inference
    h.is_main_process = True
    h.rank = int(os.environ.get("RANK", "0"))
    h.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    h.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    h.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    h.matrix_bits = 6
    h.embed_bits = 8
    return h


def load_shard_tokens(shard_path):
    """Load tokens from a training shard (binary format with 256-int32 header)."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(shard_path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {shard_path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(shard_path, dtype="<u2", count=num_tokens,
                         offset=header_bytes)
    return torch.from_numpy(tokens.astype(np.int64))


def process_shard(teacher_model, shard_path, output_dir, top_k, batch_seqs,
                  seq_len, device, shard_idx, total_shards):
    """Process a single shard: run teacher model, save top-K logits."""
    shard_name = Path(shard_path).stem  # e.g. "fineweb_train_000000"
    topk_path = os.path.join(output_dir, f"{shard_name}_teacher_topk.npy")
    vals_path = os.path.join(output_dir, f"{shard_name}_teacher_vals.npy")

    # Skip if already computed
    if os.path.exists(topk_path) and os.path.exists(vals_path):
        print(f"[{shard_idx+1}/{total_shards}] {shard_name}: already exists, skipping")
        return

    tokens = load_shard_tokens(shard_path)
    num_tokens = tokens.numel()
    num_predictions = num_tokens - 1  # predict token[i+1] from token[:i+1]

    # Pre-allocate output memmaps
    topk_indices = np.memmap(topk_path + ".tmp", dtype=np.uint16, mode="w+",
                             shape=(num_predictions, top_k))
    topk_values = np.memmap(vals_path + ".tmp", dtype=np.float16, mode="w+",
                            shape=(num_predictions, top_k))

    # Process in sequence-length chunks with batching
    num_seqs = num_predictions // seq_len
    remainder = num_predictions % seq_len
    t0 = time.perf_counter()
    positions_done = 0

    teacher_model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Process full sequences in batches
        for batch_start in range(0, num_seqs, batch_seqs):
            batch_end = min(batch_start + batch_seqs, num_seqs)
            actual_batch = batch_end - batch_start

            # Build input batch: each sequence is tokens[i*seq_len : (i+1)*seq_len]
            x = torch.zeros(actual_batch, seq_len, dtype=torch.int64, device=device)
            for bi in range(actual_batch):
                si = batch_start + bi
                start = si * seq_len
                x[bi] = tokens[start : start + seq_len].to(device)

            # Forward pass to get logits: [batch, seq_len, vocab_size]
            logits = teacher_model.forward_logits(x)

            # Get top-K for each position
            # logits shape: [batch, seq_len, vocab_size]
            topk_v, topk_i = torch.topk(logits.float(), top_k, dim=-1)

            # Write to memmap
            topk_i_cpu = topk_i.cpu().numpy().astype(np.uint16)
            topk_v_cpu = topk_v.cpu().to(torch.float16).numpy()

            for bi in range(actual_batch):
                si = batch_start + bi
                out_start = si * seq_len
                out_end = out_start + seq_len
                topk_indices[out_start:out_end] = topk_i_cpu[bi]
                topk_values[out_start:out_end] = topk_v_cpu[bi]

            positions_done += actual_batch * seq_len
            if (batch_start // batch_seqs) % 10 == 0:
                elapsed = time.perf_counter() - t0
                rate = positions_done / elapsed if elapsed > 0 else 0
                print(f"  [{shard_idx+1}/{total_shards}] {shard_name}: "
                      f"{positions_done}/{num_predictions} positions "
                      f"({rate:.0f} tok/s, {elapsed:.1f}s)")

        # Process remainder (last partial sequence)
        if remainder > 0:
            rem_start = num_seqs * seq_len
            # Pad to seq_len for the forward pass
            x_rem = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
            actual_rem = min(remainder, seq_len)
            x_rem[0, :actual_rem] = tokens[rem_start : rem_start + actual_rem].to(device)

            logits_rem = teacher_model.forward_logits(x_rem)
            topk_v_rem, topk_i_rem = torch.topk(logits_rem[:, :actual_rem].float(),
                                                  top_k, dim=-1)

            out_start = num_seqs * seq_len
            out_end = out_start + actual_rem
            topk_indices[out_start:out_end] = topk_i_rem[0].cpu().numpy().astype(np.uint16)
            topk_values[out_start:out_end] = topk_v_rem[0].cpu().to(torch.float16).numpy()
            positions_done += actual_rem

    # Flush and rename atomically
    topk_indices.flush()
    topk_values.flush()
    del topk_indices, topk_values
    os.rename(topk_path + ".tmp", topk_path)
    os.rename(vals_path + ".tmp", vals_path)

    elapsed = time.perf_counter() - t0
    print(f"[{shard_idx+1}/{total_shards}] {shard_name}: done "
          f"{positions_done} positions in {elapsed:.1f}s "
          f"({positions_done/elapsed:.0f} tok/s)")


def main():
    args = parse_args()

    # Set up distributed if launched with torchrun
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    # Import the training script to get model classes
    # We need GPT, deserialize, etc. from train_gpt_readable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # We import the module but need to handle env vars that the module reads
    os.environ.setdefault("VOCAB_SIZE", str(args.vocab_size))
    os.environ.setdefault("NUM_LAYERS", str(args.num_layers))
    os.environ.setdefault("MODEL_DIM", str(args.model_dim))
    os.environ.setdefault("EMBEDDING_DIM", str(args.embedding_dim))
    os.environ.setdefault("NUM_HEADS", str(args.num_heads))
    os.environ.setdefault("NUM_KV_HEADS", str(args.num_kv_heads))
    os.environ.setdefault("MLP_MULT", str(args.mlp_mult))
    os.environ.setdefault("LOGIT_SOFTCAP", str(args.logit_softcap))
    os.environ.setdefault("ROPE_BASE", str(args.rope_base))
    os.environ.setdefault("ROPE_DIMS", str(args.rope_dims))
    os.environ.setdefault("QK_GAIN_INIT", str(args.qk_gain_init))
    os.environ.setdefault("PARALLEL_RESIDUAL_START", str(args.parallel_residual_start))
    os.environ.setdefault("XSA_LAST_N", str(args.xsa_last_n))
    os.environ.setdefault("DATA_DIR", args.data_dir)

    import train_gpt_readable as tgr

    # Use the actual Hyperparameters class (reads env vars set above)
    # and override only the paths that differ for teacher model loading
    h = tgr.Hyperparameters()
    h.quantized_model_path = args.teacher_model
    h.compressor = args.compressor
    tgr.set_logging_hparams(h)

    print(f"Loading teacher model from {args.teacher_model}...")
    teacher_model = tgr.deserialize(h, device)

    # Enable looping if the model has depth recurrence
    if len(teacher_model.encoder_indices) != teacher_model.num_encoder_layers:
        teacher_model.looping_active = True

    # Compile for speed
    teacher_model = torch.compile(teacher_model, dynamic=False, fullgraph=True)
    print("Teacher model loaded and compiled")

    # Discover training shards
    shard_pattern = os.path.join(
        args.data_dir, "datasets", f"fineweb10B_sp{args.vocab_size}",
        "fineweb_train_*.bin")
    all_shards = sorted(glob.glob(shard_pattern))
    if not all_shards:
        raise FileNotFoundError(f"No training shards found: {shard_pattern}")

    # Filter to requested shards if specified
    if args.shards is not None:
        indices = [int(x) for x in args.shards.split(",")]
        all_shards = [all_shards[i] for i in indices if i < len(all_shards)]

    # Distribute shards across GPUs
    my_shards = all_shards[rank::world_size]
    total = len(all_shards)

    print(f"[Rank {rank}/{world_size}] Processing {len(my_shards)}/{total} shards")

    os.makedirs(args.output_dir, exist_ok=True)

    for si, shard_path in enumerate(my_shards):
        global_idx = rank + si * world_size
        process_shard(teacher_model, shard_path, args.output_dir,
                      top_k=args.top_k, batch_seqs=args.batch_seqs,
                      seq_len=args.seq_len, device=device,
                      shard_idx=global_idx, total_shards=total)
        torch.cuda.empty_cache()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    print(f"[Rank {rank}] All done!")


if __name__ == "__main__":
    main()
