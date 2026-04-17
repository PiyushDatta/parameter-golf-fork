# Non-Record Submission: SP8192 SOTA-Adapted with SWA + Half-Batch for Limited Hardware

**val_bpb = 1.1617** (3-seed mean, std 0.0010) | **~16.0 MB** | 4xA100 80GB

## 3-Seed Results (4xA100, sliding window eval stride=64)

| Seed | Post-GPTQ | Sliding BPB | Artifact Size |
|------|-----------|-------------|---------------|
| 42   | 1.1770    | **1.1608**  | ~15.997MB     |
| 137  | 1.1789    | **1.1627**  | ~15.998MB     |
| 7    | 1.1786    | **1.1616**  | ~15.997MB     |
| **Mean** | **1.1782** | **1.1617** | |
| **Std** | **0.0010** | **0.0010** | |

## Summary

SOTA-adapted architecture (PR #1493 stack) optimized for 4xA100 hardware. Two key findings: (1) **SWA dramatically outperforms EMA when training steps < 2000** (0.045 bpb difference), and (2) **halving batch size (393K tokens/step) gives 2x more steps and -0.006 bpb improvement** despite noisier gradients. All top leaderboard submissions use EMA and full batch, but on limited hardware these adaptations are critical.

## Key Techniques

1. **SP8192 tokenizer** -- 8192 BPE vocabulary for better token compression
2. **MLP 4.0x** (2048 hidden) -- fits 16MB only with LZMA code compression
3. **MuonEq-R optimizer** -- row-normalized Muon with Newton-Schulz 5 steps
4. **SDClip GPTQ** -- std-deviation based clip thresholds (k=12.85 matrix, k=20.0 embed)
5. **Mixed quantization** -- int6 for weight matrices, int8 for embeddings
6. **Parallel residuals** -- GPT-J style dual lanes from layer 7+
7. **SWA** (not EMA) -- 74 warmdown checkpoints averaged, critical for <1100 steps
8. **Brotli-11 compression** with byte-shuffle preprocessing
9. **LZMA code wrapper** -- 50KB -> 15.2KB code, enabling MLP 4.0x to fit

## Architecture

11L x 512d x 8H / 4KV, MLP 4.0x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. XSA on all 11 layers. Parallel residuals from layer 7 (attention and MLP operate on same pre-residual input). Skip gates (sigmoid-gated U-Net connections). NO depth recurrence (hurts on <1100 steps -- speed > depth).

## Training

MuonEq-R (row-normalized Muon, NS 5 steps), AdamW for embeddings/scalars. **~1912 steps** in 588s on 4xA100 80GB (~307ms/step) using **half batch (393K tokens/step)** instead of standard 786K. Muon momentum=0.95 (lower than SOTA's 0.99 -- optimized for fewer steps). Warmdown fraction=0.49. SWA collects ~131 checkpoints during warmdown. 128 training shards (12.8B tokens).

## Key Finding: SWA >> EMA Below ~1100 Steps

| Averaging | Steps | Post-GPTQ BPB | Delta |
|-----------|-------|---------------|-------|
| SWA (74 ckpts) | 1083 | **1.1666** | baseline |
| EMA (0.997) | ~900 | 1.2867 | +0.12 worse |
| EMA (0.999) | ~900 | 1.7487 | +0.58 catastrophic |
| SWA-over-EMA | ~900 | 1.3450 | +0.18 worse |

EMA includes early training weights via exponential averaging. With only ~1000 steps, early weights are garbage -- EMA can't escape them. SWA selectively averages only warmdown checkpoints (the best phase of training), avoiding early-weight contamination.

On 8xH100 with 4500+ steps, EMA works well (early weights are a small fraction). This crossover point (~1100 steps) has not been documented in other submissions.

## Hardware-Dependent Hyperparameters

| Parameter | 8xH100 (SOTA) | 4xA100 (Ours) | Why |
|-----------|---------------|---------------|-----|
| Averaging | EMA 0.9965 | **SWA** | EMA needs >2000 steps |
| Momentum | 0.99 | **0.95** | Lower momentum = faster adaptation |
| Warmdown | 0.72 | **0.49** | Less warmdown = more training at peak LR |
| Depth recurrence | 2 loops | **0 loops** | Loops cost steps; speed > depth |
| Weight decay | 0.095 | **0.085** | Less aggressive for fewer steps |

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` for principled rate-distortion clipping. int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). Byte-shuffle + Brotli-11 compression. Quant gap: ~0.012 bpb.

## 50+ Experiments Summary

Exhaustive exploration across two architecture generations:
- **SP1024 phase** (exp119-170): 44+ experiments. Plateau at 1.2272.
- **SP8192 phase** (exp171-196): 25+ experiments. Plateau at 1.1666.
- **Total improvement: -0.061 bpb** (1.2272 -> 1.1677 mean)

## Running

```bash
# Download SP8192 dataset
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# Train (4xA100)
DATA_DIR=./data/ SEED=42 torchrun --standalone --nproc_per_node=4 records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py

# Train (8xH100)
DATA_DIR=./data/ SEED=42 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py
```

## Requirements

See `requirements.txt`. Key additional package: `brotli`.
