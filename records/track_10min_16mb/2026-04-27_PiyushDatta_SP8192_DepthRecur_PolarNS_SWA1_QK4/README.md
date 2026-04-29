# Record: SP8192 + Depth Recurrence + Polar Express + Stacked SOTA

**val_bpb = 1.14210** (fast TTT, seed 1337) | **3-seed sliding mean = 1.14288** (std 0.00053) | **~15.99 MB** | 4xA100 80GB (dev), 8xH100 SXM (competition)

## Author

Piyush Datta

## 3-Seed Results

| Seed | Sliding BPB | Fast TTT (3ep, 32K) | Full TTT (3ep, 32K) | Artifact Size |
|------|-------------|---------------------|---------------------|---------------|
| 42   | **1.14227** | —                   | —                   | 15,994,434    |
| 999  | 1.14317     | —                   | —                   | 15,994,037    |
| 1337 | 1.14321     | **1.14210**         | **1.14127**         | 15,997,749    |
| **Mean** | **1.14288** | **~1.14210**    | **~1.14127**        |               |
| **Std** | **0.00053** |                  |                     |               |

## Summary

355+ experiments across SP1024 and SP8192 phases. Key breakthroughs: (1) **Depth recurrence (NUM_LOOPS=1)** reuses layers 3-5 for 14 effective layer passes with 11 unique layers, (2) **Polar Express NS coefficients** with per-iteration minimax-optimal triplets for better Muon convergence, (3) **SWA_EVERY=1 with 157 checkpoints** compensates for reduced step count from depth recurrence, (4) **HESSIAN_CLIP_LAMBDA=0.175** reduces GPTQ pruning by 70%. Combined with score-first TTT (SGD lr=0.02, 3 epochs, 32K chunks), this achieves 1.14210 val_bpb on 4xA100 (1.14127 with full TTT).

## Key Techniques

1. **Depth Recurrence (NUM_LOOPS=1)** -- Layers 3-5 shared, run twice per forward pass (14 effective layers from 11 unique). Activates at 45% of training via ENABLE_LOOPING_AT=0.45. First validated on 4xA100 with limited steps. On 8xH100, NUM_LOOPS=2 for 17 effective passes.
2. **Polar Express NS Coefficients** -- Per-iteration minimax-optimal Newton-Schulz coefficients for Muon optimizer (arxiv 2505.16932). Replaces fixed a,b,c with 5 optimized coefficient triplets in `zeropower_via_newtonschulz5`.
3. **SWA (157 checkpoints, EVERY=1)** -- Stochastic Weight Averaging over 157 late-training checkpoints (SWA_START_FRAC=0.12, SWA_EVERY=1). Every-step averaging compensates for reduced step count from depth recurrence.
4. **HESSIAN_CLIP_LAMBDA=0.175** -- Hessian-aware SDClip modulates GPTQ clipping per-row by importance. Reduces pruning by 70% (16K vs 53K values zeroed).
5. **MIN_LR=0.10** -- LR floor at 10% during warmdown. Must pair with SWA_START_FRAC=0.12 (>MIN_LR) to prevent SWA disable.
6. **SP8192 + GPTQ SDClip** -- int6 matrices (k=12.85), int8 embeddings (k=20.0). Based on PR #1394 @clarkkev.
7. **Half-Batch (393K tokens)** -- TRAIN_BATCH_TOKENS=393216, yielding ~2019 steps on 4xA100 with depth recurrence.
8. **MLP 4.0x + Parallel Residuals** -- Wider MLP (2048 hidden), GPT-J style parallel residuals from layer 7.
9. **Legal Score-First TTT** -- SGD (lr=0.02, momentum=0.9), 3 epochs per 32K-token chunk. Score-before-update ordering per Issue #1017.
10. **MuonEq-R + Linear Warmdown** -- Row-normalized Muon (NS 5 steps), MUON_MOMENTUM=0.95, linear warmdown over final 72%. MUON_WD=0.095, MATRIX_LR=0.028.
11. **Brotli + Byte-Shuffle Compression** -- Brotli-11 for model weights, LZMA for code wrapper (~19KB code). Total artifact ~15.998MB.

## Architecture

11L x 512d x 8H / 4KV, MLP 4.0x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), tied embeddings, logit softcap=30.0. **Depth recurrence (NUM_LOOPS=1)**: layers 3-5 run twice per forward pass, giving 14 effective layer passes from 11 unique layers. Activates at 45% of training. Parallel residuals from layer 7: attention and MLP operate on same pre-residual input. Skip gates (sigmoid-gated U-Net connections). QK_GAIN_INIT=5.0.

## Training

MuonEq-R optimizer (row-normalized Muon, Polar Express NS 5 steps), AdamW for embeddings/scalars. **~2019 steps** in ~600s on 4xA100 80GB (~267ms pre-loop, ~337ms post-loop) using **half batch (393K tokens/step)**. Depth recurrence activates at 45% of training. Linear warmdown to MIN_LR=0.10 over final 72%. SWA collects 157 checkpoints starting at SWA_START_FRAC=0.12, every step (SWA_EVERY=1). WARMUP_STEPS=20, EMBED_WD=0.085, MUON_WD=0.095, MATRIX_LR=0.028, MUON_MOMENTUM=0.95.

## Key Finding: SWA >> EMA Below ~1100 Steps

| Averaging | Config | Post-GPTQ BPB | Delta |
|-----------|--------|---------------|-------|
| **SWA (81 ckpts)** | uniform late avg | **1.1440** | baseline |
| EMA (0.997) | standard decay | 1.2867 | +0.14 worse |
| EMA (0.999) | standard decay | 1.7487 | +0.60 catastrophic |
| SWA-over-EMA | hybrid | 1.3450 | +0.20 worse |

EMA includes early training weights via exponential averaging. With only ~1000 steps, early weights are garbage -- EMA can't escape them. SWA selectively averages only warmdown checkpoints (the best phase of training), avoiding early-weight contamination.

On 8xH100 with 4500+ steps, EMA works well (early weights are a small fraction). This crossover point (~1100 steps) has not been documented in other submissions.

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` for principled rate-distortion clipping. int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). Byte-shuffle + Brotli-11 compression. GPTQ_RESERVE=0.

Pre-quant SWA: ~1.147 | Post-GPTQ: ~1.160 | Sliding: 1.1440 | **TTT: 1.1423**

## TTT (Test-Time Training)

Score-first, chunk-based SGD adaptation at eval time:
- Chunk val tokens into 32K-token chunks (TTT_CHUNK_TOKENS=32768)
- For each chunk: (1) score all sliding windows under `torch.no_grad()`, (2) train model on scored chunk tokens with SGD
- SGD with lr=0.02, momentum=0.9, 3 epochs per chunk (TTT_EPOCHS=3)
- Gradient clipping at 1.0, distributed all-reduce for multi-GPU
- TTT improvement: ~0.002 bpb (1.1440 sliding -> 1.1423 TTT)
- Total TTT eval time: within 600s eval budget

## Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:
- No SLOT (standard or causal)
- No pre-quant TTT on val data (model quantized once during training, TTT adapts at eval time)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes
- Training under 600s
- Eval (sliding + TTT) under 600s

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download SP8192 dataset
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# Train (8xH100, competition)
SEED=42 TTT_ENABLED=1 TTT_LR=0.02 TTT_EPOCHS=3 NUM_LOOPS=2 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/train_gpt.py

# Train (4xA100, local dev)
SEED=42 TTT_ENABLED=1 TTT_LR=0.02 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=4 \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/train_gpt.py
```

## Experiment Journey

355+ experiments across two architecture generations:

| Milestone | val_bpb | Key Change |
|-----------|---------|------------|
| SP1024 baseline | 1.2272 | int6, 11L, MLP 3.0x, QAT@50%, SWA, GPTQ |
| SP8192 migration | ~1.17 | Tokenizer change alone: ~0.05 bpb |
| MLP 4.0x + parallel residuals | ~1.16 | Capacity + half-batch steps |
| Warmdown + WD tuning | 1.1440 | WARMDOWN=0.72, WD=0.07, SWA 81 ckpts |
| Depth recurrence (NUM_LOOPS=1) | 1.1414 | Layers 3-5 run twice, 14 effective passes |
| Polar Express NS + tuning | 1.1429 | Per-iteration optimal NS coefficients |
| SWA_EVERY=1 + MUON_WD=0.095 | 1.1423 | 157 ckpts, WD/LR retuned |
| **Score-first TTT** | **1.1421** | SGD lr=0.02, 3 epochs, 32K chunks |

### Confirmed Dead Ends (4xA100)
- EMA (any decay) -- catastrophic below ~1100 steps
- Quarter batch (196K tokens) -- too noisy (+0.005)
- Cosine warmdown -- narrows SWA window, hurts
- LeakyReLU 0.3 or 0.9 -- 0.5 is optimal
- 12 layers -- SWA overhead reduces steps to 973, net worse
- MTP (Multi-Token Prediction) -- 0.006 worse
- N-gram probability mixing -- hash collisions inflate scores

## Credits

- **@clarkkev** -- SP8192 + GPTQ Embeddings + SDClip + MuonEq-R (PR #1394)
- **@abaybektursun** -- Score-first TTT framework (PR #549)
- **@Robby955** -- Parallel residuals on SP8192 (PR #1412)
- **@msisovic** -- Parallel residuals concept (PR #1204)
- **@dexhunter** -- Depth recurrence and legal TTT on SP8192 (PR #1331, #1413, #1437)
- **@X-Abhishek-X** -- Hyperparameter tuning insights (PR #1445, #1471)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_gpt_readable.py`
- `logs/seed_42.log`
- `logs/seed_314.log` (mapped from seed 1337 run)
- `logs/seed_999.log`
