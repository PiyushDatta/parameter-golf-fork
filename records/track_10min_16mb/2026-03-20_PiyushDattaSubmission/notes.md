## TEMP NOTES: Piyush Datta
baseline uses u-net/skip connections, can we do better? put router in middle, deeper layers send to router, router gives back residuals from earlier layers - but router decides

# Parameter Golf - Optimization Notes & Scratchpad

## Competition Rules (from README, issue #677, issue #1017)

### Core Constraints
- Train best LM fitting in 16MB artifact, trains in <10min on 8xH100 (SXM variant)
- Eval metric: val_bpb (bits per byte) on FineWeb validation set - LOWER is better
- **10min training** + **10min eval** (two separate 10-min budgets)
- Our hardware: 4x A100 80GB (competition uses 8xH100)
- Tokenizer: fineweb_1024_bpe.model (SP, vocab_size=1024) - CANNOT change
- Dataset: fineweb10B_sp1024 - CANNOT change
- Challenge runs March 18 to April 30, 2026

### Artifact Size Rules
- Artifact = code bytes + compressed model bytes < **16,000,000 bytes** (decimal 16MB, NOT 16 MiB/16,777,216)
- All counted code should live in the `train_gpt.py` script
- No external downloads, training dataset access, or network calls during evaluation
- Artifact must be fully self-contained and reproducible
- External libraries (PyTorch, FlashAttention, etc.) permitted, don't count toward 16MB

### Compute Budget Rules
- Training: at most 10 minutes on 8×H100 SXM GPUs
- Evaluation: at most 10 minutes **additional** on the same hardware
- **No compute may be transferred between phases**
- Procedures that consume data to modify model state (e.g. GPTQ/Hessian calibration) belong
  to the **training** budget. Performing them during evaluation is **not permitted**.
- Common violation: people train for full 10min, then do GPTQ calibration (4-40s), then 10min
  eval — this is illegal because GPTQ calibration is training-phase compute

### Four Conditions for Valid val_bpb (from issue #1017, NoesisGenesis)

These are jointly necessary and sufficient for val_bpb to be a valid information-theoretic
measure (prequential code length of a causal predictor).

**Condition 1 (Strict causal dependence)**: At position t, the predictive distribution
p_t(·) must be a function only of the submitted artifact A and the strict prefix
x_1, …, x_{t−1}. It may not depend on x_t or any future token, on any statistic computed
from future validation tokens, or on any external data not already encoded in A.

**Condition 2 (Full normalized distribution)**: Before x_t is scored, the submission must
define a single full probability distribution over the official token alphabet Σ. For every
a ∈ Σ, p_t(a) ≥ 0 and ∑ p_t(a) = 1. This distribution must be determined independently
of which token is realized at position t. It may NOT be constructed by evaluating only
the realized token and filling in remaining mass. Normalization must hold over actual
tokens, not over internal buckets, hash bins, or other latent structures.

**Condition 3 (Score-before-update)**: The score at position t is computed from p_t(x_t).
Only AFTER that score is fixed may state be updated using x_t. The current symbol may not
influence its own assigned probability, directly or indirectly.

**Condition 4 (Single left-to-right pass)**: Evaluation consists of exactly one left-to-right
pass. No rescoring, no second pass, no retrospective revision of earlier probabilities,
no selection among multiple executions based on observed validation outcomes.

### Two Tracks (from issue #1017)

**Track A — Fixed Predictor**: Model is trained, then evaluated. During evaluation, no model
state is updated from validation tokens. Score measures quality of a fixed predictor.
- Permitted: sliding-window attention, KV-cache strategies, inference optimizations
- Not permitted: eval-built n-gram caches, TTT, adaptive mixing from eval statistics

**Track B — Adaptive Compression**: Model may adapt state during eval using previously
scored tokens. Measures how good the predictor is at becoming better while predicting.
- Permitted: Score-first TTT (score chunk → train on it), per-document LoRA with
  score-before-update discipline, causal n-gram caches from already-scored tokens
- Not permitted: Score-after-adapt TTT (adapt then score same tokens), multi-pass
  rescoring, cache state at position t reflecting tokens at or beyond t

All four conditions apply to BOTH tracks.

### Common Violation Patterns (from issue #1017, Section VI)

| Pattern | Violates | Why |
|---------|----------|-----|
| Eval-built n-gram cache with state at t reflecting tokens ≥ t | Cond 1 | Future tokens influence current prediction |
| Two-pass full rescore | Cond 1, 4 | Second pass uses state from tokens after each scored position |
| Score-after-adapt TTT (adapt on chunk, then score same chunk) | Cond 3 | Current tokens influence their own probabilities |
| Oracle selection via min() across passes | Cond 4 | Multiple executions, selection on observed outcomes |
| Entropy expert in context mixer (scalar of neural dist, not a dist over Σ) | Cond 2 | Mixed result is not a normalized distribution over Σ |
| Hardcoded bytes-per-token constant | — | Incorrect metric (equally fatal) |

### Explicitly Invalid Patterns (from issue #677)

**Invalid TTT (closed PRs):**
- #410, #415, #417: two-phase/multi-epoch adaptation on validation, then reported score after
- #442, #462, #481, #486, #517, #518, #532, #555, #581, #595: adapt on validation before eval
- #512, #548, #568: multi-epoch TTT reports final-pass/final-epoch score after earlier adaptation
- #576: re-scores full validation set after TTT with fully adapted model
- #684: multi-epoch TTT trains on eval tokens, takes loss at last epoch

**Invalid eval (closed PRs):**
- #535, #544, #545, #569, #585, #593, #606, #615, #626, #639, #674: GPTQ/Hessian calibration
  uses fineweb_train_* during evaluation (this is training-phase compute, not eval)
- #573: oracle/hindsight min() selection across multiple passes
- #659: checks NLL of both model and 5-gram and picks lower — uses knowledge of true token

**Invalid n-gram caches (mass closure):**
- Hashed n-gram caches that aren't properly normalized over the full token alphabet
- The hash bucket implementation creates correlations/collisions requiring re-normalization
- Without proper normalization: sum_i P_ngram(token_i | context) >> 1, making BPB invalid
- "No sub-1.0 submission survives" under proper normalization rules (NoesisGenesis)
- Properly normalized n-gram BPB (from mhuen, built on 200M tokens, no hashing):
  1-gram: ~3.48, 2-gram: ~2.45, 3-gram: ~1.87, 4-gram: ~1.70
- Many PRs closed: #846, #868, #869, #870, #881, #888, #893, #907, #921, etc.
- valerio-oai: no new submissions merged; reviewing only PRs after #988 going forward

### Evaluation Correctness (from issue #1017)

- **val_bpb is bits per byte, NOT bits per token**
  - Must compute per-token byte lengths from sentencepiece vocabulary
  - Account for ▁ (U+2581) leading space (1 byte, part of piece string)
  - Account for byte fallback tokens (exactly 1 byte each)
  - Exclude boundary tokens (BOS, EOS, UNK, control) which encode 0 bytes
  - Use `build_sentencepiece_luts()` for lookup tables
  - `val_bpb = (total_cross_entropy_nats / log(2)) * (token_count / byte_count)`
  - Do NOT use hardcoded constants like `val_loss / (log(2) * 3.5)`
- **Evaluate on ALL validation shards** (fineweb_val_*.bin), not just first one
- **Train on ALL training shards**, not just first one
- **Do NOT reorder the validation set** — default shard/token order must be preserved

### Submission Requirements
- Must beat existing SOTA by ≥0.005 nats at p < 0.01
- Provide training logs from at least 3 independent runs (different seeds)
- PRs reviewed chronologically by creation time
- Pure systems optimizations exempt from 0.005-nats threshold
- PR adds new folder to `/records/track_10min_16mb/[date]_[description]/` with:
  1. `train_gpt.py` — compilable and runnable from records folder
  2. `README.md` — methodology and results
  3. `submission.json` — name, GitHub ID, val_bpb, metadata
  4. Training logs from 3+ seeds
  5. `requirements.txt` for any additional packages
- Tokenizer/dataset changes require proof of correct val_bpb calculation
- Seed brute-forcing or offline validation-set optimization → disqualification
- If submission does not beat rank 1, submit under `track_non_record_16mb` instead

### ⚠️ Example Compliance Issues (as of Exp 63)
1. **TTT trains on unevaluated tokens** — We do full-weight SGD on ALL val tokens BEFORE
   evaluating. Violates Conditions 3 and 4. Must switch to **score-first TTT** (Track B):
   score a chunk → lock in loss → train on scored chunk → next chunk.
2. **Eval time exceeds 10 min** — Our TTT (120ep × ~21s/ep ≈ 42min) + sliding eval (~190s)
   ≈ 45min total. Must fit all eval-time compute (including TTT) within 600s on 8xH100.
3. **Need 3-seed validation** — Must show p<0.01 with 3+ seeds.
4. **GPTQ/Hessian calibration**: If any calibration uses training data during eval phase,
   that's invalid. Must ensure all calibration happens within training budget.
5. **val_bpb computation**: Must verify we use proper byte-level BPB with sentencepiece
   lookup tables, not hardcoded constants.

## Competition Landscape (as of 2026-03-25)

### Merged Leaderboard
| Rank | PR | Author | val_bpb | Key Techniques |
|------|-----|--------|---------|----------------|
| 1 | #398 | felipe-parodi | **1.1213** | 11L EMA + aggressive TTT (20ep, all blocks unfrozen) |
| 2 | #415 | fbedev | **1.1216** | 11L XSA4 + FA3 + Two-Phase TTT |
| 3 | #388 | ElliotSlusky | **1.1231** | 11L + Tight SWA + VE128 + TTT (25ep) |
| 4 | #414 | signalrush | **1.1233** | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 |

### Top Pending PRs (from research agent)
| PR | val_bpb | Author | Key Techniques |
|----|---------|--------|----------------|
| #568 | **0.7853** | PROTEUS | 11L INT6 + LoRA TTT 5ep cosine, score every epoch |
| #512 | **0.9512** | PROTEUS | 11L INT6 + LoRA TTT 3ep |
| #573 | **1.0523** | Sarimsaljook | Multi-Pass Streaming Score-First TTT |
| #548 | **1.0865** | LoquiAuris | Per-document LoRA TTT rank-8, batched 64 docs/GPU |
| #576 | **1.1164** | cmcdnd | 33.6M params + int5 GPTQ + Score-First TTT |
| #593 | **1.1171** | abaybektursun | Full Hessian GPTQ + LeakyReLU(0.5)^2 (no TTT) |

# Experiment Runner Guide

## Overview

`run_experiment.py` is a wrapper around `train_gpt.py` that:
- Runs training with frequent validation checkpoints (every ~12s)
- Records `val_bpb` at wall-clock time boundaries (30s, 1m, 2m, ..., 10m)
- Compares against a saved baseline and **kills training early** if the experiment is worse

## Files

| File | Purpose |
|------|---------|
| `run_experiment.py` | The runner script |
| `baseline.json` | Saved baseline val_bpb at each time checkpoint (auto-generated) |
| `experiment_results/` | Directory where experiment results are saved as JSON |
| `train_gpt.py` | The actual training script (modified via env vars) |

## Baseline

The baseline is stored in `baseline.json` — a simple dict mapping seconds to `val_bpb`:

```json
{
  "30": 2.1149,
  "60": 1.7672,
  "120": 1.5372,
  "180": 1.4503,
  "240": 1.3981,
  "300": 1.3581,
  "360": 1.3349,
  "420": 1.3166,
  "480": 1.301,
  "540": 1.2913,
  "600": 1.2827
}
```

These are **pre-quant val_bpb** values measured on **4x A100 80GB** GPUs with the default `train_gpt.py` config (no TTT, EMA enabled, 11L dim=512).

The baseline only needs to be generated once. If you change the base model architecture, re-run `--baseline` to update it.

## Usage

All commands should be run from the repo root (`/data/repos/parameter-golf-fork/`). Always activate the venv first:

```bash
source .venv/bin/activate
```

### 1. Show the saved baseline

```bash
python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py --show-baseline
```

### 2. Re-generate baseline (only if needed)

```bash
with-proxy python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py --baseline
```

This runs a full 10-min training with `VAL_LOSS_EVERY=25` and saves checkpoints to `baseline.json`. Takes ~14 min (including torch.compile warmup + sliding window eval).

### 3. Run an experiment (with early stopping)

```bash
with-proxy python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py \
  --name "my_experiment_name" \
  --env "KEY1=val1,KEY2=val2"
```

The `--env` flag passes extra environment variables to `train_gpt.py`. These override hyperparameters defined in the `Hyperparameters` class. Common ones:

| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 11 | Number of transformer layers |
| `MODEL_DIM` | 512 | Model dimension |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_KV_HEADS` | 4 | KV heads (GQA) |
| `MLP_MULT` | 3.0 | MLP hidden multiplier |
| `BIGRAM_VOCAB_SIZE` | 2048 | Bigram hash table size |
| `EMA_DECAY` | 0.997 | EMA decay rate |
| `WARMDOWN_ITERS` | 3500 | LR warmdown iterations |
| `TTT_ENABLED` | 1 | Enable test-time training |
| `TTT_LR` | 1.3 | TTT learning rate |
| `TTT_MODE` | score_first | TTT mode |
| `QUANT_BITS` | 8 | Quantization bits (6 or 8) |
| `SEED` | 42 | Random seed |

### 4. Run without early stopping

```bash
with-proxy python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py \
  --name "my_experiment" \
  --env "MODEL_DIM=576" \
  --no-early-stop
```

### 5. Specify GPU count

```bash
with-proxy python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py \
  --name "my_experiment" \
  --env "MODEL_DIM=576" \
  --nproc 2
```

If `--nproc` is omitted, the script auto-detects available GPUs via `nvidia-smi`.

## Early Stopping Behavior

- At each time checkpoint (30s, 1m, 2m, ...), the script compares experiment `val_bpb` to baseline
- If the experiment is **worse by more than 0.002** at **2 consecutive checkpoints** after the 2-minute mark, it kills the training process
- This saves GPU time — no point running 10 minutes if the change is clearly worse by minute 3

## Output

During the run, you'll see real-time comparisons:

```
  [ 30s] val_bpb=2.0500  baseline=2.1149  diff=-0.0649 + BETTER
  [  1m] val_bpb=1.7200  baseline=1.7672  diff=-0.0472 + BETTER
  [  2m] val_bpb=1.5500  baseline=1.5372  diff=+0.0128 x WORSE
  [  3m] val_bpb=1.4700  baseline=1.4503  diff=+0.0197 x WORSE

  EARLY STOP: worse than baseline at 2 consecutive checkpoints. Killing training.
```

At the end, a summary table is printed and results are saved to `experiment_results/<name>_<timestamp>.json`.

## Important Notes

- The script sets `TTT_ENABLED=0` by default (training-only, no TTT eval). This is intentional — TTT eval takes 10+ additional minutes and we want fast iteration on training changes.
- The baseline was generated on **4x A100 80GB**. If you switch to different hardware, re-run `--baseline`.
- `val_bpb` during training is **pre-quantization**. The final post-quant number will be slightly worse (~+0.02 for int8).
- All experiment env vars are passed through to `train_gpt.py` — any `os.environ.get()` in `Hyperparameters` can be overridden.

## Our Current Best Config (Exp 57: val_bpb=1.0312 on 4xA100)
- 11 layers, dim=512, 8 heads, 4 KV heads (GQA), 3x MLP (**LeakyReLU(0.5)^2**)
- BigramHash(2048, dim=128) + SmearGate
- **Int8 quantization** (clip_range=127) + zlib compression (~15.8MB, fits!)
- EMA (decay=0.997)
- Value Residual (ResFormer) - cache V from layer 0, mix via learned lambda
- VE128 (ValueEmbedding) on layers 9, 10
- Gated Attention - per-head sigmoid gate after SDPA
- XSA on last 4 layers
- Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer_idx+1))
- **Full-weight SGD TTT** (120 epochs, warmup=90 (75%), LR=0.9, momentum=0.9, grad_clip=1.0)
- **TTT warmup**: linear LR ramp for first 75% of epochs, then cosine decay
- **TTT SWA**: average weights from last 25% of epochs (huge -0.018 bpb FREE)
- Adaptive TTT: TTT_MAX_SECONDS auto-sizes epoch count to time budget
- FA3 conditional import (falls back to SDPA on A100)
- Eval: sliding window stride=64, **temperature=1.0**

## Key Findings (ordered by impact)

### 1. Int8 >> Int6 Quantization (-0.058 bpb)
- Int8 (clip_range=127) + zlib ≈ 15.8MB — fits in 16MB!
- Quantization gap: 0.035 bpb (int8) vs 0.092 bpb (int6) = 2.6x reduction
- **CRITICAL WARNING**: Int8+zlib size scales with training steps!
  - 1310 steps → 15.8MB (fits)
  - 3916 steps → 21.9MB (DOES NOT FIT!)
  - On 8xH100 (~7000 steps), weights have higher entropy → may not fit
  - Must verify on actual 8xH100 run, may need to fall back to int6

### 2. TTT LR=0.9 is Optimal (-0.17 bpb vs LR=0.01)
- Full sweep: 0.01→0.02→0.03→0.05→0.10→0.20→0.50→0.70→0.85→0.90 all improved
- LR=1.0 diverges (loss → 7.0)
- Gradient clipping (1.0) is ESSENTIAL — without it, LR=0.9 diverges
- Cosine schedule: lr = lr_init * 0.5 * (1 + cos(π * epoch / total_epochs))

### 3. TTT Warmup + SWA (-0.071 bpb total, Session 6 discovery)

#### 3a. TTT Warmup (-0.016 bpb at 40 epochs)
- Linear LR warmup for first 75% of TTT epochs, then cosine decay for remaining 25%
- **Warmup sweep (40 epochs)**: 0%→1.1024, 37.5%→1.0895, 75%→**1.0865**, 87.5%→1.0889(worse)
- **Warmup enables more epochs without overfitting!**:
  - WITHOUT warmup: 50ep=1.1032(overfit), 100ep=1.1635(BAD)
  - WITH 75% warmup: 50ep=1.0725, 60ep=1.0648, 80ep=1.0618

#### 3b. TTT SWA (-0.018 bpb FREE!)
- Average model weights from last 25% of TTT epochs (cosine decay phase)
- 80ep: 1.0618 → **1.0442** (-0.018), 100ep: 1.0677 → **1.0338** (-0.034!)
- SWA rescues overfitting — 100ep overfit without SWA, but with SWA beats 80ep

#### 3c. Scaling with warmup+SWA
- 80ep=1.0442, 100ep=1.0338, 120ep=**1.0312**, 150ep=1.0313 (plateau)
- **Saturates at ~120 epochs on 4xA100** (pre-quant quality is the bottleneck)
- Adaptive TTT (TTT_MAX_SECONDS): runs 1 calibration epoch, auto-sizes to time budget
- **Competition budget**: ~21s/epoch on 8xH100, ~410s TTT budget → ~19 epochs

### 4. LeakyReLU(0.5)^2 > relu^2 (-0.008 bpb pre-quant)
- One-line change in MLP: `F.leaky_relu(x, negative_slope=0.5).square()`
- Consistent improvement across all experiments
- Used by top pending PRs (#593, #549)

### 5. Temperature=1.0 > 0.98 (-0.003 bpb)
- No temperature scaling is better than cool temperature
- Post-TTT, the model is already well-calibrated

### Negative Results
- **LoRA TTT (our implementation)**: Plateaued at 1.450 vs 1.300 full-weight. Per-document rank-8 is insufficient.
  - Note: PROTEUS gets 0.78 with LoRA TTT — different implementation (score every epoch, cosine)
- **Two-Phase TTT**: 1.386 vs 1.300 full-weight. Norm recalibration doesn't help enough.
- **TTT weight decay=0.01**: Loss → 5.0. Way too high with LR=0.9.
- **No gradient clipping**: TTT diverges at LR=0.9.
- **QAT on 4xA100**: Only 1 step runs (wallclock cap), wastes 200 training steps.

## Experiment Log (Sessions 1-3)

### Experiments 1-5 (Previous Sessions)
- Exp 1: 1xA100, 398 steps, val_bpb=1.7144 (first integration)
- Exp 3 v5: 1xA100, 344 steps, val_bpb=2.0202 (LN Scale, full stack)
- Exp 4: 4xA100, ~1000 steps, val_bpb=1.7234 (first 4-GPU run)

### Experiment 6: VE128 + FA3 + Late QAT
- 4xA100, 1112 steps (QAT overhead), pre-quant: 1.3020, post-quant+TTT: **1.3773**
- First run with VE128, FA3 conditional, Late QAT
- QAT only ran 1 step on 4xA100 (wallclock cap)

### Experiment 7: No QAT, Higher TTT LR
- 4xA100, 1312 steps (200 more without QAT), pre-quant: 1.2855, post-quant+TTT: **1.3727**
- Confirmed: disabling QAT on 4xA100 gives more steps and better results

### Experiment 8: LeakyReLU + LoRA TTT + Temperature
- 4xA100, 1308 steps, pre-quant: 1.2779 (LeakyReLU helps! -0.008)
- LoRA TTT plateaued at ~1.450 — killed early, clearly worse than full-weight TTT

### Experiment 9: LeakyReLU + Full TTT + Temperature=0.98
- 4xA100, 1307 steps, pre-quant: 1.2780, post-quant+TTT: **1.3702**

### Experiment 10: Int8 Quantization (BREAKTHROUGH)
- Same training as Exp 9, but QUANT_BITS=8 (clip_range=127)
- Post-quant+TTT: **1.3125** (-0.058 vs int6!)
- Model size: 15.77MB (fits in 16MB!)

### TTT LR Sweep (Experiments 11-22)
| Exp | TTT LR | TTT Epochs | Post-quant+TTT |
|-----|--------|------------|----------------|
| 11  | 0.01   | 20         | 1.3000         |
| 13  | 0.02   | 20         | 1.2813         |
| 14  | 0.03   | 20         | 1.2730         |
| 15  | 0.05   | 20         | 1.2583         |
| 16  | 0.10   | 20         | 1.2365         |
| 17  | 0.20   | 20         | 1.2100         |
| 18  | 0.50   | 20         | 1.1707         |
| 20  | 0.70   | 20         | 1.1586         |
| 21  | 0.85   | 20         | 1.1509         |
| 22  | 0.90   | 20         | 1.1483         |
| 19  | 1.00   | 20         | DIVERGED       |

### TTT Epoch Sweep (Experiments 23-29, 34)
| Exp | Epochs | Temp | TTT Loss | Post-quant+TTT |
|-----|--------|------|----------|----------------|
| 23  | 30     | 0.98 | 1.81     | 1.1199         |
| 24  | 50     | 0.98 | 1.64     | 1.1060         |
| 26  | 50     | 1.0  | 1.64     | 1.1032         |
| **29** | **40** | **1.0** | **1.72** | **1.1024** |
| 34  | 60     | 1.0  | 1.56     | 1.1110 (slight overfit) |
| 28  | 100    | 1.0  | 1.40     | 1.1635 (overfit!) |

### Experiment 30: Extended Training (30 min)
- 4xA100, 3916 steps (30 min), pre-quant: 1.1632
- **Int8+zlib: 21.9MB — DOES NOT FIT!** (model weights have higher entropy)
- TTT loss started at 3.01 (vs 2.17 for 10min model) — quantization worse
- **CRITICAL FINDING**: int8+zlib model size scales with training steps
- Must verify int8 fits on 8xH100, or fall back to int6

### Session 4: Score-Every-Epoch TTT + MoE

### Experiment 33: Score-Every-5 TTT (20 epochs)
- Same as Exp 22 (20 TTT epochs) but scores every 5 epochs, keeps best
- val_bpb kept improving monotonically (no overfitting in 20ep) → best=epoch 20
- Post-quant+TTT: **1.1481** (essentially same as Exp 22: 1.1483)
- Conclusion: score-every-epoch doesn't help when no overfitting occurs

### Experiment 35: Score-Every-10 TTT (60 epochs)
- 60 TTT epochs, scores every 10 epochs via fast non-overlapping eval
- Fast eval progression: epoch 10→1.210, 20→1.130, 30→1.063, 40→1.004, 50→0.947, **60→0.927**
- **BUT sliding window eval: 1.1104** (same as Exp 34!)
- **CRITICAL**: Fast eval (non-overlapping) gives very different numbers from sliding window eval
- Score-every-epoch picked epoch 60 as best → same as blind 60 epochs
- The TTT doesn't actually overfit on sliding window; it overfits on non-overlapping eval metric

### MoE Implementation
- Implemented 2-expert MoE with explicit CastedLinear per expert (no einsum)
- torch.compile-friendly: uses mask-based routing, no data-dependent branching
- Not yet tested end-to-end (torch.compile took too long on first attempt)

### Session 5: Hadamard Quant, Bigram, Nesterov, TTT Warmup

### Experiment 37: Fresh Baseline (Confirmed)
- 4xA100, 1309 steps, pre-quant: 1.2779, quant_mse: 5.04e-6
- Post-quant+TTT: **1.1024** (confirmed previous best)
- Model size: 15,680,349 bytes (fits)

### Experiment 38: Hadamard Rotation Quantization
- Applied Walsh-Hadamard Transform to weight rows before int8 quantization
- Theory: rotation spreads outliers → better quantization distribution
- **quant_mse: 5.06e-6** (virtually identical to baseline 5.04e-6)
- **Model size: 15,675,320** (marginally smaller)
- Conclusion: **No benefit at int8**. GPTQ-lite percentile search already handles outliers.
- Hadamard code removed to save ~1.4KB code space.

### Experiment 39: BiggramHash(4096)
- Pre-quant: 1.2741 (-0.0038 improvement over baseline!)
- Post-quant+TTT: 1.1101 (+0.008 worse — noise/overfitting)
- **Model size: 16,011,915 bytes — DOES NOT FIT** (over by 12KB with old code)
- After code cleanup (removed ~16KB of dead code), would fit (~15.99MB)
- Net effect unclear — pre-quant better but post-TTT noisier

### Experiment 40: Nesterov Momentum TTT
- TTT with nesterov=True, LR=0.9, momentum=0.9
- **DIVERGED**: Loss stuck at 6.07 (random level) from epoch 1
- Nesterov's look-ahead step + LR=0.9 = too aggressive, pushes past stability boundary
- **Conclusion: Nesterov incompatible with high TTT LR**

### Experiment 41: TTT Warmup=5 (NEW BEST!)
- Linear warmup for first 5 TTT epochs (LR: 0.18→0.36→0.54→0.72→0.90)
- Then cosine decay over remaining 35 epochs
- TTT final loss: 1.711 (vs baseline 1.716 — slightly better)
- Post-quant+TTT: **1.0962** (-0.0062 vs baseline 1.1024!)
- **NEW BEST on 4xA100!**

### Code Cleanup (Session 5)
- Removed Hadamard quant code (proved unhelpful), score-every-epoch TTT (proved unhelpful), LoRA TTT eval (proved worse)
- Code size: 91.6KB → 75.7KB (saved ~16KB)
- Fewer lines: 2038 → ~1680

### Session 6: TTT Warmup + SWA Discovery (Exp 42-58)

#### Thought Process

**Starting point**: Exp 41 showed warmup=5 helps TTT (1.0962 vs 1.1024). But what's
the optimal warmup ratio? And can we scale beyond 40 epochs?

**Phase 1: Warmup Ratio Sweep (Exp 42-49)**

The key question: what fraction of TTT epochs should be linear warmup vs cosine decay?

Ran sweep at 40 epochs total with warmup = 0, 3, 5, 8, 10, 15, 20, 25, 30, 35:
- Improvement was monotonic up to 75% warmup (30/40 = 1.0865)
- 87.5% warmup (35/40) was WORSE (1.0889) — not enough cosine decay epochs
- Diminishing returns: 37.5%→1.0895, 50%→1.0883, 62.5%→1.0873, 75%→1.0865

**Insight**: 75% warmup is optimal. The warmup phase slowly raises LR, letting the
model adapt gradually. The 25% cosine decay phase refines the weights at high LR
then anneals to zero. Too much warmup (87.5%) doesn't leave enough decay time.

**Phase 2: Scaling Epochs with Warmup (Exp 50-52, 54)**

Previously, 50+ epochs without warmup caused overfitting (50ep=1.1032, 100ep=1.1635).
Hypothesis: warmup's regularization effect might prevent this.

Results confirmed — warmup enables scaling:
- 50ep warmup=38: **1.0725** (vs 1.1032 without warmup)
- 60ep warmup=45: **1.0648**
- 80ep warmup=60: **1.0618** (still improving!)
- 100ep warmup=75: **1.0677** (slight overfit — cosine decay can't rescue all of it)

**Insight**: Warmup prevents overfitting by controlling how fast the model learns.
The linear ramp acts as implicit regularization. But at 100+ epochs without weight
averaging, even warmup can't prevent eventual overfitting.

**Phase 3: SWA During TTT (Exp 55-58) — BREAKTHROUGH**

The question: if epochs 60-79 produce different weight snapshots during cosine decay,
can averaging them find a flatter minimum?

Added SWA (Stochastic Weight Averaging): collect model weights from last 25% of
epochs (= the cosine decay phase), average them at the end.

Results were dramatic:
- 80ep WITHOUT SWA: 1.0618 → WITH SWA: **1.0442** (-0.018 bpb for FREE!)
- 100ep WITHOUT SWA: 1.0677 (overfit) → WITH SWA: **1.0338** (SWA rescues overfitting!)
- 120ep + SWA: **1.0312** (NEW BEST)
- 150ep + SWA: 1.0313 (plateau — pre-quant quality is now the bottleneck)

**Why SWA works so well for TTT**: During cosine decay, the optimizer explores
different regions of the loss landscape at decreasing learning rates. Each snapshot
is good but noisy. Averaging produces a smoother minimum that generalizes better.
The effect is especially powerful because TTT trains on the validation set itself —
individual snapshots can memorize noise, but the average cancels it out.

**Phase 4: Adaptive TTT for Competition**

On 8xH100, TTT gets ~410 seconds (10min eval - ~190s sliding eval). Each epoch
takes ~21s. That's only ~19 epochs — much fewer than our optimal 120.

Implemented adaptive TTT (`TTT_MAX_SECONDS`):
1. Run one calibration epoch to measure time
2. Calculate how many epochs fit in remaining budget
3. Auto-set warmup to 75% of computed epochs
4. SWA collects from last 25%

Also tested batch_seqs=64 — only 10% faster per epoch (75s vs 84s) because the
bottleneck is compute, not overhead. Not worth the fewer gradient updates.

**Final configuration (defaults in code)**:
- `TTT_EPOCHS=120` (fallback when no time limit)
- `TTT_WARMUP = 75%` of total epochs (auto-computed)
- `TTT_SWA = 1` (enabled by default, last 25% of epochs)
- `TTT_MAX_SECONDS` for adaptive sizing on competition hardware

**Total TTT schedule improvement: -0.071 bpb** (1.1024 → 1.0312)

## Final Summary Table (4xA100, 10min training)
| Exp | Pre-quant | Post-quant+TTT | Key Change |
|-----|-----------|----------------|------------|
| 4   | N/A       | 1.7234         | Old baseline |
| 6   | 1.3020    | 1.3773         | VE128+QAT |
| 7   | 1.2855    | 1.3727         | No QAT |
| 9   | 1.2780    | 1.3702         | LeakyReLU+full TTT |
| 10  | 1.2777    | 1.3125         | **Int8 quant** |
| 22  | 1.2775    | 1.1483         | TTT LR=0.90 |
| 29  | 1.2777    | 1.1024         | TTT 40ep + temp=1.0 |
| 34  | 1.2779    | 1.1110         | TTT 60ep (slight overfit) |
| 35  | 1.2778    | 1.1104         | score-every-10, TTT 60ep |
| 38  | 1.2779    | ~same           | Hadamard quant (no benefit) |
| 39  | 1.2741    | 1.1101          | bigram=4096 (doesn't fit) |
| 40  | 1.2774    | DIVERGED        | Nesterov TTT |
| **41** | **1.2775** | **1.0962** | **TTT warmup=5** |
| 42  | 1.2777    | 1.0982         | TTT warmup=3 |
| 43  | 1.2775    | 1.0941         | TTT warmup=8 |
| 44  | 1.2774    | 1.0938         | TTT warmup=10 |
| 45  | 1.2775    | 1.0895         | TTT warmup=15 |
| 46  | 1.278     | 1.0883         | TTT warmup=20 |
| 47  | 1.2776    | 1.0873         | TTT warmup=25 |
| **48** | **1.278** | **1.0865** | **TTT warmup=30 (75%, best 40ep)** |
| 49  | 1.278     | 1.0889         | TTT warmup=35 (87.5%, worse) |
| **50** | **1.2779** | **1.0725** | **50ep warmup=38 (76%)** |
| **51** | **1.278** | **1.0648** | **60ep warmup=45 (75%)** |
| 52  | 1.278     | 1.0618         | 80ep warmup=60 (75%), best no-SWA |
| 53  | 1.278     | 1.2708         | adaptive 480s batch64 (only 5ep) |
| 54  | 1.278     | 1.0677         | 100ep warmup=75, no SWA (overfit) |
| **55** | **1.278** | **1.0442** | **80ep warmup=60 + SWA** |
| **56** | **1.278** | **1.0338** | **100ep warmup=75 + SWA** |
| **57** | **1.278** | **1.0312** | **120ep warmup=90 + SWA (BEST)** |
| 58  | 1.278     | 1.0313         | 150ep warmup=112 + SWA (plateau) |

### Session 7: TTT LR Sweep (Exp 59-63, NON-COMPLIANT offline TTT)

**Note**: These experiments use offline TTT (train on all val before eval), which violates
competition rules. The absolute numbers are not competition-valid, but the LR findings
are useful for calibrating online TTT.

| Exp | TTT LR | val_bpb | Delta from LR=1.0 |
|-----|--------|---------|-------------------|
| 59  | 1.0    | 1.0269  | baseline          |
| 60  | 1.1    | 1.0244  | -0.0025           |
| 61  | 1.2    | 1.0230  | -0.0039           |
| 62  | 1.3    | 1.0202  | -0.0067           |
| **63** | **1.5** | **1.0173** | **-0.0096** |

**Finding**: Higher TTT LR keeps improving with warmup+SWA. LR=1.5 best so far.
Each 0.1 LR step gives diminishing but positive improvement.
The warmup prevents divergence at high LRs that previously failed (LR=1.0 diverged without warmup).

## Key Insight: Gap to Sub-1.0
- Pre-quant val_bpb (1310 steps, 4xA100): 1.278
- Post-quant int8 gap: +0.035 → 1.313
- Best TTT improvement (120ep+warmup+SWA, LR=1.5): -0.296 → **val_bpb=1.0173** (NON-COMPLIANT)
- **Must switch to online TTT (score-first) for competition compliance**
- Online TTT will likely give less improvement than offline TTT (fewer total gradient steps)
- On 8xH100 (~7000 steps): pre-quant ~1.16-1.18, int8 fits may be an issue

## Session 9: Score-First TTT Optimization (Exp 105-113)

### Context
Switched to compliant score-first TTT with SwiGLU + Gated Attention architecture.
Score-first TTT: score chunk → lock in BPB → train on scored chunk → next chunk.
Using env vars: TTT_ENABLED=1 TTT_OPTIMIZER=sgd TTT_LR=X TTT_EPOCHS=Y TTT_CHUNK_TOKENS=Z

### TTT LR Sweep
| Exp | LR | Epochs | Chunk Size | val_bpb | Notes |
|-----|-----|--------|------------|---------|-------|
| 107 | 0.002 | 3 | 200K | 1.3178 | First good result |
| 108 | 0.003 | 3 | 200K | 1.3229 | Overshoots |
| 109 | 0.002 | 1 | 200K | 1.3225 | Too few epochs |
| 110 | 0.002 | 3 | 100K | 1.3172 | Smaller chunks help |
| 111 | 0.002 | 3 | 50K | 1.3165 | Trend continues |
| 112 | 0.002 | 3 | 25K | 1.3156 | BEST TTT (eval 1422s/4xA100) |
| 113 | 0.001 | 3 | 25K | ??? | Was running, result unknown |

### Key Findings
- LR=0.002 optimal for SGD with 3 epochs (0.003 overshoots, 0.001 may be too slow)
- 3 epochs > 1 epoch at same LR
- Smaller chunks consistently better: 200K→100K→50K→25K all improved
- Eval time scales with chunk count: 620s→787s→980s→1422s on 4xA100
- On 8xH100 (4x faster), 25K chunks → ~356s, still within 10-min budget

### CRITICAL ISSUE
- Score-first TTT val_bpb=1.3156 is WORSE than no-TTT baseline val_bpb=1.2244
- The baseline without TTT gets 1.2244 — TTT is currently HURTING performance
- Need to either: fix TTT to actually improve, or disable it and focus on base model

## Next Steps / Ideas
1. **Understand why TTT hurts** — score-first TTT (1.3156) worse than no-TTT (1.2244)
2. **Improve base model** — architecture changes to lower pre-quant val_bpb
3. **Check GitHub PRs** for winning strategies: https://github.com/openai/parameter-golf/pulls
4. **Verify int8 fits on 8xH100** — if not, need int6 or mixed int6/int8
5. **Full Hessian GPTQ** — PR #593 shows -0.022 bpb over GPTQ-lite
6. **Wider model dim** — 576 or 640 instead of 512, fewer layers to compensate
7. **MoE** — multiple competitors report improvements
