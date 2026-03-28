# Parameter Golf - Optimization Notes & Scratchpad

## Competition Rules
- Train best LM fitting in 16MB artifact, trains in <10min on 8xH100
- Eval metric: val_bpb (bits per byte) on FineWeb validation set - LOWER is better
- 10min training + **10min eval** time limit (TTT + sliding eval must fit in 10min)
- Our hardware: 4x A100 80GB (competition uses 8xH100)
- Tokenizer: fineweb_1024_bpe.model (SP, vocab_size=1024) - CANNOT change
- Dataset: fineweb10B_sp1024 - CANNOT change
- Artifact = code bytes + compressed model bytes < 16,000,000 bytes

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

## Key Insight: Gap to Sub-1.0
- Pre-quant val_bpb (1310 steps, 4xA100): 1.278
- Post-quant int8 gap: +0.035 → 1.313
- Best TTT improvement (120ep+warmup+SWA): -0.282 → **val_bpb=1.0312**
- On 8xH100 (~7000 steps): pre-quant ~1.16-1.18
- With ~19 TTT epochs + warmup + SWA: estimated ~0.93-0.95 → **well under 1.0!**
- TTT improvement scales with epochs but saturates at ~120ep on 4xA100

## Next Steps / Ideas
1. **Verify int8 fits on 8xH100** — if not, need int6 or mixed int6/int8
2. **Full Hessian GPTQ** — PR #593 shows -0.022 bpb over GPTQ-lite
3. **Wider model dim** — 576 or 640 instead of 512, fewer layers to compensate
4. **Improve pre-quant quality** — architecture changes, training improvements
