# Parameter Golf - Optimization Notes & Scratchpad

## Competition Rules
- Train best LM fitting in 16MB artifact, trains in <10min on 8xH100
- Eval metric: val_bpb (bits per byte) on FineWeb validation set - LOWER is better
- Our hardware: 1x A100 80GB (competition uses 8xH100)
- We test locally on 1xA100 but final submission runs on 8xH100

## UPDATED Competition Landscape (as of 2026-03-22)
**WE WERE BEHIND! The competition has moved way past 1.14276!**

| Rank | PR | Author | val_bpb | Key Techniques |
|------|-----|--------|---------|----------------|
| 1 | #398 | felipe-parodi | **1.1213** | 11L EMA + aggressive TTT (20ep, all blocks unfrozen) |
| 2 | #415 | fbedev | **1.1216** | 11L XSA4 + FA3 + Two-Phase TTT |
| 3 | #388 | ElliotSlusky | **1.1231** | 11L + Tight SWA + VE128 + TTT (25ep) |
| 4 | #414 | signalrush | **1.1233** | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 |
| 5 | #401 | newjordan | **1.1243** | 11L + EMA + Tight SWA + QAT0.15 + VE128 |
| 6 | #374 | unnir | **1.1246** | 11L + Tight SWA + VE128 + Partial RoPE + LN Scale + XSA4 |

### Critical observations:
1. **EVERYONE uses 11 layers now** - we're on 10, need to upgrade
2. **EMA (decay=0.997) > SWA** by 0.003 bpb (verified)
3. **Value Residual (ResFormer)** gives -0.015 bpb for ONLY 18 params!
4. **TTT at eval is standard** - all top entries use it (SGD, 15-25 epochs)
5. **GPTQ-lite** (optimal clip search) is free improvement
6. **XSA on last 4 layers** helps
7. **BigramHash reduced to 2048 buckets** in many top entries (vs our 10240)
8. **Late QAT (last 10-15%)** better than full-run QAT

## CRITICAL: TTT Information Leakage Warning (Issue #402)
Many TTT implementations are INVALID because they adapt on tokens not yet scored.
Proper TTT must: (1) score token t, then (2) adapt on tokens <= t.
Our LoRA TTT implementation follows this correctly (score chunk first, then train).

## Key Negative Results (from PR #375 - $500 of compute)
Things that DID NOT work:
- MTP (Multi-Token Prediction): +0.028 bpb, throughput penalty
- INT4 quantization: 0.06 bpb gap wipes out param advantage
- Canon layers, memory tokens, gradient-guided quant
- Cautious WD, L1 reg, label smoothing, full-run QAT
- Meta-insight: each 1ms/step overhead costs ~0.006 bpb

Key positive from negative results PR:
- 786K > 524K batch by 0.004 bpb
- Weight decay controls compressed artifact size

## The Consensus SOTA Stack (what top entries share)
1. **11 layers**, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP
2. SmearGate + BigramHash (2048 buckets - NOT 10240!)
3. Orthogonal init + muP scaling
4. **Partial RoPE (16/64 dims) + LN Scale**
5. **Int6 + zstd-22** (NOT int5 for MLP - just int6 everywhere)
6. **Late QAT** (last 10-15% of training, NOT full run)
7. **EMA (decay=0.997)** instead of SWA
8. **XSA on last 4 layers**
9. Muon optimizer (WD=0.04)
10. **Value Residual (ResFormer)** - cache V from layer 0, mix into all layers
11. **Gated Attention** - per-head sigmoid gate after SDPA
12. **TTT at eval** (SGD, 15-25 epochs)
13. **GPTQ-lite** optimal clip search

## Our Current Script (Final v5)
- 11 layers, dim=512, 8 heads, 4 KV heads (GQA), 3x MLP (relu^2)
- BigramHash(2048, dim=128) + SmearGate
- Int6 everywhere + zlib compression (no extra deps)
- EMA (decay=0.997) - replaced SWA
- Value Residual (ResFormer) - cache V from layer 0, mix via learned lambda
- Gated Attention - per-head sigmoid gate after SDPA
- XSA on last 4 layers
- Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer_idx+1))
- GPTQ-lite clip percentile search (5 candidates)
- Full-weight SGD TTT (20 epochs, lr=0.008, momentum=0.9, cosine schedule)
- QAT disabled by default (20% overhead not worth it, available via QAT_ENABLED=1)
- Model size: ~5.4MB int6+zlib (well under 16MB cap)

## Implementation Priority (to close gap to 1.12)

### Phase 1: Critical Architecture Changes
1. **11 layers** (everyone uses this)
2. **Value Residual** (-0.015 bpb, only 18 params, trivial to add)
3. **EMA** instead of SWA (-0.003 bpb)
4. **Late QAT** (only last 10-15%, not full run)
5. **GPTQ-lite** (optimal clip percentile search)

### Phase 2: Additional Improvements
6. **Gated Attention** (-0.003 bpb)
7. **XSA on last 4 layers** (cross-sequence attention?)
8. **Partial RoPE** (16/64 dims)
9. **LN Scale**
10. **Better TTT** (SGD, more epochs, unfreeze more params)

### Phase 3: Experimental
11. **Two-Phase TTT** (norm recalibration + selective block adaptation)
12. **Dynamic Eval / Krause-style** (gradient steps during sliding window)
13. **SwiGLU** activation

## Deep Thinking: Advanced Techniques

### Sparse Circuit Discovery During Training
- Idea: identify and prune less-important attention heads/MLP neurons during training
- Track activation magnitudes and gradient flow through each circuit
- Prune bottom 5-10% of circuits mid-training, free up parameters
- Risk: may hurt convergence if circuits are pruned too early
- Better approach: use structured pruning after training to improve compression
- Related: the magnitude pruning we already do (3% unstructured) could become structured

### Variable Embedding Size
- Top entries use "VE128" (Value Embedding 128) - separate smaller dim for value projections
- This saves params: instead of V_dim=512, use V_dim=128, project back to 512
- Saves ~3x on V weights per layer: 512*128 vs 512*512 = 3/4 reduction
- Enables bigger model within 16MB budget
- Note: "VE128" in the leaderboard = Value Residual with 128 dims, NOT variable embedding

### Manifold/Ultra Connections
- Value Residual IS a form of manifold connection - maintains info flow from layer 0
- Skip connections (U-Net style) already provide cross-layer information flow
- Could add more: connect every N layers, or use a learned attention over layer outputs
- Risk: more params, more compute, may not fit in 10min

### Paired Head Attention
- Use pairs of attention heads that share K but have different Q projections
- Reduces K params by 2x, enables more heads for same budget
- Related: GQA already does this partially (4 KV heads for 8 Q heads)
- Could push further: 2 KV heads for 8 Q heads, or even 1 KV head

### MCTS / AlphaZero for Architecture Search
- Use MCTS to search over hyperparameter space
- Each "game" is a training run, "reward" is -val_bpb
- Tree search over: num_layers, dim, mlp_mult, num_heads, etc.
- Problem: each run takes 10 min on 8xH100, too expensive for search
- Better: use random search or Bayesian optimization with cheaper proxy tasks

### RL During Pretraining
- Standard pretraining uses cross-entropy loss on next-token prediction
- Could add RL reward for correct predictions on harder tokens
- E.g., upweight loss on tokens the model gets wrong
- This is essentially curriculum learning / hard example mining
- Risk: may overfit to hard cases at expense of easy ones
- Related: "focal loss" does this by downweighting easy examples

## Experiment Log

### Experiment 1: Full SOTA Integration (previous session)
- Status: COMPLETE
- 1xA100: 398 steps, ~1500ms/step
- val_bpb (pre-sliding): 1.7144 (expected poor with 398 steps)
- Model size: 15.75MB (fits!)

### Experiment 2: SOTA + QAT + LoRA TTT
- Date: 2026-03-22
- Status: COMPLETE
- 1xA100: ~1515ms/step, model size 15.5MB

### Experiment 3: Full SOTA Stack (v1-v3)
- Date: 2026-03-22
- Status: COMPLETE (v2 full pipeline verified)
- Techniques integrated:
  1. 11 layers (from 10) - DONE
  2. Value Residual (ResFormer) with learned 2-param lambda - DONE
  3. EMA (decay=0.997) replacing SWA - DONE
  4. Late QAT (last 15% of wallclock) - DONE (v3)
  5. GPTQ-lite clip percentile search (5 candidates) - DONE
  6. Gated Attention (per-head sigmoid gate, bias=4.0) - DONE
  7. XSA on last 4 layers - DONE
  8. Partial RoPE (16/64 dims) - DONE
  9. Full-weight SGD TTT (no torch.compile - faster!) - DONE
  10. BigramHash(2048) instead of 10240 - DONE
  11. Int6 everywhere (not int5 for MLP) - DONE
- Results (1xA100, 343 steps):
  - val_bpb (pre-sliding): 1.8969 (expected ~1.12 on 8xH100 with 4000+ steps)
  - Model int6+zlib: **5.38MB** (well under 16MB!)
  - Model params: 26,875,079
  - Step avg: 1737ms/step (1xA100)
  - TTT without compile: 872s/3epochs on 1xA100 (est. ~109s on 8xH100)
  - TTT with compile: 631s/3epochs (worse due to fixed 5min compile cost)
- Key finding: **Skip torch.compile for TTT** - compile overhead > actual speedup for few epochs
- Key finding: **QAT was never being activated** - fixed in v3 with wallclock-based trigger

### Experiment 3 v5: Final Script (all SOTA techniques)
- Date: 2026-03-22
- Status: COMPLETE (smoke tested, pipeline verified)
- Added vs v3:
  - LN Scale (1/sqrt(layer_idx+1)) - zero params, stabilizes deep layers
  - Updated learning rates to match SOTA: matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035
  - Two TTT modes: "full" (default) and "twophase" via TTT_MODE env var
  - Vectorized sequence pre-build for TTT
  - QAT disabled by default (20% overhead not worth it)
- Model: 5.14MB int6+zlib (well under 16MB)
- 69 steps in 2 min smoke test, pipeline works end-to-end
- Full 10-min run: 344 steps, val_bpb pre-quant=1.9217, post-quant+TTT=2.0202
- LN Scale slightly slower initial convergence but SMALLER quantization gap (0.10 vs 0.22 bpb)
- Post-quant+TTT result is BETTER with LN Scale (2.02 vs 2.14) despite worse pre-quant

### Experiment 4: 4xA100 Full Run (zlib, final script)
- Date: 2026-03-23
- Status: COMPLETE
- 4xA100, full pipeline (train + TTT + sliding eval)
- val_loss: 2.9098, **val_bpb: 1.7234** (post-quant + TTT + sliding window)
- eval_time: ~3060s (~51 min — expected, 4xA100 is much slower than 8xH100 for TTT/eval)
- Improvement over Exp3 v5 (2.0202 on 1xA100) due to more training steps with 4 GPUs

### Experiment 5 (NEXT): 8xH100 Submission
- Goal: Run on 8xH100 to get actual competitive val_bpb
- Estimated: ~55ms/step -> ~10,900 steps in 10 min
- TTT: 20 epochs full-weight SGD, ~2 min estimated
- Sliding eval: stride=64, ~2 min estimated
- Expected val_bpb: ~1.12-1.13 range (competitive with top entries)
- Run command for 8xH100:
  ```
  RUN_ID=exp5_8xH100 DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 ./records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py
  ```

## Key Findings
1. **Skip torch.compile for TTT** - fixed compile cost (~5 min) >> speedup for few epochs
2. **QAT adds ~20% overhead** - costs ~2000 training steps, disabled by default
3. **Full-weight SGD TTT > Two-Phase TTT** on under-trained models (1 GPU tests)
4. **Model size: 5.4MB** of 16MB budget — could use larger model but 11L-512 is what top entries use
5. **Two-Phase TTT may be better on well-trained models** (8xH100 with 10K+ steps) — available via TTT_MODE=twophase

## Concrete Implementation Notes

### Value Residual (ResFormer) - arXiv:2410.17897
```python
# In GPT.forward:
# After first layer, cache V vectors
# In each subsequent layer, mix cached V into current V
# Only needs a learned scalar per layer (~11 params)
# Cache: v0 = self.blocks[0].attn.c_v(first_layer_input)
# Each layer: v_mixed = (1-alpha) * v_current + alpha * v0
```

### Gated Attention - arXiv:2505.06708
```python
# In CausalSelfAttention:
# After SDPA output, apply per-head sigmoid gate
# self.attn_gate = nn.Parameter(torch.zeros(num_heads))
# y = y * torch.sigmoid(self.attn_gate)[None, :, None, None]
```

### EMA instead of SWA
```python
# During training:
# ema_decay = 0.997
# for name, param in model.named_parameters():
#     ema_state[name] = ema_decay * ema_state[name] + (1 - ema_decay) * param
# After training, load ema_state
```

### GPTQ-lite (optimal clip percentile)
```python
# During quantization:
# For each weight matrix, try clip_percentiles = [100, 99.9, 99.5, 99, 98]
# Select the one with minimum MSE after quantize/dequantize roundtrip
# Zero cost at training time, small cost at export time
```

### Late QAT
```python
# Only enable QAT in the last 15% of training
# if step > total_steps * 0.85:
#     enable_qat()
# This lets the model converge normally first, then adapt to quantization
```

### Two-Phase TTT (PR #415)
Phase 1: Norm-Only Recalibration (100 epochs, Adam lr=0.01)
- Only unfreeze LayerNorm weights + scales (~22K params)
- Fixes activation distributions damaged by int6 quantization

Phase 2: Selective-Freeze Block Adaptation (25 epochs, SGD lr=0.005)
- Unfreeze last 3 blocks + norms + scales + lm_head (~7.6M params)
- Adapts model to local text distribution

Combined: -0.021 bpb over post-SWA!

## What We Need to Read
- PR #398 (SOTA 1.1213) - full train_gpt.py
- PR #415 (Two-Phase TTT) - eval code
- PR #374 (XSA, Partial RoPE, LN Scale) - architecture details
- PR #413 (Value Residual + Gated Attention) - implementation
