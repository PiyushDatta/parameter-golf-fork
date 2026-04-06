# Parameter Golf — Complete Context (2026-04-06)

## Competition Overview
- Train best LM fitting in 16MB artifact, trains in <10min on 8xH100 SXM
- Eval: val_bpb (bits per byte) on FineWeb validation set — LOWER is better
- 10min training + 10min eval (separate budgets, no transfer)
- Tokenizer: fineweb_1024_bpe.model (SP, vocab_size=1024) — CANNOT change
- Dataset: fineweb10B_sp1024 — CANNOT change
- Artifact = code bytes + compressed model bytes < 16,000,000 bytes (decimal)
- TTT must be score-first (issue #677, #1017); GPTQ/Hessian = training budget
- N-gram caching at eval is LEGAL (confirmed by organizer)

## Our Hardware
- LOCAL: 4x A100 80GB, ~460ms/step, ~1225 steps in 600s
- COMPETITION: 8x H100 80GB, ~85ms/step, ~7000 steps

## Baseline (train_gpt_do_not_touch.py)
- Architecture: 9L, dim=512, MLP 2x (relu^2), MoE (4 experts shared+routed for both attn and MLP), GQA (8 heads, 4 KV heads), tied embeddings
- U-Net skip connections: first half stores skips, second half reuses them in reverse
- Muon optimizer (momentum=0.95, NS steps=5, nesterov=True) + Adam for embeddings/scalars
- Momentum warmup: 0.85→0.95 over 500 steps
- LR: matrix=0.04, scalar=0.04, tied_embed=0.05
- QK gain init=1.5, logit softcap=30, RoPE base=10000
- Warmdown=1200 iters (designed for ~7000 steps on H100)
- Int8 per-row quantization + zlib compression
- No QAT, no SWA, no EMA, no GPTQ
- val_bpb: **1.2244**
- Key advantage: MoE gives capacity without proportional param increase

## Our Best: val_bpb=1.2272 (exp158_momentum095)
- Architecture: 11L, dim=512, MLP 3.0x (LeakyReLU(0.5)^2), NO MoE, GQA (8H/4KV), tied embeddings
- GATED_ATTENTION=0 (disabled gated attention)
- XSA on all 11 layers
- BigramHash(2048, dim=128) + SmearGate (EngramLite)
- Value Residual (ResFormer) — V cache from layer 0, mix via learned lambda
- Partial RoPE (16/64 dims), LN Scale (1/sqrt(layer_idx+1))
- Muon: **MOMENTUM=0.95** (was 0.99), warmup 0.85→0.95 over 500 steps, NS_STEPS=4, nesterov=True
- Adam: embed_lr=0.05, scalar_lr=0.04, matrix_lr=0.04
- WD=0.04 (Muon + Adam), grad_clip=0.3, QK_GAIN=4.0
- Warmdown=600 iters (49% of 1225 steps), sqrt cooldown, LR floor=0.05
- **QAT at 50% wallclock** (int6 fake-quantize during training)
- **SWA**: ~62 checkpoints during warmdown (scale<0.7), every 5 steps
- **GPTQ**: full Hessian-guided quantization, 64 calibration batches, multi-percentile search
- **Int6 quantization** + byte-shuffle + lzma compression → ~14.9MB
- Pre-quant: 1.2454, Post-quant: 1.2272 (quant gap: -0.018)
- quant_mse: 0.000147, 1221 steps at 492ms/step
- EVAL_STRIDE=16 for competition (64 for local testing)

## Gap Analysis: Why 1.2272 vs 1.2244 (0.0028 gap)
- Baseline uses MoE (4 experts) which gives capacity without doubling params
- We can't fit MoE with int6 (would double params → blow 16MB)
- More params (27M int6) + more quant damage ≈ fewer params (18M int8) + less quant + MoE capacity
- Config is at **Nash equilibrium**: changing ANY single parameter from exp158 makes it worse

## Key Design Decisions
1. **Int6 > Int8 for us**: QAT+SWA+GPTQ tuned for int6 gives better post-quant than int8
2. **SWA > EMA on 4xA100**: With only 1225 steps, EMA includes too much early training garbage. SWA averages only warmdown checkpoints (targeted)
3. **WD=0.04 helps**: Weight decay keeps magnitudes controlled → fewer quant outliers. WD=0 is worse.
4. **Momentum=0.95 optimal**: Fewer steps → lower momentum → faster adaptation per step
5. **Warmdown=600 sweet spot**: Balances training time at full LR vs SWA checkpoint collection

## Comprehensive Dead Ends (44+ experiments, do NOT retry)

### Architecture
- 9L MLP2 int8 (exp125b: 1.2572)
- 12 layers (exp147: 1.2461, SWA overhead → only 973 steps)
- 10L/dim576 (exp155: 1.2608, 586ms/step → only 844 steps)
- MLP 3.5x + mixed precision (exp142: 1.2530, fewer steps + quant damage)
- MLP 3.5x + uniform int6 (more quant damage than capacity benefit)
- XSA_LAST_N=4 (exp169: 1.2405, QAT overhead 545ms/step → only 1124 steps)

### Quantization / Compression
- Int8+zlib with MLP 3.0x = 19.3MB — doesn't fit
- Int8+lzma (exp145: 1.2413, QAT tuned for int6)
- Mixed precision int5/6/7 (exp141b: 1.2278, tied — only 1 group promoted)
- Hadamard/OptRot rotation (exp153: 1.2322, lower MSE but worse bpb)
- GPTQ 128 cal batches (exp143: 1.2280, no improvement over 64)
- CROWN-Q (detach breaks gradients — NO-OP)

### Optimizer / Training
- LR > 0.04 (exp131: LR=0.06 overshoots)
- GRAD_CLIP=0 (exp159: 1.2334, gradients need clipping)
- WD=0.0 (exp157: 1.2350, weights too uncontrolled for int6)
- WD=0.02 (exp163: 1.2452, contaminated but trajectory worse)
- QK_GAIN_INIT=1.5 (exp161: 1.2321, baseline value worse on our setup)
- MUON_BACKEND_STEPS=3 (exp154: 1.2340, doesn't converge)
- MUON_BACKEND_STEPS=5 (exp162: 1.2296, no improvement)
- Warmdown=200 (exp160: 1.2308, SWA only ~20 ckpts)
- Warmdown=800 (exp139c: 1.2315, more SWA but worse post-GPTQ)
- Warmdown=400 + SWA_EVERY=2 (exp165: 1.2278, tied — no improvement)

### Averaging
- EMA decay=0.997 (exp166: 1.2867, catastrophic on 1223 steps)
- EMA decay=0.999 (exp167: 1.7487, even worse — more early weights)
- SWA-over-EMA (exp168: 1.3450, EMA contaminates SWA checkpoints)
- SWA_EVERY=3 (exp150: 1.2292, too much averaging)

### Activation / N-gram
- LeakyReLU 0.3 (exp133: 1.2424), LeakyReLU 0.9 (exp134: 1.2471) — 0.5 optimal
- EngramLite 8192 buckets (exp137: 1.2533, slower steps)
- N-gram probability mixing (~33 PRs closed for hash collisions)
- MTP (Multi-Token Prediction) — confirmed dead by ternary submission

### Data / Misc
- Coprime-stride loader (exp149: 1.2295, reduces data coherence)
- Seed sweep (exp152: 1.2279, variance only 0.0003)
- QAT@40% (exp132: 1.2296, QAT@50% better)
- TTT with XSA-all (hurts)

## Key Insights from 44+ Experiments

### quant_mse Paradox
- Lower quant_mse does NOT mean better val_bpb!
- EMA gives best-ever quant_mse (0.000122) but worst-ever post-quant (1.7487)
- Hadamard gives lowest non-EMA quant_mse (0.000129) but worse bpb (1.2322)
- quant_mse measures element-wise error, NOT functional model quality
- Like JPEG-compressing a blurry photo: tiny file size, useless image

### SWA is load-bearing
- Warmdown length directly determines SWA quality → quant_mse → post-quant bpb
- Can't shorten warmdown without also adjusting SWA parameters (coupled system)
- SWA_EVERY=5 with warmdown=600 gives ~62 checkpoints — optimal for 1225 steps

### Hardware-dependent hyperparameters
- Momentum 0.95 optimal for 1225 steps (0.99 optimal for 7000 steps)
- SOTA hyperparams (LeakyReLU 0.3, EngramLite 8192, QAT@15%) don't transfer from 8xH100
- EMA works on 8xH100 (7100 steps) but is catastrophic on 4xA100 (1225 steps)

## Baseline Architecture (train_gpt_do_not_touch.py) — Key Details

### U-Net Skip Connections
```
num_encoder_layers = num_layers // 2  (e.g., 4 for 9 layers)
num_decoder_layers = num_layers - num_encoder_layers  (e.g., 5)
First half (encoder): store skip connections
Second half (decoder): add skip_weight * skips in reverse order
```

### MoE Architecture
- **MoE MLP**: Shared expert (always active) + 4 routed experts (top-2 per token)
  - Router: linear → softmax → topk(2) → renormalize
  - Load balancing loss: `num_experts * (tokens_per_expert * avg_prob_per_expert).sum()`
- **MoE Attention**: 4 shared heads + 4 routed Q-expert heads (4 attn experts)
  - Shared Q projection for all tokens + expert Q projections (router picks one per token)
  - K, V shared across all experts (every token attends to every other)
  - Router probability gates the routed attention output

### Block Structure
```python
x = mix[0] * x + mix[1] * x0   # resid_mix with x0 (initial embedding)
x = x + attn_scale * attn(attn_norm(x))
x = x + mlp_scale * mlp(mlp_norm(x))
```

### Quantization
- Simple int8 per-row quantization with percentile-based clipping (99.99984th percentile)
- Per-row scales stored as fp16
- Small tensors (<65536 params) kept as fp16 passthrough
- Control tensors (attn_scale, mlp_scale, resid_mix, q_gain, skip_weight, router) kept as fp32

## User's Idea: Router-Based Skip Connections
"baseline uses u-net/skip connections, can we do better? put router in middle, deeper layers send to router, router gives back residuals from earlier layers - but router decides"

### Concept
Instead of fixed U-Net skip connections (first half stores, second half retrieves in reverse), use a **learned router** that decides WHICH earlier layer's residual to mix into each deeper layer. This makes skip connections adaptive rather than symmetric.

### Key Questions
1. Does a learned router for skip selection outperform fixed symmetric U-Net?
2. Can the router learn to route different tokens to different skip sources?
3. What's the parameter cost of the router vs the fixed skip_weights?
4. Does this help or hurt quantization (more complex routing = harder to quantize)?

## Experiment Infrastructure

### run_experiment.py
Wrapper around train_gpt.py with:
- Frequent validation checkpoints (every ~12s)
- Comparison against saved baseline
- Early stopping if worse by >0.002 at 2 consecutive checkpoints after 2 min

### Usage
```bash
source .venv/bin/activate
# Show baseline
python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py --show-baseline
# Run experiment
python records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py --name "exp_name" --env "KEY1=val1,KEY2=val2" [--no-early-stop] [--nproc N]
# Example: run exp_name with 4x A100, 2 GPUs per node, 2 nodes
python run_experiment.py --name "my_experiment" --env "FOO=bar" --no-early-stop
```

### Baseline (4x A100, pre-quant, no TTT)
| Time | val_bpb | Time | val_bpb |
|------|---------|------|---------|
| 30s  | 2.1149  | 6m   | 1.3349  |
| 1m   | 1.7672  | 7m   | 1.3166  |
| 2m   | 1.5372  | 8m   | 1.3010  |
| 3m   | 1.4503  | 9m   | 1.2913  |
| 4m   | 1.3981  | 10m  | 1.2827  |
| 5m   | 1.3581  |      |         |

## Session Rules (this session: 2026-04-06)
- **CAN touch any and every file** (including train_gpt_do_not_touch.py)
- No agent role restrictions — single agent
- GPUs available for experiments
