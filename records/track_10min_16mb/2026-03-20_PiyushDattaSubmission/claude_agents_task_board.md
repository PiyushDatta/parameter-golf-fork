# Claude Agents Task Board

## Agents

### agent_research_A (Research Agent — CPU only)
- **Role**: Reads GitHub PRs, analyzes competitor code, plans architecture/training changes, writes code diffs to `train_gpt.py`
- **Does NOT run GPU training** — prepares changes for agent_gpu_B to test
- **What I know**:
  - Full competition rules (16MB artifact, 10min train, 10min eval, score-first TTT, 4 conditions)
  - Our architecture: 11L dim=512, 8H, 4KV GQA, LeakyReLU(0.5)^2, EMA, BigramHash(2048), VE128, XSA(last 4), Gated Attention, U-Net skips, Partial RoPE(16), SmearGate
  - Current baseline: val_bpb=1.2244 (no TTT, int8+zlib, 15.8MB, 13780 steps @ 43.54ms/step on 4xA100)
  - Score-first TTT currently HURTS (1.3156 vs 1.2244) — don't use it unless we rewrite it
  - Dead ends: Hadamard quant, Nesterov TTT, LoRA TTT, Two-Phase TTT, SwiGLU, Score-first Adam, SGD LR>=0.1
  - Full experiment history in notes.md (Exp 1-113)
  - Competition SOTA: PR #1089 at val_bpb=1.1091
- **How I work**: I read PRs via `gh` CLI and WebFetch, analyze code, then either apply changes directly to `train_gpt.py` or write diffs in the Queued Changes section below. I mark changes as READY when they're applied and testable.

### agent_gpu_experiment_B (GPU Agent — runs experiments)
- **Role**: Runs training experiments on 4xA100, monitors progress, logs results
- **Has exclusive GPU access** — one run at a time for max throughput
- **What to do**:
  1. Check this task board before each run
  2. When a change is marked READY, run the experiment
  3. Log results in the Completed Experiments section AND in notes.md
  4. Update your status in the Current Status section
  5. If a run finishes and no new READY changes exist, post in the Message Board asking agent_research_A for the next change

## Rules
1. **One GPU run at a time** — all 4xA100 for max throughput
2. agent_research_A prepares code changes while agent_gpu_B runs experiments
3. Both agents update this file with status and use the Message Board to communicate
4. Detailed results go in `notes.md` experiment log
5. agent_gpu_B should re-read `train_gpt.py` before running if agent_research_A marked a change as READY
6. **DO NOT touch** `train_gpt_do_not_touch.py` (read-only reference)
7. All work stays in `/data/repos/parameter-golf-fork/records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/`
8. Full permissions granted — run all commands without asking

## Key Commands
```bash
# Setup (run once)
with-proxy uv sync && source .venv/bin/activate

# Download dataset (run once)
with-proxy python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Run training (baseline, no TTT)
source .venv/bin/activate && with-proxy RUN_ID=piyush_datta_submission \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 TTT_ENABLED=0 \
  torchrun --standalone --nproc_per_node=4 \
  ./records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py

# Run with experiment runner (has early stopping)
source .venv/bin/activate && with-proxy python \
  records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/run_experiment.py \
  --name "exp_name" --env "KEY1=val1,KEY2=val2"
```

## Current Status

### agent_research_A: STARTING
- Task: Researching top PRs (#1089, #1120, #1060, #609) to plan first improvements
- Priority list:
  1. XSA on ALL 11 layers (not just last 4) — used by ALL top submissions
  2. Full Hessian GPTQ (Cholesky error compensation) — ~0.003 bpb
  3. EngramLite (trigram+bigram hash, 8192 buckets) — ~0.003-0.005 bpb
  4. QK Gain = 4.0 (vs 1.5) — monotonic improvement from PR #1125
  5. DROP TTT — counterproductive on XSA-all stacks
  6. Int6 GPTQ (not int8) — all top submissions use int6
  7. MLP 3.5x width, Brotli compression, Coprime-stride loader

### agent_gpu_experiment_B: IDLE
- First task: Confirm current baseline (no TTT) → expect val_bpb ~1.2244
- Then wait for READY changes from agent_research_A

## Queued Changes (agent_research_A prepares, agent_gpu_B runs)

### Change 1: [RESEARCHING]
- Description: TBD — agent_research_A is researching PRs now
- Status: RESEARCHING
- Expected impact: TBD
- How to test: TBD

## Completed Experiments
(agent_gpu_B logs results here with format:)
<!-- Example:
### Exp 114: XSA on all layers
- Command: `with-proxy torchrun ... TTT_ENABLED=0 XSA_LAST_N=11`
- Pre-quant val_bpb: X.XXXX
- Post int8+zlib val_bpb: X.XXXX
- Model size: XX bytes
- Steps: XXXX @ XX.XXms/step
- Verdict: BETTER/WORSE/SAME
-->

## Message Board (async communication between agents)
<!-- Format: [agent_name] [timestamp or sequence] message -->

- [agent_research_A] #1: Starting up. I'm researching GitHub PRs for the best improvements to apply. Will prepare code changes and mark them READY for you. While you wait, please confirm the baseline by running with TTT_ENABLED=0. Current expected baseline: val_bpb=1.2244 (int8+zlib roundtrip). Log the exact number so we have a confirmed starting point.

- [agent_research_A] #2: Key context — our val_bpb=1.2244 vs competition SOTA 1.1091 (PR #1089). That's a 0.115 gap. The biggest wins from top PRs seem to be: (1) XSA on all layers, (2) better quantization (Full GPTQ or int6), (3) EngramLite hash embeddings. I'll prioritize these. TTT is counterproductive for now — all top 5 submissions don't use it.
