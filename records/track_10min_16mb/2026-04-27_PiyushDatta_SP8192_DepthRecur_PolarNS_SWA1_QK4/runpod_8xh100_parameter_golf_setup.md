# RunPod 8×H100 setup for `parameter-golf-fork`

This is a clean setup/runbook for a **fresh RunPod 8×H100 machine**:

-   do **not** rely on the small root filesystem for datasets/logs
-   keep the **repo + dataset/cache/logs on `/dev/shm`** for speed and to avoid quota/root-disk issues
-   keep the **uv environment on `/workspace`**, not `/dev/shm`, so PyTorch shared libraries load correctly
-   disable Hugging Face Xet downloads with `HF_HUB_DISABLE_XET=1`

---

## 1) Check the machine and storage

```
nvidia-smi && df -h
```

## 2) Clone the repo into `/dev/shm`

```bash
cd /dev/shm
git clone https://github.com/PiyushDatta/parameter-golf-fork.git
cd /dev/shm/parameter-golf-fork
```

---

## 3) Create a uv environment on `/workspace`

```
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/dev/shm/uv-cache
mkdir -p /dev/shm/uv-cache
uv venv /workspace/uv-envs/parameter-golf
source /workspace/uv-envs/parameter-golf/bin/activate
cd /dev/shm/parameter-golf-fork
uv sync --active
uv sync --active --reinstall-package torch
```

## 4) Verify PyTorch if needed

```
/workspace/uv-envs/parameter-golf/bin/python -c "import torch; print(torch.__file__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); import torch.distributed.run; print('ok')"
```

---

## 5) Configure Hugging Face cache + temp dirs in `/dev/shm`

```
export HF_HUB_DISABLE_XET=1
export HF_HOME=/dev/shm/hf-cache
export HUGGINGFACE_HUB_CACHE=/dev/shm/hf-cache/hub
export HF_DATASETS_CACHE=/dev/shm/hf-cache/datasets
export TMPDIR=/dev/shm
mkdir -p /dev/shm/hf-cache/hub
mkdir -p /dev/shm/hf-cache/datasets
mkdir -p /dev/shm/pg-logs
```

---

## 6) Download the dataset

```
cd /dev/shm/parameter-golf-fork
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
ls -lah data/datasets/fineweb10B_sp8192 | head
ls -lah data/tokenizers | grep 8192
```

You should see files like:

-   `data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin`
-   `data/tokenizers/fineweb_8192_bpe.model`

---

## 7) Generate 3-seed submission logs (RECOMMENDED)

The easiest way to run training + collect all logs for submission:

```bash
cd /dev/shm/parameter-golf-fork

mkdir -p /workspace/pg-tmp
mkdir -p /workspace/torchinductor-cache
mkdir -p /workspace/triton-cache

export HF_HUB_DISABLE_XET=1
export HF_HOME=/dev/shm/hf-cache
export HUGGINGFACE_HUB_CACHE=/dev/shm/hf-cache/hub
export HF_DATASETS_CACHE=/dev/shm/hf-cache/datasets
export TMPDIR=/workspace/pg-tmp
export TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor-cache
export TRITON_CACHE_DIR=/workspace/triton-cache

rm -rf /dev/shm/torchinductor_root /dev/shm/triton

/workspace/uv-envs/parameter-golf/bin/python \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/generate_submission_logs.py \
  --nproc 8 \
  --data-dir /dev/shm/parameter-golf-fork/data/
```

This will:
1. Run training with 3 seeds (42, 314, 999) sequentially
2. Save per-seed logs to `records/.../logs/seed_42.log`, etc.
3. Parse all results and write `records/.../logs/summary.json`
4. Print a summary table with mean/std val_bpb and artifact sizes

All config defaults are baked into `train_gpt.py` — no env vars needed.

Each seed takes ~12-15 min (10 min training + GPTQ + eval). Total: ~40-45 min.

**Key defaults baked into train_gpt.py:**
- `MATRIX_LR=0.028` (optimized for Polar Express NS)
- `MUON_WD=0.095, EMBED_WD=0.085`
- `NUM_LOOPS=1, ENABLE_LOOPING_AT=0.45` (depth recurrence)
- `SWA_EVERY=1, MIN_LR=0.10, SWA_START_FRAC=0.12`
- `WARMUP_STEPS=20, QK_GAIN_INIT=4.0`
- `HESSIAN_CLIP_LAMBDA=0.175`
- Polar Express Newton-Schulz coefficients

**For 8xH100 optimized config** (different from baked defaults):
```bash
/workspace/uv-envs/parameter-golf/bin/python \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/generate_submission_logs.py \
  --nproc 8 \
  --data-dir /dev/shm/parameter-golf-fork/data/ \
  --env "NUM_LOOPS=2,ENABLE_LOOPING_AT=0.35,LOOP_START=4,LOOP_END=5,TRAIN_BATCH_TOKENS=786432,MUON_MOMENTUM=0.99,MUON_MOMENTUM_WARMUP_START=0.92,MUON_MOMENTUM_WARMUP_STEPS=1500,MATRIX_LR=0.022,SWA_ENABLED=0,GPTQ_RESERVE_SECONDS=12,TTT_LR=0.005,MIN_LR=0.0,QK_GAIN_INIT=5.25"
```

**Dry run** (preview commands without running):

```bash
/workspace/uv-envs/parameter-golf/bin/python \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/generate_submission_logs.py \
  --nproc 8 --dry-run
```

**Single seed** (quick test before full 3-seed run):

```bash
/workspace/uv-envs/parameter-golf/bin/python \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/generate_submission_logs.py \
  --nproc 8 --seeds 42 \
  --data-dir /dev/shm/parameter-golf-fork/data/
```

---

## 7b) Manual single run (alternative)

If you prefer to run training manually:

```bash
cd /dev/shm/parameter-golf-fork

# Set up env vars as in step 7 above, then:
DATA_DIR=./data /workspace/uv-envs/parameter-golf/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/train_gpt.py
```

To save the console stream:

```bash
DATA_DIR=./data /workspace/uv-envs/parameter-golf/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4/train_gpt.py \
  2>&1 | tee /dev/shm/pg-logs/train_console.log
```

---

## 8) Save logs/results somewhere persistent

Because `/dev/shm` is temporary, copy logs back out after the run.

**If you used `generate_submission_logs.py`** (step 7), logs are already at:
-   `records/.../logs/seed_42.log`, `seed_314.log`, `seed_999.log`
-   `records/.../logs/summary.json` (parsed results with mean/std)

Copy the whole submission directory to `/workspace`:

```bash
mkdir -p /workspace/pg-results
cp -r /dev/shm/parameter-golf-fork/records/track_10min_16mb/2026-04-27_PiyushDatta_SP8192_DepthRecur_PolarNS_SWA1_QK4 /workspace/pg-results/
```

**If you ran manually** (step 7b), grab the latest log:

```bash
mkdir -p /workspace/pg-results
LATEST_LOG=$(ls -t /dev/shm/parameter-golf-fork/logs/*.txt | head -n 1)
cp "$LATEST_LOG" /workspace/pg-results/
cp /dev/shm/pg-logs/train_console.log /workspace/pg-results/ 2>/dev/null || true
```
