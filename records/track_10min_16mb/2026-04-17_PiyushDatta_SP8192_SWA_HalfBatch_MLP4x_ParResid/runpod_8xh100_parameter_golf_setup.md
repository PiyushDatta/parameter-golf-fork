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

## 7) Launch training

Use the **venv Python** explicitly.

```
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

DATA_DIR=./data  /workspace/uv-envs/parameter-golf/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py
```

The training script writes its own run log under `./logs/` and prints the exact path near the top as:

```text
logfile: logs/<run_id>.txt
```

If you want the full console stream saved too, run through `tee`:

```bash
DATA_DIR=./data /workspace/uv-envs/parameter-golf/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py \
  2>&1 | tee /dev/shm/pg-logs/train_console.log
```

---

## 8) Save logs/results somewhere persistent

Because `/dev/shm` is temporary, copy logs back out after the run.

The safest pattern is:

1. Copy the per-run script log from `./logs/<run_id>.txt`
2. Optionally copy the `tee` console log if you used it
3. If you want to keep the record folder up to date, copy the run log to `train.log`
4. For seeded runs, also copy it to `train_seed<seed>.log`

```bash
mkdir -p /workspace/pg-results

# The script prints "logfile: logs/<run_id>.txt". Grab the latest one if needed.
LATEST_LOG=$(ls -t /dev/shm/parameter-golf-fork/logs/*.txt | head -n 1)
cp "$LATEST_LOG" /workspace/pg-results/

# If you used tee for the console stream:
cp /dev/shm/pg-logs/train_console.log /workspace/pg-results/ 2>/dev/null || true

# Update the record directory log files if desired.
cp "$LATEST_LOG" /dev/shm/parameter-golf-fork/records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train.log

# Example for a seeded run:
SEED=7
cp "$LATEST_LOG" "/dev/shm/parameter-golf-fork/records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_seed${SEED}.log"
```

If the run writes any result files or artifacts, copy those too.
