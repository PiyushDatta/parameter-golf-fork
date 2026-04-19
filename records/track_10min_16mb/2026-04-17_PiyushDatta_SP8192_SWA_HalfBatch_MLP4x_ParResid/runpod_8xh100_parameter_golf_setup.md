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

## 6) Download the SP1024 dataset for the March 20 submission

```
cd /dev/shm/parameter-golf-fork
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
ls -lah data/datasets/fineweb10B_sp1024 | head
ls -lah data/tokenizers
```

You should see files like:

-   `data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin`
-   `data/tokenizers/fineweb_1024_bpe.model`

---

## 7) Launch training

Use the **venv Python** explicitly.

```bash
cd /dev/shm/parameter-golf-fork

export HF_HUB_DISABLE_XET=1
export HF_HOME=/dev/shm/hf-cache
export HUGGINGFACE_HUB_CACHE=/dev/shm/hf-cache/hub
export HF_DATASETS_CACHE=/dev/shm/hf-cache/datasets
export TMPDIR=/dev/shm

/workspace/uv-envs/parameter-golf/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py \
  2>&1 | tee /dev/shm/pg-logs/train_sp1024_8xh100.log
```

---

## 8) Save logs/results somewhere persistent

Because `/dev/shm` is temporary, copy logs back out after the run.

```bash
mkdir -p /workspace/pg-results
cp /dev/shm/pg-logs/train_sp1024_8xh100.log /workspace/pg-results/
```

If the run writes any result files or artifacts, copy those too.
