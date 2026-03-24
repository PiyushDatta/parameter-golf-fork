# 11L Full SOTA Stack

## Summary

11-layer transformer with the full competitive SOTA stack targeting sub-1.13 val_bpb.

## Architecture

- 11 layers, dim=512, 8 heads, 4 KV heads (GQA), 3x MLP (relu^2)
- Value Residual (ResFormer) - cache V from layer 0, mix via learned lambda
- Gated Attention - per-head sigmoid gate after SDPA
- XSA on last 4 layers - cross-sequence attention via value projection subtraction
- Partial RoPE (16/64 head dims)
- LN Scale (1/sqrt(layer_idx+1)) - depth-dependent norm scaling
- SmearGate + BigramHash(2048, dim=128)
- U-Net skip connections (encoder/decoder halves)
- Per-layer attn_scale, mlp_scale, resid_mix
- Logit softcap=30, qk_gain_init=1.5, tied embeddings

## Training

- Muon optimizer (lr=0.025, momentum=0.99, WD=0.04) for matrix params
- AdamW (lr=0.035 embed, lr=0.025 scalar) for non-matrix params
- EMA (decay=0.997, every step)
- 786K tokens/step, seq_len=2048
- Wall-clock adaptive warmdown (3500 steps)
- 20 warmup steps with state reset

## Post-Training

- 3% magnitude pruning
- GPTQ-lite int6 quantization (optimal clip percentile search over 5 candidates)
- zlib compression
- Dequantize roundtrip validation

## Evaluation

- Full-weight SGD TTT (lr=0.008, momentum=0.9, 20 epochs, cosine schedule)
- Sliding window eval (stride=64)

## Setup

No extra dependencies required (uses only stdlib + PyTorch/numpy/sentencepiece).

## Run Command

```bash
RUN_ID=piyush_datta_submission \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
./records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py
```

For local 1xGPU testing (reduced TTT):
```bash
TTT_EPOCHS=2 TTT_BATCH_SEQS=16 EVAL_STRIDE=0 \
torchrun --standalone --nproc_per_node=1 \
./records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/train_gpt.py
```

## Key Results (1xA100, 344 steps)

- val_bpb (pre-quant): 1.9217
- Model int6+zlib: ~5.4MB (well under 16MB cap)
- Step avg: ~1748ms/step
- Estimated ~10,000+ steps on 8xH100
- final_int8_zlib_roundtrip val_loss:2.9098 val_bpb:1.7234 eval_time:3060331ms
- final_int8_zlib_roundtrip_exact val_loss:2.90983668 val_bpb:1.72337375

## Included Files

- `train_gpt.py` - training script
- `submission.json` - leaderboard metadata
- `notes.md` - experiment log and implementation notes
