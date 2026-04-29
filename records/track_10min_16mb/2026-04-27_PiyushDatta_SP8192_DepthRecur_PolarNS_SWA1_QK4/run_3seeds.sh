#!/bin/bash
# Run 3-seed training + eval for parameter-golf submission
# Use on 8xH100: bash run_3seeds.sh
# Use on 4xA100: NPROC=4 bash run_3seeds.sh

set -e

NPROC=${NPROC:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Parameter Golf 3-Seed Submission Run ==="
echo "GPUs: $NPROC"
echo "Script: $SCRIPT_DIR/train_gpt.py"
echo ""

for SEED in 42 314 999; do
    echo "=========================================="
    echo "Starting seed=$SEED at $(date)"
    echo "=========================================="

    SEED=$SEED \
    TTT_LORA_RANK=0 \
    TTT_EPOCHS=1 \
    TTT_CHUNK_TOKENS=65536 \
    DATA_DIR=./data/ \
    torchrun --standalone --nproc_per_node=$NPROC \
        "$SCRIPT_DIR/train_gpt.py" \
        2>&1 | tee "$SCRIPT_DIR/train_seed${SEED}.log"

    echo ""
    echo "Seed $SEED complete at $(date)"
    echo "Result:"
    grep "quantized_ttt\|quantized_sliding\|Total submission" "$SCRIPT_DIR/train_seed${SEED}.log" | tail -3
    echo ""
done

echo "=== All 3 seeds complete ==="
echo ""
echo "Results summary:"
for SEED in 42 314 999; do
    BPB=$(grep "quantized_ttt" "$SCRIPT_DIR/train_seed${SEED}.log" 2>/dev/null | grep -o "val_bpb:[0-9.]*" | tail -1)
    SIZE=$(grep "Total submission" "$SCRIPT_DIR/train_seed${SEED}.log" 2>/dev/null | grep -o "[0-9]* bytes" | tail -1)
    echo "  Seed $SEED: $BPB artifact=$SIZE"
done
