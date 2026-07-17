#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORIES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)

for c in "${CATEGORIES[@]}"; do
    echo "========================================"
    echo "Training category: $c"
    echo "========================================"
    python tools/inpformer/train.py \
        --dataset visa \
        --root ./datasets/visa \
        --category "$c" \
        --image-size 448 \
        --train-batch-size 16 \
        --eval-batch-size 16 \
        --num-workers 8 \
        --encoder-name dinov2reg_vit_base_14 \
        --inp-num 6 \
        --decoder-depth 8 \
        --bottleneck-dropout 0.0 \
        --max-steps 5000 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --warmup-iters 100 \
        --early-stop-patience 20 \
        --seed 1 \
        --output-dir "./output_inpformer/visa_${c}" \
        --project-name INP-Former_Anomalib
done
