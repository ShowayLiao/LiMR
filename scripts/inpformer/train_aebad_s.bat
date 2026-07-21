@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat

python tools\inpformer\train.py ^
    --dataset aebad_s ^
    --root I:\exp\datasets\AeBAD\AeBAD_S ^
    --category AeBAD_S ^
    --image-size 256 ^
    --train-batch-size 6 ^
    --eval-batch-size 6 ^
    --num-workers 4 ^
    --aebad-s-domain-shift same ^
    --encoder-name dinov2reg_vit_base_14 ^
    --inp-num 6 ^
    --decoder-depth 8 ^
    --bottleneck-dropout 0.0 ^
    --max-steps 5000 ^
    --lr 1e-3 ^
    --weight-decay 1e-4 ^
    --warmup-iters 100 ^
    --early-stop-patience 20 ^
    --seed 1 ^
    --output-dir ./output_inpformer/aebad_s ^
    --project-name INP-Former_Anomalib

pause
