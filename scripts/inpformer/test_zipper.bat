@echo off
cd /d %~dp0..\..
call .venv\Scripts\activate.bat
set CKPT=./output_inpformer/zipper_nylon/lightning_logs/version_0/checkpoints/epoch=xxx.ckpt

python tools\inpformer\test.py ^
    --dataset ZipperAD ^
    --root ./datasets/ZipperAD ^
    --category nylon ^
    --image-size 448 ^
    --crop-size 392 ^
    --eval-batch-size 16 ^
    --num-workers 8 ^
    --encoder-name dinov2reg_vit_base_14 ^
    --inp-num 6 ^
    --decoder-depth 8 ^
    --checkpoint %CKPT% ^
    --output-dir ./output_inpformer/test_zipper_nylon
pause
