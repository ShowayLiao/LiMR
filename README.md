<div align="center">

# LiMR

**Lightweight Masked Reconstruction for Real-Time Sensor-Driven Anomaly Detection in Industrial IoT**

<p align="center">
  🌐 <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11606365">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://github.com/ShowayLiao/LiMR_cpp">⚡ C++ Implementation</a>
</p>

[![pytorch](https://img.shields.io/badge/pytorch-2.6%2B-orange)]()
[![lightning](https://img.shields.io/badge/lightning-2.2%2B-blue)]()
[![anomalib](https://img.shields.io/badge/anomalib-v2.3-green)]()

<img src="./assets/framework.png" alt="LiMR Framework" width="600" />





</div>

---

- [LiMR](#limr)
- [📊 Performance](#-performance)
  - [AeBAD-S (Aero-Engine Blade)](#aebad-s-aero-engine-blade)
  - [MVTec AD](#mvtec-ad)
  - [Jetson AGX Xavier Edge Deployment](#jetson-agx-xavier-edge-deployment)
- [🚀 Quick Start](#-quick-start)
  - [⌨️ Training](#️-training)
  - [⌨️ Testing](#️-testing)
- [📦 Supported Datasets](#-supported-datasets)
- [⚙️ Parameter Reference](#️-parameter-reference)
  - [Data Parameters](#data-parameters)
  - [Model Parameters](#model-parameters)
  - [Training Parameters](#training-parameters)
- [📜 Batch Scripts](#-batch-scripts)
- [📂 Output Structure](#-output-structure)
- [⚡ TensorRT High-Speed Deployment](#-tensorrt-high-speed-deployment)
- [❓ FAQ](#-faq)
  - [Acknowledgements](#acknowledgements)

---

This is the anomalib-integrated implementation of **LiMR**, a lightweight teacher-student architecture for visual anomaly detection. The model uses a frozen teacher encoder to extract multi-scale features, while a lightweight MobileViTv2-based student encoder-decoder reconstructs semantic features under masked reconstruction — achieving competitive accuracy at high throughput.

> 📄 **Original repository:** [ShowayLiao/LiMR](https://github.com/ShowayLiao/LiMR/tree/main)
> 📘 **Anomalib upstream README:** [originalREADME.md](originalREADME.md)

---

# 📊 Performance

## AeBAD-S (Aero-Engine Blade)

LiMR achieves **state-of-the-art** performance on AeBAD-S while drastically reducing model complexity.

| Method | Params (M) | FLOPs (G) | Latency (ms) | Throughput (img/s) | AUROC (%) | PRO (%) |
|:-------|:-----------|:----------|:-------------|:-------------------|:----------|:--------|
| MMR    | 170.98 | 18.60 | 33.83 | 29.56 | 84.7 | 89.1 |
| PatchCore | 68.95 | 11.46 | 79.62 | 12.56 | 71.0 | 87.8 |
| RD | 91.75 | 31.61 | 25.98 | 38.49 | 81.0 | 85.6 |
| SimpleNet | 11.68 | 1.83 | 63.81 | 15.67 | 58.4 | 68.3 |
| **LiMR-175** | **40.06** | **10.21** | **23.26** | **42.99** | **85.3** | **91.1** |
| LiMR-175 (TensorRT) | 40.06 | 10.21 | **8.26** | **121.07** | 85.3 | 91.1 |

> Compared to MMR: **76.6% fewer params**, **45.1% lower FLOPs**, **45.5% higher throughput**, while surpassing MMR in both AUROC and PRO.

<img src="./assets/aebad-s.png" alt="LiMR Framework" width="600" />

## MVTec AD

LiMR maintains competitive accuracy on the standard MVTec AD benchmark.

| Method | Image AUROC (%) | Pixel AUROC (%) |
|:-------|:----------------|:----------------|
| PatchCore | 99.1 | 98.1 |
| RD | 98.5 | 97.8 |
| SimpleNet | **99.6** | **98.1** |
| MMR | 98.4 | 97.2 |
| **LiMR** | **97.5** | **96.9** |

> LiMR achieves comparable accuracy to SOTA methods while using a lightweight CNN-ViT hybrid architecture — ideal for resource-constrained deployment.

<img src="./assets/mvtec.png" alt="LiMR Framework" width="600" />

## Jetson AGX Xavier Edge Deployment

| Method | VRAM (GB) | Latency (ms) | Throughput (FPS) | AUROC (%) |
|:-------|:----------|:-------------|:-----------------|:----------|
| LiMR | 1.44 | 82.22 | 12.16 | 85.32 |
| LiMR (TensorRT FP32) | 1.20 | 32.01 | 31.18 | 85.32 |
| LiMR (TensorRT FP16) | **0.48** | **15.66** | **63.87** | 85.23 |
| MMR | 1.64 | 106.32 | 9.41 | 84.73 |

> TensorRT FP16 achieves **63.87 FPS** real-time inference with only 0.48 GB VRAM on edge devices.

<img src="./assets/print.png" alt="LiMR Framework" width="600" />

<img src="./assets/latency.png" alt="LiMR Framework" width="600" />



---

# 🚀 Quick Start

## ⌨️ Training

```bash
# Activate environment
.venv\Scripts\activate.bat

# MVTec (single category)
python tools\limr\train.py --dataset mvtec --root ./datasets/MVTec --category bottle \
    --image-size 224 --backbone resnet50 --alpha 1.75 --epochs 200

# AeBAD-S
python tools\limr\train.py --dataset aebad_s --root I:\exp\datasets\AeBAD\AeBAD_S \
    --category AeBAD_S --image-size 256 --backbone resnet34 --alpha 1.75 \
    --fpn-output-dim 64 128 256 512 --block-dropout 0.0 --block-ffn-dropout 0.0 \
    --block-attn-dropout 0.0 --frozen-stages 3 --epochs 200 --seed 54

# AeBAD-V
python tools\limr\train.py --dataset aebad_v --root I:\exp\datasets\AeBAD\AeBAD_V \
    --category AeBAD_V --image-size 256 --backbone resnet34 --alpha 1.75 \
    --fpn-output-dim 64 128 256 512 --block-dropout 0.0 --block-ffn-dropout 0.0 \
    --block-attn-dropout 0.0 --frozen-stages 3 --epochs 200 --seed 54

# RealIAD
python tools\limr\train.py --dataset realiad --root ./datasets/RealIAD --category <category> \
    --realiad-resolution 256 --realiad-json ./datasets/RealIAD/RealIAD.json

# VISA
python tools\limr\train.py --dataset visa --root ./datasets/VISA --category candle \
    --image-size 224 --backbone resnet50

# Folder (custom dataset)
python tools\limr\train.py --dataset folder --root ./datasets/my_data \
    --folder-normal-dir normal --folder-abnormal-dir abnormal --folder-mask-dir mask \
    --category my_dataset
```

## ⌨️ Testing

```bash
# Test anomalib checkpoint (.ckpt)
python tools\limr\test.py --dataset mvtec --root ./datasets/MVTec --category bottle \
    --image-size 224 --backbone resnet50 --alpha 1.75 --checkpoint ./output_limr/xxx.ckpt

# Test original LiMR weights (.pth)
python tools\limr\test.py --dataset aebad_s --root I:\exp\datasets\AeBAD\AeBAD_S \
    --category AeBAD_S --image-size 256 --backbone resnet34 --alpha 1.75 \
    --block-dropout 0.0 --block-ffn-dropout 0.0 --block-attn-dropout 0.0 \
    --frozen-stages 3 --seed 54 --original-checkpoint I:\exp\LiMR\best_student_model_175.pth
```

> 📘 **Note:** Testing must use the same `backbone`, `alpha`, `frozen-stages`, `fpn-output-dim`, and `block-*dropout` settings as training, otherwise weight loading will be incomplete.

---

# 📦 Supported Datasets

| `--dataset` | Dataset | Notes |
|:------------|:--------|:------|
| `mvtec` | MVTec AD | 15 industrial categories |
| `mvtecad2` | MVTec AD 2 | Extended categories |
| `mvtec_loco` | MVTec LOCO | Logical constraints |
| `btech` | BeanTech | 3 categories |
| `bmad` | BMAD | Biomedical anomaly |
| `mpdd` | MPDD | Metal parts |
| `vad` | VAD | Vehicle anomaly |
| `visa` | VISA | 12 categories |
| `realiad` | RealIAD | 30 categories, multi-resolution |
| `kolektor` | KolektorSDD2 | Surface defect |
| `aebad_s` | AeBAD-S | Static images + 4 domain shifts |
| `aebad_v` | AeBAD-V | Video frames |
| `folder` | Folder | Custom dataset |

---

# ⚙️ Parameter Reference

## Data Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--dataset` | Dataset name | `mvtec` |
| `--root` | Dataset root directory | `./datasets/MVTec` |
| `--category` | Sub-category / object class | `bottle` |
| `--image-size` | Input image size (square) | `256` |
| `--train-batch-size` | Training batch size | `16` |
| `--eval-batch-size` | Evaluation batch size | `16` |
| `--num-workers` | DataLoader workers | `6` |

## Model Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--backbone` | Teacher encoder | `resnet50` |
| `--alpha` | Student width multiplier (1.0=100%) | `1.75` |
| `--mask-ratio` | Training mask ratio | `0.4` |
| `--test-mask-ratio` | Test mask ratio | `0.0` |
| `--fpn-output-dim` | FPN output channels per layer | auto-detect |
| `--block-dropout` | MobileViTBlockv2 dropout | `0.1` |
| `--block-ffn-dropout` | FFN dropout | `0.0` |
| `--block-attn-dropout` | Attention dropout | `0.0` |
| `--frozen-stages` | Frozen encoder stages (1-3) | `3` |

## Training Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--epochs` | Max training epochs | `200` |
| `--lr` | Learning rate | `0.001` |
| `--weight-decay` | Weight decay | `0.05` |
| `--warmup-epochs` | LR warmup epochs | `15` |
| `--early-stop-patience` | Early stopping patience | `10` |
| `--seed` | Random seed | `54` |

<details>
<summary><strong>Test & Output Parameters</strong></summary>

| Parameter | Description |
|:----------|:------------|
| `--checkpoint` | anomalib Lightning ckpt path (`.ckpt`) |
| `--original-checkpoint` | Original LiMR weights path (`.pth`) |
| `--output-dir` | Output directory |
| `--project-name` | W&B project name |

</details>

---

# 📜 Batch Scripts

Double-click any `.bat` file to run. Ensure `.venv` is created beforehand.

| Script | Purpose |
|:-------|:--------|
| `scripts\limr\train_aebad_s.bat` | Train AeBAD-S |
| `scripts\limr\train_aebad_v.bat` | Train AeBAD-V |
| `scripts\limr\train_mvtec.bat` | Train MVTec (single category) |
| `scripts\limr\train_mvtec_all.bat` | Train MVTec (all 15 categories) |
| `scripts\limr\train_visa.bat` | Train VISA |
| `scripts\limr\train_realiad.bat` | Train RealIAD |
| `scripts\limr\train_folder.bat` | Train custom dataset |
| `scripts\limr\test_aebad_s.bat` | Test AeBAD-S |
| `scripts\limr\test_aebad_v.bat` | Test AeBAD-V |
| `scripts\limr\test_mvtec.bat` | Test MVTec |
| `scripts\limr\test_visa.bat` | Test VISA |
| `scripts\limr\test_realiad.bat` | Test RealIAD |
| `scripts\limr\test_folder.bat` | Test custom dataset |

---

# 📂 Output Structure

After training, the following is generated under `--output-dir`:

```
output_limr/aebad_s/
├── LiMR/AeBAD_S/AeBAD_S/v34/
│   └── weights/
│       ├── lightning/model.ckpt   # Best checkpoint
│       └── onnx/model.onnx        # ONNX export
├── inference_speed.json           # Inference benchmark
└── wandb/                         # W&B logs (if enabled)
```

---

# ⚡ TensorRT High-Speed Deployment

After training, the ONNX model is auto-exported to `weights/onnx/model.onnx` under `--output-dir`. Use the [LiMR C++ TensorRT](https://github.com/ShowayLiao/LiMR_cpp) repository for high-speed inference.

---

# ❓ FAQ

<details>
<summary><strong>Out of memory (OOM)?</strong></summary>

- Reduce `--train-batch-size` (e.g., `8` or `6`)
- Reduce `--num-workers`
- Switch `--backbone` to `resnet18`

</details>

<details>
<summary><strong>Weight loading errors during testing?</strong></summary>

Ensure test parameters match training. When using `--original-checkpoint` to load legacy weights, you must use `--backbone resnet34` (default in the original repo).

</details>

<details>
<summary><strong>How does AeBAD multi-shift testing work?</strong></summary>

AeBAD testing automatically iterates over all `domain_shift` subdirectories (`same`, `background`, `illumination`, `view`) and reports cross-shift average metrics at the end. Each shift's individual results are also logged to W&B.

</details>

---

<!-- # 📚 Reference

If you find LiMR useful in your research or work, please cite the original paper:

```tex
@inproceedings{liao2024limr,
  title     = {Lightweight Masked Reconstruction for Real-Time Sensor-Driven
               Anomaly Detection in Industrial IoT},
  author    = {Liao, Showay and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2024},
}
``` -->

## Acknowledgements

We acknowledge the excellent open-source implementations that this work builds upon:

- [ConvMAE](https://github.com/Alpha-VL/ConvMAE)
- [MobileViTv2](https://github.com/apple/ml-cvnets)
- [MobileViTv2-pytorch](https://github.com/HowardLi0816/MobileViTv2_pytorch)
- [MMR](https://github.com/zhangzilongc/MMR)
