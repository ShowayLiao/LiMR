<div align="center">

# LiMR

**面向工业物联网的轻量级掩码重建实时传感器驱动异常检测**

<p align="center">
  🌐 <a href="README.md">English</a>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11606365">📄 论文</a> &nbsp;|&nbsp;
  <a href="https://github.com/ShowayLiao/LiMR_cpp">⚡ C++ 实现</a>
</p>

[![pytorch](https://img.shields.io/badge/pytorch-2.6%2B-orange)]()
[![lightning](https://img.shields.io/badge/lightning-2.2%2B-blue)]()
[![anomalib](https://img.shields.io/badge/anomalib-v2.3-green)]()

<img src="assets/framework.png" alt="LiMR Framework" width="600" />

</div>

---

- [LiMR](#limr)
- [📊 性能表现](#-性能表现)
  - [AeBAD-S（航空发动机叶片）](#aebad-s航空发动机叶片)
  - [MVTec AD](#mvtec-ad)
  - [Jetson AGX Xavier 边缘部署](#jetson-agx-xavier-边缘部署)
- [🚀 快速开始](#-快速开始)
  - [⌨️ 训练](#️-训练)
  - [⌨️ 测试](#️-测试)
- [📦 支持的数据集](#-支持的数据集)
- [⚙️ 参数参考](#️-参数参考)
  - [数据参数](#数据参数)
  - [模型参数](#模型参数)
  - [训练参数](#训练参数)
- [📜 批处理脚本](#-批处理脚本)
- [📂 输出结构](#-输出结构)
- [⚡ TensorRT 高速部署](#-tensorrt-高速部署)
- [❓ 常见问题](#-常见问题)
  - [致谢](#致谢)

---

这是基于 anomalib 框架集成的 **LiMR** 实现，一种用于视觉异常检测的轻量级教师-学生架构。模型使用冻结的教师编码器提取多尺度特征，同时轻量级 MobileViTv2 学生编码器-解码器在掩码重建下重构语义特征——以高吞吐量实现有竞争力的检测精度。

> 📄 **原始仓库：** [ShowayLiao/LiMR](https://github.com/ShowayLiao/LiMR/tree/main)
> 📘 **Anomalib 上游 README：** [originalREADME.md](originalREADME.md)

---

# 📊 性能表现

## AeBAD-S（航空发动机叶片）

LiMR 在 AeBAD-S 上达到**最先进**性能，同时大幅降低模型复杂度。

| 方法 | 参数量 (M) | FLOPs (G) | 延迟 (ms) | 吞吐量 (img/s) | AUROC (%) | PRO (%) |
|:-----|:-----------|:----------|:----------|:---------------|:----------|:--------|
| MMR    | 170.98 | 18.60 | 33.83 | 29.56 | 84.7 | 89.1 |
| PatchCore | 68.95 | 11.46 | 79.62 | 12.56 | 71.0 | 87.8 |
| RD | 91.75 | 31.61 | 25.98 | 38.49 | 81.0 | 85.6 |
| SimpleNet | 11.68 | 1.83 | 63.81 | 15.67 | 58.4 | 68.3 |
| **LiMR-175** | **40.06** | **10.21** | **23.26** | **42.99** | **85.3** | **91.1** |
| LiMR-175 (TensorRT) | 40.06 | 10.21 | **8.26** | **121.07** | 85.3 | 91.1 |

> 相较 MMR：**参数减少 76.6%**，**FLOPs 降低 45.1%**，**吞吐量提升 45.5%**，同时 AUROC 和 PRO 均超越 MMR。

<img src="assets/aebad-s.png" alt="AeBAD-S Results" width="600" />

## MVTec AD

LiMR 在标准 MVTec AD 基准上保持有竞争力的精度。

| 方法 | 图像 AUROC (%) | 像素 AUROC (%) |
|:-----|:---------------|:---------------|
| PatchCore | 99.1 | 98.1 |
| RD | 98.5 | 97.8 |
| SimpleNet | **99.6** | **98.1** |
| MMR | 98.4 | 97.2 |
| **LiMR** | **97.5** | **96.9** |

> LiMR 在使用轻量级 CNN-ViT 混合架构的同时达到与 SOTA 方法相当的精度——非常适合资源受限场景的部署。

<img src="assets/mvtec.png" alt="MVTec AD Results" width="600" />

## Jetson AGX Xavier 边缘部署

| 方法 | 显存 (GB) | 延迟 (ms) | 吞吐量 (FPS) | AUROC (%) |
|:-----|:----------|:----------|:-------------|:----------|
| LiMR | 1.44 | 82.22 | 12.16 | 85.32 |
| LiMR (TensorRT FP32) | 1.20 | 32.01 | 31.18 | 85.32 |
| LiMR (TensorRT FP16) | **0.48** | **15.66** | **63.87** | 85.23 |
| MMR | 1.64 | 106.32 | 9.41 | 84.73 |

> TensorRT FP16 在边缘设备上仅需 0.48 GB 显存，即可实现 **63.87 FPS** 实时推理。

<img src="assets/print.png" alt="Printed Material Dataset" width="600" />

<img src="assets/latency.png" alt="Latency Comparison" width="600" />

---

# 🚀 快速开始

## ⌨️ 训练

```bash
# 激活环境
.venv\Scripts\activate.bat

# MVTec（单个类别）
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

# Folder（自定义数据集）
python tools\limr\train.py --dataset folder --root ./datasets/my_data \
    --folder-normal-dir normal --folder-abnormal-dir abnormal --folder-mask-dir mask \
    --category my_dataset
```

## ⌨️ 测试

```bash
# 测试 anomalib 检查点 (.ckpt)
python tools\limr\test.py --dataset mvtec --root ./datasets/MVTec --category bottle \
    --image-size 224 --backbone resnet50 --alpha 1.75 --checkpoint ./output_limr/xxx.ckpt

# 测试原始 LiMR 权重 (.pth)
python tools\limr\test.py --dataset aebad_s --root I:\exp\datasets\AeBAD\AeBAD_S \
    --category AeBAD_S --image-size 256 --backbone resnet34 --alpha 1.75 \
    --block-dropout 0.0 --block-ffn-dropout 0.0 --block-attn-dropout 0.0 \
    --frozen-stages 3 --seed 54 --original-checkpoint I:\exp\LiMR\best_student_model_175.pth
```

> 📘 **注意：** 测试时必须使用与训练相同的 `backbone`、`alpha`、`frozen-stages`、`fpn-output-dim` 和 `block-*dropout` 参数，否则权重加载将不完整。

---

# 📦 支持的数据集

| `--dataset` | 数据集 | 说明 |
|:------------|:-------|:-----|
| `mvtec` | MVTec AD | 15 个工业类别 |
| `mvtecad2` | MVTec AD 2 | 扩展类别 |
| `mvtec_loco` | MVTec LOCO | 逻辑约束异常 |
| `btech` | BeanTech | 3 个类别 |
| `bmad` | BMAD | 生物医学异常 |
| `mpdd` | MPDD | 金属零件 |
| `vad` | VAD | 车辆异常 |
| `visa` | VISA | 12 个类别 |
| `realiad` | RealIAD | 30 个类别，多分辨率 |
| `kolektor` | KolektorSDD2 | 表面缺陷 |
| `aebad_s` | AeBAD-S | 静态图像 + 4 种域偏移 |
| `aebad_v` | AeBAD-V | 视频帧 |
| `folder` | Folder | 自定义数据集 |

---

# ⚙️ 参数参考

## 数据参数

| 参数 | 描述 | 默认值 |
|:-----|:-----|:-------|
| `--dataset` | 数据集名称 | `mvtec` |
| `--root` | 数据集根目录 | `./datasets/MVTec` |
| `--category` | 子类别 / 物体类别 | `bottle` |
| `--image-size` | 输入图像尺寸（正方形） | `256` |
| `--train-batch-size` | 训练批次大小 | `16` |
| `--eval-batch-size` | 评估批次大小 | `16` |
| `--num-workers` | DataLoader 工作线程数 | `6` |

## 模型参数

| 参数 | 描述 | 默认值 |
|:-----|:-----|:-------|
| `--backbone` | 教师编码器 | `resnet50` |
| `--alpha` | 学生网络宽度乘数（1.0=100%） | `1.75` |
| `--mask-ratio` | 训练掩码比例 | `0.4` |
| `--test-mask-ratio` | 测试掩码比例 | `0.0` |
| `--fpn-output-dim` | FPN 每层输出通道数 | 自动检测 |
| `--block-dropout` | MobileViTBlockv2 dropout | `0.1` |
| `--block-ffn-dropout` | FFN dropout | `0.0` |
| `--block-attn-dropout` | 注意力 dropout | `0.0` |
| `--frozen-stages` | 冻结编码器阶段数（1-3） | `3` |

## 训练参数

| 参数 | 描述 | 默认值 |
|:-----|:-----|:-------|
| `--epochs` | 最大训练轮数 | `200` |
| `--lr` | 学习率 | `0.001` |
| `--weight-decay` | 权重衰减 | `0.05` |
| `--warmup-epochs` | 学习率预热轮数 | `15` |
| `--early-stop-patience` | 早停耐心值 | `10` |
| `--seed` | 随机种子 | `54` |

<details>
<summary><strong>测试与输出参数</strong></summary>

| 参数 | 描述 |
|:-----|:-----|
| `--checkpoint` | anomalib Lightning 检查点路径（`.ckpt`） |
| `--original-checkpoint` | 原始 LiMR 权重路径（`.pth`） |
| `--output-dir` | 输出目录 |
| `--project-name` | W&B 项目名称 |

</details>

---

# 📜 批处理脚本

双击任意 `.bat` 文件即可运行。请确保已提前创建 `.venv` 环境。

| 脚本 | 用途 |
|:-----|:-----|
| `scripts\limr\train_aebad_s.bat` | 训练 AeBAD-S |
| `scripts\limr\train_aebad_v.bat` | 训练 AeBAD-V |
| `scripts\limr\train_mvtec.bat` | 训练 MVTec（单类别） |
| `scripts\limr\train_mvtec_all.bat` | 训练 MVTec（全部 15 个类别） |
| `scripts\limr\train_visa.bat` | 训练 VISA |
| `scripts\limr\train_realiad.bat` | 训练 RealIAD |
| `scripts\limr\train_folder.bat` | 训练自定义数据集 |
| `scripts\limr\train_material.bat` | 训练 Material 数据集（全部 4 类） |
| `scripts\limr\test_aebad_s.bat` | 测试 AeBAD-S |
| `scripts\limr\test_aebad_v.bat` | 测试 AeBAD-V |
| `scripts\limr\test_mvtec.bat` | 测试 MVTec |
| `scripts\limr\test_visa.bat` | 测试 VISA |
| `scripts\limr\test_realiad.bat` | 测试 RealIAD |
| `scripts\limr\test_folder.bat` | 测试自定义数据集 |

---

# 📂 输出结构

训练完成后，以下内容将生成在 `--output-dir` 下：

```
output_limr/aebad_s/
├── LiMR/AeBAD_S/AeBAD_S/v34/
│   └── weights/
│       ├── lightning/model.ckpt   # 最佳检查点
│       └── onnx/model.onnx        # ONNX 导出
├── inference_speed.json           # 推理速度基准测试
└── wandb/                         # W&B 日志（如已启用）
```

---

# ⚡ TensorRT 高速部署

训练完成后，ONNX 模型已自动导出至 `--output-dir` 下的 `weights/onnx/model.onnx`。使用 [LiMR C++ TensorRT](https://github.com/ShowayLiao/LiMR_cpp) 仓库进行高速推理。

---

# ❓ 常见问题

<details>
<summary><strong>显存不足 (OOM)？</strong></summary>

- 降低 `--train-batch-size`（例如 `8` 或 `6`）
- 减少 `--num-workers`
- 切换 `--backbone` 为 `resnet18`

</details>

<details>
<summary><strong>测试时权重加载报错？</strong></summary>

确保测试参数与训练时一致。使用 `--original-checkpoint` 加载原始权重时，必须使用 `--backbone resnet34`（原始仓库默认值）。

</details>

<details>
<summary><strong>AeBAD 多域偏移测试如何工作？</strong></summary>

AeBAD 测试会自动遍历所有 `domain_shift` 子目录（`same`、`background`、`illumination`、`view`），并在最后报告跨域平均指标。每个域偏移的单独结果也会记录到 W&B。

</details>

---

## 致谢

我们感谢本工作所基于的优秀开源实现：

- [ConvMAE](https://github.com/Alpha-VL/ConvMAE)
- [MobileViTv2](https://github.com/apple/ml-cvnets)
- [MobileViTv2-pytorch](https://github.com/HowardLi0816/MobileViTv2_pytorch)
- [MMR](https://github.com/zhangzilongc/MMR)
