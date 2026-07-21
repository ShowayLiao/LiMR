#!/usr/bin/env python3
"""INP-Former 训练脚本：使用 INP-Former 模型进行异常检测模型的训练与评估。

支持的数据集（通过注册表自动扩展）：
  - mvtec, mvtecad2, mvtec_loco, btech, bmad, mpdd, vad, visa, kolektor, folder
  - realiad, aebad_s, aebad_v

用法示例:
  # MVTec-AD 单类别训练
  python tools/inpformer/train.py --dataset mvtec --root ./datasets/MVTec --category bottle

  # VisA 训练
  python tools/inpformer/train.py --dataset visa --root ./datasets/VisA --category candle

  # RealIAD 训练
  python tools/inpformer/train.py --dataset realiad --root ./datasets/Real-IAD --category end_cap --realiad-resolution 1024

  # Folder 数据集训练
  python tools/inpformer/train.py --dataset folder --root ./datasets/my_data --folder-normal-dir normal --folder-abnormal-dir abnormal
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch import seed_everything

from anomalib.callbacks import TimerCallback
from anomalib.data import (
    AeBAD_S, AeBAD_V,
    BMAD, BTech, Folder, Kolektor, MPDD, MVTecAD, MVTecAD2, MVTecLOCO,
    RealIAD, VAD, Visa, ZipperAD,
)
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUPR, AUROC, F1Max, PBn, Evaluator
from anomalib.models import INP_Former

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

    class WandbLogger:
        def __init__(self, *args, **kwargs): pass

    class wandb:
        @staticmethod
        def finish(): pass
        @staticmethod
        def log(*args, **kwargs): pass


# ---------------------------------------------------------------------------
# 数据集注册表
#   key       : --dataset 命令行参数的值
#   value[0]  : 数据集类
#   value[1]  : 从 args 构建该类构造参数的函数，返回 dict
# ---------------------------------------------------------------------------
def _build_standard_dataset(args):
    """适用于 root + category 模式的标准数据集（MVTecAD, Visa, BTech 等）。"""
    return {
        "root": args.root,
        "category": args.category,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }


def _build_realiad(args):
    return {
        "root": args.root,
        "category": args.category,
        "resolution": args.realiad_resolution,
        "json_path": args.realiad_json,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "test_split_mode": args.realiad_test_split_mode,
    }


def _build_kolektor(args):
    return {
        "root": args.root,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }


def _build_folder(args):
    return {
        "root": args.root,
        "normal_dir": args.folder_normal_dir,
        "normal_test_dir": args.folder_normal_test_dir,
        "abnormal_dir": args.folder_abnormal_dir,
        "mask_dir": args.folder_mask_dir,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "name": args.category or "folder_dataset",
    }


def _build_aebad_s(args):
    return {
        "root": args.root,
        "category": args.category,
        "domain_shift": args.aebad_s_domain_shift,
        "image_size": (args.image_size, args.image_size),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "train_augmentations": v2.Compose([
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.7, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
        ]),
    }


def _build_aebad_v(args):
    return {
        "root": args.root,
        "category": args.category,
        "domain_shift": args.aebad_v_domain_shift,
        "image_size": (args.image_size, args.image_size),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "train_augmentations": v2.Compose([
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.7, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
        ]),
    }


DATASET_REGISTRY = {
    "mvtec":       (MVTecAD,    _build_standard_dataset),
    "mvtecad2":    (MVTecAD2,   _build_standard_dataset),
    "mvtec_loco":  (MVTecLOCO,  _build_standard_dataset),
    "btech":       (BTech,      _build_standard_dataset),
    "bmad":        (BMAD,       _build_standard_dataset),
    "mpdd":        (MPDD,       _build_standard_dataset),
    "vad":         (VAD,        _build_standard_dataset),
    "visa":        (Visa,       _build_standard_dataset),
    "zipperad":    (ZipperAD,   _build_standard_dataset),
    "realiad":     (RealIAD,    _build_realiad),
    "kolektor":    (Kolektor,   _build_kolektor),
    "folder":      (Folder,     _build_folder),
    "aebad_s":     (AeBAD_S,    _build_aebad_s),
    "aebad_v":     (AeBAD_V,    _build_aebad_v),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=f"INP-Former 训练脚本。支持数据集: {', '.join(DATASET_REGISTRY)}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # 数据集参数
    # ============================================================
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=list(DATASET_REGISTRY),
                        help="数据集名称")
    parser.add_argument("--root", type=str, default="./datasets/MVTec",
                        help="数据集根目录")
    parser.add_argument("--category", type=str, default="bottle",
                        help="训练类别")
    parser.add_argument("--image-size", type=int, default=448,
                        help="输入图像尺寸")
    parser.add_argument("--train-batch-size", type=int, default=16,
                        help="训练批次大小")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="评估批次大小")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="数据加载线程数")

    # ---- RealIAD 专用参数 ----
    parser.add_argument("--realiad-resolution", type=str, default="256",
                        help="RealIAD 图像分辨率")
    parser.add_argument("--realiad-json", type=str, default=None,
                        help="RealIAD JSON 配置文件路径")
    parser.add_argument("--realiad-test-split-mode", type=str, default="from_dir",
                        choices=["none", "from_dir", "synthetic"],
                        help="RealIAD 测试集构建模式")

    # ---- Folder 专用参数 ----
    parser.add_argument("--folder-normal-dir", type=str, default="normal",
                        help="Folder 数据集正常图像子目录")
    parser.add_argument("--folder-normal-test-dir", type=str, default=None,
                        help="Folder 数据集正常测试图像子目录")
    parser.add_argument("--folder-abnormal-dir", type=str, default="abnormal",
                        help="Folder 数据集异常图像子目录")
    parser.add_argument("--folder-mask-dir", type=str, default=None,
                        help="Folder 数据集掩码子目录")

    # ---- AeBAD 专用参数 ----
    parser.add_argument("--aebad-s-domain-shift", type=str, default="same",
                        choices=["same", "view"],
                        help="AeBAD_S 测试 domain shift")
    parser.add_argument("--aebad-v-domain-shift", type=str, default="video1",
                        choices=["video1", "video2", "video3"],
                        help="AeBAD_V 测试 domain shift")

    # ============================================================
    # INP-Former 模型参数
    # ============================================================
    parser.add_argument("--encoder-name", type=str, default="dinov2reg_vit_base_14",
                        help="预训练编码器名称")
    parser.add_argument("--inp-num", type=int, default=6,
                        help="内在正常原型 (INP) 数量")
    parser.add_argument("--decoder-depth", type=int, default=8,
                        help="解码器 Transformer 层数")
    parser.add_argument("--bottleneck-dropout", type=float, default=0.0,
                        help="瓶颈层 Dropout 概率")

    # ============================================================
    # 训练参数
    # ============================================================
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="最大训练步数")
    parser.add_argument("--epochs", type=int, default=None,
                        help="最大训练轮数（设置后将覆盖 max-steps）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--warmup-iters", type=int, default=100,
                        help="学习率预热迭代数")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # ============================================================
    # 输出参数
    # ============================================================
    parser.add_argument("--output-dir", type=str, default="./output_inpformer",
                        help="输出目录")
    parser.add_argument("--project-name", type=str, default="INP-Former_Anomalib",
                        help="WandB 项目名称")
    parser.add_argument("--run-name", type=str, default=None,
                        help="WandB 运行名称（默认自动生成）")

    # ============================================================
    # Benchmark 参数
    # ============================================================
    parser.add_argument("--warmup-iterations", type=int, default=10,
                        help="推理测速预热迭代次数")
    parser.add_argument("--measure-iterations", type=int, default=100,
                        help="推理测速测量迭代次数")

    return parser.parse_args()


def build_model(args: argparse.Namespace, evaluator) -> INP_Former:
    """构建 INP-Former 模型。

    Args:
        args: 命令行参数。
        evaluator: 评估器实例。

    Returns:
        INP_Former 模型实例。
    """
    return INP_Former(
        encoder_name=args.encoder_name,
        inp_num=args.inp_num,
        bottleneck_dropout=args.bottleneck_dropout,
        decoder_depth=args.decoder_depth,
        evaluator=evaluator,
    )


def measure_inference_speed(
    model: INP_Former,
    datamodule,
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """测量模型推理速度。

    提供两种延迟指标:
      - 总体延迟 (end-to-end) : 包含数据传输 + 模型推理
      - 纯推理时间 (pure)     : 仅模型推理，不含传输

    Args:
        model:  已训练的模型。
        datamodule: 数据模块。
        device: 设备类型 ("cuda" 或 "cpu")。
        warmup: 预热迭代次数。
        iterations: 测量迭代次数。

    Returns:
        包含速度测量结果的字典。
    """
    print("\n" + "=" * 80)
    print("推理速度测量")
    print("=" * 80)

    model = model.to(device)
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    # ---- 预热 + 测量 (复用同一个 DataLoader) ----
    print(f"预热阶段 ({warmup} 次迭代)...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    total_iterations = warmup + iterations
    total_time_e2e = 0.0
    total_time_pure = 0.0
    total_images = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= total_iterations:
                break

            batch_size = batch["image"].shape[0]

            if use_cuda:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                e3 = torch.cuda.Event(enable_timing=True)

                e0.record()
                images = batch["image"].to(device)
                _ = model(images)
                e1.record()
                torch.cuda.synchronize()

                images = batch["image"].to(device)
                e2.record()
                _ = model(images)
                e3.record()
                torch.cuda.synchronize()

                iter_e2e = e0.elapsed_time(e1) / 1000.0
                iter_pure = e2.elapsed_time(e3) / 1000.0
                del e0, e1, e2, e3
            else:
                t0 = time.perf_counter()
                images = batch["image"].to(device)
                _ = model(images)
                iter_e2e = time.perf_counter() - t0

                images = batch["image"].to(device)
                t2 = time.perf_counter()
                _ = model(images)
                iter_pure = time.perf_counter() - t2

            if i < warmup:
                del images
                continue

            total_time_e2e += iter_e2e
            total_time_pure += iter_pure
            total_images += batch_size

            del images
            if (i - warmup + 1) % 10 == 0:
                print(f"  iter {i - warmup + 1}/{iterations}: "
                      f"e2e={iter_e2e*1000:.2f}ms, "
                      f"pure={iter_pure*1000:.2f}ms, "
                      f"batch={batch_size}")

    del test_loader
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    avg_e2e_per_img = total_time_e2e / total_images * 1000
    avg_pure_per_img = total_time_pure / total_images * 1000
    fps_e2e = total_images / total_time_e2e
    fps_pure = total_images / total_time_pure

    print("\n" + "=" * 80)
    print("推理速度结果")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"总图像: {total_images}")
    print()
    print("【总体延迟（含数据传-推理）】")
    print(f"  平均每张: {avg_e2e_per_img:.2f} ms")
    print(f"  吞吐量: {fps_e2e:.2f} FPS")
    print()
    print("【纯推理时间（仅模型计算）】")
    print(f"  平均每张: {avg_pure_per_img:.2f} ms")
    print(f"  吞吐量: {fps_pure:.2f} FPS")
    print("=" * 80)

    return {
        "device": device,
        "total_images": total_images,
        "end_to_end": {
            "avg_ms_per_image": round(avg_e2e_per_img, 2),
            "fps": round(fps_e2e, 2),
        },
        "pure_inference": {
            "avg_ms_per_image": round(avg_pure_per_img, 2),
            "fps": round(fps_pure, 2),
        },
    }


def main():
    """主流程：构建数据集 → 构建模型 → 训练 → 测试 → 测速。"""
    args = parse_args()

    seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 根据注册表选择数据集 ----
    dataset_cls, builder = DATASET_REGISTRY[args.dataset]
    datamodule = dataset_cls(**builder(args))

    # ---- 打印配置 ----
    print("=" * 80)
    print(f"INP-Former 训练 - 数据集: {args.dataset}")
    print("=" * 80)
    for key, val in sorted(vars(args).items()):
        print(f"  {key}: {val}")
    print()

    # ---- 构建评估器 ----
    evaluator = Evaluator(
        test_metrics=[
            # --- 像素级指标 (pixel-level) ---
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),      # PRO-AUC: 缺陷区域发现能力
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),      # Pixel AUROC: 像素级定位能力
            AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),        # Pixel AUPR: 缺陷区域纯净度
            F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),       # Pixel F1-max: 缺陷分割质量
            # --- 图像级指标 (image-level) ---
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),       # Image AUROC: 图片级异常判断能力
            AUPR(fields=["pred_score", "gt_label"], prefix="image_"),         # Image AUPR: 异常筛选纯度
            F1Max(fields=["pred_score", "gt_label"], prefix="image_"),        # Image F1-max: 实际NG/OK判断能力
            PBn(fpr=0.05, fields=["pred_score", "gt_label"]),                # Recall@FPR=5%: 工业漏检控制能力
        ],
    )

    # ---- 构建模型 ----
    print("=" * 80)
    print("构建 INP-Former 模型")
    print("=" * 80)
    model = build_model(args, evaluator)

    # ---- 日志 ----
    run_name = args.run_name
    if run_name is None:
        run_name = f"INP-Former_{args.encoder_name}_{args.dataset}_{args.category}"

    logger = None
    if WANDB_AVAILABLE:
        logger = WandbLogger(
            project=args.project_name,
            name=run_name,
            config=vars(args),
        )

    # ---- 回调 ----
    callbacks = [
        TimerCallback(),
        EarlyStopping(
            monitor="train_loss_epoch",
            patience=args.early_stop_patience,
            mode="min",
            min_delta=0.001,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ---- 训练引擎 ----
    trainer_kwargs = {
        "default_root_dir": output_dir,
        "logger": logger,
        "callbacks": callbacks,
    }
    if args.epochs is not None:
        trainer_kwargs["max_epochs"] = args.epochs
    else:
        trainer_kwargs["max_steps"] = args.max_steps

    engine = Engine(**trainer_kwargs)

    # ---- 训练 ----
    print("=" * 80)
    print("开始训练")
    print("=" * 80)
    engine.fit(model=model, datamodule=datamodule)

    # ---- 测试 ----
    print("=" * 80)
    print("开始测试评估")
    print("=" * 80)

    if args.dataset in ("aebad_s", "aebad_v"):
        # AeBAD 多 domain-shift 测试
        if args.dataset == "aebad_s":
            good_test_dir = Path(args.root) / "test" / "good"
            domain_shifts = sorted(
                d.name for d in good_test_dir.iterdir()
                if d.is_dir()
            ) if good_test_dir.is_dir() else ["same", "view"]
        else:
            test_dir = Path(args.root) / "test"
            domain_shifts = sorted(
                d.name for d in test_dir.iterdir()
                if d.is_dir()
            ) if test_dir.is_dir() else ["video1", "video2", "video3"]

        all_shift_metrics: dict[str, list[float]] = {}

        for shift in domain_shifts:
            print(f"\n  >>> domain_shift = {shift}")
            for metric in evaluator.test_metrics:
                metric.reset()
            evaluator._update_count = 0
            builder_kwargs = builder(args)
            builder_kwargs["domain_shift"] = shift
            test_dm = dataset_cls(**builder_kwargs)
            results = engine.test(model=model, datamodule=test_dm)
            shift_metrics = {}
            if results:
                for k, v in results[0].items():
                    if isinstance(v, (int, float)):
                        all_shift_metrics.setdefault(k, []).append(v)
                        shift_metrics[f"{k}_{shift}"] = v
            if WANDB_AVAILABLE and wandb.run is not None and shift_metrics:
                wandb.log(shift_metrics)
            del test_dm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if all_shift_metrics:
            print("\n" + "=" * 80)
            print("AeBAD 多 domain-shift 平均结果")
            print("=" * 80)
            avg_metrics = {}
            for metric_name, values in all_shift_metrics.items():
                avg = sum(values) / len(values)
                avg_metrics[metric_name] = avg
                print(f"  {metric_name}: {avg:.4f}  (shifts: {[f'{v:.4f}' for v in values]})")
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(avg_metrics)
    else:
        engine.test(model=model, datamodule=datamodule)

    # ---- 清理 ----
    for metric in evaluator.test_metrics:
        metric.reset()
    evaluator._update_count = 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 推理速度测量 ----
    print("=" * 80)
    print("开始推理速度测量")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speed_results = measure_inference_speed(
        model=model,
        datamodule=datamodule,
        device=device,
        warmup=args.warmup_iterations,
        iterations=args.measure_iterations,
    )

    # ---- 保存结果 ----
    speed_path = output_dir / "inference_speed.json"
    with open(speed_path, "w") as f:
        json.dump(speed_results, f, indent=2)
    print(f"推理速度结果已保存: {speed_path}")

    if WANDB_AVAILABLE:
        wandb.log({"inference_speed": speed_results})
        wandb.finish()

    print("=" * 80)
    print("训练与评估全部完成")
    print(f"输出目录: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
