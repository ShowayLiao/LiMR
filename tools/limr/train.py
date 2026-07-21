#!/usr/bin/env python3
"""LiMR 训练脚本：支持多种数据集的训练与评估。"""

import argparse
import json
import random
import time
import gc
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
    RealIAD, VAD, Visa,
)
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUPR, AUROC, F1Max, PBn, Evaluator
from anomalib.models import LiMR

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
    "realiad":     (RealIAD,    _build_realiad),
    "kolektor":    (Kolektor,   _build_kolektor),
    "folder":      (Folder,     _build_folder),
    "aebad_s":     (AeBAD_S,    _build_aebad_s),
    "aebad_v":     (AeBAD_V,    _build_aebad_v),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"LiMR 训练脚本。支持数据集: {', '.join(DATASET_REGISTRY)}",
    )

    # 数据集选择
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=list(DATASET_REGISTRY),
                        help="数据集名称")
    parser.add_argument("--root", type=str, default="./datasets/MVTec",
                        help="数据集根目录")
    parser.add_argument("--category", type=str, default="bottle",
                        help="数据集子类别")
    parser.add_argument("--image-size", type=int, default=256,
                        help="输入图像尺寸")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)

    # RealIAD 专用参数
    parser.add_argument("--realiad-resolution", type=str, default="256",
                        help="RealIAD 图像分辨率")
    parser.add_argument("--realiad-json", type=str, default=None,
                        help="RealIAD JSON 配置文件路径")
    parser.add_argument("--realiad-test-split-mode", type=str, default="from_dir",
                        choices=["none", "from_dir", "synthetic"],
                        help="RealIAD 测试集构建模式（默认 from_dir，从训练集采样正常样本混入测试集）")

    # Folder 专用参数
    parser.add_argument("--folder-normal-dir", type=str, default="normal",
                        help="Folder 数据集正常图像子目录")
    parser.add_argument("--folder-normal-test-dir", type=str, default=None,
                        help="Folder 数据集正常测试图像子目录")
    parser.add_argument("--folder-abnormal-dir", type=str, default="abnormal",
                        help="Folder 数据集异常图像子目录")
    parser.add_argument("--folder-mask-dir", type=str, default=None,
                        help="Folder 数据集掩码子目录")

    # AeBAD 专用参数
    parser.add_argument("--aebad-s-domain-shift", type=str, default="same",
                        choices=["same", "view"],
                        help="AeBAD_S 测试 domain shift")
    parser.add_argument("--aebad-v-domain-shift", type=str, default="video1",
                        choices=["video1", "video2", "video3"],
                        help="AeBAD_V 测试 domain shift")

    # LiMR 模型参数
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--test-mask-ratio", type=float, default=0.0)
    parser.add_argument("--fpn-output-dim", type=int, nargs="+", default=None)
    parser.add_argument("--block-dropout", type=float, default=0.1,
                        help="MobileViTBlockv2 dropout rate")
    parser.add_argument("--block-ffn-dropout", type=float, default=0.0,
                        help="FFN dropout rate in LiMViT blocks (match original paper: 0.0)")
    parser.add_argument("--block-attn-dropout", type=float, default=0.0,
                        help="Attention dropout rate in LiMViT blocks (match original paper: 0.0)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--frozen-stages", type=int, default=3,
                        help="Number of encoder stages to freeze (1=stem, 2=stem+stage0, 3=stem+stage0+stage1)")
    parser.add_argument("--seed", type=int, default=54,
                        help="Random seed for reproducibility")

    # 输出
    parser.add_argument("--output-dir", type=str, default="./output_limr")
    parser.add_argument("--project-name", type=str, default="LiMR")

    # Benchmark
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=100)

    return parser.parse_args()


def measure_inference_speed(model: LiMR, datamodule,
                            device: str = "cuda",
                            warmup: int = 10,
                            iterations: int = 100) -> dict:
    """测量模型推理速度。"""
    print("\n" + "=" * 80)
    print("推理速度测量")
    print("=" * 80)

    datamodule.setup("test")

    model = model.to(device)
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    speed_num_workers = min(datamodule.num_workers, 4)

    # ------------------------------------------------------------------
    # 预热阶段：使用独立 dataloader，完成后立即释放
    # ------------------------------------------------------------------
    print(f"预热 ({warmup} iter)...")
    warmup_loader = datamodule.test_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(warmup_loader):
            if i >= warmup:
                break
            images = batch["image"].to(device)
            _ = model(images)
            del images
    del warmup_loader
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 测量阶段：使用全新独立 dataloader，降低 num_workers 减少内存压力
    # ------------------------------------------------------------------
    _original_num_workers = datamodule.num_workers
    datamodule.num_workers = speed_num_workers
    measure_loader = datamodule.test_dataloader()
    datamodule.num_workers = _original_num_workers

    print(f"测量 ({iterations} iter, num_workers={speed_num_workers})...")
    total_time_e2e = 0.0
    total_time_pure = 0.0
    total_images = 0

    with torch.no_grad():
        for i, batch in enumerate(measure_loader):
            if i >= iterations:
                break

            batch_size = batch["image"].shape[0]
            total_images += batch_size

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

            total_time_e2e += iter_e2e
            total_time_pure += iter_pure

            del images
            if i % 10 == 9:
                print(f"  iter {i+1}/{iterations}: "
                      f"e2e={iter_e2e*1000:.2f}ms "
                      f"pure={iter_pure*1000:.2f}ms "
                      f"batch={batch_size}")

    del measure_loader
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    avg_e2e_per_img = total_time_e2e / total_images
    avg_pure_per_img = total_time_pure / total_images

    print(f"\n【总体延迟】{avg_e2e_per_img*1000:.2f} ms/img  {total_images/total_time_e2e:.2f} FPS")
    print(f"【纯推理】  {avg_pure_per_img*1000:.2f} ms/img  {total_images/total_time_pure:.2f} FPS")

    return {
        "device": device,
        "total_images": total_images,
        "end_to_end": {"avg_ms_per_image": round(avg_e2e_per_img * 1000, 2), "fps": round(total_images / total_time_e2e, 2)},
        "pure_inference": {"avg_ms_per_image": round(avg_pure_per_img * 1000, 2), "fps": round(total_images / total_time_pure, 2)},
    }


def main():
    args = parse_args()

    seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 根据 --dataset 选择数据集类并构建参数
    dataset_cls, builder = DATASET_REGISTRY[args.dataset]
    datamodule = dataset_cls(**builder(args))

    print("=" * 80)
    print(f"LiMR 训练 - 数据集: {args.dataset}")
    print("=" * 80)
    for key, val in sorted(vars(args).items()):
        print(f"  {key}: {val}")
    print()

    # 评估指标
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

    # 模型
    model = LiMR(
        backbone=args.backbone,
        alpha=args.alpha,
        mask_ratio=args.mask_ratio,
        test_mask_ratio=args.test_mask_ratio,
        fpn_output_dim=tuple(args.fpn_output_dim) if args.fpn_output_dim else None,
        block_dropout=args.block_dropout,
        block_ffn_dropout=args.block_ffn_dropout,
        block_attn_dropout=args.block_attn_dropout,
        frozen_stages=args.frozen_stages,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        evaluator=evaluator,
    )

    # ------------------------------------------------------------------
    # 算力测量（FLOPs / 参数量）
    # ------------------------------------------------------------------
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        dummy = torch.randn(1, 3, 224, 224)
        flops = FlopCountAnalysis(model.model, dummy)
        params = parameter_count(model.model)[""]
        flops_g = flops.total() / 1e9
        params_m = params / 1e6
        print("=" * 80)
        print(f"模型算力: {flops_g:.2f} GFLOPs  |  参数量: {params_m:.2f}M")
        print("=" * 80)
    except ImportError:
        flops_g, params_m = None, None
        print("fvcore 未安装，跳过 FLOPs 计算。安装: pip install fvcore")
    except Exception as e:
        flops_g, params_m = None, None
        print(f"FLOPs 计算失败: {e}")

    # 日志
    logger = None
    if WANDB_AVAILABLE:
        logger = WandbLogger(
            project=args.project_name,
            name=f"LiMR_{args.backbone}_{args.dataset}_{args.category}",
            config=vars(args),
        )

    # 回调
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

    engine = Engine(
        default_root_dir=output_dir,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
    )

    print("=" * 80)
    print("训练")
    print("=" * 80)
    engine.fit(model=model, datamodule=datamodule)

    print("=" * 80)
    print("测试")
    print("=" * 80)

    if args.dataset in ("aebad_s", "aebad_v"):
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

        # collect per-shift metrics for averaging
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
            # collect metrics from results (1st dataloader)
            shift_metrics = {}
            if results:
                for k, v in results[0].items():
                    if isinstance(v, (int, float)):
                        all_shift_metrics.setdefault(k, []).append(v)
                        shift_metrics[f"{k}_{shift}"] = v
            # 将每个 shift 的结果分别推送到 WandB
            if WANDB_AVAILABLE and wandb.run is not None and shift_metrics:
                wandb.log(shift_metrics)
            del test_dm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # print average across all domain shifts
        if all_shift_metrics:
            print("\n" + "=" * 80)
            print("AeBAD 多 domain-shift 平均结果")
            print("=" * 80)
            avg_metrics = {}
            for metric_name, values in all_shift_metrics.items():
                avg = sum(values) / len(values)
                avg_metrics[metric_name] = avg
                print(f"  {metric_name}: {avg:.4f}  (shifts: {[f'{v:.4f}' for v in values]})")
            # 将平均值写入 WandB（覆盖被 Lightning callback 写的最后一个 shift 值）
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(avg_metrics)
    else:
        engine.test(model=model, datamodule=datamodule)

    for metric in evaluator.test_metrics:
        metric.reset()
    evaluator._update_count = 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 80)
    print("推理速度")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speed = measure_inference_speed(
        model=model,
        datamodule=datamodule,
        device=device,
        warmup=args.warmup_iterations,
        iterations=args.measure_iterations,
    )

    with open(output_dir / "inference_speed.json", "w") as f:
        json.dump(speed, f, indent=2)

    if WANDB_AVAILABLE:
        wandb_log = {
            "speed_total_images": speed["total_images"],
            "speed_e2e_ms": speed["end_to_end"]["avg_ms_per_image"],
            "speed_e2e_fps": speed["end_to_end"]["fps"],
            "speed_pure_ms": speed["pure_inference"]["avg_ms_per_image"],
            "speed_pure_fps": speed["pure_inference"]["fps"],
        }
        if flops_g is not None:
            wandb_log["flops_g"] = flops_g
            wandb_log["params_m"] = params_m
        wandb.log(wandb_log)
        wandb.finish()

    # ------------------------------------------------------------------
    # 导出 ONNX
    # ------------------------------------------------------------------
    print("=" * 80)
    print("导出 ONNX")
    print("=" * 80)
    try:
        # 放入与 ckpt 同级的版本目录（to_onnx 会自动追加 weights/onnx/）
        best_ckpt = Path(engine.trainer.checkpoint_callback.best_model_path)
        ckpt_version_dir = best_ckpt.parent.parent.parent  # lightning/weights/ -> v34/
        onnx_path = model.to_onnx(ckpt_version_dir, input_size=(224, 224), opset_version=18)
        print(f"ONNX 已导出至: {onnx_path}")
    except Exception as e:
        print(f"ONNX 导出失败: {e}")

    print("=" * 80)
    print("完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
