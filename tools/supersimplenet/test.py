#!/usr/bin/env python3
"""SuperSimpleNet 测试脚本：使用训练好的 SuperSimpleNet 模型进行数据集评估。

支持的数据集（通过注册表自动扩展）：
  - mvtec, mvtecad2, mvtec_loco, btech, bmad, mpdd, vad, visa, kolektor, folder
  - realiad, aebad_s, aebad_v

用法示例:
  # MVTec-AD 单类别测试
  python tools/supersimplenet/test.py --dataset mvtec --root ./datasets/MVTec --category bottle \
      --checkpoint ./output_supersimplenet/SuperSimpleNet/MVTecAD/bottle/v1/weights/lightning/model.ckpt

  # VisA 测试
  python tools/supersimplenet/test.py --dataset visa --root ./datasets/VisA --category capsules \
      --checkpoint ./output_supersimplenet/SuperSimpleNet/Visa/capsules/v1/weights/lightning/model.ckpt
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from lightning.pytorch import seed_everything

from anomalib.data import (
    AeBAD_S, AeBAD_V,
    BMAD, BTech, Folder, Kolektor, MPDD, MVTecAD, MVTecAD2, MVTecLOCO,
    RealIAD, VAD, Visa,
)
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUPR, AUROC, F1Max, PBn, Evaluator
from anomalib.models import Supersimplenet


# ---------------------------------------------------------------------------
# 数据集注册表
# ---------------------------------------------------------------------------
def _build_standard_dataset(args):
    return {
        "root": args.root,
        "category": args.category,
        "image_size": (args.image_size, args.image_size),
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
        description=f"SuperSimpleNet 测试脚本。支持数据集: {', '.join(DATASET_REGISTRY)}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 数据集 ----
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=list(DATASET_REGISTRY))
    parser.add_argument("--root", type=str, default="./datasets/MVTec")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)

    # ---- RealIAD ----
    parser.add_argument("--realiad-resolution", type=str, default="256")
    parser.add_argument("--realiad-json", type=str, default=None)
    parser.add_argument("--realiad-test-split-mode", type=str, default="from_dir",
                        choices=["none", "from_dir", "synthetic"])

    # ---- Folder ----
    parser.add_argument("--folder-normal-dir", type=str, default="normal")
    parser.add_argument("--folder-normal-test-dir", type=str, default=None)
    parser.add_argument("--folder-abnormal-dir", type=str, default="abnormal")
    parser.add_argument("--folder-mask-dir", type=str, default=None)

    # ---- AeBAD ----
    parser.add_argument("--aebad-s-domain-shift", type=str, default="same",
                        choices=["same", "view"])
    parser.add_argument("--aebad-v-domain-shift", type=str, default="video1",
                        choices=["video1", "video2", "video3"])

    # ---- 模型 ----
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="anomalib Lightning checkpoint 路径 (.ckpt)")
    parser.add_argument("--perlin-threshold", type=float, default=0.2)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2.tv_in1k")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--adapt-cls-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # ---- 输出 ----
    parser.add_argument("--output-dir", type=str, default="./output_supersimplenet_test")

    # ---- Benchmark ----
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=100)

    return parser.parse_args()


def measure_inference_speed(model, datamodule, device="cuda", warmup=10, iterations=100) -> dict:
    print("\n" + "=" * 80)
    print("推理速度测量")
    print("=" * 80)

    datamodule.setup("test")

    model = model.to(device)
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    speed_num_workers = min(datamodule.num_workers, 4)

    # ---- 预热 ----
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

    # ---- 测量 ----
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
                print(f"  iter {i+1}/{iterations}: e2e={iter_e2e*1000:.2f}ms pure={iter_pure*1000:.2f}ms batch={batch_size}")

    del measure_loader
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    avg_e2e = total_time_e2e / total_images * 1000
    avg_pure = total_time_pure / total_images * 1000
    print(f"\n【总体延迟】{avg_e2e:.2f} ms/img  {total_images/total_time_e2e:.2f} FPS")
    print(f"【纯推理】  {avg_pure:.2f} ms/img  {total_images/total_time_pure:.2f} FPS")

    return {
        "device": device,
        "total_images": total_images,
        "end_to_end": {"avg_ms_per_image": round(avg_e2e, 2), "fps": round(total_images / total_time_e2e, 2)},
        "pure_inference": {"avg_ms_per_image": round(avg_pure, 2), "fps": round(total_images / total_time_pure, 2)},
    }


def main():
    args = parse_args()

    if not args.checkpoint:
        raise ValueError("必须指定 --checkpoint")

    seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls, builder = DATASET_REGISTRY[args.dataset]
    datamodule = dataset_cls(**builder(args))

    print("=" * 80)
    print(f"SuperSimpleNet 测试 - 数据集: {args.dataset}")
    print("=" * 80)
    for key, val in sorted(vars(args).items()):
        print(f"  {key}: {val}")
    print()

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

    model = Supersimplenet(
        perlin_threshold=args.perlin_threshold,
        backbone=args.backbone,
        layers=list(args.layers),
        adapt_cls_features=args.adapt_cls_features,
        evaluator=evaluator,
    )

    engine = Engine()

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
        for shift in domain_shifts:
            print(f"\n  >>> domain_shift = {shift}")
            for metric in evaluator.test_metrics:
                metric.reset()
            evaluator._update_count = 0
            builder_kwargs = builder(args)
            builder_kwargs["domain_shift"] = shift
            test_dm = dataset_cls(**builder_kwargs)
            engine.test(model=model, datamodule=test_dm, ckpt_path=args.checkpoint)
            del test_dm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        engine.test(model=model, datamodule=datamodule, ckpt_path=args.checkpoint)

    for metric in evaluator.test_metrics:
        metric.reset()
    evaluator._update_count = 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speed = measure_inference_speed(
        model=model,
        datamodule=datamodule,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup=args.warmup_iterations,
        iterations=args.measure_iterations,
    )

    with open(output_dir / "inference_speed.json", "w") as f:
        json.dump(speed, f, indent=2)

    print(f"结果已保存: {output_dir}")


if __name__ == "__main__":
    main()
