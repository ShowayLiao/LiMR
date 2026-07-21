#!/usr/bin/env python3
"""LiMR 测试脚本：支持多种数据集的加载与评估。"""

import argparse
import json
import time
import gc
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
from anomalib.models import LiMR


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
        "image_size": (args.image_size, args.image_size),
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
        "image_size": (args.image_size, args.image_size),
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
        description=f"LiMR 测试脚本。支持数据集: {', '.join(DATASET_REGISTRY)}",
    )

    # 数据集
    parser.add_argument("--dataset", type=str, default="mvtec",
                        choices=list(DATASET_REGISTRY))
    parser.add_argument("--root", type=str, default="./datasets/MVTec")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    # RealIAD
    parser.add_argument("--realiad-resolution", type=str, default="256")
    parser.add_argument("--realiad-json", type=str, default=None)
    parser.add_argument("--realiad-test-split-mode", type=str, default="from_dir",
                        choices=["none", "from_dir", "synthetic"])

    # Folder
    parser.add_argument("--folder-normal-dir", type=str, default="normal")
    parser.add_argument("--folder-normal-test-dir", type=str, default=None)
    parser.add_argument("--folder-abnormal-dir", type=str, default="abnormal")
    parser.add_argument("--folder-mask-dir", type=str, default=None)

    # AeBAD
    parser.add_argument("--aebad-s-domain-shift", type=str, default="same",
                        choices=["same", "view"])
    parser.add_argument("--aebad-v-domain-shift", type=str, default="video1",
                        choices=["video1", "video2", "video3"])

    # 模型
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="anomalib Lightning checkpoint 路径 (.ckpt)")
    parser.add_argument("--original-checkpoint", type=str, default=None,
                        help="原始 LiMR 仓库权重文件路径 (.pth)")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--test-mask-ratio", type=float, default=0.0)
    parser.add_argument("--fpn-output-dim", type=int, nargs="+", default=None)
    parser.add_argument("--block-dropout", type=float, default=0.0,
                        help="MobileViTBlockv2 dropout rate")
    parser.add_argument("--block-ffn-dropout", type=float, default=0.0,
                        help="FFN dropout rate in LiMViT blocks")
    parser.add_argument("--block-attn-dropout", type=float, default=0.0,
                        help="Attention dropout rate in LiMViT blocks")
    parser.add_argument("--frozen-stages", type=int, default=3,
                        help="Number of encoder stages to freeze")
    parser.add_argument("--seed", type=int, default=54,
                        help="Random seed for reproducibility")

    # 输出
    parser.add_argument("--output-dir", type=str, default="./output_limr_test")

    # Benchmark
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=100)

    return parser.parse_args()


_ORIGINAL_KEY_MAP = {
    "encoder.conv_1": "model.encoder.stem",
    "encoder.layer_1": "model.encoder.stage0",
    "encoder.layer_2": "model.encoder.stage1",
    "encoder.layer_3": "model.encoder.stage2",
    "encoder.layer_4": "model.encoder.stage3",
    "encoder.layer_5": "model.encoder.stage4",
    "decoder.": "model.decoder.",
    "decoder_FPN_pos_embed": "model.decoder_FPN_pos_embed",
}

_SKIP_KEYS = {"norm.weight", "norm.bias"}


def _load_original_checkpoint(model: LiMR, pth_path: str):
    checkpoint = torch.load(pth_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        src_state = checkpoint["model_state_dict"]
    else:
        src_state = checkpoint

    remapped = {}
    skipped = []
    for k, v in src_state.items():
        if k in _SKIP_KEYS:
            skipped.append(k)
            continue
        new_k = k
        mapped = False
        for old_prefix, new_prefix in _ORIGINAL_KEY_MAP.items():
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, new_prefix, 1)
                mapped = True
                break
        if not mapped:
            skipped.append(k)
            continue
        remapped[new_k] = v

    if skipped:
        print(f"跳过的 key ({len(skipped)}): {skipped}")
    print(f"原始 key 数: {len(src_state)}, 映射成功: {len(remapped)}, 跳过: {len(skipped)}")

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"未加载 (目标缺少): {len(missing)} keys")
        for k in sorted(missing)[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"未匹配 (源多余): {len(unexpected)} keys")

    print(f"权重加载完成: {len(remapped) - len(unexpected)}/{len(remapped)} keys loaded")


def measure_inference_speed(model, datamodule, device="cuda", warmup=10, iterations=100) -> dict:
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

    if not args.checkpoint and not args.original_checkpoint:
        raise ValueError("必须指定 --checkpoint 或 --original-checkpoint 之一")

    seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls, builder = DATASET_REGISTRY[args.dataset]
    datamodule = dataset_cls(**builder(args))

    print("=" * 80)
    print(f"LiMR 测试 - 数据集: {args.dataset}")
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
        evaluator=evaluator,
    )

    if args.original_checkpoint:
        print("=" * 80)
        print(f"从原始权重加载: {args.original_checkpoint}")
        print("=" * 80)
        _load_original_checkpoint(model, args.original_checkpoint)
        ckpt_path = None
    else:
        ckpt_path = args.checkpoint

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
            engine.test(model=model, datamodule=test_dm, ckpt_path=ckpt_path)
            del test_dm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

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
