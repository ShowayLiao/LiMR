#!/usr/bin/env python3
"""INP-Former 测试脚本：加载已训练模型权重进行评估与推理速度测量。

支持从 Lightning Checkpoint (.ckpt) 恢复完整模型状态，并执行测试集评估。

用法示例:
  # MVTec-AD 测试
  python tools/inpformer/test.py --dataset MVTecAD --root ./datasets/MVTec \\
      --category bottle --checkpoint ./output_inpformer/lightning_logs/.../checkpoints/epoch=xxx.ckpt

  # VisA 测试
  python tools/inpformer/test.py --dataset Visa --root ./datasets/VisA \\
      --category candle --checkpoint ./output_inpformer/.../epoch=xxx.ckpt

  # RealIAD 测试
  python tools/inpformer/test.py --dataset RealIAD --root ./datasets/Real-IAD \\
      --category end_cap --resolution 1024 --checkpoint ./output_inpformer/.../epoch=xxx.ckpt
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUPR, AUROC, F1Max, PBn, Evaluator
from anomalib.models import INP_Former


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="INP-Former 模型测试脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # 数据集参数
    # ============================================================
    parser.add_argument("--dataset", type=str, default="MVTecAD",
                        choices=["MVTecAD", "Visa", "RealIAD"],
                        help="数据集名称 (VisA 使用 Visa)")
    parser.add_argument("--root", type=str, default="./datasets/MVTec",
                        help="数据集根目录")
    parser.add_argument("--category", type=str, default="bottle",
                        help="测试类别")
    parser.add_argument("--resolution", type=str, default=None,
                        help="RealIAD 分辨率 (256/512/1024)")
    parser.add_argument("--json-path", type=str, default=None,
                        help="RealIAD JSON 路径")
    parser.add_argument("--image-size", type=int, default=448,
                        help="缩放尺寸")
    parser.add_argument("--crop-size", type=int, default=392,
                        help="中心裁剪尺寸")
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="评估批次大小")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="数据加载线程数")

    # ============================================================
    # 模型参数
    # ============================================================
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重文件路径 (.ckpt)")
    parser.add_argument("--encoder-name", type=str, default="dinov2reg_vit_base_14",
                        help="预训练编码器名称（需与训练时一致）")
    parser.add_argument("--inp-num", type=int, default=6,
                        help="INP 数量（需与训练时一致）")
    parser.add_argument("--decoder-depth", type=int, default=8,
                        help="解码器层数（需与训练时一致）")

    # ============================================================
    # 输出参数
    # ============================================================
    parser.add_argument("--output-dir", type=str, default="./output_inpformer_test",
                        help="结果输出目录")

    # ============================================================
    # Benchmark 参数
    # ============================================================
    parser.add_argument("--warmup-iterations", type=int, default=10,
                        help="推理测速预热迭代次数")
    parser.add_argument("--measure-iterations", type=int, default=100,
                        help="推理测速测量迭代次数")

    return parser.parse_args()


def build_datamodule(args: argparse.Namespace):
    """根据参数构建测试数据集模块。

    Args:
        args: 命令行参数。

    Returns:
        Anomalib 数据模块实例。
    """
    datamodule_kwargs = {
        "root": args.root,
        "category": args.category,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }

    if args.dataset == "MVTecAD":
        datamodule_kwargs["image_size"] = (args.image_size, args.image_size)
        return MVTecAD(**datamodule_kwargs)

    if args.dataset == "Visa":
        datamodule_kwargs["image_size"] = (args.image_size, args.image_size)
        return Visa(**datamodule_kwargs)

    if args.dataset == "RealIAD":
        if args.resolution is None:
            args.resolution = "1024"
        datamodule_kwargs["resolution"] = args.resolution
        if args.json_path is not None:
            datamodule_kwargs["json_path"] = args.json_path
        datamodule_kwargs["test_split_mode"] = "from_dir"
        from anomalib.data import RealIAD
        return RealIAD(**datamodule_kwargs)

    raise ValueError(f"不支持的数据集: {args.dataset}")


def measure_inference_speed(
    model: INP_Former,
    datamodule,
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """测量模型推理速度。

    Args:
        model: 已加载权重的模型。
        datamodule: 数据模块。
        device: 设备类型。
        warmup: 预热迭代次数。
        iterations: 测量迭代次数。

    Returns:
        速度测量结果字典。
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

    avg_e2e_per_img = total_time_e2e / total_images * 1000
    avg_pure_per_img = total_time_pure / total_images * 1000
    fps_e2e = total_images / total_time_e2e
    fps_pure = total_images / total_time_pure

    print(f"\n【总体延迟】{avg_e2e_per_img:.2f} ms/img, {fps_e2e:.2f} FPS")
    print(f"【纯推理】{avg_pure_per_img:.2f} ms/img, {fps_pure:.2f} FPS")

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
    """主流程：加载模型 → 测试评估 → 推理测速。"""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 打印配置 ----
    print("=" * 80)
    print("INP-Former 测试配置")
    print("=" * 80)
    for key, val in vars(args).items():
        print(f"  {key}: {val}")
    print()

    # ---- 数据集 ----
    print("=" * 80)
    print(f"构建测试数据集: {args.dataset}")
    print("=" * 80)
    datamodule = build_datamodule(args)

    # ---- 评估器 ----
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

    # ---- 模型 ----
    print("=" * 80)
    print("构建 INP-Former 模型（加载权重）")
    print("=" * 80)
    model = INP_Former(
        encoder_name=args.encoder_name,
        inp_num=args.inp_num,
        decoder_depth=args.decoder_depth,
        evaluator=evaluator,
    )

    # ---- 测试 ----
    engine = Engine()

    print("=" * 80)
    print("测试评估")
    print("=" * 80)
    _ = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.checkpoint,
    )

    # ---- 推理速度测量 ----
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
    print(f"\n推理速度结果已保存: {speed_path}")
    print(f"测试结果目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
