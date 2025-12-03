import random
import numpy as np
import torch
import logging

# 假设你把上一条回复的类保存为了 realtime_detect.py
from models.LiMR.realtime_detect import RealTimeAnomalyDetector
from .load_method import LiMR

LOGGER = logging.getLogger(__name__)


def demo_realtime(cfg=None):
    """
    Load model and start real-time anomaly detection using webcam.
    """
    # ---------------- 1. Set Seed (保持与原代码一致) ----------------
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)

    LOGGER.info("Initializing Real-time Demo...")

    # ---------------- 2. Check Configuration ----------------
    # 因为目前的 TRT Pipeline 还没有实现单帧推理接口，这里强制检查
    if cfg.TEST.TensorRT.enable:
        LOGGER.warning("Real-time demo currently assumes PyTorch inference. "
                       "Please ensure 'infer_single_image' is implemented if using TensorRT. "
                       "Falling back/Proceeding with current config...")
        # 如果你没有修改 TRT 代码，建议这里暂时强制关闭 TRT，或者确保 cfg 传入时就是 False
        # cfg.TEST.TensorRT.enable = False

    # ---------------- 3. Load LiMR Pipeline ----------------
    LOGGER.info("Loading LiMR model...")

    if cfg.TRAIN.method in ['LiMR']:
        # LiMR(cfg) 返回的是 (pipeline, model)，我们只需要 pipeline
        # 注意：这里会加载权重文件，请确保 cfg.TRAIN.resume_model_path 配置正确
        try:
            pipeline, _ = LiMR(cfg)
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            raise e
    else:
        raise NotImplementedError("Method {} does not include in target methods".format(cfg.TRAIN.method))

    LOGGER.info("Model loaded successfully.")

    # ---------------- 4. Start Real-time Detector ----------------
    # 可以在 cfg 中添加新的字段来控制摄像头ID或初始阈值，这里暂时硬编码或给默认值
    camera_id = 0
    initial_threshold = 128

    LOGGER.info(f"Starting camera feed (ID: {camera_id})...")
    LOGGER.info("Interactive Controls:")
    LOGGER.info("  [q] - Quit")
    LOGGER.info("  [w] - Increase Threshold")
    LOGGER.info("  [s] - Decrease Threshold")

    try:
        # 初始化实时检测器
        detector = RealTimeAnomalyDetector(
            pipeline=pipeline,
            camera_id=camera_id,
            threshold=initial_threshold
        )

        # 开始运行主循环
        detector.run()

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Error during real-time detection: {e}")
    finally:
        LOGGER.info("Real-time demo finished.")