#!/usr/bin/env python3


# modified from https://github.com/zhangzilongc/MMR
# copyright (c) 2023 Z.Zhang et al.
# copyright (c) 2025 S.Liao et al.
# Licensed under the Apache License, Version 2.0 (the "LICENSE-APACHE-2.0-ZHANG2023");

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
# 使用CfgNode类来定义配置文件, 用于存储配置信息
# CfgNode类是一个类似于字典的类，可以通过点号访问键值对
# _C = CfgNode()创建一个CfgNode对象
_C = CfgNode()

_C.NUM_GPUS = 1
_C.RNG_SEED = 54

_C.OUTPUT_ROOT_DIR = './log_low_level_AD'
_C.OUTPUT_DIR = './logs_and_models'
_C.MODE = CfgNode()
_C.MODE = 'auto'

# -----------------------------------------------------------------------------
# Dataset是一个CfgNode对象，用于存储数据集的配置信息
_C.DATASET = CfgNode()
_C.DATASET.name = 'mvtec'
_C.DATASET.subdatasets = ["bottle",
                          "cable",
                          "capsule",
                          "carpet",
                          "grid",
                          "hazelnut",
                          "leather",
                          "metal_nut",
                          "pill",
                          "screw",
                          "tile",
                          "toothbrush",
                          "transistor",
                          "wood",
                          "zipper",
                          ]
_C.DATASET.resize = 256
# final image shape
_C.DATASET.imagesize = 224
_C.DATASET.domain_shift_category = "same"

# -----------------------------------------------------------------------------
# Model是一个CfgNode对象，用于存储模型的配置信息
_C.TRAIN = CfgNode()
_C.TRAIN.enable = True
_C.TRAIN.save_model = False
_C.TRAIN.method = 'PatchCore'
_C.TRAIN.change = 'default'
_C.TRAIN.backbone = 'resnet50'
_C.TRAIN.dataset_path = '/usr/sdd/zzl_data/MV_Tec'
_C.TRAIN.save_model_path = 'E:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\MMR\save_model'
_C.TRAIN.resume = False
_C.TRAIN.resume_model_path = 'E:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\MMR\save_model'
# -----------------------------------------------------------------------------
# MMR是一个CfgNode对象，用于存储MMR的配置信息
# for LiMR
_C.TRAIN.LiMR = CfgNode()
_C.TRAIN.LiMR.DA_low_limit = 0.2
_C.TRAIN.LiMR.DA_up_limit = 1.
_C.TRAIN.LiMR.layers_to_extract_from = ["layer1", "layer2", "layer3"]
# _C.TRAIN.LiMR.freeze_encoder = False
_C.TRAIN.LiMR.feature_compression = False
_C.TRAIN.LiMR.scale_factors = (4.0, 2.0, 1.0)
_C.TRAIN.LiMR.FPN_output_dim = (256, 512, 1024)
_C.TRAIN.LiMR.load_pretrain_model = True
_C.TRAIN.LiMR.model_chkpt = "./mae_visualize_vit_base.pth"
_C.TRAIN.LiMR.finetune_mask_ratio = 0.75
_C.TRAIN.LiMR.test_mask_ratio = 0.75
_C.TRAIN.LiMR.embed_dims = 768
_C.TRAIN.LiMR.num_heads = 12
_C.TRAIN.LiMR.alpha = 1.0
_C.TRAIN.LiMR.decoder = 'fpn'
_C.TRAIN.LiMR.block_dropout = 0.4
_C.TRAIN.LiMR.block_attn_dropout = 0.0
_C.TRAIN.LiMR.block_ffn_dropout = 0.1


# -----------------------------------------------------------------------------
# TRAIN_SETUPS是一个CfgNode对象，用于存储开始训练设置的配置信息
_C.TRAIN_SETUPS = CfgNode()
_C.TRAIN_SETUPS.batch_size = 64
_C.TRAIN_SETUPS.num_workers = 8
_C.TRAIN_SETUPS.learning_rate = 0.005
_C.TRAIN_SETUPS.epochs = 200
_C.TRAIN_SETUPS.weight_decay = 0.05
_C.TRAIN_SETUPS.warmup_epochs = 40
_C.TRAIN_SETUPS.save_interval= 10
_C.TRAIN_SETUPS.tolerance = 0.01
_C.TRAIN_SETUPS.patience = 10

# -----------------------------------------------------------------------------
# TEST是一个CfgNode对象，用于存储测试的配置信息
_C.TEST = CfgNode()
_C.TEST.enable = False
_C.TEST.method = 'PatchCore'
_C.TEST.save_segmentation_images = False
_C.TEST.save_video_segmentation_images = False
_C.TEST.dataset_path = '/usr/sdd/zzl_data/MV_Tec'
_C.TEST.model_path = 'E:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\MMR\save_model'
# -----------------------------------------------------------------------------
_C.TEST.TensorRT = CfgNode()
_C.TEST.TensorRT.enable = False
_C.TEST.TensorRT.stu_path = './LiMR_student.engine'
_C.TEST.TensorRT.tea_path = './LiMR_teacher.engine'
# -----------------------------------------------------------------------------
# TEST_SETUPS是一个CfgNode对象，用于存储测试设置的配置信息
_C.TEST.VISUALIZE = CfgNode()
_C.TEST.VISUALIZE.Random_sample = True
_C.TEST.VISUALIZE.Sample_num = 40
# -----------------------------------------------------------------------------
_C.TEST.DEMO = CfgNode()
_C.TEST.DEMO.enable = False
_C.TEST.DEMO.video_path = 'E:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\MMR\datasets\abnormal_fast_fake_x264.mp4'

# pixel auroc, aupro
_C.TEST.pixel_mode_verify = True

# -----------------------------------------------------------------------------
# TEST_SETUPS是一个CfgNode对象，用于存储测试设置的配置信息
_C.TEST_SETUPS = CfgNode()
_C.TEST_SETUPS.batch_size = 64


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
