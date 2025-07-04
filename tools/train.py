# modified from https://github.com/zhangzilongc/MMR
# copyright (c) 2023 Z.Zhang et al.
# Licensed under the Apache License, Version 2.0 (the "LICENSE-APACHE-2.0-ZHANG2023");

# copyright (c) 2025 S.Liao et al.

import random

import numpy as np
import torch
import logging

from utils import get_dataloaders, load_backbones
from utils.common import freeze_paras, scratch_MAE_decoder
from .load_method import LiMR


from models.LiMR import LiMR_base, LiMR_pipeline

import timm.optim.optim_factory as optim_factory
import time

LOGGER = logging.getLogger(__name__)


def train(cfg=None):
    """
    include data loader load, model load, optimizer, training and test.
    """
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)

    LOGGER.info("load dataset!")
    # --------------get train dataloader (include each category)-----------------
    train_dataloaders = get_dataloaders(cfg=cfg, mode='train')



    # --------------get test dataloader (include each category)------------------
    if cfg.DATASET.name in ["aebad_S", "aebad_V", "mvtec"]:
        if cfg.DATASET.name == "aebad_S":
            measured_list = ["same", "background", "illumination", "view"]
        elif cfg.DATASET.name == "aebad_V":
            measured_list = ["video1", "video2", "video3"]
        else:
            measured_list = ["same"]

        test_dataloader_dict = {}
        for each_class in measured_list:
            cfg.DATASET.domain_shift_category = each_class
            test_dataloaders_ = get_dataloaders(cfg=cfg, mode='test')
            test_dataloader_dict[each_class] = test_dataloaders_
    else:
        raise NotImplementedError("DATASET {} does not include in target datasets".format(cfg.DATASET.name))

    # --------------initialize metric------------------
    result_collect = {"AUROC": [],
                      "Pixel-AUROC": [],
                      "per-region-overlap (PRO)": [],
                      "time": []}

    # -------------training phase start---------------
    for idx, individual_dataloader in enumerate(train_dataloaders):
        LOGGER.info("current individual_dataloader is {}.".format(individual_dataloader.name))
        LOGGER.info("the data in current individual_dataloader {} are {}.".format(individual_dataloader.name,
                                                                                  len(individual_dataloader.dataset)))

        if cfg.TRAIN.method == 'LiMR':
            pipeline,start_epoch = LiMR(cfg=cfg)
            LOGGER.info("use LiMR base model to train!")
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        begin_time = time.time()
        pipeline.fit(individual_dataloader)
        LOGGER.info("Training complete,using time:{}h {}m {:.2f}s".format((time.time() - begin_time) // 3600,
                                                                          (time.time() - begin_time) % 3600 // 60,
                                                                          (time.time() - begin_time) % 60))
        LOGGER.info("finish training!")


        # -------------test phase start-------------------
        LOGGER.info("start testing!")
        for each_class in measured_list:
            LOGGER.info(f"current domain shift mode is {each_class}!")
            test_dataloaders = test_dataloader_dict[each_class]

            torch.cuda.empty_cache()
            measured_test_dataloaders = test_dataloaders[idx]
            LOGGER.info("current test individual_dataloader is {}.".format(measured_test_dataloaders.name))
            LOGGER.info("the test data in current individual_dataloader {} are {}.".format(measured_test_dataloaders.name,
                                                                                           len(measured_test_dataloaders.dataset)))
            LOGGER.info("Computing evaluation metrics.")
            """
                                prediction
                            ______1________0____
                          1 |    TP   |   FN   |
            ground truth  0 |    FP   |   TN   |
    
            ACC = (TP + TN) / (TP + FP + FN + TN)
    
            precision = TP / (TP + FP)
    
            recall (TPR) = TP / (TP + FN)
    
            FPR（False Positive Rate）= FP / (FP + TN)
            """
            if cfg.TRAIN.method == 'LiMR':
                auc_sample, auroc_pixel, pro_auc, time_use = pipeline.evaluation(
                    test_dataloader=measured_test_dataloaders)
            else:
                raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

            result_collect["AUROC"].append(auc_sample)
            LOGGER.info("{}'s Image_Level AUROC is {:2f}.%".format(individual_dataloader.name, auc_sample * 100))

            result_collect["Pixel-AUROC"].append(auroc_pixel)
            LOGGER.info(
                "{}'s Full_Pixel_Level AUROC is {:2f}.%".format(individual_dataloader.name, auroc_pixel * 100))

            result_collect["per-region-overlap (PRO)"].append(pro_auc)
            LOGGER.info(
                "{}'s per-region-overlap (PRO) AUROC is {:2f}.%".format(individual_dataloader.name, pro_auc * 100))
            result_collect["time"].append(time_use / cfg.TEST_SETUPS.batch_size * 1000)
            LOGGER.info(
                "{}'s detect time is {:2f}ms".format(measured_test_dataloaders.name,
                                                     time_use / cfg.TEST_SETUPS.batch_size * 1000))

    for key, values in result_collect.items():
        LOGGER.info(
            "Mean {} is {:2f}.%".format(key, np.mean(np.array(values)) * 100)
            if key != "time" else "Mean {} is {:2f}ms".format(key, np.mean(np.array(values))))
    LOGGER.info("Method testing phase complete!")





def train_model(cfg=None):
    """
        include data loader load, model load, optimizer, training
    """
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)

    LOGGER.info("load train dataset!")
    # 加载数据集
    train_dataloaders = get_dataloaders(cfg=cfg, mode='train')

    # 开始逐个类别训练
    for idx, individual_dataloader in enumerate(train_dataloaders):
        LOGGER.info("current individual_dataloader is {}.".format(individual_dataloader.name))
        LOGGER.info("the data in current individual_dataloader {} are {}.".format(individual_dataloader.name,
                                                                                  len(individual_dataloader.dataset)))

        # 加载模型
        if cfg.TRAIN.method in ['LiMR']:
            pipeline,start_epoch = LiMR(cfg)
            LOGGER.info("use LiMR base model to train!")
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))


        # 开始训练
        begin_time = time.time()
        pipeline.fit(individual_dataloader)
        LOGGER.info("Training complete,using time:{}h {}m {:.2f}s".format((time.time() - begin_time) // 3600,
                                                                          (time.time() - begin_time) % 3600 // 60,
                                                                          (time.time() - begin_time) % 60))
        LOGGER.info("finish training!")



