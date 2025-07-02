# modified from https://github.com/zhangzilongc/MMR
# copyright (c) 2023 Z.Zhang et al.
# Licensed under the Apache License, Version 2.0 (the "LICENSE-APACHE-2.0-ZHANG2023");

# copyright (c) 2025 S.Liao et al.


import random

import numpy as np
import torch
import logging

from utils import get_dataloaders
from .load_method import LiMR

LOGGER = logging.getLogger(__name__)

def test_model(cfg=None):
    """
    include data loader load, model load, optimizer and test.
    """
    # set seed
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)


    # 确定需要测量的数据类别（比如mydata中逐个类别测量R1-avg,R2-avg,L1-avg,L2-avg,R1-more,R2-more,L1-more,L2-more,R1-less,R2-less,L1-less,L2-less）
    LOGGER.info("load test dataset!")
    if cfg.DATASET.name in ["aebad_S", "aebad_V", "mvtec","mydata"]:
        if cfg.DATASET.name == "aebad_S":
            measured_list = ["same", "background", "illumination", "view"]
        elif cfg.DATASET.name == "aebad_V":
            measured_list = ["video1", "video2", "video3"]
            # measured_list = ["big", "small", "mid"]
        else:
            measured_list = ["same"]

        test_dataloader_dict = {}

        for each_class in measured_list:
            cfg.DATASET.domain_shift_category = each_class
            test_dataloaders_ = get_dataloaders(cfg=cfg, mode='test')# 这里是一个列表，列表中元素对应着mydata下的各个类别（这里是mix一个类别）
            test_dataloader_dict[each_class] = test_dataloaders_
    else:
        raise NotImplementedError("DATASET {} does not include in target datasets".format(cfg.DATASET.name))


    # initialize result collect
    result_collect = {"AUROC": [],
                      "Pixel-AUROC": [],
                      "per-region-overlap (PRO)": [],
                      "time": []}

    # load LiMR base model
    if cfg.TRAIN.method in ['LiMR']:
        pipeline,_ = LiMR(cfg)
    else:
        raise NotImplementedError("method {} does not include in target methods".format(cfg.TRAIN.method))

    # test each subdataset
    for idx,subdataset in enumerate(cfg.DATASET.subdatasets):

        for each_class in measured_list:

            # load corresponding test dataloader
            LOGGER.info(f"current domain shift mode is {each_class}!")
            test_dataloaders = test_dataloader_dict[each_class]
            measured_test_dataloaders = test_dataloaders[idx]

            LOGGER.info("current test individual_dataloader is {}.".format(measured_test_dataloaders.name))
            LOGGER.info("the test data in current individual_dataloader {} are {}.".format(measured_test_dataloaders.name,
                                                                                   len(measured_test_dataloaders.dataset)))

            # 测量数据说明
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

            # test
            auc_sample, auroc_pixel, pro_auc,time_use = pipeline.evaluation(
                test_dataloader=measured_test_dataloaders)

            # measure result collect
            result_collect["AUROC"].append(auc_sample)
            LOGGER.info("{}'s Image_Level AUROC is {:2f}.%".format(measured_test_dataloaders.name, auc_sample * 100))

            result_collect["Pixel-AUROC"].append(auroc_pixel)
            LOGGER.info(
                "{}'s Full_Pixel_Level AUROC is {:2f}.%".format(measured_test_dataloaders.name, auroc_pixel * 100))

            result_collect["per-region-overlap (PRO)"].append(pro_auc)
            LOGGER.info(
                "{}'s per-region-overlap (PRO) AUROC is {:2f}.%".format(measured_test_dataloaders.name, pro_auc * 100))

            result_collect["time"].append(time_use/cfg.TEST_SETUPS.batch_size*1000)
            LOGGER.info(
                "{}'s detect time is {:2f}ms".format(measured_test_dataloaders.name, time_use/cfg.TEST_SETUPS.batch_size*1000))


    for key, values in result_collect.items():
        LOGGER.info(
            "Mean {} is {:2f}.%".format(key, np.mean(np.array(values)) * 100)
                                        if key != "time" else "Mean {} is {:2f}ms".format(key, np.mean(np.array(values))))