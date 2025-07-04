# modified from https://github.com/zhangzilongc/MMR
# copyright (c) 2023 Z.Zhang et al.
# Licensed under the Apache License, Version 2.0 (the "LICENSE-APACHE-2.0-ZHANG2023");

# we add function to save each batch results and visualize feature maps yeilded from pretrained model by pca
# copyright (c) 2025 S.Liao et al.


import numpy as np
from numpy import ndarray
from sklearn import metrics
import cv2
import os
import pandas as pd
from skimage import measure
from statistics import mean
from sklearn.metrics import auc
import random
from torchvision.datasets.folder import default_loader
import logging
import math
from pathlib import Path


from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def freeze_paras(backbone):
    for para in backbone.parameters():
        para.requires_grad = False


def freeze_MAE_paras(MAE_model):
    for name, param in MAE_model.named_parameters():
        if "decoder" not in name and name != "mask_token":
            param.requires_grad = False


def scratch_MAE_decoder(checkpoint):
    for key_indv in list(checkpoint["model"].keys()):
        if "decoder" in key_indv or key_indv == "mask_token":
            checkpoint["model"].pop(key_indv)
    return checkpoint


def compute_imagewise_retrieval_metrics(
        anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):# 前者为预测，后者为实际
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # flatten
    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    mean_AP = metrics.average_precision_score(flat_ground_truth_masks.astype(int),
                                              flat_anomaly_segmentations)

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "mean_AP": mean_AP
    }

def compute_pro(anomaly_map: np.ndarray, gt_mask: np.ndarray, label: np.ndarray, num_th: int = 200):
    assert isinstance(anomaly_map, np.ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_mask, np.ndarray), "type(masks) must be ndarray"
    assert anomaly_map.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_mask.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert anomaly_map.shape == gt_mask.shape, "amaps.shape and masks.shape must be same"
    assert set(gt_mask.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    current_amap = anomaly_map[label != 0]
    current_mask = gt_mask[label != 0].astype(int)

    binary_amaps = np.zeros_like(current_amap[0], dtype=np.bool_)
    pro_auc_list = []

    for anomaly_mask, mask in zip(current_amap, current_mask):
        df = pd.DataFrame(columns=["pro", "fpr", "threshold"])

        min_th = anomaly_mask.min()
        max_th = anomaly_mask.max()
        delta = (max_th - min_th) / num_th


        for th in np.arange(min_th, max_th, delta):
            binary_amaps[anomaly_mask <= th] = 0
            binary_amaps[anomaly_mask > th] = 1

            pros = []
            # for connect region
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amaps[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

            inverse_masks = 1 - mask
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()


            fpr = fp_pixels / inverse_masks.sum()
            if pros:
                df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)
            else:
                continue

        # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
        df = df[df["fpr"] < 0.3]
        df["fpr"] = df["fpr"] / df["fpr"].max()

        pro_auc = auc(df["fpr"], df["pro"])

        pro_auc_list.append(pro_auc)

        del df

    return pro_auc_list


def save_image(cfg, segmentations: ndarray, masks_gt, ima_path, ima_name_list, individual_dataloader):
    """
    segmentations: normalized segmentations.

    add mask_AD pred mask
    """
    save_fig_path = os.path.join(cfg.OUTPUT_DIR, "image_save")
    os.makedirs(save_fig_path, exist_ok=True)

    sample_num = len(segmentations)

    segmentations_max, segmentations_min = np.max(segmentations), np.min(segmentations)

    # visualize for random sample
    if cfg.TEST.VISUALIZE.Random_sample:
        sample_idx = random.sample(range(sample_num), cfg.TEST.VISUALIZE.Sample_num)
    else:
        sample_idx = [i for i in range(sample_num)]

    segmentations_random_sample = [segmentations[idx_random] for idx_random in sample_idx]
    mask_random_sample = [masks_gt[idx_random] for idx_random in sample_idx]
    ima_path_random_sample = [ima_path[idx_random] for idx_random in sample_idx]
    ima_name_random_sample = [ima_name_list[idx_random] for idx_random in sample_idx]

    temp_individual_name = os.path.join(save_fig_path, individual_dataloader.name)
    os.makedirs(temp_individual_name, exist_ok=True)

    for idx, (seg_each, mask_each, ori_path_each, name_each) in enumerate(zip(segmentations_random_sample,
                                                                              mask_random_sample,
                                                                              ima_path_random_sample,
                                                                              ima_name_random_sample)):
        anomaly_type = name_each.split("/")[2]
        temp_anomaly_name = os.path.join(temp_individual_name, anomaly_type)
        os.makedirs(temp_anomaly_name, exist_ok=True)
        file_name = name_each.replace("/", "_").split(".")[0]

        mask_numpy = np.squeeze((255 * np.stack(mask_each)).astype(np.uint8))

        original_ima = individual_dataloader.dataset.transform_mask(default_loader(ori_path_each))
        original_ima = (original_ima.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        original_ima = cv2.cvtColor(original_ima, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(11, 10))
        sns.heatmap(seg_each, vmin=segmentations_min, vmax=segmentations_max, xticklabels=False,
                    yticklabels=False, cmap="jet", cbar=True)
        plt.savefig(os.path.join(temp_anomaly_name, f'{file_name}_sns_heatmap.jpg'),
                    bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # min-max normalize for all images
        seg_each = (seg_each - segmentations_min) / (segmentations_max - segmentations_min)

        # only for seg_each that range in (0, 1)
        seg_each = np.clip(seg_each * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(seg_each, cv2.COLORMAP_JET)

        if heatmap.shape != original_ima.shape:
            raise Exception("ima shape is not consistent!")

        heatmap_on_image = np.float32(heatmap) / 255 + np.float32(original_ima) / 255
        heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
        heatmap_on_image = np.uint8(255 * heatmap_on_image)

        cv2_ima_save(temp_anomaly_name,
                     file_name,
                     ori_ima=original_ima,
                     mask_ima=mask_numpy,
                     heat_ima=heatmap,
                     heat_on_ima=heatmap_on_image)
    LOGGER.info("image save complete!")

def save_batch_images(
        cfg,
        segmentations: np.ndarray,
        masks_gt: np.ndarray,
        individual_dataloader,  # 输入形状为 [B, C, H, W]
        ima_paths: list,
        ima_names: list,
        visualize_random: bool = True,
        student_output: list = None,
        teacher_output: list = None,
):
    """
    保存批量图像分割结果（适配 PyTorch 风格输入）

    参数:
    - cfg: 配置对象
    - segmentations: 分割结果数组 [B, H, W] 或 [B, 1, H, W]
    - masks_gt: 真实掩码数组 [B, H, W]
    - individual_dataloader: 数据加载器用于获取原始图像
    - ima_paths: 图像完整路径列表 (长度=B)
    - ima_names: 图像文件名列表 (长度=B)
    - visualize_random: 是否启用随机跳过
    """
    # 输入验证
    assert len({len(segmentations), len(masks_gt), len(ima_paths), len(ima_names)}) == 1, \
        "输入批次长度不一致"

    # 创建主保存目录
    save_root = Path(cfg.OUTPUT_DIR) / "image_save"
    save_root.mkdir(parents=True, exist_ok=True)
    segmentations_max, segmentations_min = np.max(segmentations), np.min(segmentations)

    # 遍历批次
    for idx in range(len(segmentations)):
        # try:
            # 随机跳过逻辑
            if visualize_random and random.random() > 0.2:
                continue

            # ===================== 数据预处理 =====================
            # 处理原始图像 [C, H, W] -> [H, W, C] 并转换到0-255范围
            # 使用数据加载器的预处理方法获取原始图像张量
            img_tensor = individual_dataloader.dataset.transform_mask(default_loader(ima_paths[idx]))
            original_ima = img_tensor.numpy()
            original_ima = (original_ima * 255).astype(np.uint8)  # 转换为0-255范围
            original_ima = np.transpose(original_ima, (1, 2, 0))  # CHW -> HWC

            # 转换颜色通道顺序（假设transform返回RGB，需要转为BGR供OpenCV使用）
            original_img = cv2.cvtColor(original_ima, cv2.COLOR_RGB2BGR)

            #---------------------
            # 处理分割结果
            seg = segmentations[idx].squeeze()
            # 归一化分割结果到0-1范围
            seg_normalized = (seg - segmentations_min) / (segmentations_max - segmentations_min + 1e-8)
            seg_normalized = np.clip(seg_normalized, 0, 1)  # 确保在0-1范围内
            seg_norm = (seg_normalized * 255).astype(np.uint8)
            #---------------------

            # 处理真实掩码 [H, W]
            mask = masks_gt[idx].squeeze().astype(np.float32)
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)

            # ===================== 路径处理 =====================
            # 从完整路径中提取类别名（假设路径结构为.../class_name/test/...）
            ima_path = Path(ima_paths[idx])
            # 找到父目录链中包含"test"的层级（适配不同数据集结构）
            path_parts = ima_path.parts
            try:
                test_index = path_parts.index("test")
                anomaly_type = path_parts[test_index + 1]
            except (ValueError, IndexError):
                anomaly_type = "unknown_class"  # 异常处理

            save_dir = save_root / anomaly_type
            save_dir.mkdir(exist_ok=True)
            file_stem = Path(ima_names[idx]).stem

            # ===================== 可视化处理 =====================
            # 生成热图
            heatmap = cv2.applyColorMap(seg_norm, cv2.COLORMAP_JET)

            # 验证尺寸一致性（确保热图与原始图像尺寸匹配）
            if heatmap.shape[:2] != original_img.shape[:2]:
                heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

            # 图像叠加
            overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)

            # ===================== 保存结果 =====================
            cv2.imwrite(str(save_dir / f"{file_stem}_original.jpg"), original_img)
            cv2.imwrite(str(save_dir / f"{file_stem}_mask.jpg"), mask_uint8)
            cv2.imwrite(str(save_dir / f"{file_stem}_heatmap.jpg"), heatmap)
            cv2.imwrite(str(save_dir / f"{file_stem}_overlay.jpg"), overlay)

            # ================= 保存sns热图数据 =====================
            plt.figure(figsize=(11, 10))
            sns.heatmap(seg, vmin=segmentations_min, vmax=segmentations_max, xticklabels=False,
                        yticklabels=False, cmap="jet", cbar=True)
            plt.savefig(os.path.join(save_dir, f'{file_stem}_sns_heatmap.jpg'),
                        bbox_inches='tight', pad_inches=0.0)
            plt.close()

            # # ===================== 保存层间输出结果 =====================
            visualize_student_layers(student_output,idx,save_dir, file_stem,"student")
            visualize_student_layers(teacher_output,idx,save_dir, file_stem,"teacher")



def save_video_segmentations(cfg, segmentations: ndarray, scores: ndarray, ima_path, ima_name_list,
                             individual_dataloader):
    # 确定视频输出路径
    save_fig_path = os.path.join(cfg.OUTPUT_DIR, "video_save")
    os.makedirs(save_fig_path, exist_ok=True)

    sample_num = len(segmentations)

    # obtain the max segmentations
    segmentations_max, segmentations_min = np.max(segmentations), np.min(segmentations)

    # 将sample_idx变为顺序序列
    sample_idx = [i for i in range(sample_num)]
    # 将segmentations和scores按照sample_idx的顺序进行采样
    segmentations_random_sample = [segmentations[idx_random] for idx_random in sample_idx]
    scores = scores.tolist()
    ima_path_random_sample = [ima_path[idx_random] for idx_random in sample_idx]
    ima_name_random_sample = [ima_name_list[idx_random] for idx_random in sample_idx]

    temp_individual_name = os.path.join(save_fig_path, individual_dataloader.name)
    os.makedirs(temp_individual_name, exist_ok=True)

    for seg_each, score_each, ori_path_each, name_each in zip(segmentations_random_sample,
                                                              scores,
                                                              ima_path_random_sample,
                                                              ima_name_random_sample):
        # 文件命名
        anomaly_type = name_each.split("\\")[-2]
        temp_anomaly_name = os.path.join(temp_individual_name, anomaly_type)
        os.makedirs(temp_anomaly_name, exist_ok=True)
        file_name = name_each.replace(cfg.TRAIN.dataset_path, '').replace('\\','_').split('.')[0]
        # name_each.replace('E:\\lsw\\abnormal-detection-of-blades-MMR_0.1022.lsw\\LiMR\\datasets\\', '').replace('\\','_').split('.')[0]

        # 将图片加载后进行尺寸变换处理
        original_ima = individual_dataloader.dataset.transform_mask(default_loader(ori_path_each))
        # 将tensor变为numpy，映射到255，换为uint8，把原来的1,2维度变为0,1，原来的0维度变为2维度
        original_ima = (original_ima.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        # 将tensor处理的BGR图像换为RGB
        original_ima = cv2.cvtColor(original_ima, cv2.COLOR_BGR2RGB)

        # 对segment进行归一化
        seg_each = (seg_each - segmentations_min) / (segmentations_max - segmentations_min)
        # 如果最大值超过255，将其置换为255，若最小值小于0，则替换为0
        seg_each = np.clip(seg_each * 255, 0, 255).astype(np.uint8)
        # 将灰度值（即单一维度的seg）映射到从最小值到最大值的颜色图（从蓝色到红色）
        heatmap = cv2.applyColorMap(seg_each, cv2.COLORMAP_JET)

        if heatmap.shape != original_ima.shape:
            raise Exception("ima shape is not consistent!")

        # 两边归一化后相加，再归一化，映射回255的uint8
        heatmap_on_image = np.float32(heatmap) / 255 + np.float32(original_ima) / 255
        heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
        heatmap_on_image = np.uint8(255 * heatmap_on_image)

        str_score_each = str(score_each).replace(".", "_")

        cv2.imwrite(os.path.join(temp_anomaly_name, f'{file_name}_heatmap_{str_score_each}.jpg'), heatmap_on_image)# 写文件
    LOGGER.info("image save complete!")


def save_single_video_segmentation(cfg, segmentation: np.ndarray, score: float, image_path: str, image_name: str, dataloader, visualize_random: bool = True):
    # 确定视频输出路径
    save_fig_path = os.path.join(cfg.OUTPUT_DIR, "video_save")
    os.makedirs(save_fig_path, exist_ok=True)

    # 获取最大和最小分割值
    segmentation_max = np.max(segmentation)
    segmentation_min = np.min(segmentation)

    # 随机决定是否保存图像
    if visualize_random and random.random() > 0.5:
        return

    # 加载并处理图像
    original_ima = dataloader.dataset.transform_mask(default_loader(image_path))
    original_ima = (original_ima.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    original_ima = cv2.cvtColor(original_ima, cv2.COLOR_BGR2RGB)

    # 归一化分割结果并应用颜色映射
    seg_normalized = (segmentation - segmentation_min) / (segmentation_max - segmentation_min)
    seg_normalized = np.clip(seg_normalized * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(seg_normalized, cv2.COLORMAP_JET)

    if heatmap.shape != original_ima.shape:
        raise Exception("Image shape is not consistent!")

    # 合并图像和热图
    heatmap_on_image = np.float32(heatmap) / 255 + np.float32(original_ima) / 255
    heatmap_on_image = heatmap_on_image / np.max(heatmap_on_image)
    heatmap_on_image = np.uint8(255 * heatmap_on_image)

    # 构建文件名
    anomaly_type = image_name.split("\\")[-2]
    temp_anomaly_name = os.path.join(save_fig_path, anomaly_type)
    os.makedirs(temp_anomaly_name, exist_ok=True)
    file_name = image_name.replace(cfg.TRAIN.dataset_path, '').replace('\\', '_').split('.')[0]
    str_score_each = str(score).replace(".", "_")

    # 保存图像
    cv2.imwrite(os.path.join(temp_anomaly_name, f'{file_name}_heatmap_{str_score_each}.jpg'), heatmap_on_image)
    LOGGER.info(f"Image saved: {file_name}_heatmap_{str_score_each}.jpg")

def cv2_ima_save(dir_path, file_name, ori_ima, mask_ima, heat_ima, heat_on_ima):
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_original.jpg'), ori_ima)
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_mask.jpg'), mask_ima)
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_heatmap.jpg'), heat_ima)
    cv2.imwrite(os.path.join(dir_path, f'{file_name}_hm_on_ima.jpg'), heat_on_ima)


def load_model(checkpoint_path,model):

    if os.path.isfile(checkpoint_path):

        # 加载时指定设备映射
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        start_epoch = checkpoint['epoch']

        # 加载模型参数
        model_state_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        msg = model.load_state_dict(filtered_dict, strict=False)
        LOGGER.info(f"Model load state dict msg: {msg}")

        return model, start_epoch
    else:
        raise Exception("Checkpoint not found in {}".format(checkpoint_path))

def find_updated_model(root_path,file_type=".pth"):
    # 找到路径下所有的pth文件

    # Initialize variables to keep track of the file with the maximum epoch
    max_epoch = -1
    max_epoch_file = None

    # Loop through all files in the given directory
    for file in os.listdir(root_path):
        # Check if the file is a .pth file
        if file.endswith(file_type):
            # Extract the epoch number from the file name
            epoch_str = file.split('_epoch_')[-1].split('.pth')[0]
            try:
                epoch = int(epoch_str)
                # Update the maximum epoch and file name if necessary
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_epoch_file = file
            except ValueError:
                # Skip files that do not have a valid epoch number
                continue

    return os.path.join(root_path, max_epoch_file)


def channel_weight_pca(tensor):
    """
    Args:
        tensor: 输入张量 (C, H, W)
    Returns:
        fused_tensor: 加权融合结果 (1, H, W)  # 保持输出维度统一性
        channel_weights: 通道权重 (C,)
    """
    C, H, W = tensor.shape

    # 1. 数据预处理：转换为(H*W, C)的2D矩阵
    features = tensor.permute(1, 2, 0).reshape(-1, C).cpu().numpy()  # 网页4的转置方案

    # 2. 数据标准化（与原始实现一致）
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 3. PCA分析（保留全部主成分）
    pca = PCA(n_components=C)
    pca.fit(scaled_features)

    # 4. 计算通道权重（基于方差贡献率与特征向量）
    weights = np.sum(pca.explained_variance_ratio_[:, None] * pca.components_, axis=0)  # [7]
    weights = weights / np.sum(np.abs(weights))  # 符号敏感归一化

    # 5. 加权求和（调整einsum表达式适应CHW格式）
    weighted_map = torch.einsum('chw,c->hw', tensor, torch.tensor(weights, device=tensor.device))
    fused_tensor = weighted_map.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度 -> (1,1,H,W)

    return fused_tensor, weights


def visualize_student_layers(student_output, batch_idx,save_dir,file_stem,type):
    """
    可视化学生模型所有层的通道热力图
    Args:
        student_output (list/tensor): 各层输出，形状为 [Layers, B, C, H, W]
        batch_idx (int): 选择要可视化的批次索引
    """
    # 参数校验
    if student_output is None:
        raise ValueError("输入不能为空")
    num_layers = len(student_output)

    # 创建动态布局
    grid_size = math.ceil(math.sqrt(num_layers))
    row_num = grid_size
    col_num = grid_size

    # 创建画布和子图网格
    fig, axs = plt.subplots(row_num, col_num, figsize=(col_num * 4, row_num * 3))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)  # 调整子图间距[8](@ref)

    # 统一颜色映射范围
    vmin, vmax = 0, 1

    # 遍历所有层
    for layer_idx in range(num_layers):
        # 计算子图坐标
        row = layer_idx // col_num
        col = layer_idx % col_num

        # 提取当前层数据并计算通道权重
        layer_data = student_output[layer_idx][batch_idx]
        # 上采样到原始分辨率
        # layer_data = F.interpolate(layer_data, size=(224, 224), mode='bilinear', align_corners=False)
        layer_data = F.interpolate(layer_data.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()

        pic, _ = channel_weight_pca(layer_data)  # 假设已实现该函数

        # 归一化
        pic = (pic - pic.min()) / (pic.max() - pic.min() + 1e-8)

        # 绘制热力图到对应子图
        ax = axs[row, col] if row_num > 1 else axs[col]
        sns.heatmap(pic.squeeze().cpu().detach().numpy(),
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=False,
                    yticklabels=False,
                    cmap="jet",
                    cbar=False)  # 禁用单个颜色条[1,8](@ref)

        # 设置子图标题
        ax.set_title(f'Layer {layer_idx}', fontsize=8)

    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 右侧位置[8](@ref)
    fig.colorbar(ax.collections[0], cax=cbar_ax)

    # 隐藏空白子图
    for layer_idx in range(num_layers, row_num * col_num):
        row = layer_idx // col_num
        col = layer_idx % col_num
        axs[row, col].axis('off')

    plt.savefig(os.path.join(save_dir, f'{file_stem}_{type}_layers.jpg'),
                bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)