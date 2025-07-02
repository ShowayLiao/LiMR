
import matplotlib
matplotlib.use('TkAgg')
import timm
from torchvision.datasets.folder import default_loader

from utils import parse_args, load_config
from models.LiMR.LiMR import LiMR_base

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Dict


class FeatureVisualizer:
    def __init__(self, model, target_layers, img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],model_type='resnet'):
        """
        特征可视化类

        参数:
            model: 预训练模型
            target_layers: 目标层名称列表
            img_size: 输入图像尺寸
            mean: 归一化均值
            std: 归一化标准差
        """
        self.model = model
        self.target_layers = target_layers
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.forward_dict = {}
        self.model_type = model_type

        # 注册前向传播钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册前向传播钩子捕获特征图"""

        class ForwardHook:
            def __init__(self, storage, key):
                self.storage = storage
                self.key = key

            def __call__(self, module, input, output):
                self.storage[self.key] = output.detach()

        for idx,layer_name in enumerate(self.target_layers):
            if self.model_type in ['resnet', 'wide_resnet']:
                layer = self.model.__dict__["_modules"][layer_name][-1]
            elif self.model_type == 'mymodel':
                layer = model.decoder.__dict__['_modules']['conv_proj'].__dict__['_modules'][str(idx)]
            elif self.model_type == 'mobilevit':
                layer = model.__dict__["_modules"]["stages"][idx+1]
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            hook = ForwardHook(self.forward_dict, layer_name)
            layer.register_forward_hook(hook)

    def preprocess_image(self, image_path):
        """图像预处理流水线"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return transform(default_loader(image_path)).unsqueeze(0)

    @staticmethod
    def normalize_feature(feature):
        """特征归一化函数"""
        min_val = torch.min(feature)
        max_val = torch.max(feature)
        return (feature - min_val) / (max_val - min_val + 1e-8)

    def visualize_features(self, image_path, save_prefix="feature"):
        """
        可视化特征热力图并叠加原图

        参数:
            image_path: 输入图像路径
            save_prefix: 保存文件前缀
        """
        # 前向传播获取特征
        image_tensor = self.preprocess_image(image_path)
        _ = self.model(image_tensor)

        # 逆归一化处理
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std]
        )
        image_vis = inv_normalize(image_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
        image_vis = (image_vis * 255).astype(np.uint8)

        # 遍历各层特征
        for idx, layer_name in enumerate(self.target_layers):
            feature = self.forward_dict[layer_name]

            # 维度修正处理 (关键修复点)
            if feature.dim() == 4:
                # 保持通道维度 (B, C, H, W)
                normalized = self.normalize_feature(feature.sum(dim=1)).unsqueeze(1)

                # 插值维度修正：保持四维输入
                heatmap = F.interpolate(
                    input=normalized,
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                ).mean(dim=1).squeeze().cpu().numpy()  # 合并通道维度
            else:
                raise ValueError(f"特征图维度异常：{feature.shape}，应为四维张量")

            # 生成热力图
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # 图像叠加
            superimposed = cv2.addWeighted(image_vis, 0.6, heatmap_colored, 0.4, 0)

            # 可视化布局
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1).imshow(image_vis)
            plt.axis('off')
            plt.title("Original")
            plt.subplot(1, 3, 2).imshow(heatmap_colored)
            plt.axis('off')
            plt.title("Heatmap")
            plt.subplot(1, 3, 3).imshow(superimposed)
            plt.axis('off')
            plt.title(f"{layer_name} Overlay")

            plt.savefig(f"{save_prefix}_{layer_name}.png", bbox_inches='tight', dpi=300)
            plt.close()


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output

class BackwardHook:
    def __init__(self, hook_dict, layer_name: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name

    def __call__(self, module, grad_input, grad_output):
        self.hook_dict[self.layer_name] = grad_output[0].detach()

def each_patch_loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_tem = a[item].permute(0, 2, 3, 1)
        b_tem = b[item].permute(0, 2, 3, 1)

        loss += torch.mean(1 - cos_loss(a_tem.contiguous().view(-1, a_tem.shape[-1]),
                                        b_tem.contiguous().view(-1, b_tem.shape[-1])))
    return loss

def freeze_paras(backbone):
    for para in backbone.parameters():
        para.requires_grad = False

def normalize_feature(feature):
    min_val = torch.min(feature)
    max_val = torch.max(feature)
    return (feature - min_val) / (max_val - min_val + 1e-8)


if __name__ == "__main__":

    model = timm.create_model('wide_resnet50_2', pretrained=True)
    # model = timm.create_model('mobilevitv2_050', pretrained=True)
    # model_chkpt = r"H:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\LiMR\logs_and_models\aebad_S224\mobileViTMAE\benchmark-17b16(woONNX)-MMV_e-mFPN-layer4\54_2025_04_04_10_49\mobileViTMAE_benchmark-17b16(woONNX)-MMV_e-mFPN-layer4_weights_epoch_130.pth"
    #
    # args = parse_args()
    # cfg = load_config(args, path_to_config=args.cfg_files[0])
    # cfg.TRAIN.enable = False
    # cfg.TEST.enable = True
    # model = mobilevitMAE_base(cfg=cfg,
    #                             scale_factors=cfg.TRAIN.LiMR.scale_factors,
    #                             FPN_output_dim=cfg.TRAIN.LiMR.FPN_output_dim,
    #                             alpha=cfg.TRAIN.LiMR.alpha)
    # checkpoint = torch.load(model_chkpt)
    # msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # print(msg)


    visualizer = FeatureVisualizer(
        model=model,
        target_layers=["layer1", "layer2", "layer3", "layer4"],
        img_size=224,
        # model_type='mymodel'
        # model_type='mobilevit'
    )

    # 生成可视化结果
    visualizer.visualize_features(
        image_path=r"H:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\MMR\datasets\AeBAD\AeBAD_S\test\groove\background\IMG_8396.png",
        save_prefix="demo"
    )



