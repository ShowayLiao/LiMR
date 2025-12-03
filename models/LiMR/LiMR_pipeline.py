from utils.common import save_batch_images,save_single_video_segmentation,visualize_student_layers
from .utils import ForwardHook, cal_anomaly_map, each_patch_loss_function
from utils import compute_pixelwise_retrieval_metrics, compute_pro

from PIL import Image

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

import plotly.graph_objects as go
import plotly.io as io
import torch
import logging
import numpy as np
import os
import time
import cv2
from torchvision import transforms


LOGGER = logging.getLogger(__name__)


class LiMR_pipeline_:

    def __init__(self,
                 teacher_model,
                 LiMR_model,
                 optimizer,# tuple
                 scheduler,# tuple
                 device,
                 cfg):
        # register forward hook
        self.teacher_outputs_dict = {}
        for idx,extract_layer in enumerate(cfg.TRAIN.LiMR.layers_to_extract_from):
            forward_hook = ForwardHook(self.teacher_outputs_dict, extract_layer)
            network_layer = teacher_model.__dict__["_modules"][extract_layer]# resnet

            network_layer[-1].register_forward_hook(forward_hook)

        # send models to device
        self.teacher_model = teacher_model.to(device)
        self.LiMR_model = LiMR_model.to(device)

        # define optimizer and scheduler
        self.encoder_optimizer = optimizer[0]
        self.decoder_optimizer = optimizer[1]
        self.encoder_scheduler = scheduler[0]
        self.decoder_scheduler = scheduler[1]

        self.device = device
        self.cfg = cfg

        # define image and mask transform
        transform_mask = [
            transforms.ToPILImage(),
            transforms.Resize((cfg.DATASET.resize, cfg.DATASET.resize)),
            transforms.CenterCrop(cfg.DATASET.imagesize),  # 围绕中心裁剪到符合imagesize的尺寸
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(transform_mask)

    def fit(self, individual_dataloader,start_epoch=0):

        # record time(not accurate)
        self.time_save = []

        # record loss
        self.loss_save = []
        if self.cfg.TRAIN.resume:
            self.load_loss()

        # set train mode
        self.teacher_model.eval()
        self.LiMR_model.train()

        # start training
        for epoch in range(start_epoch,self.cfg.TRAIN_SETUPS.epochs):
            # initialize time
            self.begin_time = time.time()

            # show learning rate
            LOGGER.info("encoder current lr is %.7f" % self.encoder_optimizer.param_groups[0]['lr'])
            LOGGER.info("decoder current lr is %.7f" % self.decoder_optimizer.param_groups[0]['lr'])

            # record iter loss
            loss_list = []

            for image in individual_dataloader:
                # -----------------load image---------------------
                if isinstance(image, dict):
                    image = image["image"].to(self.device)
                else:
                    image = image.to(self.device)

                #-----------------forward teacher---------------------
                self.teacher_outputs_dict.clear()
                with torch.no_grad():
                    _ = self.teacher_model(image)
                multi_scale_features = [self.teacher_outputs_dict[key]
                                        for key in self.cfg.TRAIN.LiMR.layers_to_extract_from]

                # ----------------forward student---------------------
                reverse_features = self.LiMR_model(image,
                                                  mask_ratio=self.cfg.TRAIN.LiMR.finetune_mask_ratio)# bn(inputs))
                multi_scale_reverse_features = [reverse_features[key]
                                                for key in self.cfg.TRAIN.LiMR.layers_to_extract_from]

                # ----------------calculate loss---------------------
                loss_multilayer = each_patch_loss_function(multi_scale_features, multi_scale_reverse_features)
                loss = loss_multilayer

                # ----------------backward and optimize---------------------
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                print(loss)
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                loss_list.append(loss.item())

            # -----------------step scheduler---------------------
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()

            # -----------------log information---------------------
            self.time_save.append(time.time() - self.begin_time)
            self.loss_save.append(np.mean(loss_list))
            time_infor = "average using time {}h {}m {:.2f}s".format((np.mean(self.time_save))//3600,
                                                    (np.mean(self.time_save))%3600//60,
                                                    (np.mean(self.time_save))%60)

            LOGGER.info('epoch [{}/{}], loss:{:.4f},using time {}'.format(epoch + 1,
                                                                           self.cfg.TRAIN_SETUPS.epochs,
                                                                           self.loss_save[-1],
                                                                           time_infor))






            # -------------------save model and checkpoint---------------------
            if self.cfg.TRAIN.save_model and (epoch + 1) % self.cfg.TRAIN_SETUPS.save_interval == 0:
                self.save_model_and_checkpoint(epoch,
                                               self.LiMR_model,
                                               self.encoder_optimizer,
                                               self.decoder_optimizer,
                                               self.encoder_scheduler,
                                               self.decoder_scheduler,
                                               self.cfg)

            # -------------------early stop---------------------
            if len(self.loss_save) > self.cfg.TRAIN_SETUPS.patience*2:
                if abs(np.mean(self.loss_save[-self.cfg.TRAIN_SETUPS.patience:]) - np.mean(self.loss_save[-self.cfg.TRAIN_SETUPS.patience*2:-self.cfg.TRAIN_SETUPS.patience]))< self.cfg.TRAIN_SETUPS.tolerance:
                    LOGGER.info("loss is not decrease, stop training!")
                    break


    def evaluation(self, test_dataloader=None):

        # set model to eval mode
        self.teacher_model.eval()
        self.LiMR_model.eval()

        # initialize lists to store results
        labels_gt = []
        labels_prediction = []

        # initialize metric list
        aupro_list = []
        pauroc_list = []

        # initialize time list(not accurate)
        time_use = []

        with torch.no_grad():
            for image in test_dataloader:
                # ----------------load image instance---------------------
                if isinstance(image, dict):
                    # load label(1 or 0)
                    label_current = image["is_anomaly"].numpy()
                    # load ground truth mask
                    mask_current = image["mask"].squeeze(1).numpy()
                    labels_gt.extend(label_current.tolist())

                    # load path and name
                    ima_path_list = image["image_path"]
                    ima_name_list = image["image_name"]
                    # load image
                    image = image["image"].to(self.device)

                else:
                    raise Exception("the format of DATA error!")

                # record time
                start_time = time.time()
                # -----------------forward teacher---------------------
                self.teacher_outputs_dict.clear()
                _ = self.teacher_model(image)
                multi_scale_features = [self.teacher_outputs_dict[key]
                                        for key in self.cfg.TRAIN.LiMR.layers_to_extract_from]

                """
                try masking in test. Although it will produce higher abnormal scores, 
                but it simultaneously produce larger error for complex normal part or high variance area
                """
                # -------------------LiMR network forward---------------------
                reverse_features = self.LiMR_model(image,
                                                  mask_ratio=self.cfg.TRAIN.LiMR.test_mask_ratio)

                # 输出结果转化为list
                multi_scale_reverse_features = [reverse_features[key]
                                                for key in self.cfg.TRAIN.LiMR.layers_to_extract_from]

                # -----------------calculate anomaly map---------------------
                # return anomaly_map np.array (batch_size, imagesize, imagesize)
                anomaly_map, _ = cal_anomaly_map(multi_scale_features, multi_scale_reverse_features, image.shape[-1],
                                                 amap_mode='a')

                # record consumption time
                time_use.append(time.time()-start_time)

                # -----------------calculate AUROC---------------------
                for item in range(len(anomaly_map)):
                    anomaly_map[item] = gaussian_filter(anomaly_map[item], sigma=4)

                labels_prediction.extend(np.max(anomaly_map.reshape(anomaly_map.shape[0], -1), axis=1))


                # pixel-level AUROC and pro-AUROC
                if self.cfg.TEST.pixel_mode_verify:
                    # PRO-AUROC
                    if set(mask_current.astype(int).flatten()) == {0, 1}:
                        aupro_list.extend(compute_pro(anomaly_map, mask_current.astype(int), label_current))
                        # P-AUROC
                        pixel_scores = compute_pixelwise_retrieval_metrics(
                            [anomaly_map.tolist()], [mask_current.astype(int).tolist()]
                        )
                        pauroc_list.append(pixel_scores["auroc"])

                else:
                    pauroc_list = 0
                    aupro_list = 0

                # visualize anomaly map and save images
                if self.cfg.TEST.save_segmentation_images:
                    save_batch_images(cfg=self.cfg,
                               segmentations=anomaly_map,
                               masks_gt=mask_current,
                               individual_dataloader = test_dataloader,
                               ima_paths=ima_path_list,
                               ima_names=ima_name_list,
                               visualize_random=self.cfg.TEST.VISUALIZE.Random_sample,
                               student_output = multi_scale_reverse_features,
                               teacher_output = multi_scale_features)

            # I-AUROC
            auroc_samples = round(roc_auc_score(labels_gt, labels_prediction), 3)


            """
            if normalizing the mask for each image, it will highlight the abnormal part, but it will
            hidden the effect in the normal image
            """


        return auroc_samples, round(np.mean(pauroc_list), 3), round(np.mean(aupro_list), 3),np.mean(time_use)

    def infer_single_image(self,
                           single_image,
                           mask=None,
                           save_path=None,
                           img_label=None):
        """
        单张图片异常检测推理函数

        参数说明：
            single_image: 输入图片，支持两种格式：
                          - numpy.ndarray：如cv2读取的BGR格式图片（shape: [H, W, 3]）
                          - PIL.Image.Image：RGB格式图片
            mask: 可选，像素级标注mask（0=正常，1=异常），格式同single_image，用于计算像素级指标
            save_path: 可选，异常热力图保存路径（如"./anomaly_heatmap.jpg"），若为None则不保存
            img_label: 可选，图片级标签（0=正常，1=异常），用于计算PRO-AUROC；若为None，将根据mask自动判断

        返回结果：
            result: 字典包含以下键：
                    - "image_level_anomaly_score": 图片级异常分数（越大越可能异常）
                    - "anomaly_map": 异常图（numpy.ndarray，shape: [H, W]），每个像素值为对应位置的异常分数
                    - "pixel_metrics": 像素级指标字典（仅当提供mask时非None），包含"P-AUROC"和"PRO-AUROC"
        """
        # -------------------------- 1. 输入图片格式处理与预处理 --------------------------
        # 1.1 统一图片格式为PIL.Image
        if isinstance(single_image, np.ndarray):
            # 若为cv2读取的BGR数组，转为RGB
            if single_image.shape[-1] == 3:
                single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)
            # 转为PIL Image（处理单通道灰度图也兼容）
            single_image = Image.fromarray(single_image)
        elif isinstance(single_image, Image.Image):
            # 若已为PIL Image，直接使用（确保为RGB格式）
            if single_image.mode != "RGB":
                single_image = single_image.convert("RGB")
        else:
            raise TypeError("不支持的图片格式！请输入 numpy.ndarray（BGR/RGB）或 PIL.Image.Image（RGB）")

        # 1.2 图片预处理（与训练时保持一致，包含归一化，适配教师模型预训练需求）
        # 注：若教师模型未使用ImageNet预训练，需修改Normalize的均值/方差
        image_transform = transforms.Compose([
            transforms.Resize((self.cfg.DATASET.resize, self.cfg.DATASET.resize)),  # 缩放
            transforms.CenterCrop(self.cfg.DATASET.imagesize),  # 中心裁剪
            transforms.ToTensor(),  # 转为Tensor（[C, H, W]，值归一化到0-1）
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                                 std=[0.229, 0.224, 0.225])  # ImageNet方差
        ])

        # 1.3 预处理并增加batch维度（模型要求输入为[batch_size, C, H, W]）
        image_tensor = image_transform(single_image).unsqueeze(0).to(self.device)  # shape: [1, 3, H, W]

        # -------------------------- 2. 模型切换为评估模式 --------------------------
        self.teacher_model.eval()
        self.LiMR_model.eval()

        # -------------------------- 3. 核心推理流程（与原evaluation一致） --------------------------
        with torch.no_grad():  # 关闭梯度计算，加速推理
            # 3.1 教师模型前向传播：提取多尺度特征
            self.teacher_outputs_dict.clear()  # 清空历史特征
            _ = self.teacher_model(image_tensor)  # 教师模型仅用于特征提取，输出无用
            # 获取配置中指定层的特征
            multi_scale_features = [
                self.teacher_outputs_dict[key]
                for key in self.cfg.TRAIN.LiMR.layers_to_extract_from
            ]

            # 3.2 LiMR模型前向传播：生成反向特征
            reverse_features = self.LiMR_model(
                image_tensor,
                mask_ratio=self.cfg.TRAIN.LiMR.test_mask_ratio  # 测试时的mask比例（与训练配置一致）
            )
            # 获取与教师模型对应的反向特征
            multi_scale_reverse_features = [
                reverse_features[key]
                for key in self.cfg.TRAIN.LiMR.layers_to_extract_from
            ]

            # 3.3 计算异常图（像素级异常分数）
            # 注：image_tensor.shape[-1]为图片边长（H=W，因预处理已保证）
            anomaly_map, _ = cal_anomaly_map(
                multi_scale_features,
                multi_scale_reverse_features,
                out_size=image_tensor.shape[-1],
                amap_mode='a'  # 与原evaluation一致的异常图计算模式
            )

            # 3.4 高斯滤波平滑异常图（减少噪声，与原evaluation一致）
            anomaly_map_np = anomaly_map  # Tensor转numpy（shape: [1, H, W]）
            anomaly_map_np[0] = gaussian_filter(anomaly_map_np[0], sigma=4)  # 单张图无需循环

            # 3.5 计算图片级异常分数（取异常图最大值，与原evaluation逻辑一致）
            img_level_score = np.max(anomaly_map_np.reshape(1, -1), axis=1)[0]

            # 3.6 调整异常图格式（去除batch维度，便于后续使用）
            anomaly_map_np = anomaly_map_np.squeeze(0)  # shape: [H, W]

        # -------------------------- 4. 可选：像素级指标计算（需提供mask） --------------------------
        pixel_metrics = None
        if mask is not None:
            # 4.1 统一mask格式并与异常图尺寸对齐
            if isinstance(mask, np.ndarray):
                # 调整mask尺寸与异常图一致（ nearest插值避免标签模糊）
                mask = cv2.resize(
                    mask,
                    dsize=(anomaly_map_np.shape[1], anomaly_map_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                mask = (mask > 0).astype(int)  # 确保二值化（0=正常，1=异常）
            elif isinstance(mask, Image.Image):
                # mask预处理（与图片尺寸一致）
                mask_transform = transforms.Compose([
                    transforms.Resize((self.cfg.DATASET.resize, self.cfg.DATASET.resize)),
                    transforms.CenterCrop(self.cfg.DATASET.imagesize),
                    transforms.ToTensor()  # 转为[1, H, W]
                ])
                mask = mask_transform(mask).squeeze(0).cpu().numpy()  # shape: [H, W]
                mask = (mask > 0).astype(int)
            else:
                raise TypeError("不支持的mask格式！请输入 numpy.ndarray 或 PIL.Image.Image")

            # 4.2 计算像素级AUROC（P-AUROC）
            pixel_gt = mask.flatten()  # 标签展平
            pixel_pred = anomaly_map_np.flatten()  # 预测分数展平
            try:
                pauroc = round(roc_auc_score(pixel_gt, pixel_pred), 3)
            except ValueError:
                # 若mask全为0或全为1（无正负样本），无法计算AUROC，设为-1
                pauroc = -1
                LOGGER.warning("mask全为正常或全为异常，无法计算P-AUROC")

            # 4.3 计算PRO-AUROC（需图片级标签）
            if img_label is None:
                # 自动判断图片级标签：mask有异常像素则为1，否则为0
                img_label = 1 if np.sum(mask) > 0 else 0
            try:
                # compute_pro输入格式：[异常图列表], [mask列表], [图片级标签列表]
                aupro = compute_pro([anomaly_map_np], [mask], [img_label])[0]
                aupro = round(aupro, 3)
            except Exception as e:
                aupro = -1
                LOGGER.warning(f"计算PRO-AUROC失败: {str(e)}")

            # 整理像素级指标
            pixel_metrics = {
                "P-AUROC": pauroc,  # 像素级AUROC
                "PRO-AUROC": aupro  # 像素级PRO-AUROC
            }

        # -------------------------- 5. 可选：保存异常热力图 --------------------------
        if save_path is not None:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 异常图归一化到0-255（便于显示）
            # anomaly_norm = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min())
            anomaly_uint8 = (anomaly_map_np * 255).astype(np.uint8)

            # 转为Jet热力图（更直观）
            anomaly_heatmap = cv2.applyColorMap(anomaly_uint8, cv2.COLORMAP_JET)
            # （可选）与原图叠加显示（如需叠加，需先将原图缩放到异常图尺寸）
            origin_resized = cv2.resize(np.array(single_image), (anomaly_heatmap.shape[1], anomaly_heatmap.shape[0]))
            origin_bgr = cv2.cvtColor(origin_resized, cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(origin_bgr, 0.5, anomaly_heatmap, 0.5, 0)

            # 保存热力图
            cv2.imwrite(save_path, anomaly_heatmap)
            cv2.imwrite(save_path.replace(".jpg", "_blended.jpg"), blended)
            LOGGER.info(f"异常热力图已保存至: {save_path}")

            # save_path的文件夹分离出来

            file_dir = os.path.split(save_path)[0]
            file_stem = os.path.splitext(os.path.basename(save_path))[0]


            visualize_student_layers(multi_scale_reverse_features, 0, file_dir, file_stem, "student")
            visualize_student_layers(multi_scale_features, 0, file_dir, file_stem, "teacher")


        # -------------------------- 6. 整理返回结果 --------------------------
        result = {
            "image_level_anomaly_score": round(img_level_score, 4),
            "anomaly_map": anomaly_map_np,  # [H, W] numpy数组，像素值为异常分数
            "pixel_metrics": pixel_metrics  # 仅当提供mask时非None
        }

        return result




    def save_model_and_checkpoint(self,
                                  epoch,
                                  model,
                                  encoder_optimizer,
                                  decoder_optimizer,
                                  encoder_scheduler,
                                  decoder_scheduler,
                                  cfg):
        """
        save checkpoint and model weights

        :param epoch: current epoch
        :param model: saved model
        :param optimizer: optimizer
        :param cfg: config
        """

        # save checkpoint
        filename = f'{self.cfg.TRAIN.method}_{self.cfg.TRAIN.change}_weights_epoch_{epoch+1}.pth'
        save_path = os.path.join(cfg.OUTPUT_DIR, filename)


        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'encoder_scheduler_state_dict': encoder_scheduler.state_dict(),
            'decoder_scheduler_state_dict': decoder_scheduler.state_dict(),
        }, save_path)

        # save loss curve figure
        self.draw_loss(os.path.join(cfg.OUTPUT_DIR, f'loss_{epoch+1}.html'))

        # save loss value
        save_path_txt = os.path.join(self.cfg.OUTPUT_DIR, f'loss.txt')

        with open(save_path_txt, 'w') as f:
            for item in self.loss_save:
                f.write("%s\n" % item)

    def draw_loss(self,file_path):
        loss_values = self.loss_save

        fig = go.Figure()

        fig.add_trace(go.Scatter(y=loss_values, mode='lines+markers', name='reconstruction loss'))

        fig.update_layout(
            title='loss',
            title_x=0.5,
            font=dict(
                family='Times New Roman',
            ),
            title_font=dict(
                size=24,
                color='black'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='epoch',
            xaxis_tickfont_color='black',
            yaxis_title='loss',
            yaxis_tickfont_color='black',
            # 边框为黑色
            # paper_bgcolor='black',
            xaxis=dict(
                gridcolor='lightgray',
                linecolor='black',
                showgrid=True,
                showline=True
            ),
            yaxis=dict(
                gridcolor='lightgray',
                linecolor='black',
                showgrid=True,
                showline=True
            )
        )

        io.write_html(fig, file_path)



    def load_loss(self):
        weight_path = self.cfg.TRAIN.resume_model_path
        if os.path.exists(weight_path):

            path = os.path.split(weight_path)[0]

            loss_txt = os.path.join(path, 'loss.txt')

            if os.path.exists(loss_txt):
                with open(loss_txt, 'r') as f:
                    self.loss_save = [float(line.strip()) for line in f]
            else:
                raise Exception("loss.txt not found in {}".format(path))

        else:
            raise Exception("model not found in {}".format(weight_path))









