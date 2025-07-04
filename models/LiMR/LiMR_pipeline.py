from utils.common import save_batch_images,save_single_video_segmentation
from .utils import ForwardHook, cal_anomaly_map, each_patch_loss_function
from utils import compute_pixelwise_retrieval_metrics, compute_pro

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



