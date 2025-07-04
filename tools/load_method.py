import torch
import logging

from utils import load_backbones
from utils.common import freeze_paras,load_model

from models.LiMR import LiMR_pipeline,LiMR_base
from torch.optim.lr_scheduler import CosineAnnealingLR


LOGGER = logging.getLogger(__name__)



# 定义编码器Warmup策略
def encoder_warmup_lambda(epoch):
    if epoch < 10:
        return (epoch + 1) / 10  # 线性增长[4](@ref)
    else:
        return 0.5 ** (epoch // 25)  # 原StepLR逻辑


# 带warmup的cos
class WarmupCosineScheduler(CosineAnnealingLR):
    def __init__(self, warmup_epochs, **kwargs):
        self.warmup_epochs = warmup_epochs
        super().__init__(**kwargs)
        # self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]



    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1)/self.warmup_epochs
                   for base_lr in self.base_lrs]  # 线性预热[5](@ref)
        return super().get_lr()


def LiMR(cfg):

    # --------initialize device----------------
    cur_device = torch.device("cuda:0")

    # --------initialize teacher model----------------
    cur_model = load_backbones(cfg.TRAIN.backbone)
    freeze_paras(cur_model)

    # --------initialize LiMR model----------------
    base = LiMR_base(cfg=cfg,
                            scale_factors=cfg.TRAIN.LiMR.scale_factors,
                            FPN_output_dim=cfg.TRAIN.LiMR.FPN_output_dim,
                            alpha = cfg.TRAIN.LiMR.alpha)

    #-----------load pretrained model and freeze extractor-------------
    if cfg.TRAIN.LiMR.load_pretrain_model:
        checkpoint = torch.load(cfg.TRAIN.LiMR.model_chkpt)
        LOGGER.info("train the decoder FPN of LiMR from scratch!")

        msg = base.encoder.load_state_dict(checkpoint, strict=False)
        LOGGER.info("MAE load meg: {}".format(msg))

        forzen_msg = "frozen encoder"

        for idx in range(1,3):
            for name, param in base.encoder.named_parameters():
                if "layer_{}".format(idx) in name:
                    param.requires_grad = False
                    forzen_msg+= ", {}".format(name)

        LOGGER.info(forzen_msg)

    else:
        LOGGER.info("MAE train/test from scratch!")

    # -------group encoder parameters for optimizer----------------
    encoder_params, encoder_bias_params = [], []
    for name, param in base.encoder.named_parameters():
        if param.requires_grad:
            if name.endswith('.bias'):
                encoder_bias_params.append(param)
            else:
                encoder_params.append(param)

    # -------group decoder parameters for optimizer----------------
    decoder_params, decoder_bias_params = [], []
    for name, param in base.decoder.named_parameters():
        if name.endswith('.bias'):
            decoder_bias_params.append(param)
        else:
            decoder_params.append(param)

    #-------apply weight decay to encoder and decoder parameters----------------
    encoder_param_groups = [
        {'params': encoder_params, 'lr': cfg.TRAIN_SETUPS.learning_rate, 'weight_decay': cfg.TRAIN_SETUPS.weight_decay},
        {'params': encoder_bias_params, 'lr': cfg.TRAIN_SETUPS.learning_rate, 'weight_decay': 0},
    ]
    decoder_param_groups = [
        {'params': decoder_params, 'lr': cfg.TRAIN_SETUPS.learning_rate, 'weight_decay': cfg.TRAIN_SETUPS.weight_decay},
        {'params': decoder_bias_params, 'lr': cfg.TRAIN_SETUPS.learning_rate, 'weight_decay': 0}
    ]

    # -------initialize optimizers----------------
    optimizer_encoder = torch.optim.AdamW(
        encoder_param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=False
    )
    optimizer_decoder = torch.optim.AdamW(
        decoder_param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=False,
    )

    # -------initialize schedulers(cosine annealing with warmup)----------------
    scheduler_encoder = WarmupCosineScheduler(
        optimizer=optimizer_encoder,
        warmup_epochs=cfg.TRAIN_SETUPS.warmup_epochs,
        T_max=cfg.TRAIN_SETUPS.epochs-cfg.TRAIN_SETUPS.warmup_epochs,
        eta_min=1e-5,
    )

    scheduler_decoder = WarmupCosineScheduler(
        optimizer=optimizer_decoder,
        warmup_epochs=cfg.TRAIN_SETUPS.warmup_epochs,
        T_max=cfg.TRAIN_SETUPS.epochs-cfg.TRAIN_SETUPS.warmup_epochs,
        eta_min=1e-5,
    )

    # -------load model if resume training or testing----------------
    start_epoch = 0
    if cfg.TRAIN.resume:
        # TODO LiMR does not support resume training yet!
        raise NotImplementedError("LiMR does not support resume training yet!")
    if cfg.TEST.enable:
        base,  start_epoch = load_model(cfg.TEST.model_path,
                                            base,)

    # -------initialize LiMR pipeline----------------
    LiMR_instance = LiMR_pipeline(teacher_model=cur_model,
                                 LiMR_model=base,
                                 optimizer=(optimizer_encoder, optimizer_decoder),
                                 scheduler=(scheduler_encoder, scheduler_decoder),
                                 device=cur_device,
                                 cfg=cfg)

    return LiMR_instance, start_epoch