import torch
from thop import profile
from models.LiMR import LiMR_base, LiMR_pipeline
from utils import parse_args, load_config
from utils import load_backbones


def count_model(model,checkpoint):
    """
    计算模型参数量
    :param model: 模型
    :return: 参数量
    """
    input = torch.randn(1, 3, 224, 224)# 规定batch_size=1
    macs, _ = profile(model, inputs=(input,))

    # 提取模型权重
    model_weights = checkpoint['model_state_dict']

    # 计算参数量
    total_params = sum(p.numel() for p in model_weights.values())


    return total_params / 1e6,macs / 1e9# 前者单位为M，后者单位为GFLOPs


if __name__ == '__main__':

    args = parse_args()
    cfg = load_config(args, path_to_config=args.cfg_files[0])

    # 初始化学生模型
    mymodel = LiMR_base(cfg=cfg,
                            scale_factors=cfg.TRAIN.MMR.scale_factors,
                            FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim,
                            alpha = cfg.TRAIN.MMR.alpha)

    checkpoint = torch.load(r"H:\lsw\abnormal-detection-of-blades-MMR_0.1022.lsw\MMR\logs_and_models\aebad_S224\mobileViTMAE\benchmark-17b16-LiMR-175-resnet26\54_2025_05_17_15_40\mobileViTMAE_benchmark-17b16-LiMR-175-resnet26_weights_epoch_140.pth")


    cur_model = load_backbones(cfg.TRAIN.backbone)


    s_para,s_flops =count_model(mymodel,checkpoint)

    print(f"学生实际参数量: {s_para:.2f} M")
    print(f"学生实际FLOPs: {s_flops:.4f} GFLOPs")

    t_para,t_flops = count_model(cur_model,{'model_state_dict':cur_model.state_dict()})

    print(f"教师实际参数量: {t_para:.2f} M")
    print(f"教师实际FLOPs: {t_flops:.4f} GFLOPs")


