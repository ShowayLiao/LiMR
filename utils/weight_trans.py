import timm
from models.LiMR.LiMR import LiMR_base
from utils import parse_args, load_config
import torch

mapping = {
    # Stem
    "conv_1.0.": "stem.conv.",
    "conv_1.1.": "stem.bn.",

    # Layer 1 (对应stage0)
    "layer_1.0.block.exp_1x1":      "stages.0.0.conv1_1x1.conv",
    "layer_1.0.block.exp_1x1_bn":   "stages.0.0.conv1_1x1.bn",
    "layer_1.0.block.conv_3x3":     "stages.0.0.conv2_kxk.conv",
    "layer_1.0.block.conv_3x3_bn":  "stages.0.0.conv2_kxk.bn",
    "layer_1.0.block.red_1x1":      "stages.0.0.conv3_1x1.conv",
    "layer_1.0.block.red_1x1_bn":   "stages.0.0.conv3_1x1.bn",

    # Layer 2 (对应stage1)
    "layer_2.0.block.exp_1x1":      "stages.1.0.conv1_1x1.conv",
    "layer_2.0.block.exp_1x1_bn":   "stages.1.0.conv1_1x1.bn",
    "layer_2.0.block.conv_3x3":     "stages.1.0.conv2_kxk.conv",
    "layer_2.0.block.conv_3x3_bn":  "stages.1.0.conv2_kxk.bn",
    "layer_2.0.block.red_1x1":      "stages.1.0.conv3_1x1.conv",
    "layer_2.0.block.red_1x1_bn":   "stages.1.0.conv3_1x1.bn",
    "layer_2.1.block.exp_1x1":      "stages.1.1.conv1_1x1.conv",
    "layer_2.1.block.exp_1x1_bn":   "stages.1.1.conv1_1x1.bn",
    "layer_2.1.block.conv_3x3":     "stages.1.1.conv2_kxk.conv",
    "layer_2.1.block.conv_3x3_bn":  "stages.1.1.conv2_kxk.bn",
    "layer_2.1.block.red_1x1":      "stages.1.1.conv3_1x1.conv",
    "layer_2.1.block.red_1x1_bn":   "stages.1.1.conv3_1x1.bn",

    # Layer 3 (对应stage2)
    "layer_3.0.block.exp_1x1":      "stages.2.0.conv1_1x1.conv",
    "layer_3.0.block.exp_1x1_bn":   "stages.2.0.conv1_1x1.bn",
    "layer_3.0.block.conv_3x3":     "stages.2.0.conv2_kxk.conv",
    "layer_3.0.block.conv_3x3_bn":  "stages.2.0.conv2_kxk.bn",
    "layer_3.0.block.red_1x1":      "stages.2.0.conv3_1x1.conv",
    "layer_3.0.block.red_1x1_bn":   "stages.2.0.conv3_1x1.bn",

    "layer_3.1.local_rep.0.0":                          "stages.2.1.conv_kxk.conv",
    "layer_3.1.local_rep.0.1":                          "stages.2.1.conv_kxk.bn",
    "layer_3.1.local_rep.1":                            "stages.2.1.conv_1x1",

    "layer_3.1.global_rep.0.pre_norm_attn.0":           "stages.2.1.transformer.0.norm1",
    "layer_3.1.global_rep.0.pre_norm_attn.1":           "stages.2.1.transformer.0.attn",
    "layer_3.1.global_rep.0.pre_norm_ffn.0":            "stages.2.1.transformer.0.norm2",
    "layer_3.1.global_rep.0.pre_norm_ffn.1":            "stages.2.1.transformer.0.mlp.fc1",
    "layer_3.1.global_rep.0.pre_norm_ffn.4":            "stages.2.1.transformer.0.mlp.fc2",

    # block2
    "layer_3.1.global_rep.1.pre_norm_attn.0":           "stages.2.1.transformer.1.norm1",
    "layer_3.1.global_rep.1.pre_norm_attn.1":           "stages.2.1.transformer.1.attn",
    "layer_3.1.global_rep.1.pre_norm_ffn.0":            "stages.2.1.transformer.1.norm2",
    "layer_3.1.global_rep.1.pre_norm_ffn.1":            "stages.2.1.transformer.1.mlp.fc1",
    "layer_3.1.global_rep.1.pre_norm_ffn.4":            "stages.2.1.transformer.1.mlp.fc2",

    "layer_3.1.global_rep.2":                           "stages.2.1.norm",

    "layer_3.1.conv_proj.0":                            "stages.2.1.conv_proj.conv",
    "layer_3.1.conv_proj.1":                            "stages.2.1.conv_proj.bn",

    # Layer 4 (对应stage3)
    "layer_4.0.block.exp_1x1":      "stages.3.0.conv1_1x1.conv",
    "layer_4.0.block.exp_1x1_bn":   "stages.3.0.conv1_1x1.bn",
    "layer_4.0.block.conv_3x3":     "stages.3.0.conv2_kxk.conv",
    "layer_4.0.block.conv_3x3_bn":  "stages.3.0.conv2_kxk.bn",
    "layer_4.0.block.red_1x1":      "stages.3.0.conv3_1x1.conv",
    "layer_4.0.block.red_1x1_bn":   "stages.3.0.conv3_1x1.bn",

    "layer_4.1.local_rep.0.0":                          "stages.3.1.conv_kxk.conv",
    "layer_4.1.local_rep.0.1":                          "stages.3.1.conv_kxk.bn",
    "layer_4.1.local_rep.1":                            "stages.3.1.conv_1x1",

    "layer_4.1.global_rep.0.pre_norm_attn.0":           "stages.3.1.transformer.0.norm1",
    "layer_4.1.global_rep.0.pre_norm_attn.1":           "stages.3.1.transformer.0.attn",
    "layer_4.1.global_rep.0.pre_norm_ffn.0":            "stages.3.1.transformer.0.norm2",
    "layer_4.1.global_rep.0.pre_norm_ffn.1":            "stages.3.1.transformer.0.mlp.fc1",
    "layer_4.1.global_rep.0.pre_norm_ffn.4":            "stages.3.1.transformer.0.mlp.fc2",

    # block2
    "layer_4.1.global_rep.1.pre_norm_attn.0":           "stages.3.1.transformer.1.norm1",
    "layer_4.1.global_rep.1.pre_norm_attn.1":           "stages.3.1.transformer.1.attn",
    "layer_4.1.global_rep.1.pre_norm_ffn.0":            "stages.3.1.transformer.1.norm2",
    "layer_4.1.global_rep.1.pre_norm_ffn.1":            "stages.3.1.transformer.1.mlp.fc1",
    "layer_4.1.global_rep.1.pre_norm_ffn.4":            "stages.3.1.transformer.1.mlp.fc2",

    "layer_4.1.global_rep.2.pre_norm_attn.0":           "stages.3.1.transformer.2.norm1",
    "layer_4.1.global_rep.2.pre_norm_attn.1":           "stages.3.1.transformer.2.attn",
    "layer_4.1.global_rep.2.pre_norm_ffn.0":            "stages.3.1.transformer.2.norm2",
    "layer_4.1.global_rep.2.pre_norm_ffn.1":            "stages.3.1.transformer.2.mlp.fc1",
    "layer_4.1.global_rep.2.pre_norm_ffn.4":            "stages.3.1.transformer.2.mlp.fc2",

    "layer_4.1.global_rep.3.pre_norm_attn.0":           "stages.3.1.transformer.3.norm1",
    "layer_4.1.global_rep.3.pre_norm_attn.1":           "stages.3.1.transformer.3.attn",
    "layer_4.1.global_rep.3.pre_norm_ffn.0":            "stages.3.1.transformer.3.norm2",
    "layer_4.1.global_rep.3.pre_norm_ffn.1":            "stages.3.1.transformer.3.mlp.fc1",
    "layer_4.1.global_rep.3.pre_norm_ffn.4":            "stages.3.1.transformer.3.mlp.fc2",

    "layer_4.1.global_rep.4":                           "stages.3.1.norm",

    "layer_4.1.conv_proj.0":                            "stages.3.1.conv_proj.conv",
    "layer_4.1.conv_proj.1":                            "stages.3.1.conv_proj.bn",

    # Layer 5 (对应stage4)
    "layer_5.0.block.exp_1x1":      "stages.4.0.conv1_1x1.conv",
    "layer_5.0.block.exp_1x1_bn":   "stages.4.0.conv1_1x1.bn",
    "layer_5.0.block.conv_3x3":     "stages.4.0.conv2_kxk.conv",
    "layer_5.0.block.conv_3x3_bn":  "stages.4.0.conv2_kxk.bn",
    "layer_5.0.block.red_1x1":      "stages.4.0.conv3_1x1.conv",
    "layer_5.0.block.red_1x1_bn":   "stages.4.0.conv3_1x1.bn",
    "layer_5.1.local_rep.0.0":                          "stages.4.1.conv_kxk.conv",
    "layer_5.1.local_rep.0.1":                          "stages.4.1.conv_kxk.bn",
    "layer_5.1.local_rep.1":                            "stages.4.1.conv_1x1",
    "layer_5.1.global_rep.0.pre_norm_attn.0":           "stages.4.1.transformer.0.norm1",
    "layer_5.1.global_rep.0.pre_norm_attn.1":           "stages.4.1.transformer.0.attn",
    "layer_5.1.global_rep.0.pre_norm_ffn.0":            "stages.4.1.transformer.0.norm2",
    "layer_5.1.global_rep.0.pre_norm_ffn.1":            "stages.4.1.transformer.0.mlp.fc1",
    "layer_5.1.global_rep.0.pre_norm_ffn.4":            "stages.4.1.transformer.0.mlp.fc2",
    # block2
    "layer_5.1.global_rep.1.pre_norm_attn.0":           "stages.4.1.transformer.1.norm1",
    "layer_5.1.global_rep.1.pre_norm_attn.1":           "stages.4.1.transformer.1.attn",
    "layer_5.1.global_rep.1.pre_norm_ffn.0":            "stages.4.1.transformer.1.norm2",
    "layer_5.1.global_rep.1.pre_norm_ffn.1":            "stages.4.1.transformer.1.mlp.fc1",
    "layer_5.1.global_rep.1.pre_norm_ffn.4":            "stages.4.1.transformer.1.mlp.fc2",
    "layer_5.1.global_rep.2.pre_norm_attn.0":           "stages.4.1.transformer.2.norm1",
    "layer_5.1.global_rep.2.pre_norm_attn.1":           "stages.4.1.transformer.2.attn",
    "layer_5.1.global_rep.2.pre_norm_ffn.0":            "stages.4.1.transformer.2.norm2",
    "layer_5.1.global_rep.2.pre_norm_ffn.1":            "stages.4.1.transformer.2.mlp.fc1",
    "layer_5.1.global_rep.2.pre_norm_ffn.4":            "stages.4.1.transformer.2.mlp.fc2",

    "layer_5.1.global_rep.3":                           "stages.4.1.norm",
    "layer_5.1.conv_proj.0":                            "stages.4.1.conv_proj.conv",
    "layer_5.1.conv_proj.1":                            "stages.4.1.conv_proj.bn",


    # Head
    "head.": "head."
}


if __name__ == "__main__":
    # load your model and state dict
    args = parse_args()
    cfg = load_config(args, path_to_config=args.cfg_files[0])
    mymodel = LiMR_base(cfg=cfg,
                                scale_factors=cfg.TRAIN.MMR.scale_factors,
                                FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim,
                                alpha=cfg.TRAIN.MMR.alpha)
    mydict = mymodel.encoder.state_dict()

    # load the pretrained model
    model = timm.create_model("mobilevitv2_200.cvnets_in1k", pretrained=True)
    # pretrained_model = timm.create_model('mobilevitv2_050', pretrained=True)
    pretrained_state_dict = model.state_dict()

    # for k, v in mydict.items():
    #     print(k)
    new_dict = {}

    # trans the keys of the pretrained model
    for k,v in pretrained_state_dict.items():
        for map_to,map_from in mapping.items():
            if k.startswith(map_from):
                k = k.replace(map_from,map_to)
                new_dict[k] = v
                break

    # check the missing keys
    missing = set(mydict.keys()) - set(new_dict.keys())
    if missing:
        print("未加载参数:", missing)
        print(len(missing))

    # save new weights
    torch.save(new_dict, "../mobilevitv2_200_layer4.pth")
    #
    # pretrain_dict = torch.load("mobilevitv2_075.pth")
    # missing_keys, unexpected_keys = mymodel.encoder.load_state_dict(pretrain_dict, strict=False)
    #
    # # 打印结果
    # print("Missing keys:", missing_keys)
    # print(len(missing_keys))
    # print("Unexpected keys:", unexpected_keys)
    # print(len(unexpected_keys))
