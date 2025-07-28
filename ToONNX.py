import torch
from models.LiMR import LiMR_base,MMR_base
from utils import parse_args, load_config, load_backbones
from utils.common import freeze_paras,load_model
import tensorrt as trt


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output

class teacher_multi_layer(torch.nn.Module):
    def __init__(self, model):
        super(teacher_multi_layer, self).__init__()
        self.model = model
        self.teacher_outputs_dict = {}
        self.layers_to_extract_from = ["layer1", "layer2", "layer3"]
        for idx,extract_layer in enumerate(self.layers_to_extract_from):
            forward_hook = ForwardHook(self.teacher_outputs_dict, extract_layer)
            network_layer = model.__dict__["_modules"][extract_layer]# resnet

            network_layer[-1].register_forward_hook(forward_hook)
    def forward(self, x):
        self.teacher_outputs_dict.clear()
        _ = self.model(x)
        return [self.teacher_outputs_dict[key] for key in self.layers_to_extract_from ]


class student_multi_layer(torch.nn.Module):
    def __init__(self, model):
        super(student_multi_layer, self).__init__()
        self.model = model
        self.layers_to_extract_from = ["layer1", "layer2", "layer3"]
    def forward(self, x):
        outputs = self.model(x)
        return [outputs[key] for key in self.layers_to_extract_from]


def LiMR_onnx():
    # ---------------initialize parameters---------------
    args = parse_args()
    cfg = load_config(args, path_to_config=args.cfg_files[0])
    device = torch.device("cuda")
    # ---------------initialize student model---------------
    student = LiMR_base(cfg=cfg,
                        scale_factors=cfg.TRAIN.LiMR.scale_factors,
                        FPN_output_dim=cfg.TRAIN.LiMR.FPN_output_dim,
                        alpha=cfg.TRAIN.LiMR.alpha).to(device)

    checkpoint = torch.load('best_student_model_175.pth', map_location=device)
    # pre_student,_ = load_model('best_student_model_175.pth', student, )
    msg = student.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(msg)

    # freeze_paras(pre_student.eval())
    multi_student = student_multi_layer(student).to(device).eval()


    # ---------------initialize teacher model---------------
    teacher = load_backbones(cfg.TRAIN.backbone)
    # freeze_paras(teacher)
    multi_teacher = teacher_multi_layer(teacher).to(device).eval()

    # ---------------dummy input---------------
    dummy_input = torch.randn(1, 3, cfg.DATASET.imagesize, cfg.DATASET.imagesize).to(device).to(torch.float32)

    # ---------------onnx export---------------

    torch.onnx.export(
        multi_student,
        dummy_input,
        "LiMR_student.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )

    torch.onnx.export(
        multi_teacher,
        dummy_input,
        "LiMR_teacher.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )

    print("DONE")


def build_engine(onnx_path: str, engine_path: str) -> None:
    # 1. 初始化日志系统和运行时环境
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)  # ✅ 必须显式创建Runtime对象

    # 2. 创建构建器和网络定义（显式批次）
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 3. 解析ONNX模型
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"❌ ONNX解析错误 #{i + 1}: {parser.get_error(i)}")
            return

    # 4. 配置构建参数[3,5](@ref)
    config = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16加速，会因layernorm没有显式定义而导致优化失败，输出全为NaN
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB工作空间

    # 5. 设置动态输入配置（支持批量处理）[5](@ref)
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, 224, 224),
                      opt=(8, 3, 224, 224),
                      max=(32, 3, 224, 224))
    config.add_optimization_profile(profile)

    # 6. 构建并序列化引擎[3,4](@ref)
    # ✅ 直接使用build_serialized_network获取序列化引擎
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌ 引擎构建失败")
        return

    # 7. 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)  # ✅ 直接保存序列化数据

    print(f"✅ TensorRT引擎已成功保存至: {engine_path}")
    # print(f"  输入动态范围: 最小={profile.min}, 最优={profile.opt}, 最大={profile.max}")


if __name__ == "__main__":
    # LiMR_onnx()
    build_engine('./LiMR_student.onnx', './LiMR_student.engine')
    build_engine('./LiMR_teacher.onnx', './LiMR_teacher.engine')







