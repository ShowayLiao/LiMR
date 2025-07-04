import torch
from models.LiMR import LiMR_base,MMR_base
from utils import parse_args, load_config, load_backbones


def caculate_time(student,teacher,dummy_input,device=torch.device("cuda"),cfg=None):

    # 启用模型优化
    # teacher = torch.compile(teacher)
    # student = torch.compile(student)
    teacher.eval()
    student.eval()

    # 预热（包含异步传输和no_grad）
    with torch.no_grad():
        for _ in range(30):
            inputs = dummy_input.to(device, non_blocking=True)
            _ = teacher(inputs)
            _ = student(inputs)

    # CUDA事件计时
    def measure_speed(model, inputs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_time = 0
        for _ in range(100):  # 测量100次取平均
            start_event.record()
            _ = model(inputs)
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)
        return total_time / 100   # 返回秒

    inputs = dummy_input.to(device, non_blocking=True)
    teacher_time = measure_speed(teacher, inputs)
    student_time = measure_speed(student, inputs)

    print(f"Teacher推理时间: {teacher_time:.4f}ms | Student推理时间: {student_time:.4f}ms")


def LiMR_main():
    # ---------------初始化参数---------------
    args = parse_args()
    cfg = load_config(args, path_to_config=args.cfg_files[0])
    device = torch.device("cuda")
    # ---------------初始化模型---------------
    student = LiMR_base(cfg=cfg,
                        scale_factors=cfg.TRAIN.LiMR.scale_factors,
                        FPN_output_dim=cfg.TRAIN.LiMR.FPN_output_dim,
                        alpha=cfg.TRAIN.LiMR.alpha).to(device)

    teacher = load_backbones(cfg.TRAIN.backbone).to(device)
    dummy_input = torch.randn(cfg.TEST_SETUPS.batch_size, 3, cfg.DATASET.imagesize, cfg.DATASET.imagesize)

    # ----------------计算时间------------------
    caculate_time(student, teacher, dummy_input, device=device, cfg=cfg)


def MMR_main():
    # ---------------初始化参数---------------
    args = parse_args()
    cfg = load_config(args, path_to_config=args.cfg_files[0])
    device = torch.device("cuda")
    # ---------------初始化模型---------------
    student = MMR_base(cfg=cfg,
                        embed_dim=768,
                        num_heads=12,
                        scale_factors=cfg.TRAIN.LiMR.scale_factors,
                        FPN_output_dim=(256,512,768,1024)).to(device)

    teacher = load_backbones('wideresnet50').to(device)
    dummy_input = torch.randn(cfg.TEST_SETUPS.batch_size, 3, cfg.DATASET.imagesize, cfg.DATASET.imagesize)

    # ----------------计算时间------------------
    caculate_time(student, teacher, dummy_input, device=device, cfg=cfg)


if __name__ == '__main__':
    LiMR_main()