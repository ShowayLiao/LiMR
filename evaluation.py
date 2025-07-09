import torch
from models.LiMR import LiMR_base,MMR_base
from utils import parse_args, load_config, load_backbones
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import onnxruntime as ort
import time


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


def calculate_onnx_time(stu_model_path,tea_model_path, dummy_input, device="cuda", warmup_iters=30, test_iters=100):
    """
    测量ONNX模型的推理时间（支持GPU）

    参数:
        model_path (str): ONNX模型路径
        dummy_input (torch.Tensor): 测试输入张量
        device (str): 推理设备，可选 "cuda" 或 "cpu"
        warmup_iters (int): 预热迭代次数
        test_iters (int): 正式测试迭代次数
    """
    # 配置ONNX Runtime会话选项
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

    # 创建ONNX Runtime会话
    session_stu = ort.InferenceSession(stu_model_path, options, providers=providers)
    input_name_stu  = session_stu.get_inputs()[0].name
    session_tea = ort.InferenceSession(tea_model_path, options, providers=providers)
    input_name_tea = session_tea.get_inputs()[0].name

    # 将输入数据转为NumPy格式（ONNX Runtime的输入要求）
    input_np = dummy_input.cpu().numpy().astype(np.float32)  # 确保数据类型匹配

    # 预热阶段（包含数据传输）
    for _ in range(warmup_iters):
        _ = session_stu.run(None, {input_name_stu : input_np})
        _ = session_tea.run(None, {input_name_tea : input_np})

    # CUDA事件计时（仅GPU有效）


    # 执行测速
    if device == "cuda":
        stu_avg_time = measure_speed_ort(session_stu,input_name_stu,test_iters,input_np)
        tea_avg_time = measure_speed_ort(session_tea,input_name_tea,test_iters,input_np)
    else:
        # CPU测速使用time模块
        start_time = time.perf_counter()
        for _ in range(test_iters):
            _ = session_stu.run(None, {input_name_stu: input_np})
        stu_avg_time = (time.perf_counter() - start_time) * 1000 / test_iters  # 转毫秒
        start_time = time.perf_counter()
        for _ in range(test_iters):
            _ = session_tea.run(None, {input_name_tea: input_np})
        tea_avg_time = (time.perf_counter() - start_time) * 1000 / test_iters

    print(f"ONNX Student模型平均推理时间: {stu_avg_time:.4f}ms | ONNX Teacher模型平均推理时间: {tea_avg_time:.4f}ms")

def measure_speed_ort(session,input_name,test_iters,input_np):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0

    for _ in range(test_iters):
        start_event.record()
        _ = session.run(None, {input_name: input_np})
        end_event.record()
        torch.cuda.synchronize()  # 确保异步操作完成
        total_time += start_event.elapsed_time(end_event)  # 毫秒

    return total_time / test_iters




def caculate_time_trt(student_path,teacher_path,input_data, warmup_iters=30, test_iters=100):
    # 初始化tensorRT
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    # ---------学生初始化----------
    with open(student_path, "rb") as f:
        engine_stu = runtime.deserialize_cuda_engine(f.read())
    context_stu = engine_stu.create_execution_context()
    context_stu.set_input_shape(engine_stu.get_tensor_name(0), input_data.shape)

    bindings_stu = []
    for idx in range(engine_stu.num_io_tensors):
        binding_name = engine_stu.get_tensor_name(idx)
        if engine_stu.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            size = input_data.nbytes  # 输入按实际数据大小
            print(size)
        else:
            shape = context_stu.get_tensor_shape(binding_name)  # 输出按引擎形状
            dtype = trt.nptype(engine_stu.get_tensor_dtype(binding_name))
            size = abs(int(np.prod(shape))) * np.dtype(dtype).itemsize
        device_mem = cuda.mem_alloc(size)  # DeviceAllocation对象
        bindings_stu.append(device_mem)  # 转换为地址用于绑定
        context_stu.set_tensor_address(binding_name, device_mem)  # 绑定设备内存地址
        # 不要将device_mem转换为int再存储，这样会让对象失去引用，从而回收，导致分配的内存地址失效

    stream_stu = cuda.Stream()

    #----------教师初始化----------
    with open(teacher_path, "rb") as f:
        engine_tea = runtime.deserialize_cuda_engine(f.read())
    context_tea = engine_tea.create_execution_context()
    context_tea.set_input_shape(engine_tea.get_tensor_name(0), input_data.shape)

    bindings_tea = []
    for idx in range(engine_tea.num_io_tensors):
        binding_name = engine_tea.get_tensor_name(idx)
        if engine_tea.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            size = input_data.nbytes
            print(size)
        else:
            shape = context_tea.get_tensor_shape(binding_name)
            dtype = trt.nptype(engine_tea.get_tensor_dtype(binding_name))
            size = abs(int(np.prod(shape))) * np.dtype(dtype).itemsize
        device_mem = cuda.mem_alloc(size)
        bindings_tea.append(device_mem)
        context_tea.set_tensor_address(binding_name, device_mem)

    stream_tea = cuda.Stream()

    # -------创建锁页内存-------

    host_input = cuda.register_host_memory(
        np.ascontiguousarray(input_data.astype(np.float32))
    )

    # ---------预热阶段---------
    for _ in range(warmup_iters):
        # 学生模型预热
        cuda.memcpy_htod_async(bindings_stu[0], host_input, stream_stu)
        context_stu.execute_async_v3(stream_handle=stream_stu.handle)

        # 教师模型预热
        cuda.memcpy_htod_async(bindings_tea[0], host_input, stream_tea)
        context_tea.execute_async_v3(stream_handle=stream_tea.handle)

    stream_stu.synchronize()
    stream_tea.synchronize()

    # ----------------正式测试----------------

    # 创建独立事件对象（学生模型）
    start_event_stu = cuda.Event()
    end_event_stu = cuda.Event()

    # 教师模型同理
    start_event_tea = cuda.Event()
    end_event_tea = cuda.Event()

    total_time_stu = 0
    total_time_tea = 0

    for _ in range(test_iters):
        # 每次迭代都拷贝新数据（避免重复计算）
        cuda.memcpy_htod_async(bindings_stu[0], input_data, stream_stu)
        cuda.memcpy_htod_async(bindings_tea[0], input_data, stream_tea)

        # 学生模型推理计时
        start_event_stu.record(stream_stu)
        context_stu.execute_async_v3(stream_handle=stream_stu.handle)
        end_event_stu.record(stream_stu)

        # 教师模型推理计时
        start_event_tea.record(stream_tea)
        context_tea.execute_async_v3(stream_handle=stream_tea.handle)
        end_event_tea.record(stream_tea)


        stream_stu.synchronize()  # 同步学生流
        stream_tea.synchronize()  # 同步教师流

        # 累加时间（确保事件完成）
        total_time_stu += start_event_stu.time_till(end_event_stu)  # 毫秒
        total_time_tea += start_event_tea.time_till(end_event_tea)

    # 计算平均时间
    avg_time_stu = total_time_stu / test_iters
    avg_time_tea = total_time_tea / test_iters

    print(f"TensorRT Student模型平均推理时间: {avg_time_stu:.4f}ms | TensorRT Teacher模型平均推理时间: {avg_time_tea:.4f}ms")
    # print(f"TensorRT 并行模型平均推理时间: {avg_time_stu:.4f}ms")

def inference_trt(engine_path, input_data):
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 1. 设置动态输入形状
    context.set_input_shape(engine.get_tensor_name(0), input_data.shape)

    # 2. 按需分配显存（含动态形状支持）
    bindings = []
    for idx in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            size = input_data.nbytes  # 输入按实际数据大小
            print(size)
        else:
            shape = context.get_tensor_shape(binding_name)  # 输出按引擎形状
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
            size = abs(int(np.prod(shape))) * np.dtype(dtype).itemsize
        device_mem = cuda.mem_alloc(size)  # DeviceAllocation对象
        bindings.append(device_mem)  # 转换为地址用于绑定
        context.set_tensor_address(binding_name, device_mem)  # 绑定设备内存地址
        # 不要将device_mem转换为int再存储，这样会让对象失去引用，从而回收，导致分配的内存地址失效

    # 3. 创建锁页内存
    host_input = cuda.register_host_memory(
        np.ascontiguousarray(input_data.astype(np.float32))
    )
    stream = cuda.Stream()

    # 4. 异步拷贝与推理
    cuda.memcpy_htod_async(bindings[0], host_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)

    # 5. 输出处理（修复索引）
    output_datas = []
    for idx in range(engine.num_io_tensors):
        if not engine.get_tensor_mode(engine.get_tensor_name(idx)) == trt.TensorIOMode.INPUT:
            shape = context.get_tensor_shape(engine.get_tensor_name(idx))
            dtype = trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(idx)))
            output_data = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh_async(output_data, bindings[idx], stream)
            output_datas.append(output_data)

    stream.synchronize()
    return output_datas

def inference_onnx(onnx_path, input_data):
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_data.astype(np.float32)})
    print(output)

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

    device = torch.device("cuda")

    # -------origin------
    # LiMR_main()

    # --------onnx--------
    # calculate_onnx_time('./LiMR_student.onnx','./LiMR_teacher.onnx', torch.randn(1, 3, 224, 224).to(device))

    #---------tensorrt--------
    # inference_trt('./LiMR_student.engine', np.random.randn(1, 3, 224, 224).astype(np.float32))
    caculate_time_trt('./LiMR_student.engine', './LiMR_teacher.engine', np.random.randn(1, 3, 224, 224).astype(np.float32))

