import torch
from torchvision import transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit# This is necessary for initializing CUDA driver
import numpy as np
from .utils import cal_anomaly_map
import time
from scipy.ndimage import gaussian_filter
from utils import compute_pixelwise_retrieval_metrics, compute_pro
from utils.common import save_batch_images
from sklearn.metrics import roc_auc_score
import logging

LOGGER = logging.getLogger(__name__)

class LiMR_pipeline_trt:
    def __init__(self,cfg):
        self.cfg = cfg
        self.input_shape = (self.cfg.TEST_SETUPS.batch_size,3,self.cfg.DATASET.imagesize,self.cfg.DATASET.imagesize)
        self.device = cuda.Device(0)
        self.peak_used = 0


    def evaluation(self,test_dataloader=None):

        # --------------initialize TensorRT engine----------------
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        # initialize student TensorRT engine
        with open(self.cfg.TEST.TensorRT.stu_path, "rb") as f:
            engine_stu = runtime.deserialize_cuda_engine(f.read())
        context_stu = engine_stu.create_execution_context()
        context_stu.set_input_shape(engine_stu.get_tensor_name(0), self.input_shape)

        bindings_stu = []
        for idx in range(engine_stu.num_io_tensors):
            binding_name = engine_stu.get_tensor_name(idx)
            # if engine_stu.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            #     size = input_data.nbytes  # 输入按实际数据大小
            #     print(size)
            # else:
            shape = context_stu.get_tensor_shape(binding_name)  # 输出按引擎形状
            dtype = trt.nptype(engine_stu.get_tensor_dtype(binding_name))
            size = abs(int(np.prod(shape))) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            bindings_stu.append(device_mem)  # 转换为地址用于绑定
            context_stu.set_tensor_address(binding_name, device_mem)  # 绑定设备内存地址
            # 不要将device_mem转换为int再存储，这样会让对象失去引用，从而回收，导致分配的内存地址失效

        stream_stu = cuda.Stream()

        # ------------------initialize teacher TensorRT engine----------------------

        with open(self.cfg.TEST.TensorRT.tea_path, "rb") as f:
            engine_tea = runtime.deserialize_cuda_engine(f.read())

        context_tea = engine_tea.create_execution_context()
        context_tea.set_input_shape(engine_tea.get_tensor_name(0), self.input_shape)
        bindings_tea = []
        for idx in range(engine_tea.num_io_tensors):
            binding_name = engine_tea.get_tensor_name(idx)
            shape = context_tea.get_tensor_shape(binding_name)
            dtype = trt.nptype(engine_tea.get_tensor_dtype(binding_name))
            size = abs(int(np.prod(shape))) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            bindings_tea.append(device_mem)
            context_tea.set_tensor_address(binding_name, device_mem)

        stream_tea = cuda.Stream()
        # ---------------------------warmup---------------------------
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        host_dummy_input = cuda.register_host_memory(
            np.ascontiguousarray(dummy_input)
        )

        for _ in range(30):
            # copy input data to device memory
            cuda.memcpy_htod_async(bindings_stu[0], host_dummy_input, stream_stu)
            cuda.memcpy_htod_async(bindings_tea[0], host_dummy_input, stream_tea)

            # execute inference
            context_stu.execute_async_v3(stream_handle=stream_stu.handle)
            context_tea.execute_async_v3(stream_handle=stream_tea.handle)

        stream_stu.synchronize()
        stream_tea.synchronize()


        # ---------------------------initialize result collect---------------------
        # initialize lists to store results
        labels_gt = []
        labels_prediction = []

        # initialize metric list
        aupro_list = []
        pauroc_list = []

        # initialize time list(not accurate)
        time_use_inf = []
        time_use_post = []

        # --------------------------------inference--------------------------------
        for image in test_dataloader:
            # prepare input data
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
                image = image["image"]

                # create host memory for input
                host_input = cuda.register_host_memory(
                    np.ascontiguousarray(image.numpy().astype(np.float32))
                )

                # copy input data to device memory
                cuda.memcpy_htod_async(bindings_stu[0], host_input, stream_stu)
                cuda.memcpy_htod_async(bindings_tea[0], host_input, stream_tea)

            else:
                raise Exception("the format of DATA error!")

            # measure inference time
            start_event = cuda.Event()
            end_event = cuda.Event()

            # ------------------inference------------------
            # inference
            start_event.record(stream_tea)
            self.get_peak_memory_usage()  # 获取峰值内存使用情况
            context_tea.execute_async_v3(stream_handle=stream_tea.handle)
            self.get_peak_memory_usage()  # 获取峰值内存使用情况
            context_stu.execute_async_v3(stream_handle=stream_tea.handle)
            self.get_peak_memory_usage()  # 获取峰值内存使用情况
            end_event.record(stream_tea)
            stream_tea.synchronize()
            stream_stu.synchronize()

            time_use_inf.append(start_event.time_till(end_event))

            start_time = time.perf_counter()

            # get output data
            output_datas_tea = []
            for idx in range(engine_tea.num_io_tensors):
                if not engine_tea.get_tensor_mode(engine_tea.get_tensor_name(idx)) == trt.TensorIOMode.INPUT:
                    shape = context_tea.get_tensor_shape(engine_tea.get_tensor_name(idx))
                    dtype = trt.nptype(engine_tea.get_tensor_dtype(engine_tea.get_tensor_name(idx)))
                    output_data = np.empty(shape, dtype=dtype)
                    cuda.memcpy_dtoh_async(output_data, bindings_tea[idx], stream_tea)
                    output_datas_tea.append(torch.from_numpy(output_data))

            output_datas_stu = []
            for idx in range(engine_stu.num_io_tensors):
                if not engine_stu.get_tensor_mode(engine_stu.get_tensor_name(idx)) == trt.TensorIOMode.INPUT:
                    shape = context_stu.get_tensor_shape(engine_stu.get_tensor_name(idx))
                    dtype = trt.nptype(engine_stu.get_tensor_dtype(engine_stu.get_tensor_name(idx)))
                    output_data = np.empty(shape, dtype=dtype)
                    cuda.memcpy_dtoh_async(output_data, bindings_stu[idx], stream_stu)
                    output_datas_stu.append(torch.from_numpy(output_data))

            # ------------------post-process------------------
            anomaly_map, _ = cal_anomaly_map(output_datas_tea, output_datas_stu, image.shape[-1],
                                             amap_mode='a')
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

            end_time = time.perf_counter()
            time_use_post.append(end_time - start_time)

            if self.cfg.TEST.save_segmentation_images:
                save_batch_images(cfg=self.cfg,
                                  segmentations=anomaly_map,
                                  masks_gt=mask_current,
                                  individual_dataloader=test_dataloader,
                                  ima_paths=ima_path_list,
                                  ima_names=ima_name_list,
                                  visualize_random=self.cfg.TEST.VISUALIZE.Random_sample,
                                  student_output=output_datas_stu,
                                  teacher_output=output_datas_tea)

        # I-AUROC
        auroc_samples = round(roc_auc_score(labels_gt, labels_prediction), 3)

        LOGGER.info(f"TensorRT Student模型平均推理时间: {sum(time_use_inf) / len(time_use_inf):.4f}ms | TensorRT Teacher模型平均推理时间: {sum(time_use_post) / len(time_use_post):.4f}ms")
        LOGGER.info(f"TensorRT Student模型峰值内存使用量: {self.peak_used / (1024 ** 2):.2f} MB")

        return auroc_samples, round(np.mean(pauroc_list), 3), round(np.mean(aupro_list), 3), sum(time_use_inf) / len(time_use_inf)+ sum(time_use_post) / len(time_use_post)

    def get_peak_memory_usage(self):
        """
        获取当前CUDA设备的峰值内存使用情况
        :return: 峰值内存使用量（单位：字节）
        """
        peak_used = cuda.mem_get_info()[1]
        if peak_used > self.peak_used:
            self.peak_used = peak_used

