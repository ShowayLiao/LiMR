import cv2
import numpy as np
import time
from PIL import Image


class RealTimeAnomalyDetector:
    def __init__(self, pipeline, camera_id=0, threshold=128):
        """
        初始化实时检测器
        :param pipeline: 已经初始化好的 LiMR_pipeline_ 实例
        :param camera_id: 摄像头ID，通常为0
        :param threshold: 初始异常分割阈值 (0-255)
        """
        self.pipeline = pipeline
        self.cap = cv2.VideoCapture(camera_id)
        self.threshold = threshold

        # 从配置中获取预处理参数，用于显示对齐
        self.resize_h = pipeline.cfg.DATASET.resize
        self.resize_w = pipeline.cfg.DATASET.resize
        self.crop_size = pipeline.cfg.DATASET.imagesize

    def preprocess_for_display(self, frame):
        """
        对原始摄像头帧进行与模型一致的预处理（Resize + CenterCrop），
        以便可视化时掩膜能与图像对齐。
        """
        # 1. Resize (保持宽高比或强制缩放，根据cfg逻辑，这里假设是强制缩放)
        # 注意：OpenCV的resize参数是 (width, height)
        frame_resized = cv2.resize(frame, (self.resize_w, self.resize_h))

        # 2. CenterCrop
        h, w, _ = frame_resized.shape
        start_x = w // 2 - self.crop_size // 2
        start_y = h // 2 - self.crop_size // 2

        # 边界保护
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, start_x + self.crop_size)
        end_y = min(h, start_y + self.crop_size)

        cropped_frame = frame_resized[start_y:end_y, start_x:end_x]
        return cropped_frame

    def run(self):
        print(f"[-] 开始实时检测... 按 'q' 退出")
        print(f"[-] 按 'w' 增加阈值, 按 's' 减少阈值")

        if not self.cap.isOpened():
            print("[!] 无法打开摄像头")
            return

        fps_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[!] 无法读取视频帧")
                break

            # ---------------- 1. 预处理与推理 ----------------
            # 将 OpenCV 的 BGR 图像转换为 pipeline 需要的格式
            # pipeline.infer_single_image 内部处理了 BGR->RGB 和 transforms
            # 但我们需要一个裁剪后的图用于显示
            display_frame = self.preprocess_for_display(frame)

            # 记录推理时间
            t0 = time.time()

            # 调用你提供的 pipeline 推理接口
            # 注意：传入原始 frame 即可，infer_single_image 会自己处理 transform
            result = self.pipeline.infer_single_image(frame)

            infer_time = (time.time() - t0) * 1000

            # ---------------- 2. 数据解析 ----------------
            anomaly_map = result['anomaly_map']  # shape [H, W]
            image_score = result['image_level_anomaly_score']

            # ---------------- 3. 后处理与可视化 ----------------
            # 3.1 归一化异常图到 0-255
            if anomaly_map.max() - anomaly_map.min() != 0:
                norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
                anomaly_map_uint8 = (norm_map * 255).astype(np.uint8)
            else:
                anomaly_map_uint8 = anomaly_map.astype(np.uint8)

            # 3.2 生成热力图 (用于叠加)
            heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)

            # 3.3 根据阈值生成二值 Mask
            _, mask_binary = cv2.threshold(anomaly_map_uint8, self.threshold, 255, cv2.THRESH_BINARY)

            # 3.4 寻找轮廓 (用于圈出缺陷)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ---------------- 4. 图像叠加绘制 ----------------
            # 确保 display_frame 和 heatmap 尺寸一致 (防止 rounding error)
            if display_frame.shape[:2] != heatmap.shape[:2]:
                display_frame = cv2.resize(display_frame, (heatmap.shape[1], heatmap.shape[0]))

            # 绘制热力图叠加 (0.7 原图 + 0.3 热力图)
            vis_img = cv2.addWeighted(display_frame, 0.7, heatmap, 0.3, 0)

            # 绘制缺陷轮廓 (红色)
            cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)

            # 如果有缺陷，在左上角标记
            if len(contours) > 0:
                cv2.putText(vis_img, "DEFECT DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ---------------- 5. 状态显示 ----------------
            # 计算 FPS
            now = time.time()
            fps = 1 / (now - fps_time)
            fps_time = now

            # 在画面上打印信息
            info_text = [
                f"FPS: {fps:.1f}",
                f"Infer Time: {infer_time:.1f}ms",
                f"Score: {image_score:.4f}",
                f"Threshold: {self.threshold} (w/s to adj)"
            ]

            for i, text in enumerate(info_text):
                cv2.putText(vis_img, text, (10, 25 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow('LiMR Real-time Detection', vis_img)
            # 显示原始 mask (可选，方便调试)
            # cv2.imshow('Mask', mask_binary)

            # ---------------- 6. 按键交互 ----------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):  # 增加阈值
                self.threshold = min(255, self.threshold + 5)
            elif key == ord('s'):  # 减少阈值
                self.threshold = max(0, self.threshold - 5)

        self.cap.release()
        cv2.destroyAllWindows()


# ================= 使用示例 =================
if __name__ == "__main__":
    # 假设你已经定义并加载了 config, teacher_model, LiMR_model 等
    # 这里只是模拟调用的过程，请替换为你实际的初始化代码

    # 1. 初始化 Pipeline (使用你提供的 PyTorch 类)
    # pipeline = LiMR_pipeline_(teacher_model, LiMR_model, optimizer, scheduler, device, cfg)

    # 确保模型已经加载了权重
    # load_checkpoint(...)

    # 2. 启动实时检测
    # detector = RealTimeAnomalyDetector(pipeline, camera_id=0)
    # detector.run()
    pass