import streamlit as st
import cv2
import numpy as np
import torch
import logging
import os
import time
import sys

# ---------------------------------------------------------
# 1. 导入你现有的项目模块
# ---------------------------------------------------------
# 假设这些模块都在当前目录下，或者在PYTHONPATH中
try:
    from config import get_cfg
    from utils import setup_logging, set_output_dir
    from tools.load_method import LiMR
    # 如果有其他特定的utils引用，请确保在这里导入
except ImportError as e:
    st.error(f"导入项目模块失败，请确保 app.py 在项目根目录。\n错误详情: {e}")
    st.stop()

# 设置日志
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------
# 2. 辅助类与函数 (适配你的架构)
# ---------------------------------------------------------

class MockArgs:
    """
    模拟 argparse 解析后的对象。
    load_config 函数需要 args.opts 属性。
    """

    def __init__(self, opts=None):
        self.opts = opts if opts else []


def load_config_wrapper(path_to_config, ui_opts):
    """
    复用你提供的 load_config 逻辑
    """
    # 1. 初始化 MockArgs，传入 UI 生成的覆盖参数
    args = MockArgs(opts=ui_opts)

    # 2. 调用原有的 get_cfg (来自于你的 config.py)
    cfg = get_cfg()

    # 3. 加载 YAML 文件
    if path_to_config and os.path.exists(path_to_config):
        cfg.merge_from_file(path_to_config)
    else:
        raise FileNotFoundError(f"找不到配置文件: {path_to_config}")

    # 4. 应用命令行(这里是UI)的覆盖
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # 5. 设置输出目录 (复用你的 utils)
    # 注意：防止重复创建多层目录，这里稍微小心处理
    cfg = set_output_dir(cfg)

    # 6. 设置日志 (复用你的 utils)
    setup_logging(cfg)

    return cfg


@st.cache_resource
def get_pipeline(config_path, ui_opts, device_id):
    """
    加载模型 Pipeline。
    使用 cache_resource 确保改变无关参数时不会重新加载模型。
    """
    # 设置设备环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # 加载配置
    cfg = load_config_wrapper(config_path, ui_opts)

    # 加载模型 (复用你的 LiMR)
    st.info(f"正在加载模型... (Device: {device_id})")
    pipeline, _ = LiMR(cfg)

    return pipeline, cfg


# ---------------------------------------------------------
# 3. 图像处理与可视化逻辑
# ---------------------------------------------------------
def process_frame(frame, pipeline, threshold, cfg):
    # ================= 1. 原始画面准备 =================
    # 转换颜色 BGR -> RGB，用于界面显示的第一个流（Raw Input）
    raw_view = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================= 2. 模型预处理 =================
    resize_h = cfg.DATASET.resize
    resize_w = cfg.DATASET.resize
    crop_size = cfg.DATASET.imagesize

    # 模拟模型的预处理步骤：先 Resize
    frame_resized = cv2.resize(frame, (resize_w, resize_h))
    h, w, _ = frame_resized.shape
    # 再 CenterCrop
    start_x = max(0, w // 2 - crop_size // 2)
    start_y = max(0, h // 2 - crop_size // 2)

    # 这是模型真正“看到”的底图 (224x224)
    model_input_view = frame_resized[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # ================= 3. 推理 =================
    t0 = time.time()
    result = pipeline.infer_single_image(frame)
    infer_time = (time.time() - t0) * 1000

    # ================= 4. 后处理与生成视图 =================
    anomaly_map = result['anomaly_map']
    score = result['image_level_anomaly_score']

    # 4.1 归一化热力图
    if anomaly_map.max() - anomaly_map.min() != 0:
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        anomaly_map_uint8 = (norm_map * 255).astype(np.uint8)
    else:
        anomaly_map_uint8 = anomaly_map.astype(np.uint8)

    # 4.2 生成纯热力图 (Heatmap Stream)
    heatmap_color = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)

    # 4.3 生成二值 Mask 和 轮廓
    _, mask_binary = cv2.threshold(anomaly_map_uint8, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4.4 生成最终检测图 (Detection Stream)
    # 确保尺寸匹配
    if model_input_view.shape[:2] != heatmap_color.shape[:2]:
        model_input_view = cv2.resize(model_input_view, (heatmap_color.shape[1], heatmap_color.shape[0]))

    # 叠加：原图底片 + 热力图 + 红色轮廓
    final_view = cv2.addWeighted(model_input_view, 0.7, heatmap_color, 0.3, 0)
    cv2.drawContours(final_view, contours, -1, (0, 0, 255), 2)

    # ================= 5. 图像优化 (解决模糊) =================
    # 将模型视图 (通常较小, 如224x224) 放大到 400x400 以便于人眼观察
    # 使用 INTER_CUBIC 插值，会让热力图看起来更平滑，不全是马赛克
    display_size = (400, 400)

    heatmap_view = cv2.resize(heatmap_color, display_size, interpolation=cv2.INTER_CUBIC)
    final_view = cv2.resize(final_view, display_size, interpolation=cv2.INTER_CUBIC)

    # 颜色转换 BGR -> RGB
    heatmap_view = cv2.cvtColor(heatmap_view, cv2.COLOR_BGR2RGB)
    final_view = cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB)

    # 返回三个视图 + 指标
    return raw_view, heatmap_view, final_view, score, infer_time


# ---------------------------------------------------------
# 4. Streamlit 主程序
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="LiMR Parameter Tuning", layout="wide")

    # ================= 侧边栏：参数配置 =================
    st.sidebar.title("🛠️ 参数配置")

    # 1. 基础文件路径
    st.sidebar.subheader("1. 基础设置")
    default_cfg_path = r"H:\lsw\LiMR\method_config\AeBAD_S\LiMR.yaml"
    cfg_path = st.sidebar.text_input("配置文件路径 (.yaml)", value=default_cfg_path)
    device_id = st.sidebar.text_input("显卡 ID (Device)", value="0")

    # 2. 覆盖参数 (这些会生成 opts 列表)
    st.sidebar.subheader("2. 覆盖参数 (Overwrite)")

    # 模型路径
    default_model_path = "./best_student_model_175.pth"
    model_path = st.sidebar.text_input("模型权重路径 (Model Path)", value=default_model_path)

    # 图像尺寸
    resize_val = st.sidebar.number_input("Resize 大小", value=256)
    crop_val = st.sidebar.number_input("Crop 大小 (ImageSize)", value=224)

    # 输出目录
    output_dir = st.sidebar.text_input("日志输出目录", value="./logs_webui")

    # TensorRT 开关
    use_trt = st.sidebar.checkbox("开启 TensorRT", value=False)

    # ---------------- 构建 opts 列表 ----------------
    # 这就是模拟命令行 python main.py --opts KEY VALUE ...
    # 这里的 Key 必须和 YAML 文件中的层级结构严格对应
    ui_opts = [
        "TEST.model_path", model_path,
        "DATASET.resize", str(resize_val),
        "DATASET.imagesize", str(crop_val),
        "OUTPUT_ROOT_DIR", output_dir,
        "TEST.TensorRT.enable", str(use_trt),
        # 强制开启测试模式，关闭训练模式，防止误触
        "TRAIN.enable", "False",
        "TEST.enable", "True"
    ]

    load_btn = st.sidebar.button("🚀 加载/重载模型", type="primary")

    # ================= 主界面：展示与控制 =================
    st.title("🔍 LiMR 实时缺陷检测 WebUI")

    # Session State 管理
    if 'pipeline' not in st.session_state:
        st.session_state['pipeline'] = None
    if 'cfg' not in st.session_state:
        st.session_state['cfg'] = None

    # 加载逻辑
    if load_btn:
        if not os.path.exists(cfg_path):
            st.error(f"找不到配置文件: {cfg_path}")
        else:
            try:
                # 清除旧缓存 (可选，如果显存紧张)
                # st.cache_resource.clear()

                pipeline, cfg = get_pipeline(cfg_path, ui_opts, device_id)
                st.session_state['pipeline'] = pipeline
                st.session_state['cfg'] = cfg
                st.success(f"模型加载成功！\nBackbone: {cfg.TRAIN.backbone} | Model: {model_path}")
            except Exception as e:
                st.error(f"加载失败: {e}")
                import traceback
                st.code(traceback.format_exc())
    # 只有模型加载成功才显示视频流区域
    if st.session_state['pipeline']:
        pipeline = st.session_state['pipeline']
        cfg = st.session_state['cfg']

        st.divider()

        # --- 布局修改：创建三列用于显示视频流 ---
        st.subheader("📺 实时检测监控")
        col_raw, col_heat, col_res = st.columns(3)

        with col_raw:
            st.markdown("**1. 原始画面 (Raw)**")
            view_raw = st.empty()  # 占位符1

        with col_heat:
            st.markdown("**2. 实时热力图 (Heatmap)**")
            view_heat = st.empty()  # 占位符2

        with col_res:
            st.markdown("**3. 缺陷检测结果 (Result)**")
            view_res = st.empty()  # 占位符3

        # --- 控制区 ---
        st.divider()
        ctrl_col1, ctrl_col2 = st.columns([1, 3])

        with ctrl_col1:
            st.subheader("🎛️ 操作面板")
            # 【修改点】更语义化的开关名称
            run_check = st.toggle("🚀 开启/关闭 检测", value=False)
            threshold = st.slider("异常阈值 (Threshold)", 0, 255, 100)

        with ctrl_col2:
            st.subheader("📊 实时指标")
            # 使用列来横向排列指标
            m1, m2, m3 = st.columns(3)
            kpi_score = m1.empty()
            kpi_time = m2.empty()
            kpi_fps = m3.empty()

            # 初始化指标显示
            kpi_score.metric("异常分数", "0.0000")
            kpi_time.metric("推理耗时", "0 ms")
            kpi_fps.metric("FPS", "0.0")

        # --- 循环逻辑 ---
        if run_check:
            cap = cv2.VideoCapture(0)
            prev_time = time.time()

            while run_check:
                ret, frame = cap.read()
                if not ret:
                    st.warning("无法读取摄像头画面")
                    break

                try:
                    # 获取三个视图
                    img_raw, img_heat, img_res, score, infer_time = process_frame(frame, pipeline, threshold, cfg)

                    # 分别更新三个画面
                    # width=None (默认自适应列宽) 或者指定 width=350
                    view_raw.image(img_raw, channels="RGB", use_container_width=True)
                    view_heat.image(img_heat, channels="RGB", use_container_width=True)
                    view_res.image(img_res, channels="RGB", use_container_width=True)

                    # 更新指标
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time

                    # 阈值颜色警告
                    score_color = "normal" if score < 0.5 else "off"  # 简单示例
                    kpi_score.metric("异常分数", f"{score:.4f}")
                    kpi_time.metric("推理耗时", f"{infer_time:.1f} ms")
                    kpi_fps.metric("FPS", f"{fps:.1f}")

                except Exception as e:
                    st.error(f"推理运行出错: {e}")
                    import traceback
                    print(traceback.format_exc())
                    break

            cap.release()
            # 循环结束后，显示最后一张图或者黑屏，或者保持最后一帧
    else:
        st.info("👈 请在左侧配置路径并点击 '加载模型' 按钮开始。")


if __name__ == "__main__":
    main()