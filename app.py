import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import torch
from collections import Counter

# 页面配置
st.set_page_config(
    page_title="YOLOv8 Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
    <style>
    .gradient-title {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: inline;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 16px;
    }
    img {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 4px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .result-card {
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
        background: rgba(102,126,234,0.06);
    }
    .desc-card {
        border-left: 4px solid #764ba2;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
        background: rgba(118,75,162,0.06);
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    lang = st.selectbox("", ["中文", "EN"], label_visibility="collapsed")
    zh = lang == "中文"
    st.markdown("### ⚙️ " + ("设置" if zh else "Settings"))

    model_path = st.text_input(
        "模型路径" if zh else "Model Path",
        value="best.pt"
    )
    conf_threshold = st.slider(
        "置信度阈值" if zh else "Confidence Threshold",
        0.1, 0.9, 0.25, 0.05,
        help="置信度越高，检测结果更严格" if zh else
             "Higher threshold = stricter, fewer detections"
    )
    mode = st.radio(
        "功能模式" if zh else "Mode",
        ["🖼️ " + ("图片检测" if zh else "Image Detection"),
         "🎬 " + ("视频追踪" if zh else "Video Tracking")]
    )

    vl_backend = "Qwen3.5-0.8B"
    if "图片" in mode or "Image" in mode:
        st.divider()
        enable_caption = st.toggle(
            "🤖 " + ("AI智能描述" if zh else "AI Caption"),
            value=False,
            help="开启后对每个检测目标生成自然语言描述，速度较慢" if zh else
                 "Generate natural language description for each detected object. Slower."
        )
    else:
        enable_caption = False

    st.divider()
    st.markdown("**" + ("支持类别（20类）" if zh else "Supported Classes (20)") + "**")
    st.caption("VOC2012 标准类别，涵盖人、车辆、动物、家具等常见物体" if zh else
               "VOC2012 standard classes: people, vehicles, animals, furniture and more.")
    classes_zh = [
        "✈️ 飞机", "🚲 自行车", "🐦 鸟", "⛵ 船",
        "🍾 瓶子", "🚌 公共汽车", "🚗 汽车", "🐱 猫",
        "🪑 椅子", "🐄 牛", "🍽️ 餐桌", "🐶 狗",
        "🐴 马", "🏍️ 摩托车", "🧍 人", "🌿 盆栽",
        "🐑 羊", "🛋️ 沙发", "🚂 火车", "📺 显示器"
    ]
    classes_en = [
        "✈️ aeroplane", "🚲 bicycle", "🐦 bird", "⛵ boat",
        "🍾 bottle", "🚌 bus", "🚗 car", "🐱 cat",
        "🪑 chair", "🐄 cow", "🍽️ diningtable", "🐶 dog",
        "🐴 horse", "🏍️ motorbike", "🧍 person", "🌿 pottedplant",
        "🐑 sheep", "🛋️ sofa", "🚂 train", "📺 tvmonitor"
    ]
    classes = classes_zh if zh else classes_en
    for i in range(0, 20, 2):
        c1, c2 = st.columns(2)
        c1.caption(classes[i])
        if i + 1 < 20:
            c2.caption(classes[i + 1])

# 标题区
st.markdown('🎯 <span class="gradient-title">YOLOv8 ' + ("目标检测 & 追踪" if zh else "Detection & Tracking") + '</span>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">' + (
    "基于 YOLOv8s 在 VOC2012 数据集训练，支持图片检测与视频多目标追踪，集成 Qwen3.5 多模态描述"
    if zh else
    "Trained on VOC2012 with YOLOv8s · Image detection, video tracking & Qwen3.5 multimodal captioning"
) + '</div>', unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_yolo(path):
    return YOLO(path)

@st.cache_resource
def load_qwen35():
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    model_id = '/path/to/Qwen3.5-0.8B'
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    vl_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    return processor, vl_model

VOC_CLASSES_STR = "aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor"

def get_qwen35_description(crop_img, processor, vl_model, zh=True):
    question = f"Choose the most likely category from: {VOC_CLASSES_STR}. Answer with only the category name."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_img},
        {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[crop_img], return_tensors="pt").to("cuda")
    gen_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    with torch.no_grad():
        output_ids = vl_model.generate(**gen_inputs, max_new_tokens=16, do_sample=False)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

yolo_model = load_yolo(model_path)

# 图片检测
if "图片" in mode or "Image" in mode:
    st.info("💡 " + (
        "上传图片后，系统将自动识别图中的目标类别、位置和置信度。开启 AI 描述可获取每个目标的自然语言说明。"
        if zh else
        "Upload an image to detect objects. Enable AI Caption for natural language descriptions of each target."
    ))
    uploaded_file = st.file_uploader(
        "上传图片" if zh else "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        with st.spinner("检测中..." if zh else "Detecting..."):
            results = yolo_model(img_array, conf=conf_threshold)
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**" + ("原始图片" if zh else "Original") + "**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**" + ("检测结果" if zh else "Detection Result") + "**")
            st.image(result_img_rgb, use_container_width=True)

        st.divider()

        boxes = results[0].boxes
        if len(boxes) > 0:
            cls_counts = Counter([yolo_model.names[int(b.cls)] for b in boxes])
            cols = st.columns(min(len(cls_counts), 4))
            for i, (cls_name, count) in enumerate(cls_counts.items()):
                with cols[i % 4]:
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{count}</div>
                            <div class="metric-label">{cls_name}</div>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**" + ("详细结果" if zh else "Details") + "**")

            if enable_caption:
                with st.spinner("加载 VLM 模型..." if zh else "Loading VLM..."):
                    processor, vl_model = load_qwen35()

            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            for box in sorted(boxes, key=lambda b: float(b.conf), reverse=True):
                cls_name = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))

                desc_html = ""
                if enable_caption:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop_bgr = img_bgr[y1:y2, x1:x2]
                    if crop_bgr.size > 0:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        crop_pil = Image.fromarray(crop_rgb)
                        desc = get_qwen35_description(crop_pil, processor, vl_model, zh)
                        desc_html = f'<div class="desc-card">🤖 {desc}</div>'

                st.markdown(f"""
                    <div class="result-card">
                        <b>{cls_name}</b> &nbsp; {bar} &nbsp; {conf:.1%}
                    </div>
                    {desc_html}
                """, unsafe_allow_html=True)
        else:
            st.info("未检测到物体，尝试降低置信度阈值" if zh else "No objects detected. Try lowering the confidence threshold.")

# 视频追踪
elif "视频" in mode or "Video" in mode:
    st.info("💡 " + (
        "上传视频后，系统将使用 ByteTrack 算法对每一帧进行目标检测与追踪，并为每个目标分配唯一 ID。处理完成后可在线预览或下载结果。"
        if zh else
        "Upload a video and ByteTrack will detect and assign unique IDs to each object across frames. Preview or download the result when done."
    ))
    uploaded_video = st.file_uploader(
        "上传视频" if zh else "Upload Video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded_video:
        video_id = uploaded_video.name + str(uploaded_video.size)
        if 'processed_video_id' not in st.session_state or st.session_state.processed_video_id != video_id:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            st.session_state.original_path = tfile.name

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            col1, col2, col3 = st.columns(3)
            col1.metric("总帧数" if zh else "Total Frames", total_frames)
            col2.metric("FPS", f"{fps:.1f}")
            col3.metric("分辨率" if zh else "Resolution", f"{w}×{h}")

            st.divider()
            progress_bar = st.progress(0, text="处理中..." if zh else "Processing...")

            out_path = tfile.name.replace('.mp4', '_tracked.mp4')
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model.track(frame, conf=conf_threshold, tracker="bytetrack.yaml", persist=True, verbose=False)
                annotated = results[0].plot()
                out.write(annotated)
                frame_idx += 1
                progress_bar.progress(frame_idx / total_frames,
                    text=f"{'处理中' if zh else 'Processing'}... {frame_idx}/{total_frames} {'帧' if zh else 'frames'}")

            cap.release()
            out.release()

            st.session_state.processed_video_id = video_id
            st.session_state.out_path = out_path
            st.session_state.frame_idx = frame_idx

        if 'out_path' in st.session_state and os.path.exists(st.session_state.out_path):
            st.success(f"✅ {'处理完成！共' if zh else 'Done! Total'} {st.session_state.frame_idx} {'帧' if zh else 'frames'}")
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**▶️ " + ("原始视频" if zh else "Original Video") + "**")
                if 'original_path' in st.session_state and os.path.exists(st.session_state.original_path):
                    with open(st.session_state.original_path, 'rb') as f:
                        st.video(f.read())
            with col2:
                st.markdown("**🎯 " + ("追踪结果" if zh else "Tracking Result") + "**")
                with open(st.session_state.out_path, 'rb') as f:
                    st.video(f.read())

            st.divider()
            with open(st.session_state.out_path, 'rb') as f:
                st.download_button(
                    "⬇️ " + ("下载追踪结果视频" if zh else "Download Tracked Video"),
                    f,
                    file_name="tracked.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
