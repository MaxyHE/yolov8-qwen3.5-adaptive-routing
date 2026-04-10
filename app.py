import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("YOLOv8 目标检测 Demo")
st.write("上传图片，自动检测20类物体")

model_path = st.sidebar.text_input("模型路径", value="yolov8s.pt")
conf_threshold = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.25)

uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", use_column_width=True)

    with st.spinner("检测中..."):
        model = YOLO(model_path)
        img_array = np.array(image)
        results = model(img_array, conf=conf_threshold)
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.image(result_img_rgb, caption="检测结果", use_column_width=True)

    boxes = results[0].boxes
    if len(boxes) > 0:
        st.write(f"检测到 {len(boxes)} 个物体：")
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"- {cls_name}：置信度 {conf:.2f}")
    else:
        st.write("未检测到物体，尝试降低置信度阈值")
