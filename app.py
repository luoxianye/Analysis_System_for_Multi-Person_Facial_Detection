# app.py

import streamlit as st
from utils import uploaded_file_to_bgr, bgr_to_rgb
from face_detector import FaceDetector

st.set_page_config(page_title="课堂状态分析系统", layout="wide")
st.title("基于多人人脸检测与表情识别的课堂状态分析系统")

detector = FaceDetector()

uploaded_file = st.file_uploader("上传课堂图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bgr = uploaded_file_to_bgr(uploaded_file)

    detections = detector.detect(image_bgr)
    result_bgr = detector.draw_faces(image_bgr, detections)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图片")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)

    with col2:
        st.subheader("人脸检测结果")
        st.image(bgr_to_rgb(result_bgr), use_container_width=True)

    st.metric("检测人数", len(detections))

    with st.expander("人脸框坐标"):
        st.write(detections)