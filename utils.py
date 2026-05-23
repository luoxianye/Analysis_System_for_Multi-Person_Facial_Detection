# utils.py

import cv2
import numpy as np
from PIL import Image


def uploaded_file_to_bgr(uploaded_file):
    """
    将 Streamlit 上传文件转换为 OpenCV BGR 图像。
    """
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr


def bgr_to_rgb(image_bgr):
    """
    OpenCV BGR 图像转 RGB，用于 Streamlit 展示。
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def crop_face(image_bgr, box, margin=0.0):
    """
    根据人脸框裁剪人脸区域。
    box: (x, y, w, h)
    margin: 额外扩展比例（默认 0.0，因 MediaPipe 检测器已预扩展边距）。
    """
    h_img, w_img = image_bgr.shape[:2]
    x, y, w, h = box

    dx = int(w * margin)
    dy = int(h * margin)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w_img, x + w + dx)
    y2 = min(h_img, y + h + dy)

    return image_bgr[y1:y2, x1:x2]