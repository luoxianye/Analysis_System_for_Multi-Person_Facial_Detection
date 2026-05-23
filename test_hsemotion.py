# test_hsemotion.py
import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

img_path = "test_images/test.jpg"
image_bgr = cv2.imread(img_path)

if image_bgr is None:
    raise FileNotFoundError(
        f"未找到测试图片：{img_path}\n"
        "请在 test_images/ 下放一张已经裁剪好的人脸图片，并命名为 test.jpg。"
    )

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

fer = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
emotion, scores = fer.predict_emotions(image_rgb, logits=False)

print("emotion:", emotion)
print("scores:", scores)
