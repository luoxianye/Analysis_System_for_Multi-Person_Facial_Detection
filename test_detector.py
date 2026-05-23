# test_detector.py
import cv2
from face_detector import FaceDetector

img_path = "test_images/classroom.jpg"
image_bgr = cv2.imread(img_path)

if image_bgr is None:
    raise FileNotFoundError(
        f"未找到测试图片：{img_path}\n"
        "请在 test_images/ 下放一张多人课堂图片，并命名为 classroom.jpg。"
    )

detector = FaceDetector(min_detection_confidence=0.5, model_selection=1)
detections = detector.detect(image_bgr)

print("检测到的人脸数量:", len(detections))
print("检测结果:", detections)

result = detector.draw_faces(image_bgr, detections)
cv2.imwrite("results/test_detector_result.jpg", result)
print("结果图片已保存到 results/test_detector_result.jpg")
