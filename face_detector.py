# face_detector.py

import cv2


class FaceDetector:
    """
    基于 OpenCV Haar Cascade 的人脸检测器。
    优点：安装简单，适合大作业基础版。
    缺点：复杂角度、遮挡、低清图像下可能漏检。
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Haar Cascade 加载失败，请检查 OpenCV 安装。")

    def detect(self, image_bgr):
        """
        输入 BGR 图像，返回人脸框列表。
        返回格式：[{"box": (x, y, w, h)}, ...]
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        results = []
        for (x, y, w, h) in faces:
            results.append({
                "box": (int(x), int(y), int(w), int(h))
            })

        return results

    def draw_faces(self, image_bgr, detections):
        """
        在图像上绘制人脸框。
        """
        output = image_bgr.copy()

        for item in detections:
            x, y, w, h = item["box"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return output