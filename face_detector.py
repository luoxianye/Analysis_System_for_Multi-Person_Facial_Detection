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
        在图像上绘制人脸框（仅框，无标签）。
        """
        output = image_bgr.copy()

        for item in detections:
            x, y, w, h = item["box"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return output

    def draw_results(self, image_bgr, detections):
        """
        在图像上绘制人脸框，并标注表情标签与置信度。
        """
        from config import EMOTION_CN

        output = image_bgr.copy()

        for i, item in enumerate(detections):
            x, y, w, h = item["box"]

            label_en = item.get("emotion", "Face")
            confidence = item.get("confidence", None)

            # 使用中文标签
            label_cn = EMOTION_CN.get(label_en, label_en)

            if confidence is not None:
                text = f"{label_cn} {confidence:.2f}"
            else:
                text = label_cn

            # 不同表情使用不同颜色
            color_map = {
                "Happy": (0, 255, 255),      # 黄色
                "Neutral": (0, 255, 0),       # 绿色
                "Sad": (255, 0, 0),           # 蓝色
                "Angry": (0, 0, 255),         # 红色
                "Surprise": (255, 0, 255),    # 品红
                "Fear": (128, 0, 128),        # 紫色
                "Disgust": (128, 128, 0),     # 青色
            }
            color = color_map.get(label_en, (0, 255, 0))

            # 画人脸框
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # 画标签背景
            text_y = y - 10 if y - 10 > 20 else y + h + 20
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                output,
                (x, text_y - th - 6),
                (x + tw + 4, text_y + 4),
                color,
                -1,
            )

            # 画标签文字
            cv2.putText(
                output,
                text,
                (x + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # 画序号
            cv2.putText(
                output,
                f"#{i + 1}",
                (x, y + h + 15) if y + h + 15 < output.shape[0] else (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return output