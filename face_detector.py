# face_detector.py
import cv2
import mediapipe as mp


class FaceDetector:
    """
    基于 MediaPipe Face Detection 的多人脸检测器。

    相比 OpenCV Haar Cascade：
    1. 对多人画面更稳定；
    2. 对低清、轻微侧脸、复杂背景更鲁棒；
    3. 适合图片、视频抽帧和摄像头实时检测。
    """

    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        """
        参数说明：
        min_detection_confidence:
            人脸检测置信度阈值，默认 0.5。
            如果漏检较多，可调低到 0.35；
            如果误检较多，可调高到 0.6。

        model_selection:
            0: 短距离模型，适合摄像头近距离人脸；
            1: 全距离模型，适合课堂图片、多人场景。
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image_bgr):
        """
        输入 BGR 图像，返回人脸框列表。

        返回格式：
        [
            {
                "box": (x, y, w, h),
                "det_score": 0.93
            },
            ...
        ]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        img_h, img_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        results = self.detector.process(image_rgb)
        detections = []

        if not results.detections:
            return detections

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * img_w)
            y = int(bbox.ymin * img_h)
            w = int(bbox.width * img_w)
            h = int(bbox.height * img_h)

            # 为人脸框增加一点边距，让表情识别模型看到更完整的脸部区域
            pad_x = int(w * 0.12)
            pad_y = int(h * 0.15)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            final_w = x2 - x1
            final_h = y2 - y1

            # 过滤太小的人脸框，避免误检或裁剪失败
            if final_w < 20 or final_h < 20:
                continue

            score = float(det.score[0]) if det.score else 0.0

            detections.append({
                "box": (x1, y1, final_w, final_h),
                "det_score": score,
            })

        return detections

    def draw_faces(self, image_bgr, detections):
        """
        在图像上绘制人脸框，不标注表情。
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

        color_map = {
            "Happy": (0, 255, 255),
            "Neutral": (0, 255, 0),
            "Sad": (255, 0, 0),
            "Angry": (0, 0, 255),
            "Surprise": (255, 0, 255),
            "Fear": (128, 0, 128),
            "Disgust": (128, 128, 0),
            "Unknown": (180, 180, 180),
        }

        for i, item in enumerate(detections):
            x, y, w, h = item["box"]
            label_en = item.get("emotion", "Face")
            confidence = item.get("confidence", None)

            label_cn = EMOTION_CN.get(label_en, label_en)

            if confidence is not None:
                text = f"{label_cn} {confidence:.2f}"
            else:
                text = label_cn

            color = color_map.get(label_en, (0, 255, 0))

            # 绘制人脸框
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # 标签位置：优先放在人脸框上方，如果空间不足则放在下方
            text_y = y - 10 if y - 10 > 20 else y + h + 25

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )

            cv2.rectangle(
                output,
                (x, text_y - th - 6),
                (x + tw + 6, text_y + 4),
                color,
                -1,
            )

            cv2.putText(
                output,
                text,
                (x + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # 绘制序号
            cv2.putText(
                output,
                f"#{i + 1}",
                (x, min(y + h + 18, output.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return output