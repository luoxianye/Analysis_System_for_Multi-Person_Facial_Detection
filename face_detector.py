# face_detector.py
import cv2


class FaceDetector:
    """
    基于 InsightFace SCRFD 的多人脸检测器。

    适用场景：
    - 课堂视频；
    - 多人画面；
    - 远景小脸；
    - 视频帧质量不稳定；
    - MediaPipe 漏检较多的情况。
    """

    def __init__(
        self,
        det_thresh=0.30,
        det_size=(960, 960),
        min_face_size=8,
        use_gpu=False,
        model_name="buffalo_l",
    ):
        """
        参数说明：
            det_thresh:
                人脸检测阈值。漏检多时降低到 0.25；误检多时提高到 0.40~0.50。
            det_size:
                检测输入尺寸。小脸漏检多时使用 (960, 960) 或 (1280, 1280)。
                如果速度太慢，改成 (640, 640)。
            min_face_size:
                过滤过小检测框。课堂后排人脸小，可以设置为 8 或 10。
            use_gpu:
                是否使用 GPU。普通课程演示建议 False，使用 CPU 更容易部署。
            model_name:
                InsightFace 模型包。默认 buffalo_l，检测模型为 SCRFD-10GF。
        """
        from insightface.app import FaceAnalysis

        self.det_thresh = det_thresh
        self.det_size = det_size
        self.min_face_size = min_face_size
        self.use_gpu = use_gpu
        self.model_name = model_name

        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
            providers=providers,
        )

        self.app.prepare(
            ctx_id=ctx_id,
            det_size=det_size,
            det_thresh=det_thresh,
        )

        print(
            f"[FaceDetector] SCRFD loaded: "
            f"model={model_name}, det_thresh={det_thresh}, "
            f"det_size={det_size}, providers={providers}"
        )

    def detect(self, image_bgr):
        """
        输入 BGR 图像，返回人脸框列表。

        返回格式：
        [
            {
                "box": (x, y, w, h),
                "det_score": 0.93,
                "kps": [[x1, y1], ...]
            },
            ...
        ]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        img_h, img_w = image_bgr.shape[:2]
        faces = self.app.get(image_bgr)

        detections = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            w = x2 - x1
            h = y2 - y1

            if w < self.min_face_size or h < self.min_face_size:
                continue

            item = {
                "box": (x1, y1, w, h),
                "det_score": float(getattr(face, "det_score", 0.0)),
            }

            if hasattr(face, "kps") and face.kps is not None:
                item["kps"] = face.kps.astype(int).tolist()

            detections.append(item)

        # 为了显示更稳定，按 x 坐标排序
        detections.sort(key=lambda d: d["box"][0])
        return detections

    def draw_faces(self, image_bgr, detections):
        """
        在图像上绘制人脸框，不标注表情。
        """
        output = image_bgr.copy()

        for item in detections:
            x, y, w, h = item["box"]
            score = item.get("det_score", 0.0)

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"Face {score:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output

    def draw_results(self, image_bgr, detections):
        """
        在图像上绘制人脸框，并标注表情标签（英文）与置信度。
        """
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
            det_score = item.get("det_score", None)

            if confidence is not None:
                text = f"{label_en} {confidence:.2f}"
            elif det_score is not None:
                text = f"Face {det_score:.2f}"
            else:
                text = label_en

            color = color_map.get(label_en, (0, 255, 0))

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

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