# expression_recognizer.py
import cv2
import numpy as np


class ExpressionRecognizer:
    """
    基于 HSEmotionONNX 的表情识别模块。

    输入：裁剪后的人脸 BGR 图像。
    输出：项目统一的表情标签和置信度。
    """

    def __init__(self, model_path=None, backend="hsemotion_onnx", model_name="enet_b0_8_best_vgaf"):
        """
        参数说明：
        model_path:
            保留参数，用于兼容旧代码。HSEmotionONNX 不需要手动传入模型路径。

        backend:
            推荐固定为 "hsemotion_onnx"。
            如果旧代码传入 "keras" 或 "deepface"，这里会自动切换为 HSEmotionONNX。

        model_name:
            enet_b0_8_best_vgaf: 推荐，适合视频、课堂、多人场景；
            enet_b0_8_best_afew: 适合普通图片；
            enet_b2_7: 7 类模型，通常更大，速度稍慢。
        """
        self.backend = "hsemotion_onnx"
        self.model_name = model_name
        self.fer = None

        if backend != "hsemotion_onnx":
            print(
                f"[ExpressionRecognizer] 检测到旧 backend={backend}，"
                "已自动切换为 hsemotion_onnx。"
            )

        self._load_hsemotion_onnx(model_name)

    def _load_hsemotion_onnx(self, model_name):
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError as e:
            raise RuntimeError(
                "HSEmotionONNX 未安装。\n"
                "请运行：python -m pip install hsemotion-onnx onnx onnxruntime"
            ) from e

        self.fer = HSEmotionRecognizer(model_name=model_name)
        print(f"[ExpressionRecognizer] 已加载 HSEmotionONNX 模型：{model_name}")

    def predict(self, face_bgr):
        """
        输入裁剪后的人脸 BGR 图像，返回：
        {
            "label": "Happy",
            "confidence": 0.92
        }
        """
        if face_bgr is None or face_bgr.size == 0:
            return {
                "label": "Unknown",
                "confidence": 0.0,
            }

        try:
            return self._predict_with_hsemotion_onnx(face_bgr)
        except Exception as e:
            print(f"[ExpressionRecognizer] 表情识别失败：{e}")
            return {
                "label": "Unknown",
                "confidence": 0.0,
            }

    def _predict_with_hsemotion_onnx(self, face_bgr):
        """
        HSEmotionONNX 表情预测。
        """
        # OpenCV 图像是 BGR，HSEmotionONNX 更适合输入 RGB 图像
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        emotion, scores = self.fer.predict_emotions(face_rgb, logits=False)

        # HSEmotionONNX 输出标签映射到项目统一标签
        mapping = {
            "Happiness": "Happy",
            "Happy": "Happy",

            "Neutral": "Neutral",
            "Contempt": "Neutral",

            "Sadness": "Sad",
            "Sad": "Sad",

            "Anger": "Angry",
            "Angry": "Angry",

            "Surprise": "Surprise",
            "Fear": "Fear",
            "Disgust": "Disgust",
        }

        label = mapping.get(str(emotion), "Neutral")

        # logits=False 时，scores 通常是概率分布，最大值作为置信度
        if isinstance(scores, np.ndarray):
            confidence = float(np.max(scores))
        elif isinstance(scores, list):
            confidence = float(max(scores))
        elif isinstance(scores, dict):
            confidence = float(scores.get(emotion, max(scores.values())))
        else:
            confidence = 0.0

        return {
            "label": label,
            "confidence": confidence,
        }
