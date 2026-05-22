# expression_recognizer.py

import cv2
import numpy as np
from pathlib import Path
from config import EMOTION_LABELS, BASE_DIR


class ExpressionRecognizer:
    """
    表情识别模块。
    优先使用本地 Keras 模型；
    如果没有本地模型，可以改为调用 DeepFace。
    """

    def __init__(self, model_path=None, backend="deepface"):
        """
        model_path: Keras 模型文件路径（仅 backend="keras" 时需要）
        backend: "keras" 或 "deepface"
        """
        self.backend = backend
        self.model = None

        if backend == "keras":
            self._load_keras_model(model_path)
        elif backend == "deepface":
            self._load_deepface()
        else:
            raise ValueError("backend 只能是 'keras' 或 'deepface'")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_keras_model(self, model_path):
        # 如果未指定路径，自动查找 models 目录下的模型
        if model_path is None:
            candidates = [
                BASE_DIR / "models" / "emotion_model.h5",
                BASE_DIR / "models" / "cnn3_best_weights.h5",
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    break
            else:
                raise FileNotFoundError(
                    "models 目录下未找到模型文件。"
                    "请将 .h5 模型放入 models/ 目录，"
                    "或使用 backend='deepface'。"
                )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        from tensorflow.keras.models import load_model
        self.model = load_model(str(model_path))
        num_classes = self.model.output_shape[-1]
        print(
            f"[ExpressionRecognizer] 已加载 Keras 模型：{model_path.name}"
            f"（{num_classes} 类）"
        )

    def _load_deepface(self):
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            print("[ExpressionRecognizer] 已加载 DeepFace 后端")
        except ImportError:
            raise RuntimeError(
                "DeepFace 未安装，请运行：pip install deepface tf-keras"
            )
        except Exception as e:
            raise RuntimeError(f"DeepFace 加载失败：{e}")

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------

    def predict(self, face_bgr):
        """
        输入裁剪后的人脸 BGR 图像，返回：
        {
            "label": "Happy",
            "confidence": 0.92
        }
        """
        if face_bgr is None or face_bgr.size == 0:
            return {"label": "Unknown", "confidence": 0.0}

        if self.backend == "keras":
            return self._predict_with_keras(face_bgr)
        if self.backend == "deepface":
            return self._predict_with_deepface(face_bgr)

    # ------------------------------------------------------------------
    # Keras 预测
    # ------------------------------------------------------------------

    def _predict_with_keras(self, face_bgr):
        """
        模型输入为 48x48 灰度图，输出 7 或 8 类 logits。
        """
        from config import EMOTION_LABELS, EMOTION_LABELS_8

        num_classes = self.model.output_shape[-1]

        # 选择对应的标签列表
        if num_classes == 8:
            labels = EMOTION_LABELS_8
        elif num_classes == 7:
            labels = EMOTION_LABELS
        else:
            labels = [str(i) for i in range(num_classes)]

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype("float32") / 255.0

        x = np.expand_dims(normalized, axis=-1)   # (48,48,1)
        x = np.expand_dims(x, axis=0)             # (1,48,48,1)

        logits = self.model.predict(x, verbose=0)[0]

        # 模型输出为 logits（无 softmax），手动计算概率
        logits_stable = logits - np.max(logits)
        probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = labels[idx] if idx < len(labels) else str(idx)

        return {"label": label, "confidence": confidence}

    # ------------------------------------------------------------------
    # DeepFace 预测
    # ------------------------------------------------------------------

    def _predict_with_deepface(self, face_bgr):
        """
        使用 DeepFace 分析表情。
        DeepFace 返回的表情名称是小写英文，需要映射到项目统一标签。
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # DeepFace.analyze 可能返回 dict 或 list[dict]
        result = self.DeepFace.analyze(
            img_path=face_rgb,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        if isinstance(result, list):
            result = result[0]

        emotion_scores = result.get("emotion", {})
        dominant = result.get("dominant_emotion", "neutral")

        # DeepFace 小写标签 → 项目标准标签
        mapping = {
            "happy": "Happy",
            "neutral": "Neutral",
            "sad": "Sad",
            "angry": "Angry",
            "surprise": "Surprise",
            "fear": "Fear",
            "disgust": "Disgust",
        }

        label = mapping.get(dominant.lower(), "Neutral")
        # DeepFace 返回的是 0-100 的百分比，转为 0-1
        confidence = float(emotion_scores.get(dominant, 0.0)) / 100.0

        return {"label": label, "confidence": confidence}
