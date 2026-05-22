# config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
IMAGE_RESULT_DIR = RESULT_DIR / "images"
VIDEO_RESULT_DIR = RESULT_DIR / "videos"
RECORD_FILE = BASE_DIR / "records.csv"

IMAGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 标准 FER2013 7 类表情标签
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# 扩展 8 类标签（增加 Contempt，部分模型使用）
EMOTION_LABELS_8 = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
    "Contempt",
]

EMOTION_CN = {
    "Happy": "开心",
    "Neutral": "平静",
    "Sad": "悲伤",
    "Angry": "生气",
    "Surprise": "惊讶",
    "Fear": "害怕",
    "Disgust": "厌恶",
    "Contempt": "蔑视",
}

# 课堂状态判断阈值，可以后续调整
GOOD_THRESHOLD = 0.70
LOW_THRESHOLD = 0.40
SURPRISE_THRESHOLD = 0.30
HIGH_NEUTRAL_THRESHOLD = 0.60