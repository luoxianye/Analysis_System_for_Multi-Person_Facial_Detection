# analyzer.py

from collections import Counter
from config import (
    EMOTION_LABELS,
    GOOD_THRESHOLD,
    LOW_THRESHOLD,
    SURPRISE_THRESHOLD,
    HIGH_NEUTRAL_THRESHOLD,
)


def summarize_expressions(detections):
    """
    对所有人脸检测结果进行表情汇总统计。

    参数:
        detections: [
            {"box": (x,y,w,h), "emotion": "Happy", "confidence": 0.9},
            ...
        ]

    返回:
        {
            "total": 总人数,
            "counts": {"Happy": 2, "Neutral": 3, ...},
            "ratios": {"Happy": 0.33, "Neutral": 0.50, ...},
            "main_expression": "Neutral",
            "status": "课堂状态良好",
        }
    """
    total = len(detections)
    counter = Counter()

    for item in detections:
        emotion = item.get("emotion", "Unknown")
        counter[emotion] += 1

    # 按标准标签顺序构建计数
    counts = {label: counter.get(label, 0) for label in EMOTION_LABELS}

    # 计算比例
    if total > 0:
        ratios = {label: counts[label] / total for label in EMOTION_LABELS}
        main_expression = max(counts, key=counts.get)
    else:
        ratios = {label: 0.0 for label in EMOTION_LABELS}
        main_expression = "None"

    # 课堂状态判断
    status = judge_classroom_status(total, counts, ratios, main_expression)

    return {
        "total": total,
        "counts": counts,
        "ratios": ratios,
        "main_expression": main_expression,
        "status": status,
    }


def judge_classroom_status(
    total,
    counts,
    ratios,
    main_expression,
    good_threshold=None,
    low_threshold=None,
    surprise_threshold=None,
    high_neutral_threshold=None,
):
    """
    根据表情比例判断课堂状态。

    规则说明：
    - 低落表情（Sad + Angry）占比 ≥ low_threshold → "课堂状态较低落或需要关注"
    - 惊讶占比 ≥ surprise_threshold → "课堂注意力波动较大"
    - 积极表情（Happy + Neutral）占比 ≥ good_threshold → "课堂状态良好"
    - 中性占比 ≥ high_neutral_threshold → "课堂状态平稳"
    - 否则 → "课堂状态一般"
    - 无人时 → "未检测到学生"

    阈值可通过参数自定义，默认使用 config.py 中的全局值。
    """
    if good_threshold is None:
        good_threshold = GOOD_THRESHOLD
    if low_threshold is None:
        low_threshold = LOW_THRESHOLD
    if surprise_threshold is None:
        surprise_threshold = SURPRISE_THRESHOLD
    if high_neutral_threshold is None:
        high_neutral_threshold = HIGH_NEUTRAL_THRESHOLD

    if total == 0:
        return "未检测到学生"

    positive_ratio = ratios.get("Happy", 0) + ratios.get("Neutral", 0)
    low_ratio = ratios.get("Sad", 0) + ratios.get("Angry", 0)
    surprise_ratio = ratios.get("Surprise", 0)
    neutral_ratio = ratios.get("Neutral", 0)

    if low_ratio >= low_threshold:
        return "课堂状态较低落或需要关注"

    if surprise_ratio >= surprise_threshold:
        return "课堂注意力波动较大"

    if positive_ratio >= good_threshold:
        return "课堂状态良好"

    if main_expression == "Neutral" or neutral_ratio >= high_neutral_threshold:
        return "课堂状态平稳"

    return "课堂状态一般"


def build_stats_dataframe(summary):
    """
    将统计结果转换为 pandas DataFrame，方便展示和导出。
    """
    import pandas as pd

    rows = []
    for label in EMOTION_LABELS:
        rows.append({
            "表情": label,
            "人数": summary["counts"].get(label, 0),
            "比例": f"{summary['ratios'].get(label, 0) * 100:.1f}%",
        })
    return pd.DataFrame(rows)
