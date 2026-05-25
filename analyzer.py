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


def add_warning_levels(frame_df, window_size=5):
    """
    对逐帧分析 DataFrame 添加滑动窗口指标和三级预警等级。

    参数:
        frame_df: 逐帧分析 DataFrame，至少需包含以下列：
            Happy_ratio, Neutral_ratio, Sad_ratio, Angry_ratio, Surprise_ratio
        window_size: 滑动窗口大小（帧数），默认 5

    返回:
        添加了以下新列的 DataFrame：
        - positive_ratio: 积极表情占比 (Happy + Neutral)
        - low_ratio: 低落表情占比 (Sad + Angry)
        - attention_wave_ratio: 注意力波动占比 (Surprise)
        - window_positive_mean: 滑动窗口积极占比均值
        - window_low_mean: 滑动窗口低落占比均值
        - window_neutral_mean: 滑动窗口中性占比均值
        - warning_level: 预警等级 (Green / Yellow / Red / None)

    三级预警规则：
        🔴 Red:   连续 ≥ 3 帧低落或注意力波动
        🟡 Yellow: 连续 ≥ 5 帧中性占比过高（≥ 60%）
        🟢 Green:  连续 ≥ 10 帧状态良好（积极占比 ≥ 70%）
    """
    import pandas as pd

    if frame_df.empty:
        return frame_df

    df = frame_df.copy()

    # ---- 计算组合指标 ----
    happy = df.get("Happy_ratio", pd.Series([0] * len(df)))
    neutral = df.get("Neutral_ratio", pd.Series([0] * len(df)))
    sad = df.get("Sad_ratio", pd.Series([0] * len(df)))
    angry = df.get("Angry_ratio", pd.Series([0] * len(df)))
    surprise = df.get("Surprise_ratio", pd.Series([0] * len(df)))

    df["positive_ratio"] = happy + neutral
    df["low_ratio"] = sad + angry
    df["attention_wave_ratio"] = surprise

    # ---- 滑动窗口均值 ----
    df["window_positive_mean"] = (
        df["positive_ratio"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    df["window_low_mean"] = (
        df["low_ratio"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    df["window_neutral_mean"] = (
        neutral
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    # ---- 三级预警（基于连续帧计数）----
    df["warning_level"] = "None"

    good_streak = 0      # 连续良好帧数
    neutral_streak = 0   # 连续中性偏高帧数
    low_streak = 0       # 连续低落/波动帧数

    for idx in df.index:
        current_positive = df.loc[idx, "positive_ratio"]
        current_neutral = df.loc[idx, "Neutral_ratio"]
        current_low = df.loc[idx, "low_ratio"]
        current_surprise = df.loc[idx, "attention_wave_ratio"]

        # 积极占比 ≥ 70% → 良好帧
        if current_positive >= 0.70:
            good_streak += 1
        else:
            good_streak = 0

        # 中性占比 ≥ 60% → 中性偏高帧
        if current_neutral >= 0.60:
            neutral_streak += 1
        else:
            neutral_streak = 0

        # 低落 ≥ 40% 或 惊讶 ≥ 30% → 低落/波动帧
        if current_low >= 0.40 or current_surprise >= 0.30:
            low_streak += 1
        else:
            low_streak = 0

        # 优先级: Red > Yellow > Green
        if low_streak >= 3:
            df.loc[idx, "warning_level"] = "Red"
        elif neutral_streak >= 5:
            df.loc[idx, "warning_level"] = "Yellow"
        elif good_streak >= 10:
            df.loc[idx, "warning_level"] = "Green"
        else:
            df.loc[idx, "warning_level"] = "None"

    return df


def summarize_warnings(df):
    """
    汇总预警信息。

    参数:
        df: 经 add_warning_levels 处理后的 DataFrame

    返回:
        {
            "green_count": 绿色预警帧数,
            "yellow_count": 黄色预警帧数,
            "red_count": 红色预警帧数,
            "total_warnings": 有预警的总帧数,
            "max_warning": 最高预警等级 (Red > Yellow > Green),
            "summary_text": 预警摘要文字,
        }
    """
    if df.empty:
        return {
            "green_count": 0,
            "yellow_count": 0,
            "red_count": 0,
            "total_warnings": 0,
            "max_warning": "None",
            "summary_text": "无足够数据用于预警分析。",
        }

    green_count = int((df["warning_level"] == "Green").sum())
    yellow_count = int((df["warning_level"] == "Yellow").sum())
    red_count = int((df["warning_level"] == "Red").sum())
    total_warnings = green_count + yellow_count + red_count

    # 最高预警等级
    if red_count > 0:
        max_warning = "Red"
        summary_text = (
            f"🔴 视频中出现 {red_count} 帧低落或注意力波动片段，"
            "建议重点关注该时段课堂情况。"
        )
    elif yellow_count > 0:
        max_warning = "Yellow"
        summary_text = (
            f"🟡 视频中存在 {yellow_count} 帧中性占比偏高片段，"
            "课堂状态较平稳但互动可能不足。"
        )
    elif green_count > 0:
        max_warning = "Green"
        summary_text = (
            f"🟢 视频中存在 {green_count} 帧连续状态良好的片段，"
            "课堂整体表现积极。"
        )
    else:
        max_warning = "None"
        summary_text = "未触发明显预警，各帧状态波动在正常范围内。"

    return {
        "green_count": green_count,
        "yellow_count": yellow_count,
        "red_count": red_count,
        "total_warnings": total_warnings,
        "max_warning": max_warning,
        "summary_text": summary_text,
    }
