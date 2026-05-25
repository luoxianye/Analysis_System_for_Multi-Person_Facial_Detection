# visualization.py

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from config import EMOTION_LABELS, EMOTION_CN

# 设置中文字体，避免中文乱码
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 表情对应的颜色方案
EMOTION_COLORS = {
    "Happy": "#FFD700",      # 金色
    "Neutral": "#90EE90",    # 浅绿
    "Sad": "#6495ED",        # 矢车菊蓝
    "Angry": "#FF6B6B",      # 珊瑚红
    "Surprise": "#FF69B4",   # 热粉
    "Fear": "#9370DB",       # 中紫
    "Disgust": "#20B2AA",    # 浅海绿
    "Contempt": "#A9A9A9",   # 深灰
}


def _get_labels_and_values(summary):
    """
    从 summary 中提取有序的表情标签和对应数值。
    返回 labels_cn, counts, ratios。
    """
    labels_cn = []
    counts = []
    ratios = []

    for label_en in EMOTION_LABELS:
        labels_cn.append(EMOTION_CN.get(label_en, label_en))
        counts.append(summary["counts"].get(label_en, 0))
        ratios.append(summary["ratios"].get(label_en, 0.0))

    return labels_cn, counts, ratios


def _get_colors():
    """返回按 EMOTION_LABELS 顺序的颜色列表。"""
    return [EMOTION_COLORS.get(label, "#CCCCCC") for label in EMOTION_LABELS]


def plot_expression_bar(summary):
    """
    表情人数柱状图。
    每根柱子代表一种表情的人数。
    """
    labels_cn, counts, _ = _get_labels_and_values(summary)
    colors = _get_colors()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels_cn, counts, color=colors, edgecolor="white", linewidth=0.8)

    # 在柱子上标注数值
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("表情类别", fontsize=12)
    ax.set_ylabel("人数", fontsize=12)
    ax.set_title(f"表情人数分布（共 {summary['total']} 人）", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, max(counts) + max(1, max(counts) * 0.2) if counts else 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


def plot_expression_ratio_bar(summary):
    """
    表情比例柱状图。
    每根柱子代表一种表情的占比（0-1）。
    """
    labels_cn, _, ratios = _get_labels_and_values(summary)
    colors = _get_colors()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels_cn, ratios, color=colors, edgecolor="white", linewidth=0.8)

    # 在柱子上标注百分比
    for bar, ratio in zip(bars, ratios):
        if ratio > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{ratio * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("表情类别", fontsize=12)
    ax.set_ylabel("比例", fontsize=12)
    ax.set_title("表情比例分布", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    return fig


def plot_expression_pie(summary):
    """
    表情比例饼图。
    只显示人数 > 0 的表情类别。
    """
    labels_cn, counts, _ = _get_labels_and_values(summary)
    colors = _get_colors()

    # 过滤掉人数为 0 的类别
    filtered = [
        (label, count, color)
        for label, count, color in zip(labels_cn, counts, colors)
        if count > 0
    ]

    if not filtered:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=16)
        ax.set_title("表情比例饼图", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    pie_labels, pie_counts, pie_colors = zip(*filtered)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        pie_counts,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title(f"表情比例饼图（共 {summary['total']} 人）", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_emotion_summary_dashboard(summary):
    """
    综合仪表盘：同时展示柱状图（左）和饼图（右）。
    返回一个包含两个子图的 figure。
    """
    labels_cn, counts, ratios = _get_labels_and_values(summary)
    colors = _get_colors()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左：柱状图（人数）
    bars = ax1.bar(labels_cn, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax1.set_title(f"表情人数分布（共 {summary['total']} 人）", fontsize=13, fontweight="bold")
    ax1.set_xlabel("表情类别")
    ax1.set_ylabel("人数")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_ylim(0, max(counts) + max(1, max(counts) * 0.25) if counts else 1)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 右：饼图
    filtered = [
        (label, count, color)
        for label, count, color in zip(labels_cn, counts, colors)
        if count > 0
    ]
    if filtered:
        pie_labels, pie_counts, pie_colors = zip(*filtered)
        wedges, texts, autotexts = ax2.pie(
            pie_counts,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")
    else:
        ax2.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=16)
    ax2.set_title(f"表情比例分布", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"课堂表情分析仪表盘 — 主要表情：{EMOTION_CN.get(summary['main_expression'], summary['main_expression'])}  |  "
        f"课堂状态：{summary['status']}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    return fig



def plot_timeline(df):
    """
    绘制时序表情比例折线图。

    参数:
        df: 逐帧分析 DataFrame，需包含 video_time_sec 或 frame_no 列，
            以及各表情 _ratio 列

    返回:
        matplotlib Figure 对象
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "无时序数据", ha="center", va="center", fontsize=16)
        ax.set_title("课堂表情时序分析", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    # 时间轴
    if "video_time_sec" in df.columns:
        x = df["video_time_sec"]
        x_label = "视频时间（秒）"
    else:
        x = df["frame_no"]
        x_label = "帧序号"

    fig, ax = plt.subplots(figsize=(14, 6))

    # 需要绘制的表情比例列（颜色对应）
    ratio_cols = {
        "Happy_ratio": ("开心", "#FFD700"),
        "Neutral_ratio": ("平静", "#90EE90"),
        "Sad_ratio": ("悲伤", "#6495ED"),
        "Angry_ratio": ("生气", "#FF6B6B"),
        "Surprise_ratio": ("惊讶", "#FF69B4"),
    }

    for col, (label_cn, color) in ratio_cols.items():
        if col in df.columns:
            ax.plot(
                x, df[col],
                marker="o", markersize=4,
                linewidth=1.5, color=color,
                label=label_cn, alpha=0.85,
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("表情比例", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("课堂表情时序折线图", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_timeline_with_warnings(df):
    """
    绘制带预警标记的时序表情比例折线图。

    参数:
        df: 经 add_warning_levels 处理后的 DataFrame

    返回:
        matplotlib Figure 对象
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.text(0.5, 0.5, "无时序数据", ha="center", va="center", fontsize=16)
        ax.set_title("课堂表情时序分析（含预警标记）", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    # 时间轴
    if "video_time_sec" in df.columns:
        x = df["video_time_sec"]
        x_label = "视频时间（秒）"
    else:
        x = df["frame_no"]
        x_label = "帧序号"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ---- 上图：表情比例折线 ----
    ratio_cols = {
        "Happy_ratio": ("开心", "#FFD700"),
        "Neutral_ratio": ("平静", "#90EE90"),
        "Sad_ratio": ("悲伤", "#6495ED"),
        "Angry_ratio": ("生气", "#FF6B6B"),
        "Surprise_ratio": ("惊讶", "#FF69B4"),
    }

    for col, (label_cn, color) in ratio_cols.items():
        if col in df.columns:
            ax1.plot(
                x, df[col],
                marker="o", markersize=3,
                linewidth=1.5, color=color,
                label=label_cn, alpha=0.85,
            )

    ax1.set_ylabel("表情比例", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("课堂表情时序折线图（含预警标记）", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- 下图：预警等级色带 ----
    warning_colors = {
        "Green": "#4CAF50",
        "Yellow": "#FFC107",
        "Red": "#F44336",
        "None": "#E0E0E0",
    }

    if "warning_level" in df.columns:
        # 逐段绘制色带
        for i in range(len(df) - 1):
            color = warning_colors.get(
                df.loc[df.index[i], "warning_level"], "#E0E0E0"
            )
            ax2.fill_between(
                [x.iloc[i], x.iloc[i + 1]],
                0, 1,
                color=color, alpha=0.7,
            )

        # 最后一个点
        if len(df) > 0:
            last_color = warning_colors.get(
                df.loc[df.index[-1], "warning_level"], "#E0E0E0"
            )
            ax2.axvline(x=x.iloc[-1], color=last_color, linewidth=3, alpha=0.7)

    ax2.set_ylabel("预警等级", fontsize=12)
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Green（连续良好）"),
        Patch(facecolor="#FFC107", label="Yellow（中性偏高）"),
        Patch(facecolor="#F44336", label="Red（低落/波动）"),
        Patch(facecolor="#E0E0E0", label="None（无预警）"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=4)

    fig.tight_layout()
    return fig


def plot_warning_summary_chart(warning_summary):
    """
    绘制预警统计汇总图（柱状图）。

    参数:
        warning_summary: summarize_warnings() 的返回值

    返回:
        matplotlib Figure 对象
    """
    labels = ["Green", "Yellow", "Red"]
    counts = [
        warning_summary["green_count"],
        warning_summary["yellow_count"],
        warning_summary["red_count"],
    ]
    colors = ["#4CAF50", "#FFC107", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
            )

    ax.set_ylabel("预警帧数", fontsize=12)
    ax.set_title("三级预警统计", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) + max(1, max(counts) * 0.3) if any(counts) else 5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig
