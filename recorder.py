# recorder.py

from datetime import datetime
from pathlib import Path
import pandas as pd
from config import RECORD_FILE, EMOTION_LABELS


def build_record(image_name, summary, result_path=None):
    """
    根据单次检测的统计摘要，构造一条 CSV 记录。

    参数:
        image_name: 原始文件名
        summary: summarize_expressions() 的返回值
        result_path: 保存的结果图片路径（可选）

    返回:
        dict 类型的单条记录（与 build_video_record 列结构一致）
    """
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_type": "image",
        "source_name": image_name,
        "total_people": summary["total"],
        "main_expression": summary["main_expression"],
        "classroom_status": summary["status"],
        "result_path": result_path or "",
        # 视频专用字段，图片记录填空
        "video_duration_sec": "",
        "sampled_frames": "",
        "total_face_samples": "",
        "avg_people_per_frame": "",
        "max_people_per_frame": "",
    }

    for label in EMOTION_LABELS:
        record[f"{label}_count"] = summary["counts"].get(label, 0)
        record[f"{label}_ratio"] = round(summary["ratios"].get(label, 0.0), 4)

    return record


def append_record(image_name, summary, result_path=None, file_path=None):
    """
    追加保存一条检测记录到 CSV 文件。

    参数:
        image_name: 原始文件名
        summary: summarize_expressions() 的返回值
        result_path: 保存的结果图片路径
        file_path: CSV 文件路径，默认使用 config.RECORD_FILE
    """
    if file_path is None:
        file_path = RECORD_FILE

    file_path = Path(file_path)
    record = build_record(image_name, summary, result_path)
    df = pd.DataFrame([record])

    if file_path.exists():
        df.to_csv(
            file_path, mode="a", index=False, header=False, encoding="utf-8-sig"
        )
    else:
        df.to_csv(
            file_path, mode="w", index=False, header=True, encoding="utf-8-sig"
        )


def load_records(file_path=None):
    """
    加载历史检测记录。

    参数:
        file_path: CSV 文件路径，默认使用 config.RECORD_FILE

    返回:
        pandas DataFrame，如果文件不存在或读取失败则返回空 DataFrame
    """
    if file_path is None:
        file_path = RECORD_FILE

    file_path = Path(file_path)

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except pd.errors.ParserError:
        # CSV 列数不一致时（如旧版图片记录和视频记录混合），
        # 使用 on_bad_lines='skip' 跳过有问题的行
        return pd.read_csv(
            file_path, encoding="utf-8-sig", on_bad_lines="skip"
        )


def build_video_record(video_name, total_summary, frame_count, csv_path=None):
    """
    根据视频整体分析结果，构造一条 CSV 记录。

    参数:
        video_name: 原始视频文件名
        total_summary: analyze_video 返回的 total_summary
        frame_count: 抽帧总数
        csv_path: 保存的逐帧 CSV 路径（可选）

    返回:
        dict 类型的单条记录
    """
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_type": "video",
        "source_name": f"[视频] {video_name}",
        "total_people": total_summary.get("total_face_samples", total_summary["total"]),
        "main_expression": total_summary["main_expression"],
        "classroom_status": total_summary["status"],
        "result_path": csv_path or "",
        "video_duration_sec": total_summary.get("video_duration_sec", 0),
        "sampled_frames": frame_count,
        "total_face_samples": total_summary.get("total_face_samples", total_summary["total"]),
        "avg_people_per_frame": total_summary.get("avg_people_per_frame", 0),
        "max_people_per_frame": total_summary.get("max_people_per_frame", 0),
    }

    for label in EMOTION_LABELS:
        record[f"{label}_count"] = total_summary["counts"].get(label, 0)
        record[f"{label}_ratio"] = round(
            total_summary["ratios"].get(label, 0.0), 4
        )

    return record


def append_video_record(
    video_name, total_summary, frame_count, csv_path=None, file_path=None
):
    """
    追加保存一条视频分析记录到 CSV 文件。

    参数:
        video_name: 原始视频文件名
        total_summary: analyze_video 返回的 total_summary
        frame_count: 抽帧总数
        csv_path: 保存的逐帧 CSV 路径
        file_path: CSV 文件路径，默认使用 config.RECORD_FILE
    """
    if file_path is None:
        file_path = RECORD_FILE

    file_path = Path(file_path)
    record = build_video_record(video_name, total_summary, frame_count, csv_path)
    df = pd.DataFrame([record])

    if file_path.exists():
        df.to_csv(
            file_path, mode="a", index=False, header=False, encoding="utf-8-sig"
        )
    else:
        df.to_csv(
            file_path, mode="w", index=False, header=True, encoding="utf-8-sig"
        )


def delete_records(file_path=None):
    """
    清空历史记录（删除 CSV 文件）。
    """
    if file_path is None:
        file_path = RECORD_FILE

    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
        return True
    return False
