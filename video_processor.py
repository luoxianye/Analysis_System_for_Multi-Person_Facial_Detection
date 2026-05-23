# video_processor.py

import cv2
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime

from analyzer import summarize_expressions, judge_classroom_status
from utils import crop_face
from config import EMOTION_LABELS, VIDEO_RESULT_DIR


def save_uploaded_video(uploaded_video):
    """
    Streamlit 上传的视频无法直接被 cv2.VideoCapture 稳定读取，
    先保存成临时文件。

    参数:
        uploaded_video: Streamlit UploadedFile 对象

    返回:
        临时文件的路径字符串
    """
    suffix = "." + uploaded_video.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        return tmp.name


def analyze_video(
    video_path,
    detector,
    recognizer,
    interval_sec=1,
    good_threshold=None,
    low_threshold=None,
    surprise_threshold=None,
    high_neutral_threshold=None,
):
    """
    按 interval_sec 秒抽一帧进行分析。

    参数:
        video_path: 视频文件路径
        detector: FaceDetector 实例
        recognizer: ExpressionRecognizer 实例
        interval_sec: 抽帧间隔（秒）
        good_threshold / low_threshold / surprise_threshold / high_neutral_threshold:
            课堂状态判断阈值，默认使用 config.py 中的全局值

    返回:
        frame_records: 每帧统计结果列表 (list of dict)
        total_summary: 全视频汇总结果 (dict)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("视频打开失败，请检查文件格式。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # 默认 25 FPS

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps if fps > 0 else 0

    frame_interval = int(fps * interval_sec)
    if frame_interval < 1:
        frame_interval = 1

    frame_index = 0
    sampled_index = 0

    frame_records = []
    all_detections = []  # 汇聚所有人脸用于整体统计
    people_counts = []   # 每帧人数，用于计算平均/最大人数

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_sec = frame_index / fps

            # ---- 人脸检测 ----
            detections = detector.detect(frame_bgr)

            # ---- 表情识别 ----
            for item in detections:
                face_img = crop_face(frame_bgr, item["box"])
                pred = recognizer.predict(face_img)
                item["emotion"] = pred["label"]
                item["confidence"] = pred["confidence"]

            # ---- 单帧统计 ----
            summary = summarize_expressions(detections)

            # 使用自定义阈值重新判断状态
            status = judge_classroom_status(
                summary["total"],
                summary["counts"],
                summary["ratios"],
                summary["main_expression"],
                good_threshold=good_threshold,
                low_threshold=low_threshold,
                surprise_threshold=surprise_threshold,
                high_neutral_threshold=high_neutral_threshold,
            )

            people_counts.append(summary["total"])

            record = {
                "frame_no": sampled_index,
                "video_time_sec": round(timestamp_sec, 2),
                "people_in_frame": summary["total"],
                "main_expression": summary["main_expression"],
                "classroom_status": status,
            }

            for label in EMOTION_LABELS:
                record[f"{label}_count"] = summary["counts"].get(label, 0)
                record[f"{label}_ratio"] = round(
                    summary["ratios"].get(label, 0.0), 4
                )

            frame_records.append(record)
            all_detections.extend(detections)

            sampled_index += 1

        frame_index += 1

    cap.release()

    # ---- 全视频汇总 ----
    total_summary = summarize_expressions(all_detections)

    # 用自定义阈值重新判断整体状态
    total_summary["status"] = judge_classroom_status(
        total_summary["total"],
        total_summary["counts"],
        total_summary["ratios"],
        total_summary["main_expression"],
        good_threshold=good_threshold,
        low_threshold=low_threshold,
        surprise_threshold=surprise_threshold,
        high_neutral_threshold=high_neutral_threshold,
    )

    # 计算视频统计指标
    sampled_frames = sampled_index
    total_face_samples = total_summary["total"]
    avg_people_per_frame = (
        round(sum(people_counts) / len(people_counts), 2) if people_counts else 0
    )
    max_people_per_frame = max(people_counts) if people_counts else 0

    # 附加视频元信息
    total_summary["video_duration_sec"] = round(video_duration_sec, 2)
    total_summary["sampled_frames"] = sampled_frames
    total_summary["fps"] = round(fps, 2)
    total_summary["total_face_samples"] = total_face_samples
    total_summary["avg_people_per_frame"] = avg_people_per_frame
    total_summary["max_people_per_frame"] = max_people_per_frame

    return frame_records, total_summary


def records_to_dataframe(frame_records):
    """
    将帧记录列表转换为 pandas DataFrame。

    参数:
        frame_records: analyze_video 返回的 frame_records

    返回:
        pandas DataFrame
    """
    return pd.DataFrame(frame_records)


def save_frame_records_csv(frame_records, video_name):
    """
    将视频逐帧分析记录保存为 CSV 文件。

    参数:
        frame_records: analyze_video 返回的 frame_records
        video_name: 原始视频文件名

    返回:
        保存的 CSV 文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = VIDEO_RESULT_DIR / f"video_analysis_{timestamp}.csv"
    df = records_to_dataframe(frame_records)
    df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    return csv_path
