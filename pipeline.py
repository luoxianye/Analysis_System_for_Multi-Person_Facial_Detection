# pipeline.py
from __future__ import annotations

from typing import Any

from analyzer import judge_classroom_status, summarize_expressions
from utils import crop_face


def analyze_image(
    image_bgr,
    detector,
    recognizer,
    good_threshold: float | None = None,
    low_threshold: float | None = None,
    surprise_threshold: float | None = None,
    high_neutral_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    单张图片分析主流程。

    步骤：
    1. 人脸检测
    2. 人脸裁剪
    3. 表情识别
    4. 多人表情统计
    5. 课堂状态判断
    """
    detections = detector.detect(image_bgr)

    for item in detections:
        face_img = crop_face(image_bgr, item["box"])
        pred = recognizer.predict(face_img)
        item["emotion"] = pred.get("label", "Unknown")
        item["confidence"] = float(pred.get("confidence", 0.0))

    summary = summarize_expressions(detections)

    summary["status"] = judge_classroom_status(
        summary["total"],
        summary["counts"],
        summary["ratios"],
        summary["main_expression"],
        good_threshold=good_threshold,
        low_threshold=low_threshold,
        surprise_threshold=surprise_threshold,
        high_neutral_threshold=high_neutral_threshold,
    )

    return detections, summary


def analyze_and_draw_image(
    image_bgr,
    detector,
    recognizer,
    good_threshold: float | None = None,
    low_threshold: float | None = None,
    surprise_threshold: float | None = None,
    high_neutral_threshold: float | None = None,
):
    """分析图片并直接返回画框后的结果图。"""
    detections, summary = analyze_image(
        image_bgr=image_bgr,
        detector=detector,
        recognizer=recognizer,
        good_threshold=good_threshold,
        low_threshold=low_threshold,
        surprise_threshold=surprise_threshold,
        high_neutral_threshold=high_neutral_threshold,
    )
    result_img = detector.draw_results(image_bgr, detections)
    return detections, summary, result_img
