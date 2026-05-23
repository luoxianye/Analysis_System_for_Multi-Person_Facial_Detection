# webcam_demo.py
#
# 摄像头实时人脸检测与表情识别演示脚本。
# 用法：python webcam_demo.py
# 按键：
#   q — 退出
#   s — 保存当前帧截图到 results/images/
#
# 本脚本独立于 Streamlit 运行，属于迭代 9（加分项）。

import cv2
import time
from pathlib import Path
from datetime import datetime

from face_detector import FaceDetector
from expression_recognizer import ExpressionRecognizer
from utils import crop_face
from analyzer import summarize_expressions
from config import IMAGE_RESULT_DIR, EMOTION_CN


# 状态文字对应的显示颜色 (BGR)
STATUS_COLORS = {
    "课堂状态良好": (0, 255, 0),            # 绿色
    "课堂状态平稳": (255, 255, 0),           # 青色
    "课堂状态一般": (0, 255, 255),           # 黄色
    "课堂注意力波动较大": (0, 165, 255),      # 橙色
    "课堂状态较低落或需要关注": (0, 0, 255),   # 红色
    "未检测到学生": (128, 128, 128),          # 灰色
}


def get_status_color(status_text):
    """根据课堂状态文字返回对应的 BGR 颜色。"""
    return STATUS_COLORS.get(status_text, (255, 255, 255))


def main():
    print("=" * 60)
    print("  课堂状态分析系统 — 摄像头实时识别演示")
    print("=" * 60)
    print()
    print("正在初始化模型...")

    # ---- 加载模型 ----
    detector = FaceDetector(
        min_detection_confidence=0.5,
        model_selection=0,  # 短距离模型，适合摄像头近距离人脸
    )
    print("[OK] 人脸检测器加载完成（MediaPipe）")

    recognizer = ExpressionRecognizer(
        backend="hsemotion_onnx",
        model_name="enet_b0_8_best_vgaf",
    )
    print("[OK] 表情识别器加载完成（HSEmotionONNX）")

    # ---- 打开摄像头 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 摄像头打开失败！请检查摄像头是否可用。")
        return

    print("[OK] 摄像头已打开")
    print()
    print("操作提示：")
    print("  按 Q 键 — 退出程序")
    print("  按 S 键 — 保存当前帧截图")
    print()

    # FPS 计算变量
    prev_time = time.time()
    fps_display = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取摄像头帧失败，正在重试...")
            continue

        frame_count += 1

        # ---- 镜像翻转（更自然的自拍视角）----
        frame = cv2.flip(frame, 1)

        # ---- 人脸检测 ----
        detections = detector.detect(frame)

        # ---- 表情识别 ----
        for item in detections:
            face_img = crop_face(frame, item["box"])
            pred = recognizer.predict(face_img)
            item["emotion"] = pred["label"]
            item["confidence"] = pred["confidence"]

        # ---- 统计与状态判断 ----
        summary = summarize_expressions(detections)

        # ---- 绘制结果 ----
        output = detector.draw_results(frame, detections)

        # 在左上角显示课堂状态
        status_text = f"Status: {summary['status']}"
        status_color = get_status_color(summary["status"])
        cv2.putText(
            output,
            status_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # 显示人数和主要表情
        info_text = (
            f"People: {summary['total']}  |  "
            f"Main: {EMOTION_CN.get(summary['main_expression'], summary['main_expression'])}"
        )
        cv2.putText(
            output,
            info_text,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 右上角显示 FPS
        fps_text = f"FPS: {fps_display:.1f}"
        h, w = output.shape[:2]
        cv2.putText(
            output,
            fps_text,
            (w - 140, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 底部显示操作提示
        cv2.putText(
            output,
            "Q: Quit  |  S: Save Screenshot",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # ---- 显示 ----
        cv2.imshow("Classroom Emotion — Webcam Demo", output)

        # ---- FPS 计算 ----
        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / elapsed)
        prev_time = curr_time

        # ---- 按键处理 ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("\n[INFO] 用户退出。")
            break

        if key == ord("s") or key == ord("S"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = IMAGE_RESULT_DIR / f"webcam_{timestamp}.jpg"
            cv2.imwrite(str(save_path), output)
            print(f"[SAVE] 截图已保存：{save_path.name}")

    # ---- 清理 ----
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 程序结束。共处理 {frame_count} 帧。")


if __name__ == "__main__":
    main()
