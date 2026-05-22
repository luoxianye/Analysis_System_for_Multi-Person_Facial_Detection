# app.py

import streamlit as st
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime

from utils import uploaded_file_to_bgr, bgr_to_rgb, crop_face
from face_detector import FaceDetector
from expression_recognizer import ExpressionRecognizer
from analyzer import summarize_expressions, build_stats_dataframe
from recorder import append_record, append_video_record, load_records
from video_processor import (
    save_uploaded_video,
    analyze_video,
    records_to_dataframe,
    save_frame_records_csv,
)
from visualization import (
    plot_expression_bar,
    plot_expression_ratio_bar,
    plot_expression_pie,
    plot_emotion_summary_dashboard,
)
from config import (
    IMAGE_RESULT_DIR,
    EMOTION_LABELS,
    EMOTION_CN,
    GOOD_THRESHOLD,
    LOW_THRESHOLD,
    SURPRISE_THRESHOLD,
    HIGH_NEUTRAL_THRESHOLD,
)

st.set_page_config(page_title="课堂状态分析系统", layout="wide")
st.title("基于多人人脸检测与表情识别的课堂状态分析系统")

# ---------- 侧边栏 ----------
st.sidebar.title("⚙️ 功能设置")

mode = st.sidebar.radio(
    "选择输入方式",
    ["图片分析", "视频分析（加分项）", "摄像头实时识别（加分项）"],
)

# 状态判断阈值调节
st.sidebar.subheader("课堂状态阈值调节")
good_threshold = st.sidebar.slider(
    "状态良好阈值 (Happy+Neutral 占比)",
    0.50, 1.00, GOOD_THRESHOLD, 0.05,
)
low_threshold = st.sidebar.slider(
    "低落关注阈值 (Sad+Angry 占比)",
    0.10, 0.80, LOW_THRESHOLD, 0.05,
)
surprise_threshold = st.sidebar.slider(
    "注意力波动阈值 (Surprise 占比)",
    0.10, 0.80, SURPRISE_THRESHOLD, 0.05,
)
high_neutral_threshold = st.sidebar.slider(
    "平稳判定阈值 (Neutral 占比)",
    0.30, 0.90, HIGH_NEUTRAL_THRESHOLD, 0.05,
)

# ---------- 缓存模型加载 ----------
@st.cache_resource
def load_detector():
    return FaceDetector()


@st.cache_resource
def load_recognizer():
    # 优先使用本地 Keras 模型，否则回退到 DeepFace
    model_path = "models/emotion_model.h5"
    if Path(model_path).exists():
        return ExpressionRecognizer(model_path=model_path, backend="keras")
    return ExpressionRecognizer(backend="deepface")


detector = load_detector()
recognizer = load_recognizer()

# ---------- 核心分析函数 ----------
def analyze_image(image_bgr):
    """
    对整张图片执行：人脸检测 → 表情识别 → 统计汇总 → 绘制结果。
    返回 detections, result_bgr, summary。
    """
    detections = detector.detect(image_bgr)

    for item in detections:
        face_img = crop_face(image_bgr, item["box"])
        pred = recognizer.predict(face_img)
        item["emotion"] = pred["label"]
        item["confidence"] = pred["confidence"]

    result_bgr = detector.draw_results(image_bgr, detections)
    summary = summarize_expressions(detections)

    # 用侧边栏阈值重新判断状态
    from analyzer import judge_classroom_status
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

    return detections, result_bgr, summary


# ====================================================================
# 图片分析模式
# ====================================================================
if mode == "图片分析":
    uploaded_file = st.file_uploader(
        "📁 上传课堂图片", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_bgr = uploaded_file_to_bgr(uploaded_file)

        with st.spinner("正在进行人脸检测与表情识别，请稍候..."):
            detections, result_bgr, summary = analyze_image(image_bgr)

        # ---------- 图片对比展示 ----------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 原始图片")
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)

        with col2:
            st.subheader("🔍 检测与识别结果")
            st.image(bgr_to_rgb(result_bgr), use_container_width=True)

        # ---------- 核心指标 ----------
        st.subheader("📊 课堂状态分析结果")

        c1, c2, c3 = st.columns(3)
        c1.metric("👥 检测人数", summary["total"])
        c2.metric("😶 主要表情", summary["main_expression"])
        c3.metric("📋 课堂状态", summary["status"])

        # ---------- 表情统计表 ----------
        st.subheader("📈 表情分布统计")
        stats_df = build_stats_dataframe(summary)
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
        )

        # ---------- 每人表情详情 ----------
        if detections:
            with st.expander("🔎 每人表情识别详情"):
                rows = []
                for i, item in enumerate(detections):
                    label_en = item.get("emotion", "N/A")
                    label_cn = EMOTION_CN.get(label_en, label_en)
                    rows.append({
                        "序号": i + 1,
                        "人脸框": f"({item['box'][0]}, {item['box'][1]}, "
                                  f"{item['box'][2]}, {item['box'][3]})",
                        "表情(英文)": label_en,
                        "表情(中文)": label_cn,
                        "置信度": f"{item.get('confidence', 0):.2%}",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # ---------- 表情统计图表 ----------
        st.subheader("📊 表情统计图表")

        chart_tab1, chart_tab2, chart_tab3 = st.tabs(
            ["📊 综合仪表盘", "📈 柱状图", "🥧 饼图"]
        )

        with chart_tab1:
            fig_dashboard = plot_emotion_summary_dashboard(summary)
            st.pyplot(fig_dashboard)

        with chart_tab2:
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                st.subheader("人数分布")
                fig_bar = plot_expression_bar(summary)
                st.pyplot(fig_bar)
            with col_bar2:
                st.subheader("比例分布")
                fig_ratio = plot_expression_ratio_bar(summary)
                st.pyplot(fig_ratio)

        with chart_tab3:
            fig_pie = plot_expression_pie(summary)
            st.pyplot(fig_pie)

        # ---------- 保存与导出 ----------
        st.subheader("💾 结果保存与导出")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = IMAGE_RESULT_DIR / f"result_{timestamp}.jpg"
        cv2.imwrite(str(result_path), result_bgr)

        # 保存 CSV 记录
        append_record(
            image_name=uploaded_file.name,
            summary=summary,
            result_path=str(result_path),
        )

        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            # 下载本次结果图片
            with open(str(result_path), "rb") as f:
                st.download_button(
                    label="🖼️ 下载结果图片",
                    data=f.read(),
                    file_name=f"result_{timestamp}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

        with export_col2:
            # 下载本次分析 CSV
            record_df = build_stats_dataframe(summary)
            csv_data = record_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📄 下载本次统计 CSV",
                data=csv_data,
                file_name=f"analysis_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col3:
            # 下载全部历史记录
            records = load_records()
            if not records.empty:
                all_csv = records.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📚 下载全部历史记录",
                    data=all_csv,
                    file_name="records_all.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button(
                    "📚 暂无历史记录",
                    disabled=True,
                    use_container_width=True,
                )

        st.success(f"✅ 检测记录已保存。结果图片：`{result_path.name}`")

        # ---------- 历史记录 ----------
        st.subheader("📜 最近检测记录")
        if not records.empty:
            st.dataframe(
                records.tail(10),
                use_container_width=True,
            )
        else:
            st.info("暂无历史记录。")

        # ---------- 原始检测数据 ----------
        with st.expander("🧾 原始检测数据 (JSON)"):
            st.write(detections)

# ====================================================================
# 视频分析模式
# ====================================================================
elif mode == "视频分析（加分项）":
    uploaded_video = st.file_uploader(
        "🎬 上传课堂视频", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        interval_sec = st.slider("⏱ 抽帧间隔（秒）", 1, 5, 1)

        # 保存上传视频到临时文件
        video_path = save_uploaded_video(uploaded_video)

        with st.spinner("正在分析视频，请稍等..."):
            frame_records, total_summary = analyze_video(
                video_path,
                detector,
                recognizer,
                interval_sec=interval_sec,
                good_threshold=good_threshold,
                low_threshold=low_threshold,
                surprise_threshold=surprise_threshold,
                high_neutral_threshold=high_neutral_threshold,
            )

        # ---------- 视频整体统计 ----------
        st.subheader("📊 视频整体统计")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 总识别人数样本", total_summary["total"])
        c2.metric("😶 主要表情", total_summary["main_expression"])
        c3.metric("📋 整体课堂状态", total_summary["status"])
        c4.metric("🎞 抽帧数", total_summary.get("sampled_frames", 0))

        # 视频元信息
        with st.expander("📹 视频信息"):
            st.write(f"视频时长: {total_summary.get('video_duration_sec', 0):.1f} 秒")
            st.write(f"视频帧率: {total_summary.get('fps', 0):.1f} FPS")
            st.write(f"抽帧间隔: {interval_sec} 秒")
            st.write(f"实际抽帧数: {total_summary.get('sampled_frames', 0)}")

        # ---------- 表情分布统计表 ----------
        st.subheader("📈 视频整体表情分布")
        total_stats_df = build_stats_dataframe(total_summary)
        st.dataframe(total_stats_df, use_container_width=True, hide_index=True)

        # ---------- 表情统计图表 ----------
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("人数分布")
            fig_bar = plot_expression_bar(total_summary)
            st.pyplot(fig_bar)
        with chart_col2:
            st.subheader("比例分布")
            fig_pie = plot_expression_pie(total_summary)
            st.pyplot(fig_pie)

        # ---------- 逐帧分析记录 ----------
        st.subheader("🎞 逐帧分析记录")
        df = records_to_dataframe(frame_records)
        st.dataframe(df, use_container_width=True)

        # 绘制逐帧人数变化折线图
        if not df.empty:
            st.subheader("📉 逐帧人数变化趋势")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(
                df["video_time_sec"],
                df["total_people"],
                marker="o",
                linestyle="-",
                color="#4A90D9",
            )
            ax.set_xlabel("视频时间（秒）")
            ax.set_ylabel("检测人数")
            ax.set_title("逐帧检测人数变化")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # ---------- 保存与导出 ----------
        st.subheader("💾 结果保存与导出")

        # 保存逐帧 CSV
        csv_path = save_frame_records_csv(frame_records, uploaded_video.name)

        # 保存视频整体记录到 records.csv
        append_video_record(
            video_name=uploaded_video.name,
            total_summary=total_summary,
            frame_count=len(frame_records),
            csv_path=str(csv_path),
        )

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            # 下载逐帧分析 CSV
            csv_data = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📄 下载逐帧分析 CSV",
                data=csv_data,
                file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            # 下载全部历史记录
            records = load_records()
            if not records.empty:
                all_csv = records.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📚 下载全部历史记录",
                    data=all_csv,
                    file_name="records_all.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button(
                    "📚 暂无历史记录",
                    disabled=True,
                    use_container_width=True,
                )

        st.success(f"✅ 视频分析完成。逐帧记录已保存至: `{csv_path.name}`")

        # ---------- 历史记录 ----------
        st.subheader("📜 最近检测记录")
        records = load_records()
        if not records.empty:
            st.dataframe(records.tail(10), use_container_width=True)
        else:
            st.info("暂无历史记录。")

# ====================================================================
# 摄像头实时识别模式（占位，迭代 9 实现）
# ====================================================================
elif mode == "摄像头实时识别（加分项）":
    st.warning("⚠️ 摄像头实时识别功能将在后续迭代中实现。")
    st.info(
        "当前已完成功能：\n"
        "- ✅ 图片上传与分析\n"
        "- ✅ 视频上传与分析\n"
        "- ✅ 多人人脸检测\n"
        "- ✅ 表情识别\n"
        "- ✅ 表情分布统计\n"
        "- ✅ 课堂状态判断\n"
        "- ✅ CSV 记录保存\n"
        "- ✅ 历史记录查看\n"
        "- ✅ 逐帧趋势图"
    )