# app.py

import streamlit as st
import pandas as pd
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from utils import uploaded_file_to_bgr, bgr_to_rgb, crop_face
from face_detector import FaceDetector
from expression_recognizer import ExpressionRecognizer
from analyzer import summarize_expressions, build_stats_dataframe, judge_classroom_status
from analyzer import add_warning_levels, summarize_warnings
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
    plot_timeline,
    plot_timeline_with_warnings,
    plot_warning_summary_chart,
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
    # 子模式：单张 / 批量
    image_mode = st.radio(
        "选择图片分析模式",
        ["🖼️ 单张分析", "📚 批量分析"],
        horizontal=True,
    )

    # ================================================================
    # 单张分析
    # ================================================================
    if image_mode == "🖼️ 单张分析":
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
            records = load_records()
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

    # ================================================================
    # 批量分析
    # ================================================================
    else:
        uploaded_files = st.file_uploader(
            "📁 批量上传课堂图片（可一次选择多张）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.info(f"已上传 {len(uploaded_files)} 张图片，点击下方按钮开始批量分析。")

            if st.button("🚀 开始批量分析", use_container_width=True, type="primary"):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(
                        f"正在分析第 {idx + 1}/{len(uploaded_files)} 张：{uploaded_file.name}..."
                    )
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                    image_bgr = uploaded_file_to_bgr(uploaded_file)
                    detections, result_bgr, summary = analyze_image(image_bgr)

                    # 保存结果图片
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_path = IMAGE_RESULT_DIR / f"batch_{idx + 1}_{timestamp}.jpg"
                    cv2.imwrite(str(result_path), result_bgr)

                    # 保存 CSV 记录
                    append_record(
                        image_name=uploaded_file.name,
                        summary=summary,
                        result_path=str(result_path),
                    )

                    # 收集统计信息
                    batch_results.append({
                        "序号": idx + 1,
                        "文件名": uploaded_file.name,
                        "检测人数": summary["total"],
                        "主要表情": EMOTION_CN.get(
                            summary["main_expression"], summary["main_expression"]
                        ),
                        "课堂状态": summary["status"],
                        "结果图片": str(result_path),
                        "开心人数": summary["counts"].get("Happy", 0),
                        "平静人数": summary["counts"].get("Neutral", 0),
                        "悲伤人数": summary["counts"].get("Sad", 0),
                        "生气人数": summary["counts"].get("Angry", 0),
                        "惊讶人数": summary["counts"].get("Surprise", 0),
                        "开心比例": f"{summary['ratios'].get('Happy', 0) * 100:.1f}%",
                        "平静比例": f"{summary['ratios'].get('Neutral', 0) * 100:.1f}%",
                    })

                progress_bar.progress(1.0)
                status_text.text("✅ 批量分析完成！")

                # ---------- 批量汇总表 ----------
                st.subheader("📊 批量分析汇总")

                batch_df = pd.DataFrame(batch_results)

                # 汇总指标
                total_people_all = batch_df["检测人数"].sum()
                total_images = len(batch_df)

                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("📚 分析图片数", total_images)
                bc2.metric("👥 累计检测人数", total_people_all)
                bc3.metric(
                    "📋 最常见状态",
                    batch_df["课堂状态"].mode().iloc[0]
                    if not batch_df["课堂状态"].mode().empty
                    else "N/A",
                )
                bc4.metric(
                    "📷 平均每张人数",
                    f"{total_people_all / total_images:.1f}"
                    if total_images > 0 else "0",
                )

                # 汇总表格
                st.dataframe(
                    batch_df.drop(columns=["结果图片"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True,
                )

                # ---------- 批量对比图表 ----------
                st.subheader("📈 批量对比分析")

                if total_images > 1:
                    import matplotlib.pyplot as plt

                    # 各图片人数对比柱状图
                    fig1, ax1 = plt.subplots(figsize=(12, 5))
                    short_names = [
                        name if len(name) <= 15 else name[:14] + "..."
                        for name in batch_df["文件名"]
                    ]
                    bars = ax1.bar(
                        range(len(batch_df)),
                        batch_df["检测人数"],
                        color="#4A90D9",
                        edgecolor="white",
                    )
                    for bar, val in zip(bars, batch_df["检测人数"]):
                        ax1.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.2,
                            str(val),
                            ha="center", va="bottom",
                            fontweight="bold",
                        )
                    ax1.set_xticks(range(len(batch_df)))
                    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
                    ax1.set_ylabel("检测人数")
                    ax1.set_title("各图片检测人数对比", fontsize=13, fontweight="bold")
                    ax1.set_ylim(0, max(batch_df["检测人数"]) * 1.2 + 1)
                    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    ax1.grid(axis="y", alpha=0.3)
                    fig1.tight_layout()
                    st.pyplot(fig1)

                    # 各图片课堂状态分布饼图
                    status_counts = batch_df["课堂状态"].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(7, 7))
                    ax2.pie(
                        status_counts.values,
                        labels=status_counts.index,
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=["#4CAF50", "#FFC107", "#F44336", "#90EE90", "#FF9800"],
                    )
                    ax2.set_title("各图片课堂状态分布", fontsize=13, fontweight="bold")
                    fig2.tight_layout()
                    st.pyplot(fig2)

                # ---------- 单张详情展开 ----------
                st.subheader("🔎 各图片详细结果")
                for idx, uploaded_file in enumerate(uploaded_files):
                    with st.expander(
                        f"📷 图片 {idx + 1}：{uploaded_file.name}"
                        f"（{batch_df.loc[idx, '检测人数']} 人，"
                        f"{batch_df.loc[idx, '课堂状态']}）"
                    ):
                        # 重新分析（或从缓存取）
                        image_bgr = uploaded_file_to_bgr(uploaded_file)
                        _, result_bgr, _ = analyze_image(image_bgr)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(
                                uploaded_file,
                                caption="原始图片",
                                use_container_width=True,
                            )
                        with col_b:
                            st.image(
                                bgr_to_rgb(result_bgr),
                                caption="检测结果",
                                use_container_width=True,
                            )

                # ---------- 导出 ----------
                st.subheader("💾 批量结果导出")

                export_c1, export_c2 = st.columns(2)
                with export_c1:
                    batch_csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="📄 下载批量分析汇总 CSV",
                        data=batch_csv,
                        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with export_c2:
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

                st.success(
                    f"✅ 批量分析完成！共处理 {total_images} 张图片，"
                    f"累计检测 {total_people_all} 人次。"
                )

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

        # ---------- 时序分析与三级预警 ----------
        st.subheader("🚨 时序分析与三级预警")

        if not df.empty:
            # 可调窗口大小
            warn_window = st.slider(
                "滑动窗口大小（帧）", 3, 15, 5, 2,
                help="窗口越大，预警判定越平滑；窗口越小，对变化越敏感",
            )
            df_warn = add_warning_levels(df, window_size=warn_window)
            warning_summary = summarize_warnings(df_warn)

            # 预警汇总指标
            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                st.metric("🟢 Green 预警帧", warning_summary["green_count"])
            with wc2:
                st.metric("🟡 Yellow 预警帧", warning_summary["yellow_count"])
            with wc3:
                st.metric("🔴 Red 预警帧", warning_summary["red_count"])

            # 预警摘要文本框
            if warning_summary["max_warning"] == "Red":
                st.error(warning_summary["summary_text"])
            elif warning_summary["max_warning"] == "Yellow":
                st.warning(warning_summary["summary_text"])
            elif warning_summary["max_warning"] == "Green":
                st.success(warning_summary["summary_text"])
            else:
                st.info(warning_summary["summary_text"])

            # 预警统计柱状图
            if warning_summary["total_warnings"] > 0:
                st.pyplot(plot_warning_summary_chart(warning_summary))

            # 时序折线图（带预警色带）
            st.subheader("📈 表情时序折线图（含预警标记）")
            st.pyplot(plot_timeline_with_warnings(df_warn))

            # 逐帧预警详情表
            with st.expander("🔎 逐帧预警详情"):
                # 选择关键列展示
                display_cols = [
                    "frame_no", "video_time_sec", "total_people",
                    "classroom_status", "warning_level",
                ]
                # 添加存在的比率列
                for col in [
                    "Happy_ratio", "Neutral_ratio", "Sad_ratio",
                    "Angry_ratio", "Surprise_ratio",
                    "positive_ratio", "low_ratio", "attention_wave_ratio",
                    "window_positive_mean", "window_low_mean", "window_neutral_mean",
                ]:
                    if col in df_warn.columns:
                        display_cols.append(col)

                display_cols = [c for c in display_cols if c in df_warn.columns]
                st.dataframe(
                    df_warn[display_cols],
                    use_container_width=True,
                )

            # 下载含预警的 CSV
            warn_csv = df_warn.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📄 下载含预警等级的完整 CSV",
                data=warn_csv,
                file_name=f"warning_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("无帧数据可供时序分析。")

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
# 摄像头实时识别模式
# ====================================================================
elif mode == "摄像头实时识别（加分项）":
    st.subheader("📹 摄像头实时识别")

    # 子模式选择
    webcam_mode = st.radio(
        "选择摄像头模式",
        ["📸 拍照分析", "🔴 实时预览"],
        horizontal=True,
    )

    # ================================================================
    # 模式 A：拍照分析（使用 Streamlit 原生 camera_input）
    # ================================================================
    if webcam_mode == "📸 拍照分析":
        st.info("点击下方摄像头画面中的 📷 按钮拍照，系统将自动分析。")

        camera_image = st.camera_input("", label_visibility="collapsed")

        if camera_image is not None:
            # 将拍照结果转为 BGR
            image_bgr = uploaded_file_to_bgr(camera_image)

            with st.spinner("正在分析..."):
                detections, result_bgr, summary = analyze_image(image_bgr)

            # ---- 结果展示 ----
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 原始照片")
                st.image(camera_image, use_container_width=True)
            with col2:
                st.subheader("🔍 检测与识别结果")
                st.image(bgr_to_rgb(result_bgr), use_container_width=True)

            # 核心指标
            st.subheader("📊 课堂状态分析结果")
            c1, c2, c3 = st.columns(3)
            c1.metric("👥 检测人数", summary["total"])
            c2.metric("😶 主要表情", summary["main_expression"])
            c3.metric("📋 课堂状态", summary["status"])

            # 表情统计表
            st.subheader("📈 表情分布统计")
            stats_df = build_stats_dataframe(summary)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # 表情统计图表
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_bar = plot_expression_bar(summary)
                st.pyplot(fig_bar)
            with chart_col2:
                fig_pie = plot_expression_pie(summary)
                st.pyplot(fig_pie)

            # 保存记录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = IMAGE_RESULT_DIR / f"webcam_{timestamp}.jpg"
            cv2.imwrite(str(result_path), result_bgr)
            append_record(
                image_name=f"webcam_{timestamp}.jpg",
                summary=summary,
                result_path=str(result_path),
            )
            st.success(f"✅ 分析完成，结果已保存至 `{result_path.name}`")

    # ================================================================
    # 模式 B：实时预览（OpenCV 循环采集 + Streamlit 占位刷新）
    # ================================================================
    else:
        st.info(
            "实时预览模式将调用摄像头持续采集画面，并在界面中逐帧刷新分析结果。"
        )

        # 可调节参数
        live_col1, live_col2 = st.columns(2)
        with live_col1:
            frame_skip = st.slider(
                "处理间隔（每隔 N 帧处理一次，越大越流畅）",
                1, 10, 3,
                help="数值越大，画面越流畅但检测更新越慢",
            )
        with live_col2:
            max_duration = st.slider(
                "最长运行时间（秒）",
                10, 120, 60, 10,
                help="到达最大时间后自动停止",
            )

        # 启动 / 重置按钮
        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = False

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(
                "🔴 开始实时预览" if not st.session_state.webcam_active else "🔄 重新开始",
                use_container_width=True,
            ):
                st.session_state.webcam_active = True
                st.rerun()
        with btn_col2:
            if st.button("⏹ 停止", use_container_width=True, disabled=not st.session_state.webcam_active):
                st.session_state.webcam_active = False
                st.rerun()

        if not st.session_state.webcam_active:
            st.info("👆 点击「开始实时预览」启动摄像头。")
        else:
            # 创建占位区域
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            stats_placeholder = st.empty()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ 无法打开摄像头，请检查设备连接。")
                st.session_state.webcam_active = False
            else:
                frame_idx = 0
                start_time = time.time()
                fps_display = 0.0
                prev_fps_time = start_time

                # 累计统计数据
                all_detections_for_summary = []

                # 初始状态（尚未检测到任何人脸时）
                detections = []
                summary = {
                    "total": 0,
                    "counts": {label: 0 for label in EMOTION_LABELS},
                    "ratios": {label: 0.0 for label in EMOTION_LABELS},
                    "main_expression": "None",
                    "status": "未检测到学生",
                }

                try:
                    while st.session_state.webcam_active:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            time.sleep(0.05)
                            continue

                        frame_idx += 1
                        # 镜像翻转
                        frame_bgr = cv2.flip(frame_bgr, 1)

                        # 每隔 frame_skip 帧做一次检测
                        if frame_idx % frame_skip == 0:
                            detections = detector.detect(frame_bgr)
                            for item in detections:
                                face_img = crop_face(frame_bgr, item["box"])
                                pred = recognizer.predict(face_img)
                                item["emotion"] = pred["label"]
                                item["confidence"] = pred["confidence"]

                            summary = summarize_expressions(detections)
                            # 用侧边栏阈值重判
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
                            all_detections_for_summary.extend(detections)
                        else:
                            # 复用上次检测结果
                            pass

                        # 绘制人脸框
                        output_bgr = detector.draw_results(frame_bgr, detections if frame_idx % frame_skip == 0 else [])

                        # 叠加状态文字
                        status_text = summary.get("status", "分析中...")
                        status_colors = {
                            "课堂状态良好": (0, 255, 0),
                            "课堂状态平稳": (255, 255, 0),
                            "课堂状态一般": (0, 255, 255),
                            "课堂注意力波动较大": (0, 165, 255),
                            "课堂状态较低落或需要关注": (0, 0, 255),
                            "未检测到学生": (128, 128, 128),
                        }
                        color = status_colors.get(status_text, (255, 255, 255))
                        cv2.putText(
                            output_bgr, f"Status: {status_text}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA,
                        )
                        cv2.putText(
                            output_bgr,
                            f"People: {summary.get('total', 0)}  |  "
                            f"Main: {EMOTION_CN.get(summary.get('main_expression', ''), summary.get('main_expression', ''))}",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA,
                        )

                        # FPS
                        now = time.time()
                        elapsed_fps = now - prev_fps_time
                        if elapsed_fps > 0:
                            fps_display = 0.9 * fps_display + 0.1 * (1.0 / elapsed_fps)
                        prev_fps_time = now
                        h, w = output_bgr.shape[:2]
                        cv2.putText(
                            output_bgr, f"FPS: {fps_display:.1f}",
                            (w - 140, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2, cv2.LINE_AA,
                        )

                        # 底部提示
                        cv2.putText(
                            output_bgr,
                            f"Frame: {frame_idx}  |  Elapsed: {now - start_time:.0f}s",
                            (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (180, 180, 180), 1, cv2.LINE_AA,
                        )

                        # 更新画面
                        video_placeholder.image(
                            bgr_to_rgb(output_bgr),
                            channels="RGB",
                            use_container_width=True,
                        )

                        # 更新状态指标
                        with status_placeholder.container():
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("👥 检测人数", summary.get("total", 0))
                            mc2.metric(
                                "😶 主要表情",
                                EMOTION_CN.get(
                                    summary.get("main_expression", "-"),
                                    summary.get("main_expression", "-"),
                                ),
                            )
                            mc3.metric("📋 课堂状态", status_text)
                            mc4.metric("⏱ 运行时间", f"{now - start_time:.0f}s")

                        # 超时判断
                        if now - start_time > max_duration:
                            st.session_state.webcam_active = False
                            break

                        # 检查停止条件（通过 rerun 机制）
                        time.sleep(0.02)

                finally:
                    cap.release()

                # 显示最终汇总
                if all_detections_for_summary:
                    final_summary = summarize_expressions(all_detections_for_summary)
                    final_summary["status"] = judge_classroom_status(
                        final_summary["total"],
                        final_summary["counts"],
                        final_summary["ratios"],
                        final_summary["main_expression"],
                        good_threshold=good_threshold,
                        low_threshold=low_threshold,
                        surprise_threshold=surprise_threshold,
                        high_neutral_threshold=high_neutral_threshold,
                    )
                    st.subheader("📊 本次会话整体统计")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("累计检测人次", final_summary["total"])
                    c2.metric("主要表情", final_summary["main_expression"])
                    c3.metric("整体课堂状态", final_summary["status"])

                    stats_df = build_stats_dataframe(final_summary)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.pyplot(plot_expression_bar(final_summary))
                    with col_c2:
                        st.pyplot(plot_expression_pie(final_summary))

                st.success("✅ 实时预览已结束。")