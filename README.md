# 基于多人人脸检测与表情识别的课堂状态分析系统

## 1. 项目简介

本项目面向课堂状态分析场景，基于多人脸检测与表情识别，对图片、视频或摄像头画面中的学生表情进行统计，进一步给出课堂状态判断和可视化结果。

## 2. 功能特性

- 支持图片上传分析
- 支持多人人脸检测
- 支持 Happy / Neutral / Sad / Angry / Surprise / Fear / Disgust 表情识别
- 支持人数统计、表情人数统计、比例统计
- 支持课堂状态判断
- 支持视频关键帧分析
- 支持摄像头实时检测
- 支持 CSV 记录保存
- 支持统计图表展示

## 3. 技术路线

```
图片 / 视频 / 摄像头输入
→ MediaPipe 人脸检测
→ 人脸裁剪
→ HSEmotionONNX 表情识别
→ 多人表情统计
→ 课堂状态规则判断
→ Streamlit 可视化展示
→ CSV 记录保存
```

## 4. 项目结构

```text
├── app.py                    # Streamlit 主界面
├── pipeline.py               # 统一图片分析流程
├── face_detector.py          # MediaPipe 人脸检测模块
├── expression_recognizer.py  # HSEmotionONNX 表情识别模块
├── analyzer.py               # 表情统计与课堂状态判断
├── video_processor.py        # 视频抽帧与分析
├── visualization.py          # 统计图表可视化
├── recorder.py               # CSV 记录读写
├── smoother.py               # 表情平滑器
├── utils.py                  # 图像处理工具函数
├── config.py                 # 全局配置与常量
├── requirements.txt          # Python 依赖
├── pyproject.toml            # 项目工程化配置
├── README.md                 # 项目说明
├── install.ps1               # 一键安装脚本
├── run.ps1                   # 一键运行脚本
└── tests/                    # 测试目录
    ├── test_analyzer.py
    └── test_utils.py
```

## 5. 环境配置

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

或直接运行一键安装脚本：

```powershell
.\install.ps1
```

## 6. 运行方式

```powershell
python -m streamlit run app.py
```

或直接运行：

```powershell
.\run.ps1
```

## 7. 模型说明

- 人脸检测：MediaPipe Face Detection
- 表情识别：HSEmotionONNX
- 推荐表情模型：enet_b0_8_best_vgaf

## 8. 课堂状态判断规则

| 条件 | 状态 |
|---|---|
| Happy + Neutral 占比 ≥ 70% | 课堂状态良好 |
| Sad + Angry 占比 ≥ 40% | 课堂状态较低落或需要关注 |
| Surprise 占比较高 | 课堂注意力波动较大 |
| Neutral 占比最高或中性占比过高 | 课堂状态平稳 |
| 检测人数为 0 | 未检测到学生 |

## 9. 数据记录

每次检测后，系统会保存 CSV 记录，包含时间、输入文件名、人数、各表情人数和比例、主要表情、课堂状态等字段。

视频分析额外记录：
- `total_face_samples`：累计人脸样本数（不等于真实学生总人数）
- `avg_people_per_frame`：平均每帧人数
- `max_people_per_frame`：单帧最大人数
- `sampled_frames`：抽样帧数

## 10. 系统局限性

- 侧脸、遮挡、低光照会影响人脸检测
- Neutral 与 Sad 可能存在混淆
- 戴口罩、低清晰度图片会影响表情识别
- 视频统计中的累计人脸样本数不等于真实学生总人数
