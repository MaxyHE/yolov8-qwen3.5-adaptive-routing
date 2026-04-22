# yolov8-qwen3.5-adaptive-routing

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple)
![Qwen3.5](https://img.shields.io/badge/Qwen3.5--0.8B-Alibaba-red)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

基于 YOLOv8s + Qwen3.5-0.8B 的自适应路由目标检测系统。对低置信度检测框引入轻量 VLM 进行语义消歧，在保持高吞吐的同时提升分类准确率，并提供 Streamlit 交互式 Demo。

---

## 项目背景与动机

YOLOv8s 在 VOC2012 上检测性能已经较强（mAP@0.5=73.23%，FPS=97.2），但其判别式分类头在视觉相似类别上存在固有盲区：

- **小目标**：bottle、pottedplant 的 AP@0.5 仅 0.53–0.57
- **语义相近类别**：sofa/chair、diningtable/person 混淆严重
- **低置信度框**：conf < 0.5 的框分类错误率显著高于高置信度框

单纯扩大模型规模收益有限，而全量 VLM 替换会带来显存和延迟的大幅上升。本项目提出**自适应路由策略**：只对低置信度框调用 VLM，高置信度框直接沿用 YOLO 结果，在效率和准确率之间取得最优平衡。

实验结果表明，自适应路由不仅节省了 80% 的 VLM 调用，准确率（90.40%）还高于全量 VLM（89.21%）。此外，通过对比 Qwen3.5-0.8B 与 InternVL2-1B，发现模型的指令遵循能力对受限分类任务的影响远大于参数量差异。

---

## 系统架构

本系统采用两阶段串联设计：YOLOv8s 负责定位，Qwen3.5-0.8B 负责对不确定框的语义消歧。置信度阈值（conf=0.5）作为路由开关，将检测框分流到不同的分类路径，在不牺牲高置信度框处理速度的前提下，对低置信度框引入更强的语义理解能力。

![自适应路由 Pipeline：YOLOv8s 检测后按置信度分流，低置信度框送入 Qwen3.5-0.8B 做语义消歧](eval_results/pipeline.png)

核心流程：

1. YOLOv8s 对全图做检测，输出边界框 + 置信度
2. 置信度 ≥ 0.5：直接采用 YOLO 分类结果
3. 置信度 < 0.5：裁剪 ROI，送入 Qwen3.5-0.8B 做受限分类（从 VOC 20 类中选择）
4. 合并结果输出最终检测

---

## 评估方法说明

本项目的分类准确率采用**图像级 GT 类别集合**作为参考：若预测类别存在于该图像的 GT 类别集合中，则视为正确。

**局限性**：VOC2012 标注为图像级别，一张图可能同时包含多个类别。当一个检测框的预测类别与图像中其他 GT 类别匹配（而非该框对应的真实类别）时，会产生假正确；反之，同一框会被同时计入多个 GT 类别的混淆统计，产生假错误。精确的框级别评估需要人工对每个检测框进行标注。

> 因此，本文报告的分类准确率（Cls Acc）是对系统语义理解能力的近似度量，与 mAP 互补，不可直接比较。

---

## 实验结果

本节报告五组配置的量化对比结果，涵盖整体准确率、YOLO 检测指标、难类分析、per-class 可视化，以及 Qwen3.5 与 InternVL2 的消融实验。所有评估在相同数据集和检测框集合上进行，确保对比有效性。

### 整体性能对比

| 模型 | 分类准确率 | VLM 调用率 | 显存 | 推理时间 | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLO only | 88.38% | — | — | — | 97.2 |
| **Qwen3.5 Adaptive** | **90.40%** | **20.1%** | **1.62 GB** | 521 ms/crop | — |
| Qwen3.5 Full | 89.21% | 100% | 1.62 GB | 518 ms/crop | — |
| InternVL2 Adaptive | 83.73% | 20.1% | 1.76 GB | 199 ms/crop | — |
| InternVL2 Full | 65.74% | 100% | 1.76 GB | 196 ms/crop | — |

> 评估基于 VOC2012 val 集前 500 张图像，共 1334 个检测框（conf ≥ 0.25）。推理时间为单次 VLM 调用耗时；自适应模式下仅 20.1% 的框触发 VLM，系统实际延迟远低于该值。

**关键结论：**
- 自适应路由（90.40%）优于全量 VLM（89.21%）：高置信度框由 YOLO 直接判断，避免了 VLM 在这部分框上因输出格式不一致（如 `airplane` vs `aeroplane`）引入的错误；路由策略本身有价值，而非简单的模型替换
- Qwen3.5 自适应仅调用 20.1% 的框，以极低代价换取整体准确率提升

### YOLO 检测指标

| 指标 | 数值 |
| :--- | :---: |
| mAP@0.5 | 73.23% |
| mAP@0.5:0.95 | 54.08% |
| FPS（RTX 3090） | 97.2 |

### 难类分析（YOLO vs Qwen3.5 Adaptive）

难类定义：AP@0.5 < 0.65 的类别，视觉相似度高、小目标多。Delta 单位为百分点（pp）。

| 类别 | YOLO | Qwen3.5 Adaptive | Delta | AP@0.5 |
| :--- | :---: | :---: | :---: | :---: |
| pottedplant | 26.3% | 32.9% | +6.6 pp | 0.532 |
| bottle | 34.5% | 35.4% | +0.9 pp | 0.574 |
| boat | 52.9% | 48.6% | -4.3 pp | 0.569 |
| chair | 37.3% | 31.8% | -5.5 pp | 0.581 |
| diningtable | 14.1% | 22.8% | **+8.7 pp** | 0.588 |
| sofa | 21.5% | 38.5% | **+17.0 pp** | 0.637 |
| **平均** | **31.1%** | **35.0%** | **+3.9 pp** | — |

sofa、diningtable 提升显著，说明 VLM 的语义理解对形状相似类别有效。chair 和 boat 略有下降，原因有两点：一是这两类在 VOC 中视觉多样性高（chair 涵盖餐椅、沙发椅等多种形态，boat 涵盖帆船、快艇等），低置信度 crop 本身质量差，VLM zero-shot 能力同样难以判断；二是 conf < 0.5 的阈值对这两类偏低，部分本应由 YOLO 直接判断的框被错误路由到 VLM。路由阈值仍有针对类别的优化空间。

### Per-Class 准确率对比

![20类 per-class 分类准确率对比柱状图，按 YOLO 准确率升序排列，红色阴影标注难类](eval_results/per_class_acc.png)

### Qwen3.5 vs InternVL2：消融分析

**模型选型：轻量化优先**

本项目的 VLM 选型以**轻量化、可本地部署**为核心标准，目标是在资源受限场景（单张消费级 GPU）下实现实时可用的检测-语义双层架构。

当前主流轻量 VLM 的最小规格对比：

| 模型系列 | 最小版本 | 显存占用（bfloat16） |
| :--- | :---: | :---: |
| Qwen3.5 | **0.8B** | ~1.6 GB |
| Qwen3.5 | 2B | ~4 GB |
| InternVL3 | 2B | ~4 GB |
| InternVL2 | **1B** | ~1.8 GB |

Qwen3.5-0.8B 是目前公开可用的参数量最小的 early-fusion 多模态 VLM 之一（视觉 token 与文本 token 在同一 transformer 中联合建模，而非先分别编码再拼接），显存占用仅 1.62 GB。InternVL2-1B 采用 InternViT 视觉编码器 + LLM 拼接的 late-fusion 架构，两者均可在单张 RTX 3090 上与 YOLOv8s 同时运行，无需额外显存分配。

两个模型参数量相近（0.8B vs 1B），显存占用接近（1.62 GB vs 1.76 GB），但在受限分类任务上表现差异显著——这正是本项目消融实验的核心发现之一。

**指令遵循能力**

受限分类要求模型严格从 VOC 20 类中选择输出：

| 模型 | 非 VOC 输出数 | 非 VOC 输出率 | 示例 |
| :--- | :---: | :---: | :--- |
| Qwen3.5-0.8B | 0 / 1334 | **0.0%** | — |
| InternVL2-1B (Full) | 53 / 1334 | **4.0%** | airplane, motorcycle, baby, girl |
| InternVL2-1B (Adaptive) | 15 / 1334 | 1.1% | airplane, baby |

InternVL2-1B 倾向于输出自然语言描述（`airplane` 而非 `aeroplane`，`baby` 而非 `person`），语义上正确但不符合约束，导致这部分框直接计为错误。这也是 InternVL2 全量模式（65.74%）远低于自适应模式（83.73%）的主要原因——强制全量调用将指令遵循问题放大到所有框，而自适应路由将其限制在低置信度的 20.1% 框内。

**选型结论**：Qwen3.5-0.8B 在 zero-shot 受限分类任务中指令遵循更稳定、准确率更高，最终选用 Qwen3.5 作为 VLM 组件。

---

## Demo

为了直观展示系统能力，本项目基于 Streamlit 构建了交互式 Demo，支持**图片检测**与**视频多目标追踪**两种模式，并集成 VLM 语义描述功能，支持中英文界面切换。用户无需编写代码，上传图片或视频即可实时查看检测结果、置信度、VLM 描述输出。

### Streamlit 界面

![Streamlit Demo 界面截图 1](assets/streamlit_pic_1.jpg)
![Streamlit Demo 界面截图 2](assets/streamlit_pic_2.jpg)

### 图片检测模式

上传图片后，系统自动运行 YOLOv8s 检测，标注边界框、类别与置信度。开启 VLM 描述开关后，对每个检测框裁剪 ROI 并调用 Qwen3.5-0.8B 生成语义描述。

![YOLOv8s 图片检测效果](assets/demo.png)

### VLM 语义描述能力

除受限分类外，VLM 具备对复杂场景的细粒度语义描述能力，能够识别遮挡关系、目标属性等判别式模型难以表达的信息。

**案例：两只狗的遮挡场景**

| YOLO 检测结果 | Qwen3.5 语义描述 |
| :---: | :---: |
| ![YOLO 检测两只狗的边界框结果](assets/case_TwoDogs_yolo.png) | ![Qwen3.5 对遮挡场景的语义描述](assets/case_TwoDogsVLM.png) |

VLM 对局部裁剪区域生成描述，能够识别遮挡关系和细粒度属性（颜色、姿态），有效补充了判别式模型的类别信息。

### 视频多目标追踪（ByteTrack）

ByteTrack 是一种高性能多目标追踪算法，其核心思路是**不丢弃低置信度检测框**，而是将其纳入轨迹关联过程——相比传统方法只关联高置信度框，ByteTrack 能有效减少目标丢失和 ID 切换，在遮挡、快速运动等复杂场景下表现更稳定。

本项目集成 ByteTrack 对视频逐帧检测，为每个目标分配唯一 ID 并跨帧持续追踪，适用于人群、车辆等多目标密集场景。

[原始视频](assets/original_video.mp4) | [追踪结果](assets/tracked_video.mp4)

---

## 训练过程

YOLOv8s 在 VOC2012 训练集上训练 100 个 epoch，使用默认超参数配置。以下曲线记录了训练过程中的收敛情况，可用于判断模型是否过拟合以及最优 checkpoint 的选取。

### Loss & mAP 曲线

![训练过程中 Loss 与 mAP 曲线](assets/results.png)

### PR 曲线

![Precision-Recall 曲线](assets/BoxPR_curve.png)

### 混淆矩阵

![归一化混淆矩阵](assets/confusion_matrix_normalized.png)

---

## 快速开始

以下步骤适用于在本地或 Linux 服务器上复现实验、运行评估或启动 Demo。VLM 推理需要 GPU，建议显存 ≥ 16GB（实验在 RTX 3090 上进行）。

### 环境要求

- Python 3.10
- PyTorch 2.4（CUDA 12.1）
- GPU：建议 16GB+ 显存（实验在 NVIDIA RTX 3090 上进行）

### 安装依赖

```bash
# 先单独安装 PyTorch（需指定 CUDA 版本）
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 再安装其余依赖
pip install ultralytics streamlit transformers==5.5.4 accelerate einops timm
```

### 数据准备

下载 [VOC2012 数据集](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)，解压后运行格式转换：

```bash
python voc2yolo.py
```

> `voc2yolo.py` 默认读取 `VOCdevkit/VOC2012/Annotations`，请将数据集放在项目根目录下或修改脚本中的路径。

### 训练

```bash
python train.py
```

> 训练权重 `best.pt` 需自行训练生成，或联系作者获取。

### 评估

```bash
# YOLO baseline
python eval_yolo.py --data_dir /path/to/VOCdevkit/VOC2012 --model_path best.pt --yaml_path VOC2012.yaml

# Qwen3.5 自适应路由
python eval_qwen_adaptive.py --data_dir /path/to/VOCdevkit/VOC2012 --model_path best.pt --vlm_path /path/to/Qwen3.5-0.8B

# 生成分析图表
python eval_analysis.py
```

### 启动 Demo

```bash
streamlit run app.py
```

---

## 项目结构

项目分为三个主要部分：核心检测与追踪（`app.py`、`train.py`）、系统评估（`eval_*.py`）、以及分析与可视化（`eval_analysis.py`、`draw_pipeline.py`）。评估脚本共享 `eval_utils.py` 中的工具函数，结果统一输出到 `eval_results/`。

```
├── app.py                  # Streamlit Demo
├── train.py                # 训练脚本
├── voc2yolo.py             # VOC XML → YOLO txt 格式转换
├── VOC2012.yaml            # 数据集配置
├── eval_utils.py           # 评估工具函数（共享）
├── eval_yolo.py            # YOLO baseline 评估
├── eval_qwen_adaptive.py   # Qwen3.5 自适应路由评估
├── eval_qwen_full.py       # Qwen3.5 全量 VLM 评估
├── eval_ivl_adaptive.py    # InternVL2 自适应路由评估
├── eval_ivl_full.py        # InternVL2 全量 VLM 评估
├── eval_analysis.py        # 生成 per-class 准确率图表
├── draw_pipeline.py        # 生成流程图
├── save_error_crops.py     # 保存错误检测框 crop
├── remap_ivl_results.py    # InternVL2 同义词映射后处理
├── assets/                 # Demo 截图与演示素材
└── eval_results/           # 评估结果 JSON 与图表
```

---

## 技术栈

| 组件 | 技术 |
| :--- | :--- |
| 检测模型 | YOLOv8s（anchor-free，PAN-FPN neck，解耦检测头） |
| VLM | Qwen3.5-0.8B（early-fusion 架构，视觉与文本 token 联合建模，bfloat16 推理） |
| 对比模型 | InternVL2-1B（InternViT 视觉编码器 + LLM late-fusion 拼接架构） |
| 追踪算法 | ByteTrack |
| 框架 | Python 3.10 / PyTorch 2.4 / Transformers 5.5.4 |
| Demo | Streamlit |
| 数据集 | PASCAL VOC2012（20 类，5717 训练 / 5823 验证） |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
