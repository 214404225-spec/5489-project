# AIGC Image Detection — ViT-Base & ResNet-50

> **AIGC 生成图像二分类检测系统** | 基于迁移学习的真实图像与 AI 生成图像鉴别

---

## 📌 项目简介

本项目针对 **AIGC（AI Generated Content）图像检测**任务，在仅约 **5 万张训练图像**的受限场景下，构建了一套高效、可泛化的二分类检测系统，用于区分真实图像（Real）与 AI 生成图像（Fake）。

项目完整实现了从数据预处理、模型设计、对比实验到集成推理的全流程，重点探究了不同主干网络在真实测试集上的泛化能力。

---

## 🚀 项目亮点（技术要点）

| 技术点 | 说明 |
|---|---|
| **迁移学习** | 仅微调 ViT-Base 后 4 层 Encoder 与 ResNet-50 的 layer4，有效缓解数据不足导致的过拟合 |
| **频域增强** | 自定义高通滤波（High-Pass Filter）数据增强算子，突出 AI 生成图像的高频伪造痕迹 |
| **Focal Loss** | 结合标签平滑（label smoothing=0.1）与类别权重（alpha=0.75），增强对难样本的鲁棒性 |
| **混合精度训练** | 使用 `torch.cuda.amp` 的 `autocast` + `GradScaler`，显著加速训练并降低显存占用 |
| **余弦退火调度** | 采用 `CosineAnnealingLR` 学习率策略，配合早停（patience=5），防止过拟合 |
| **TTA 推理** | 测试时增强（原图 + 水平翻转），提升推理稳定性 |
| **模型集成** | 对 ViT-Base 与 ResNet-50 的预测概率进行加权融合（Ensemble），综合两模型优势 |
| **系统性消融实验** | 对比了复杂融合模型与单一主干网络在验证集及测试集上的差异，揭示过拟合与泛化的 trade-off |

---

## 🏗️ 模型架构

### ViT-Base（Vision Transformer）
- 预训练权重：`google/vit-base-patch16-224-in21k`（HuggingFace）
- 冻结前 8 层 Encoder，仅微调后 4 层（层 8–11）及 LayerNorm
- 分类头：`Dropout(0.5)` + `Linear(768, 2)`

### ResNet-50
- 预训练权重：`ResNet50_Weights.IMAGENET1K_V2`（torchvision）
- 冻结 layer1–layer3，仅微调 layer4 与 BatchNorm
- 分类头：`Dropout(0.5)` + `Linear(2048, 2)`

---

## 📊 实验结果

> 训练集约 50,000 张，验证集占比 10%，测试集为独立评测集。

| 模型 | 验证集 Acc | 测试集 Acc | 备注 |
|---|---|---|---|
| 复杂融合模型（ResNet + ViT + 频域） | 99.48% | 80.23% | **严重过拟合** |
| ResNet-50 Only | 更高 Val Acc | 81.59% | 验证集最优，泛化一般 |
| **ViT-Base Only** | 略低 Val Acc | **88.45%** | **测试集最优，泛化最强** |

**关键发现：** 在数据量受限的场景下，ViT-Base 展现出更强的跨分布泛化能力；ResNet-50 在同分布验证集上领先，但测试集泛化不及 ViT-Base，揭示了模型复杂度与泛化能力之间的 trade-off。

---

## 🔧 技术栈

- **深度学习框架：** PyTorch 2.x、torchvision、HuggingFace Transformers
- **数据增强：** Albumentations、OpenCV（自定义高通滤波）
- **训练策略：** 混合精度（AMP）、Focal Loss、余弦退火、早停、梯度裁剪
- **可视化 & 分析：** TensorBoard、Matplotlib、Seaborn、scikit-learn（混淆矩阵、F1）
- **部署 & 推理：** ONNX Runtime、TTA、模型集成

---

## 📁 项目结构

```
├── Train.ipynb          # 训练流程：数据加载、模型定义、训练、可视化、消融实验
├── Test.ipynb           # 推理流程：加载权重、TTA 推理、模型集成、生成提交文件
├── requirements.txt     # Python 依赖列表
└── README.md
```

训练完成后将自动生成：
```
checkpoints/
├── best_model_ViT-Base.pth          # ViT-Base 最优权重
├── best_model_ResNet50.pth          # ResNet-50 最优权重
├── training_config_ViT-Base.json    # 训练配置与指标
└── training_config_ResNet50.json
runs/                                # TensorBoard 日志
submission_ensemble.csv              # 最终集成预测结果
```

---

## ⚙️ 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/214404225-spec/5489-project.git
cd 5489-project
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 建议使用 Python 3.10+，并配备 CUDA 支持的 GPU（显存 ≥ 16 GB）。

### 3. 准备数据集

将训练集解压至本地路径，目录结构如下：

```
Training_set/
├── real/      # 真实图像
└── fake/      # AI 生成图像

Test_set/      # 无标签测试图像
```

在 `Train.ipynb` 的 **Section 2.1** 中修改 `TRAIN_DIR` 为实际路径；在 `Test.ipynb` 的 **Section 1** 中修改 `TEST_DIR` 与 `CHECKPOINT_DIR`。

---

## 🔄 使用方法

### 训练

按顺序执行 `Train.ipynb` 中的所有 Cell：

1. **Section 1**：初始化环境与随机种子（固定为 42，保证可复现）
2. **Section 2**：构建数据加载器（自动按 9:1 划分训练/验证集）
3. **Section 3**：定义统一训练流水线（ViT-Base 与 ResNet-50）
4. **Section 4**：顺序训练两个模型，自动保存最优权重至 `checkpoints/`
5. **Section 5**：验证集可视化与误差分析（Loss/Acc 曲线、混淆矩阵、误分类样本）
6. **Section 6**：保存训练配置与指标至 JSON 文件

### 推理

执行 `Test.ipynb` 中的所有 Cell：

1. 自动加载 ViT-Base 与 ResNet-50 的最优权重
2. 对测试集执行 TTA 推理，分别生成各模型的概率预测 CSV
3. 加权融合两模型预测，输出最终 `submission_ensemble.csv`

---

## 📦 依赖列表

详见 [requirements.txt](requirements.txt)，核心依赖包括：

```
torch==2.7.0
torchvision==0.22.0
transformers==4.57.1
timm==1.0.22
albumentations==2.0.8
opencv-python-headless==4.12.0.88
scikit-learn==1.7.2
tensorboard==2.19.0
onnxruntime==1.23.2
```

---

## 📝 备注

- 预训练 ViT 权重通过 [hf-mirror.com](https://hf-mirror.com) 镜像加速下载（适用于国内网络环境）
- 随机种子固定为 42，保证实验完全可复现
- 本项目为 COMP 5489 课程大作业，作者：汪鸿源
