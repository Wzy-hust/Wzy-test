# 医学影像多任务学习（分类 + 分割）资源导航

> 本文档面向希望在医学影像领域同时完成**图像分类**与**图像分割**（或检测）多任务学习的研究者和工程师，汇整常见架构模式、高质量开源仓库及检索技巧。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [常见多任务架构模式](#2-常见多任务架构模式)
3. [开源仓库推荐清单](#3-开源仓库推荐清单)
4. [常用数据集](#4-常用数据集)
5. [GitHub 搜索关键词 / 快捷链接](#5-github-搜索关键词--快捷链接)
6. [延伸阅读](#6-延伸阅读)

---

## 1. 背景与动机

医学影像分析中，**分类**（病灶是否存在、良恶性判断）和**分割**（病灶轮廓定位）往往互相依赖——分割结果提供空间先验，分类标签提供全局监督。联合训练具有以下优势：

- **数据高效**：两个任务共享同一特征提取器，减少标注成本。
- **正则化效果**：辅助任务充当隐式正则，缓解过拟合。
- **端到端推理**：单次前向传播同时给出诊断类别和病灶掩码。

---

## 2. 常见多任务架构模式

### 2.1 共享编码器 + 双头（Shared Encoder + Dual Head）

```
Input Image
    │
┌───▼────────────────────┐
│   共享编码器             │
│   (ResNet / EfficientNet│
│    / ViT / Swin-T 等)  │
└──────┬─────────────────┘
       │
  ┌────┴──────┐
  │           │
  ▼           ▼
分割头        分类头
(U-Net       (GAP → FC →
 Decoder)     Softmax)
```

- 编码器参数完全共享，两个解码头各自独立。
- 总损失：`L = λ₁·L_seg + λ₂·L_cls`（λ 手动设置或自动学习）。

### 2.2 U-Net + 分类分支

经典 U-Net 在**瓶颈层（bottleneck）**后接一条全局分类路径：

```
Encoder → Bottleneck ──► 分割解码器 (跳跃连接)
               │
               ▼
          Global Average Pooling
               │
               ▼
          全连接层 → 分类输出
```

- 常见于皮肤病变（ISIC）、视网膜、组织病理切片等任务。

### 2.3 任务特定适配器（Task-Specific Adapters）

在共享主干的顶部为每个任务插入轻量适配器（Adapter / LoRA / 并行卷积块），保持主干冻结或低学习率，适合预训练大模型微调场景。

### 2.4 多任务损失权重策略

| 策略 | 说明 | 参考 |
|------|------|------|
| 固定权重 | `L = λ₁L_seg + λ₂L_cls`，手动调参 | 简单有效的基线 |
| 不确定性加权（Uncertainty Weighting） | 通过任务同方差不确定性 σᵢ 自动平衡 | Kendall et al., 2018 |
| GradNorm | 动态调整梯度范数使各任务训练速度一致 | Chen et al., 2018 |
| PCGrad | 投影冲突梯度以缓解任务间干扰 | Yu et al., 2020 |
| MGDA | 多梯度下降，寻找帕累托最优解 | Sener & Koltun, 2018 |

### 2.5 注意力 / 特征解耦

- **Cross-task attention**：让分类特征图引导分割解码器关注判别区域。
- **Task-conditional normalization**：用任务 token 调制 BN/LN 统计量。

---

## 3. 开源仓库推荐清单

> **图例**  
> ✅ 原生支持多任务（分类+分割）  
> 🔧 单任务为主，但结构容易扩展到多任务  
> ⭐ 代码质量高 / 社区活跃 / 文档完善

---

### 3.1 MedicalZooPytorch

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/black0017/MedicalZooPytorch |
| **主要任务** | 3D / 2D 医学图像分割 |
| **是否多任务** | 🔧 分割为主，可扩展分类头 |
| **框架** | PyTorch |
| **亮点** | 支持 BraTS、ISEG、MICCAI 等数据集；封装了 V-Net、U-Net、ResNet3D 等多种模型；训练脚本完善，适合快速加分类分支实验 |

---

### 3.2 SegFormer（NVIDIA）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/NVlabs/SegFormer |
| **主要任务** | 语义分割 |
| **是否多任务** | 🔧 分割为主，Transformer 编码器极易接分类头 |
| **框架** | PyTorch |
| **亮点** | Mix Transformer（MiT）编码器在医学影像迁移学习中广泛使用；分层特征适合双头设计；论文 / 代码均高质量 |

---

### 3.3 TransUNet

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/Beckschen/TransUNet |
| **主要任务** | 医学图像分割（Synapse multi-organ, Cardiac） |
| **是否多任务** | 🔧 分割为主，ViT 编码器瓶颈处可接分类头 |
| **框架** | PyTorch |
| **亮点** | Transformer + U-Net 混合架构的奠基工作；预训练 ViT 权重直接可用；社区有大量基于此的多任务扩展 |

---

### 3.4 SwinUNETR（MONAI）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/Project-MONAI/MONAI / https://github.com/Project-MONAI/research-contributions |
| **主要任务** | 3D 医学图像分割（BraTS、BTCV、MSD） |
| **是否多任务** | ✅ MONAI 框架内已有多任务示例 |
| **框架** | PyTorch |
| **亮点** | MONAI 是医学影像深度学习的事实标准库；SwinUNETR 为 SOTA 3D 分割架构；框架提供损失函数、指标、数据增强全套工具，适合直接做多任务实验 |

---

### 3.5 nnU-Net

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/MIC-DKFZ/nnUNet |
| **主要任务** | 医学图像分割（自适应，支持几乎所有模态） |
| **是否多任务** | 🔧 自动化分割流程，可在分割模型上接分类分支 |
| **框架** | PyTorch |
| **亮点** | "无需调参"的分割基线，MICCAI 多项挑战冠军；是多任务系统的优秀分割头选项；2.0 版本支持残差编码器 |

---

### 3.6 DINO-MC（多任务多中心皮肤病变）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/xiaoyuan1996/DINO-MC |
| **主要任务** | 皮肤病变分类 + 分割（ISIC） |
| **是否多任务** | ✅ 原生支持分类 + 分割联合训练 |
| **框架** | PyTorch |
| **亮点** | 基于 DINO 自监督预训练；ISIC 2018 数据集；提供完整训练 / 测试脚本和配置文件 |

---

### 3.7 Lesion-based Multi-task Learning（皮肤镜多任务）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/JiahangXu/SLSDeep |
| **主要任务** | 皮肤病变分割 + 分类（ISIC 2017/2018） |
| **是否多任务** | ✅ 原生多任务（分割 Loss + 分类 Loss 联合） |
| **框架** | PyTorch |
| **亮点** | 典型"共享编码器 + 双头"实现；论文代码一一对应；适合作为多任务实现参考模板 |

---

### 3.8 SAM-Med2D（Segment Anything for Medical Imaging）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/OpenGVLab/SAM-Med2D |
| **主要任务** | 医学图像分割（多模态、多器官） |
| **是否多任务** | 🔧 分割为主，ViT-H 编码器可扩展分类头 |
| **框架** | PyTorch |
| **亮点** | 基于 SAM（Segment Anything Model）微调；覆盖 CT/MRI/内镜等多模态；大规模医学数据预训练权重公开，非常适合作为分割分支的预训练骨干 |

---

### 3.9 UNet++ with Classification（乳腺超声/病理）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/4uiiurz1/pytorch-nested-unet |
| **主要任务** | 分割（支持添加分类头） |
| **是否多任务** | 🔧 结构简洁，Issues 和 Fork 中有多任务版本 |
| **框架** | PyTorch |
| **亮点** | UNet++ 嵌套跳跃连接，分割效果强；代码量少（<500 行主文件），极易添加分类分支；适合教学 / 原型验证 |

---

### 3.10 MultiTask-Learning-PyTorch

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch |
| **主要任务** | 通用多任务视觉（分割 + 深度 + 法线 + 表面法线） |
| **是否多任务** | ✅ 完整多任务框架，支持多种 MTL 策略 |
| **框架** | PyTorch |
| **亮点** | 实现了 GradNorm、Uncertainty Weighting、PCGrad、MGDA 等主流 MTL 损失权重方法；虽非医学影像专用，但架构可直接移植；适合学习 MTL 工程实现范式 |

---

### 3.11 Med-Query（CT 多任务检测 + 分割）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/alibaba-damo-academy/Med-Query |
| **主要任务** | 3D CT 多器官/病灶检测 + 分割 |
| **是否多任务** | ✅ 原生多任务（检测 + 分割联合训练） |
| **框架** | PyTorch |
| **亮点** | 阿里达摩院出品；基于 Transformer Query 设计；检测分支可替换为分类分支；支持 BTCV、MSD 数据集 |

---

### 3.12 Chexpert-Multitask（胸部 X 光多标签分类 + 定位）

| 项目 | 信息 |
|------|------|
| **链接** | https://github.com/kamenbliznashki/chexpert |
| **主要任务** | 胸部 X 光多标签分类（CheXpert 数据集） |
| **是否多任务** | 🔧 分类为主，结合 GradCAM 可扩展为分割/定位多任务 |
| **框架** | PyTorch |
| **亮点** | CheXpert 官方竞赛代码参考；14 类病理分类；结构清晰，适合在 CheXpert 上快速搭建分类+分割多任务系统 |

---

## 4. 常用数据集

| 数据集 | 模态 | 典型任务 | 链接 |
|--------|------|---------|------|
| **BraTS** | MRI（脑肿瘤） | 分割（肿瘤区域）+ 分级（WHO Grade） | https://www.med.upenn.edu/cbica/brats/ |
| **ISIC 2018/2019/2020** | 皮肤镜 | 分割（病变轮廓）+ 分类（良恶性/7类） | https://challenge.isic-archive.com/ |
| **LIDC-IDRI** | CT（肺结节） | 分割（结节轮廓）+ 分类（恶性程度） | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| **CheXpert** | 胸部 X 光 | 多标签分类（14类病理）+ 定位 | https://stanfordmlgroup.github.io/competitions/chexpert/ |
| **DRIVE / CHASE_DB1** | 眼底 | 血管分割 + 糖网分级 | https://drive.grand-challenge.org/ |
| **PathMNIST / MedMNIST** | 多模态（统一格式） | 分类 + 分割多任务入门友好 | https://medmnist.com/ |
| **ACDC** | 心脏 MRI | 分割（心室/心肌）+ 疾病分类 | https://www.creatis.insa-lyon.fr/Challenge/acdc/ |

---

## 5. GitHub 搜索关键词 / 快捷链接

### 精准关键词（直接点击进入 GitHub 搜索）

| 关键词 | 适用场景 |
|--------|---------|
| [`multitask segmentation classification medical`](https://github.com/search?q=multitask+segmentation+classification+medical&type=repositories) | 医学影像分类+分割多任务 |
| [`multi-task learning medical image segmentation classification`](https://github.com/search?q=multi-task+learning+medical+image+segmentation+classification&type=repositories) | 完整任务描述 |
| [`joint segmentation classification unet`](https://github.com/search?q=joint+segmentation+classification+unet&type=repositories) | U-Net 双头结构 |
| [`shared encoder segmentation classification`](https://github.com/search?q=shared+encoder+segmentation+classification&type=repositories) | 共享编码器架构 |
| [`auxiliary classification loss segmentation`](https://github.com/search?q=auxiliary+classification+loss+segmentation&type=repositories) | 辅助分类损失 |
| [`multi-task unet classification pytorch`](https://github.com/search?q=multi-task+unet+classification+pytorch&type=repositories) | PyTorch U-Net 多任务 |
| [`uncertainty weighting multitask medical`](https://github.com/search?q=uncertainty+weighting+multitask+medical&type=repositories) | 不确定性加权医学多任务 |
| [`BraTS segmentation classification pytorch`](https://github.com/search?q=BraTS+segmentation+classification+pytorch&type=repositories) | BraTS 数据集多任务 |
| [`ISIC segmentation classification multitask`](https://github.com/search?q=ISIC+segmentation+classification+multitask&type=repositories) | 皮肤镜多任务 |
| [`medical image multi-task transformer`](https://github.com/search?q=medical+image+multi-task+transformer&type=repositories) | Transformer 医学多任务 |

### Papers with Code 对应链接

- [Multi-Task Learning on Medical Image Segmentation](https://paperswithcode.com/task/medical-image-segmentation)
- [Skin Lesion Segmentation](https://paperswithcode.com/task/skin-lesion-segmentation)
- [Brain Tumor Segmentation](https://paperswithcode.com/task/brain-tumor-segmentation)

---

## 6. 延伸阅读

### 综述论文

| 论文 | 说明 |
|------|------|
| *Multi-Task Learning as Multi-Objective Optimization* (Sener & Koltun, NeurIPS 2018) | MGDA 梯度平衡方法 |
| *Multi-Task Learning Using Uncertainty to Weigh Losses* (Kendall et al., CVPR 2018) | 同方差不确定性自动权重 |
| *GradNorm: Gradient Normalization for Adaptive Loss Balancing* (Chen et al., ICML 2018) | GradNorm 动态权重 |
| *A Survey on Multi-Task Learning* (Zhang & Yang, IEEE TNNLS 2021) | 全面综述 |
| *Towards a Unified Architecture for Multi-Task Medical Image Segmentation* (arXiv 2023) | 医学影像多任务统一架构 |

### 推荐起步路径

```
1. 选择数据集（推荐 ISIC 2018 或 ACDC，标注完善）
2. 复现 UNet++ 分割基线（3.9 节）
3. 在 bottleneck 层接 GAP → FC 分类头
4. 实验 Uncertainty Weighting（3.10 节 MTL 框架）
5. 替换编码器为 Swin Transformer（SwinUNETR，3.4 节）
6. 用 MONAI 的指标和数据增强替换自定义代码
```

---

> 📝 最后更新：2026-05-06  
> 欢迎通过 Issue 或 PR 补充更多仓库！
