# CoPeDiT 代码分析报告与实验适配方案

> 生成时间：2026-06-04 | 分支：`experiment/CoPeDiT_adapt`

---

## 1 方法基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | Exploiting Completeness Perception with Diffusion Transformer for Unified 3D MRI Synthesis |
| **方法简称** | CoPeDiT |
| **核心任务** | 3D MRI 缺失模态/缺失切片合成 |
| **代码仓库** | 匿名审稿版（arXiv 2026） |
| **基础框架** | MONAI + PyTorch |
| **许可证** | 未声明（root 有 LICENSE 文件引用） |

### 1.1 核心创新点

1. **Completeness-aware Prompt（完整性感知提示）**：替代传统手工 mask code，通过自监督学习生成 "completeness-aware prompt"，使模型自主感知输入的不完整程度。

2. **两阶段架构 (CoPeVAE + MDiT3D)**：
   - **Stage 1 - CoPeVAE**：基于 VQ-VAE 的 tokenizer，同时训练三个 pretext task（缺失数量检测、缺失位置定位、模态对比学习），生成 completeness-aware prompt。
   - **Stage 2 - MDiT3D**：3D Diffusion Transformer，交替进行 spatial block（空间注意力）和 modal block（模态注意力），使用 CoPeVAE 生成的 prompt 指导扩散生成。

3. **统一框架**：同一架构同时支持脑部 MRI 缺失模态合成（BraTS/IXI）和心脏 MRI 缺失切片合成（UKBB/ACDC/MSCMR）。

### 1.2 代码仓库评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 完整性 | ★★★★☆ | 训练/推理代码完整，含 4 个数据集支持 |
| 可复现性 | ★★★☆☆ | 预训练权重未发布，随机种子固定 |
| 文档质量 | ★★★☆☆ | README 清晰但缺 API 文档 |
| 代码风格 | ★★★☆☆ | 函数式为主，部分硬编码路径 |
| 模块化 | ★★★☆☆ | AutoEncoder 和 LDM 解耦良好 |

---

## 2 原代码结构

```
CoPeDiT__arxiv2026/
├── AutoEncoder/                  # Stage 1: CoPeVAE
│   ├── model/
│   │   ├── CoPeVAE_BrainMRI.py   # 脑部 VAE (含 prompt encoder)
│   │   └── MRIContrastiveLearning.py
│   ├── dataset/
│   │   └── BrainMRI_data.py      # 数据加载 (BraTS2021/IXI)
│   └── utils_brain.py            # 损失函数/辅助工具
├── LDM/                          # Stage 2: MDiT3D
│   ├── model/
│   │   ├── MDiT3D_Brain.py       # 3D DiT 模型
│   │   ├── pos_embed.py          # 3D Rotary PE
│   │   ├── rmsnorm.py            # RMSNorm
│   │   └── swiglu_ffn.py         # SwiGLU FFN
│   ├── dataset/
│   │   └── BrainMRI_data.py      # 数据加载 (collate不同)
│   ├── inferers/
│   │   └── inferer_DiT_Brain.py  # LatentDiffusionInferer
│   └── utils.py                  # EMA/logger/get_noise
├── train_CoPeVAE_Brain.{py,sh}
├── train_MDiT3D_Brain.{py,sh}
├── inference_MDiT3D_Brain.{py,sh}
├── datalist/BraTS2020/           # 数据集划分
└── assets/
```

### 2.1 原数据加载方式

**缺点：**
- 硬编码 `../../dataset/BrainMRI/` 路径
- 仅支持 BraTS2021 命名格式
- 使用 `sklearn.train_test_split` 随机划分
- 模态顺序 `[t1, t2, t1ce, flair]`

### 2.2 原模型架构

**CoPeVAE (~30M params):**
- VQ-VAE: 3D 卷积, channels=(256,384,512), codebook=8192x8
- 3个projection_head -> prompt (512D each)
- 3个分类头: 数量检测、位置定位、模态对比

**MDiT3D-B/2 (~120M params):**
- PatchEmbed_3D: patch_size=2, in_channels=8, embed_dim=768
- 16 TransformerBlock, 交替 spatial+modal block
- SwiGLU FFN, RMSNorm, 3D Rotary PE

### 2.3 原训练方式

**Stage 1 (CoPeVAE):**
- Optimizer: Adam, lr=1e-4, warmup_cosine
- Loss: L_recon(L1) + lambda1*L_vq + lambda2*L_per + lambda3*L_adv + lambda4*L_pretext
- Batch size: 2, epochs: 10000 -> adapted to 200

**Stage 2 (MDiT3D):**
- Optimizer: AdamW, lr=5e-5, warmup_cosine
- Loss: MSE on x0 prediction
- Batch size: 4 (DDP), gradient_accumulation=2
- 4 GPU DDP -> adapted to single GPU

### 2.4 原评估方式

- 生成中间切片PNG保存
- torchmetrics PSNR/SSIM/MAE + PerceptualLoss as LPIPS
- 不支持FID

---

## 3 与我的方法关键差异

| 维度 | CoPeDiT | MySparseDiffusion |
|------|---------|-------------------|
| **架构** | VQ-VAE + DiT (two-stage) | VAE-GAN + Sparse MoE |
| **生成策略** | 从可用模态+噪声扩散生成 | (待补充) |
| **完整性感知** | 自学习 prompt (3 pretext tasks) | (待补充) |
| **损失函数** | L1_recon+VQ+Perceptual+Adversarial+Pretext | (待补充) |
| **参数量** | ~150M | (待补充) |
| **训练效率** | 两阶段串行，4 GPU DDP | (待补充) |
| **推理速度** | DDIM 200 steps, ~30s/sample | (待补充) |

### 优劣势

**CoPeDiT 优势：**
- 两阶段架构解耦表示学习与生成
- Completeness-aware prompt 提供显式完整性信息
- MDiT3D 空间-模态交替注意力设计精巧

**CoPeDiT 劣势：**
- 两阶段训练耗时长
- VQ-VAE codebook 易 collapse
- 不支持动态batch内不同缺失数量
- 推理需200 DDIM steps

---

## 4 统一实验配置

### 4.1 数据集
- BraTS 2020: 369例 (train 260/val 36/test 73)
- 模态: flair, t1, t1ce, t2
- 输入尺寸: (192, 192, 64)
- 归一化: ScaleIntensityRangePercentilesd(lower=0, upper=99.5)

### 4.2 缺失模式（14种）

| Mask ID | Pattern | 缺失数 |
|---------|---------|--------|
| 1-4 | 0001,0010,0100,1000 | 3 |
| 5-10 | 0011,0101,0110,1001,1010,1100 | 2 |
| 11-14 | 0111,1011,1101,1110 | 1 |

### 4.3 评价指标
SSIM/PSNR/MSE/MAE/LPIPS/FID via $EVAL_SCRIPT

### 4.4 训练超参数

| 参数 | Stage 1 | Stage 2 |
|------|---------|---------|
| Epochs | 200 | 200 |
| Batch size | 2 | 2 |
| Optimizer | Adam | AdamW |
| LR | 1e-4 | 5e-5 |
| Schedule | warmup_cosine | warmup_cosine |

### 4.5 硬件
Quadro RTX 8000 48G

---

## 5 代码适配指南

### 5.1 数据加载替换
- 原: BraTS2021 硬编码路径 + random split
- 新: BraTS2020 datalist驱动, data/BraTS2020_data.py

### 5.2 训练适配
- 新增 train_CoPeVAE_Brain_adapted.py (Stage1)
- 新增 train_MDiT3D_Brain_adapted.py (Stage2)
- TensorBoard: Loss/train_*, LR/lr, Metrics/val_*
- 模型保存: results/task_{timestamp}/models/

### 5.3 推理适配
- 新增 eval.py
- 14种mask逐样本生成
- 输出: prediction/{mask_id}/{input,ground_truth,prediction}/

### 5.4 评估适配
- 导入EVAL_SCRIPT的ImageQualityEvaluator
- 逐mask计算指标
- 结果: prediction_metric_result/{mask_id}/result.txt

### 5.5 已知问题与修复
- xformers未安装: 自动回退F.scaled_dot_product_attention
- CUDA OOM: batch_size->1, gradient_accumulation_steps->4
- DDP通信超时: 默认单GPU

---

## 6 文件清单

| 文件 | 说明 |
|------|------|
| data/BraTS2020_data.py | 统一数据加载 |
| train_CoPeVAE_Brain_adapted.py | Stage1训练 |
| train_MDiT3D_Brain_adapted.py | Stage2训练 |
| eval.py | 推理+评估 |
| run_train.sh, run_eval.sh | 启动脚本 |
| summary.md, git_learning.md | 文档 |

---

## 7 快速开始

```bash
bash run_train.sh
bash run_eval.sh --ae_ckpt results/task_XXX/models/CoPeVAE/Autoencoder.pt --dit_ckpt results/task_XXX/models/MDiT3D/DiT.pt
tensorboard --logdir results/task_*/tensorboard/
```
