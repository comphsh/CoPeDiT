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

## 7 环境检查与依赖安装

### 7.1 确认 Python 环境

```bash
# 检查当前环境（应包含 PyTorch CUDA + MONAI）
python -c "
import torch; import monai
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'MONAI {monai.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 7.2 安装缺失依赖

```bash
# monai-generative (DDPM/DDIM scheduler, LatentDiffusionInferer 等)
pip install monai-generative==0.2.3

# 其他关键依赖（如缺少请逐条安装）
pip install nibabel tqdm einops timm torchmetrics tensorboard
```

### 7.3 检查数据集完整性

```bash
# 确认 369 例数据都在
ls /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/ | wc -l
# 应输出: 369

# 单样本应包含 4 个模态文件 + 1 个分割文件
ls /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/
# 应输出: flair.nii  t1.nii  t1ce.nii  t2.nii  seg.nii
```

### 7.4 检查 datalist 划分

```bash
# 确认划分正确（train 260 / val 36 / test 73 = 369）
wc -l datalist/BraTS2020/*.list
# 预期输出:
#    73 datalist/BraTS2020/test.list
#   296 datalist/BraTS2020/train_all.list
#   260 datalist/BraTS2020/train.list
#    36 datalist/BraTS2020/val.list
```

---

## 8 训练命令（完整）

> 工作目录：`$COMPARE_ROOT` = `/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026`

### 8.1 方式 A：一键启动两阶段训练

```bash
cd /devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026

# 标准训练（Stage 1 → Stage 2 自动串联）
bash run_train.sh

# 训练完成后，模型和日志保存在:
#   results/task_{timestamp}/
#   ├── models/
#   │   ├── CoPeVAE/Autoencoder.pt          # Stage 1 最终权重
#   │   └── MDiT3D/DiT.pt                    # Stage 2 最终权重
#   └── tensorboard/
#       ├── stage1/                           # CoPeVAE 训练日志
#       └── stage2/                           # MDiT3D 训练日志
```

### 8.2 方式 B：分阶段训练（推荐用于调试）

**Stage 1: 训练 CoPeVAE (200 epochs)**

```bash
# ========== 默认参数 (batch=2) ==========
python train_CoPeVAE_Brain_adapted.py \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_min 8e-5 \
    --lr_schedule warmup_cosine \
    --opt adam \
    --warmup_steps 5 \
    --val_interval 5 \
    --ckpt_interval 50 \
    --cache 0.2 \
    --gradient_accumulation_steps 2 \
    --seed 2025

# ========== 显存不足 (OOM) 时的应急参数 (batch=1) ==========
python train_CoPeVAE_Brain_adapted.py \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --epochs 200 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --lr_min 8e-5 \
    --lr_schedule warmup_cosine \
    --opt adam \
    --warmup_steps 5 \
    --val_interval 5 \
    --ckpt_interval 50 \
    --seed 2025

# Stage 1 完成后，模型保存在:
#   results/task_{timestamp}/models/CoPeVAE/Autoencoder.pt
```

**Stage 2: 训练 MDiT3D (200 epochs, 需 Stage 1 权重)**

```bash
# 先找到 Stage 1 的输出权重
STAGE1_CKPT=$(find results/task_*/models/CoPeVAE -name "Autoencoder.pt" 2>/dev/null | sort | tail -1)
echo "Stage 1 checkpoint: $STAGE1_CKPT"

# ========== 默认参数 (batch=2, missing_num=1) ==========
python train_MDiT3D_Brain_adapted.py \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --ae_ckpt "$STAGE1_CKPT" \
    --epochs 200 \
    --batch_size 2 \
    --missing_num 1 \
    --lr 5e-5 \
    --lr_min 2e-5 \
    --lr_schedule warmup_cosine \
    --opt adamw \
    --warmup_steps 50 \
    --val_interval 20 \
    --ckpt_interval 50 \
    --cache 0.2 \
    --gradient_accumulation_steps 2 \
    --pred_type x \
    --diff_loss l2 \
    --seed 2025

# ========== 显存不足 (OOM) 时的应急参数 (batch=1) ==========
python train_MDiT3D_Brain_adapted.py \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --ae_ckpt "$STAGE1_CKPT" \
    --epochs 200 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --missing_num 1 \
    --lr 5e-5 \
    --lr_min 2e-5 \
    --lr_schedule warmup_cosine \
    --opt adamw \
    --warmup_steps 50 \
    --val_interval 20 \
    --ckpt_interval 50 \
    --pred_type x \
    --diff_loss l2 \
    --seed 2025

# Stage 2 完成后，模型保存在:
#   results/task_{timestamp}/models/MDiT3D/DiT.pt
```

### 8.3 训练过程中监控

```bash
# 方式 1：TensorBoard（推荐）
tensorboard --logdir results/task_*/tensorboard/ --port 6006

# 方式 2：tail 日志文件
tail -f results/task_*/stage2_MDiT3D/log/*.log

# 方式 3：实时查看 GPU 使用
watch -n 2 nvidia-smi
```

### 8.4 断点续训

```bash
# Stage 1 断点续训（不提供 checkpoint 路径即可自动 resume）
python train_CoPeVAE_Brain_adapted.py \
    ... (同上参数) ...

# Stage 2 断点续训
# 修改 train_MDiT3D_Brain_adapted.py 中 args.resume_ckpt 指向已有 checkpoint
# 或直接在脚本中设置:
#   args.resume_ckpt = "results/task_XXXXXX/models/MDiT3D/DiT_epoch100.pt"
```

### 8.5 关键超参数说明

| 参数 | Stage 1 默认值 | Stage 2 默认值 | 说明 |
|------|---------------|---------------|------|
| `--epochs` | 200 | 200 | 训练轮数 |
| `--batch_size` | 2 | 2 | OOM 时降为 1 |
| `--lr` | 1e-4 | 5e-5 | 学习率 |
| `--lr_min` | 8e-5 | 2e-5 | Cosine 最低学习率 |
| `--warmup_steps` | 5 | 50 | 学习率预热步数 |
| `--val_interval` | 5 | 20 | 每隔 N epoch 验证一次 |
| `--ckpt_interval` | 50 | 50 | 每隔 N epoch 保存 period checkpoint |
| `--gradient_accumulation_steps` | 2 | 2 | 梯度累积（降低显存占用） |
| `--missing_num` | — | 1 | 训练时的缺失模态数量 (1/2/3) |
| `--pred_type` | — | x | 预测目标: noise / x (x0) / v |
| `--diff_loss` | — | l2 | 扩散损失: l1 / l2 |

---

## 9 推理命令（完整）

### 9.1 方式 A：一键启动（run_eval.sh）

```bash
cd /devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026

# 先找到训练好的 checkpoint 路径
AE_CKPT=$(find results/task_*/models/CoPeVAE -name "Autoencoder.pt" 2>/dev/null | sort | tail -1)
DIT_CKPT=$(find results/task_*/models/MDiT3D -name "DiT.pt" 2>/dev/null | sort | tail -1)
echo "AE:  $AE_CKPT"
echo "DiT: $DIT_CKPT"

# 启动评估（对 73 例测试样本 × 14 种 mask = 1022 次生成）
bash run_eval.sh \
    --ae_ckpt "$AE_CKPT" \
    --dit_ckpt "$DIT_CKPT"
```

### 9.2 方式 B：直接调用 eval.py（更灵活）

```bash
cd /devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026

python eval.py \
    --ae_ckpt results/task_XXXXXX/models/CoPeVAE/Autoencoder.pt \
    --dit_ckpt results/task_XXXXXX/models/MDiT3D/DiT.pt \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --eval_script /devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py \
    --task_dir ./results/task_eval_manual \
    --sample_steps 200 \
    --pred_type x
```

### 9.3 推理输出结构

```
results/task_{timestamp}/
└── prediction/
    ├── 1/                        # mask_id=1, pattern=0001 (仅 t2 可用)
    │   ├── input/                # 可用模态 (t2) — 模型输入
    │   │   └── BraTS20_Training_137/
    │   │       └── BraTS20_Training_137_t2.nii.gz
    │   ├── ground_truth/         # 缺失模态 (flair, t1, t1ce) — 原始真值
    │   │   └── BraTS20_Training_137/
    │   │       ├── BraTS20_Training_137_flair.nii.gz
    │   │       ├── BraTS20_Training_137_t1.nii.gz
    │   │       └── BraTS20_Training_137_t1ce.nii.gz
    │   └── prediction/           # 缺失模态 (flair, t1, t1ce) — 模型生成
    │       └── BraTS20_Training_137/
    │           ├── BraTS20_Training_137_flair.nii.gz
    │           ├── BraTS20_Training_137_t1.nii.gz
    │           └── BraTS20_Training_137_t1ce.nii.gz
    ├── 2/                        # mask_id=2, pattern=0010 (仅 t1ce 可用)
    │   └── ...
    ├── ...                        
    └── 14/                       # mask_id=14, pattern=1110 (缺 t2)
        └── ...

prediction_metric_result/
├── 1/
│   └── result.txt                # ssim  psnr  mse  mae  fid  lpips
├── 2/
│   └── result.txt
├── ...
└── 14/
    └── result.txt
```

### 9.4 推理硬件预估

| 项目 | 数值 |
|------|------|
| 测试样本数 | 73 例 |
| 每样本 mask 数 | 14 种 |
| 总生成任务 | 73 × 14 = 1022 次 DDIM 采样 |
| 每次采样步数 | 200 steps (DDIM) |
| 预计单样本耗时 | ~30-60 秒 |
| 预计总耗时 | ~6-12 小时 (单 GPU) |

---

## 10 评估命令

### 10.1 评估已在 eval.py 末尾自动执行

`eval.py` 在生成完所有图像后会自动：
1. 导入 `EVAL_SCRIPT` 的 `ImageQualityEvaluator`
2. 对每个 mask 目录，逐样本/逐模态计算 SSIM/PSNR/MSE/MAE/LPIPS
3. 将结果写入 `prediction_metric_result/{mask_id}/result.txt`

### 10.2 单独运行评估（如果已完成推理，跳过来重新计算）

```bash
# 如果推理已完成，重新计算指标（无需重复生成图像）
python -c "
import sys
sys.path.insert(0, '/devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics')
from syn_metrics import ImageQualityEvaluator
import numpy as np, torch, nibabel as nib, os

MODALITY_KEYS = ['flair', 't1', 't1ce', 't2']
MASK_LIST = [
    [0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],
    [0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0],[1,1,0,0],
    [0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],
]
TASK_DIR = 'results/task_XXXXXX'
PRED_ROOT = os.path.join(TASK_DIR, 'prediction')
METRIC_ROOT = os.path.join(TASK_DIR, 'prediction_metric_result')

evaluator = ImageQualityEvaluator(LPIPS_model_type='nomedical', device='cuda')
for mask_id in range(1, 15):
    mask = MASK_LIST[mask_id-1]
    missing_idx = [i for i,m in enumerate(mask) if m==0]
    pred_dir = os.path.join(PRED_ROOT, str(mask_id), 'prediction')
    gt_dir = os.path.join(PRED_ROOT, str(mask_id), 'ground_truth')
    all_m = {'ssim':[],'psnr':[],'mse':[],'mae':[],'lpips':[]}
    for sid in os.listdir(pred_dir):
        for mi in missing_idx:
            mn = MODALITY_KEYS[mi]
            pp = os.path.join(pred_dir, sid, f'{sid}_{mn}.nii.gz')
            gp = os.path.join(gt_dir, sid, f'{sid}_{mn}.nii.gz')
            if not os.path.exists(pp) or not os.path.exists(gp): continue
            p = nib.load(pp).get_fdata().astype(np.float32)
            g = nib.load(gp).get_fdata().astype(np.float32)
            p = (p-p.min())/(p.max()-p.min()+1e-8); p = 2*p-1
            g = (g-g.min())/(g.max()-g.min()+1e-8); g = 2*g-1
            pt = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).cuda()
            gt = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).cuda()
            m = evaluator.evaluate_all_metrics(pt, gt)
            for k in all_m: all_m[k].append(m[k])
    avg = {k: np.mean(v) if v else float('nan') for k,v in all_m.items()}
    avg['fid'] = -1.0
    os.makedirs(os.path.join(METRIC_ROOT, str(mask_id)), exist_ok=True)
    with open(os.path.join(METRIC_ROOT, str(mask_id), 'result.txt'), 'w') as f:
        f.write(f\"ssim     psnr     mse      mae      fid     lpips\n\")
        f.write(f\"{avg['ssim']:.6f}   {avg['psnr']:.6f}  {avg['mse']:.6f}  {avg['mae']:.6f}  {avg['fid']:.6f}   {avg['lpips']:.6f}\n\")
    print(f'Mask {mask_id}: SSIM={avg[\"ssim\"]:.4f} PSNR={avg[\"psnr\"]:.2f}')
print('Done.')
"
```

### 10.3 汇总所有 mask 的结果

```bash
# 一键查看 14 种 mask 的所有指标
for mask_id in $(seq 1 14); do
    f="results/task_XXXXXX/prediction_metric_result/$mask_id/result.txt"
    if [ -f "$f" ]; then
        echo "Mask $mask_id: $(tail -1 "$f")"
    fi
done
```

---

## 11 完整执行流程（从零开始）

```bash
# ==================== 第 0 步：环境准备 ====================
cd /devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026

# 确认分支
git branch    # 应显示 * experiment/CoPeDiT_adapt

# 安装依赖（如未安装）
pip install monai-generative==0.2.3

# ==================== 第 1 步：训练 CoPeVAE (Stage 1) ====================
# 约 12-24 小时，200 epochs
python train_CoPeVAE_Brain_adapted.py \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --epochs 200 --batch_size 2 --lr 1e-4 \
    --val_interval 5 --ckpt_interval 50

# 记下输出的 task 目录，例如: results/task_20260605_100000/

# ==================== 第 2 步：训练 MDiT3D (Stage 2) ====================
# 约 12-24 小时，200 epochs，依赖第 1 步的 checkpoint
python train_MDiT3D_Brain_adapted.py \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --ae_ckpt results/task_20260605_100000/models/CoPeVAE/Autoencoder.pt \
    --epochs 200 --batch_size 2 --missing_num 1 --lr 5e-5 \
    --val_interval 20 --ckpt_interval 50

# ==================== 第 3 步：推理 + 评估 ====================
# 约 6-12 小时，73 例 × 14 mask = 1022 次 DDIM 采样
python eval.py \
    --ae_ckpt results/task_20260605_100000/models/CoPeVAE/Autoencoder.pt \
    --dit_ckpt results/task_20260605_100000/models/MDiT3D/DiT.pt \
    --data_root /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --datalist_dir ./datalist/BraTS2020 \
    --task_dir ./results/task_20260605_100000

# ==================== 第 4 步：查看结果 ====================
for mask_id in $(seq 1 14); do
    f="results/task_20260605_100000/prediction_metric_result/$mask_id/result.txt"
    if [ -f "$f" ]; then
        echo "Mask $mask_id: $(tail -1 "$f")"
    fi
done
```

---

## 12 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `CUDA out of memory` | batch_size=2 时显存不足 | `--batch_size 1 --gradient_accumulation_steps 4` |
| `ModuleNotFoundError: No module named 'generative'` | monai-generative 未安装 | `pip install monai-generative==0.2.3` |
| `FileNotFoundError: Datalist not found` | datalist 路径不对 | 确认 `datalist/BraTS2020/{train,val,test}.list` 存在 |
| `RuntimeError: shape mismatch` | 训练和推理的 noise 维度不一致 | 检查 --missing_num 与 checkpoint 的训练设置匹配 |
| `NVIDIA driver too old` | CUDA 12.8 要求较新驱动 | 降级 torch 或更新驱动 |
| 训练 loss 不下降 | VQ-VAE codebook collapse | 增大 `--vq_weight` (如 2.0), 减小 `--lr` |
| `xformers not available` | 库未安装 | 自动回退到 `F.scaled_dot_product_attention`，无需处理 |
| DDP 初始化失败 | 多 GPU 通信问题 | 默认用单 GPU，加 `--distributed` 才开启 DDP |
