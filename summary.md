# CoPeDiT 代码分析报告与实验适配方案

> 生成时间：2026-06-11 | 分支：`experiment/CoPeDiT_adapt`
> 仓库：`comphsh/CoPeDiT__arxiv2026` (fork from JK-Liu7/CoPeDiT)

---

## 1 方法基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | Exploiting Completeness Perception with Diffusion Transformer for Unified 3D MRI Synthesis |
| **方法简称** | CoPeDiT |
| **发表** | arXiv 2026 |
| **核心任务** | 3D MRI 缺失模态合成 / 缺失切片合成 |
| **官方仓库** | https://github.com/JK-Liu7/CoPeDiT |
| **基础框架** | MONAI + PyTorch + monai-generative |
| **硬件需求** | GPU ≥ 48GB (Quadro RTX 8000) |

### 1.1 核心创新点

1. **Completeness-aware Prompt**：替代传统手工 mask code，3 个自监督 pretext task（缺失数量检测、缺失位置定位、模态对比学习）引导模型感知输入完整程度。

2. **两阶段架构 (CoPeVAE + MDiT3D)**：
   - **Stage 1 - CoPeVAE** (~30M)：基于 VQ-VAE 的 tokenizer，同时训练 pretext task 生成 completeness-aware prompt。
   - **Stage 2 - MDiT3D** (~120M)：3D Diffusion Transformer，交替 spatial block（空间注意力）和 modal block（模态注意力），CoPeVAE prompt 指导扩散生成。

3. **统一框架**：同一架构支持脑部 MRI 缺失模态合成（BraTS）和心脏 MRI 缺失切片合成（UKBB/ACDC/MSCMR）。

---

## 2 代码结构

```
CoPeDiT__arxiv2026/
├── AutoEncoder/                        # Stage 1: CoPeVAE
│   ├── model/
│   │   ├── CoPeVAE_BrainMRI.py         # CopeVAE (VQ-VAE + 3 pretext heads)
│   │   └── MRIContrastiveLearning.py   # InfoNCE 对比损失
│   ├── dataset/
│   │   └── BrainMRI_data.py            # 原始 BraTS2021/IXI 加载（保留）
│   └── utils_brain.py                  # 损失加权、EMA、logger
├── LDM/                                # Stage 2: MDiT3D
│   ├── model/
│   │   ├── MDiT3D_Brain.py             # 3D DiT (spatial+modal blocks)
│   │   ├── pos_embed.py                # 3D Rotary Position Embedding
│   │   ├── rmsnorm.py                  # RMSNorm (LLaMA2)
│   │   └── swiglu_ffn.py               # SwiGLU FFN (LLaMA2)
│   ├── inferers/
│   │   └── inferer_DiT_Brain.py        # LatentDiffusionInferer
│   └── utils.py                        # EMA、logger、get_noise
├── data/                               # 统一数据模块 (BraTS2020)
│   ├── __init__.py
│   └── BraTS2020_data.py               # MONAI DataLoader + 14 mask + collate
├── datalist/BraTS2020/                 # 数据集划分
│   ├── train.list (260), val.list (36), test.list (73)
│   └── divide_data.py
│
├── train_CoPeVAE_Brain_adapted.py      # Stage 1 训练入口
├── train_MDiT3D_Brain_adapted.py       # Stage 2 训练入口
├── eval.py                             # 推理 + 指标计算
├── run_train.sh                        # 训练一键启动
├── run_eval.sh                         # 推理一键启动
└── summary.md                          # 本文档
```

---

## 3 全局路径变量

```bash
# ============================================================
# 全局路径变量（按实际修改）
# ============================================================
export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026
export DATA_ROOT=/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
export DATALIST_DIR=$COMPARE_ROOT/datalist/BraTS2020

# 指标计算脚本（来自 MySparseDiffusion 项目）
export EVAL_SCRIPT=/devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py
# ============================================================
```

> **提示**：所有训练/推理脚本的 `--data_root` / `--datalist_dir` 默认值均从以上环境变量读取。本地电脑只需修改这 3 行即可移植运行。

---

## 4 统一实验配置

### 4.1 数据集

- **BraTS 2020**: 369 例
- **划分**: train 260 / val 36 / test 73 (70/10/20)
- **模态顺序**: `[flair, t1, t1ce, t2]`
- **输入尺寸**: (192, 192, 64) after CenterCrop + Resize
- **归一化**: `ScaleIntensityRangePercentilesd(lower=0, upper=99.5, b_min=0, b_max=1)` → [0, 1]

### 4.2 缺失模式（14 种）

| Mask ID | Pattern (flair,t1,t1ce,t2) | 缺失数 | 描述 |
|---------|---------------------------|--------|------|
| 1 | 0001 | 3 | 仅 t2 可用 |
| 2 | 0010 | 3 | 仅 t1ce 可用 |
| 3 | 0100 | 3 | 仅 t1 可用 |
| 4 | 1000 | 3 | 仅 flair 可用 |
| 5 | 0011 | 2 | t1ce + t2 |
| 6 | 0101 | 2 | t1 + t2 |
| 7 | 0110 | 2 | t1 + t1ce |
| 8 | 1001 | 2 | flair + t2 |
| 9 | 1010 | 2 | flair + t1ce |
| 10 | 1100 | 2 | flair + t1 |
| 11 | 0111 | 1 | 缺 flair |
| 12 | 1011 | 1 | 缺 t1 |
| 13 | 1101 | 1 | 缺 t1ce |
| 14 | 1110 | 1 | 缺 t2 |

### 4.3 评价指标

SSIM / PSNR / MSE / MAE / LPIPS（FID 固定为 -1，3D 体数据不支持逐样本 FID）

### 4.4 训练超参数

| 参数 | Stage 1 (CoPeVAE) | Stage 2 (MDiT3D) |
|------|-------------------|-------------------|
| Epochs | 200 | 200 |
| Batch size | 2 | 2 |
| Optimizer | Adam | AdamW |
| LR | 1e-4 | 5e-5 |
| LR min | 8e-5 | 2e-5 |
| Schedule | warmup_cosine | warmup_cosine |
| Warmup steps | 5 | 50 |
| Gradient accumulation | 2 | 2 |
| Missing num | 随机 (1/2/3) | 1 (可配置) |
| Pred type | — | x (x0 prediction) |
| Diff loss | — | l2 (MSE) |
| VQ weight | 1.0 | — |
| Perceptual weight | 0.05 | — |
| Adversarial weight | 0.01 | — |
| Pretext weight | 0.01 | — |

---

## 5 环境准备与冒烟测试

### 5.1 确认环境

```bash
cd $COMPARE_ROOT

# 检查 Python 依赖
python -c "
import torch; import monai; import nibabel
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'MONAI {monai.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 5.2 安装缺失依赖

```bash
pip install monai-generative==0.2.3
pip install nibabel tqdm einops timm torchmetrics tensorboard
```

### 5.3 验证数据完整性

```bash
# 确认 369 例数据
ls $DATA_ROOT/ | wc -l          # → 369

# 单样本应包含 4 模态 + seg
ls $DATA_ROOT/BraTS20_Training_001/
# → flair.nii  t1.nii  t1ce.nii  t2.nii  seg.nii

# 确认划分正确
wc -l $DATALIST_DIR/*.list
# → 260 train.list
# →  36 val.list
# →  73 test.list
```

### 5.4 快速冒烟测试（验证数据加载 + 模型前向，约 3 分钟）

```bash
cd $COMPARE_ROOT

python -c "
import sys, os, torch
sys.path.insert(0, '.')
from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from LDM.model.MDiT3D_Brain import MDiT3D_models
from data.BraTS2020_data import (
    MODALITY_KEYS, read_train_val_datalist, read_datalist,
    get_transforms, collate_fn_CoPeVAE, collate_fn_MDiT3D,
)
import argparse

# 模拟 args
class A: pass
args = A()
for k, v in {
    'spatial_dims':3, 'vae_channel':(256,384,512), 'res_channel':256,
    'code_num':8192, 'code_dim':8, 'latent_dim':8, 'proj_dim':512,
    'contrast_dim':128, 'modality_num':4, 'recon_loss':'l1',
    'lambda1':1.0, 'lambda2':0.5, 'lambda3':0.5,
    'brain_roi':(192,192,80), 'brain_size':(192,192,64),
    'cache':0.0, 'batch_size':1, 'distributed':False,
}.items():
    setattr(args, k, v)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# 测试数据加载 (train — 不接触 test)
print('=== 测试训练数据加载 ===')
train_files, val_files = read_train_val_datalist('$DATALIST_DIR', '$DATA_ROOT')
print(f'Train: {len(train_files)}, Val: {len(val_files)}')
train_transform, _ = get_transforms(args, MODALITY_KEYS, is_train=True)

from monai.data import DataLoader
from data.BraTS2020_data import MonaiDataset
ds = MonaiDataset(data=train_files, transform=train_transform, cache_rate=0.0, num_workers=2)
loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn_CoPeVAE)
batch = next(iter(loader))
x_incomp, x_missing, missing_len, missing_label = [b.to(device) for b in batch]
print(f'x_incomp: {x_incomp.shape}, x_missing: {x_missing.shape}')
print(f'missing_len: {missing_len}, missing_label: {missing_label}')

# 测试模型创建
print('=== 测试模型创建 ===')
ae = CopeVAE(args).to(device)
print(f'CoPeVAE: {sum(p.numel() for p in ae.parameters()):,} params')

with torch.no_grad():
    x_in, x_rec, l_vq, l_pretext = ae(x_incomp, x_missing, missing_len, missing_label)
    print(f'Forward OK: rec keys={list(x_rec.keys())}, vq_loss={l_vq.item():.4f}')

# 测试 Stage 2 编码
z = ae.encode_stage_2_inputs(x_incomp)
print(f'Latent shape: {z.shape}')

# 测试 MDiT3D
dit = MDiT3D_models['MDiT3D-B/2'](num_modalities=1).to(device)
print(f'MDiT3D-B/2: {sum(p.numel() for p in dit.parameters()):,} params')

# 确认 test 数据隔离（train 函数不应加载 test）
_, _, test_files = read_datalist('$DATALIST_DIR', '$DATA_ROOT')
print(f'Test files (only for eval): {len(test_files)}')
print('🎉 冒烟测试全部通过！')
"
```

---

## 6 训练命令

### 6.1 方式 A：一键启动（推荐）

```bash
cd $COMPARE_ROOT

# 标准训练（Stage 1 → Stage 2 自动串联）
bash run_train.sh

# 仅 Stage 1
bash run_train.sh --stage1-only

# 仅 Stage 2（需指定 Stage 1 权重）
bash run_train.sh --stage2-only --ae_ckpt results/task_XXXXXX/models/CoPeVAE/Autoencoder.pt

# 自定义参数
bash run_train.sh --batch_size 1 --epochs 100 --missing_num 2
```

### 6.2 方式 B：分阶段训练

#### Stage 1: CoPeVAE (200 epochs)

```bash
cd $COMPARE_ROOT

python train_CoPeVAE_Brain_adapted.py \
    --data_root "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
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

# 显存不足时 (OOM):
#   --batch_size 1 --gradient_accumulation_steps 4
```

#### Stage 2: MDiT3D (200 epochs)

```bash
cd $COMPARE_ROOT

# 先找到 Stage 1 的输出权重
STAGE1_CKPT=$(find $COMPARE_ROOT/results/task_*/models/CoPeVAE -name "Autoencoder.pt" 2>/dev/null | sort | tail -1)
echo "Stage 1 checkpoint: $STAGE1_CKPT"

python train_MDiT3D_Brain_adapted.py \
    --data_root "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
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

# 显存不足时 (OOM):
#   --batch_size 1 --gradient_accumulation_steps 4
```

### 6.3 训练输出结构

```
$COMPARE_ROOT/results/task_{timestamp}/
├── models/
│   ├── CoPeVAE/
│   │   ├── Autoencoder.pt              # 每 epoch 覆盖保存
│   │   ├── Autoencoder_best_epochXX.pt # 最佳 loss 保存
│   │   └── Autoencoder_epoch50.pt      # 周期性保存
│   └── MDiT3D/
│       ├── DiT.pt                      # 每 epoch 覆盖保存 (含 EMA)
│       └── DiT_epoch50.pt             # 周期性保存
├── tensorboard/
│   ├── stage1/                         # CoPeVAE 日志
│   └── stage2/                         # MDiT3D 日志
├── stage1_CoPeVAE/log/                 # Stage 1 文本日志
└── stage2_MDiT3D/log/                  # Stage 2 文本日志
```

### 6.4 训练过程监控

```bash
# TensorBoard（推荐）
tensorboard --logdir $COMPARE_ROOT/results/task_*/tensorboard/ --port 6006

# 监控指标:
#   Stage 1: Loss/train_total, Loss/train_rec, Loss/train_vq, Loss/train_per,
#            Loss/train_disc, Loss/train_pretext, Loss/train_len/loc/con,
#            Loss/val_total, Metrics/val_psnr/ssim/mae/lpips, LR/lr
#   Stage 2: Loss/train_diff, Loss/val_diff, LR/lr

# GPU 使用
watch -n 1 nvidia-smi

# 日志文件
tail -f $COMPARE_ROOT/results/task_*/stage2_MDiT3D/log/*.log
```

---

## 7 推理命令 (eval.py)

> **eval.py 负责两件事**：① 加载模型 → 14 种 mask 逐样本生成 → 保存 NIfTI；② 调用 `ImageQualityEvaluator` 计算指标。
> 可通过 `--metrics_only` 跳过推理，仅重算指标。

### 7.1 方式 A：一键启动

```bash
cd $COMPARE_ROOT

# 完整推理 + 指标
bash run_eval.sh \
    --ae_ckpt results/task_XXXXXX/models/CoPeVAE/Autoencoder.pt \
    --dit_ckpt results/task_XXXXXX/models/MDiT3D/DiT.pt
```

### 7.2 方式 B：直接调用 eval.py

```bash
cd $COMPARE_ROOT

# ============================================================
# 完整推理 (73 samples × 14 masks = 1022 generations)
# 预计耗时: 6-12 小时 (单 GPU, DDIM 200 steps)
# ============================================================
python eval.py \
    --ae_ckpt results/task_XXXXXX/models/CoPeVAE/Autoencoder.pt \
    --dit_ckpt results/task_XXXXXX/models/MDiT3D/DiT.pt \
    --data_root "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
    --task_dir "$COMPARE_ROOT/results/task_eval_manual" \
    --sample_steps 200 \
    --pred_type x \
    --batch_size 1

# ============================================================
# 仅重算指标（跳过推理，需要已有 prediction 目录）
# ============================================================
python eval.py \
    --ae_ckpt results/task_XXXXXX/models/CoPeVAE/Autoencoder.pt \
    --dit_ckpt results/task_XXXXXX/models/MDiT3D/DiT.pt \
    --task_dir "$COMPARE_ROOT/results/task_XXXXXX" \
    --metrics_only
```

### 7.3 推理输出结构

```
results/task_{timestamp}/
├── prediction/
│   ├── 1/                        # mask=0001: 仅 t2 可用 (3-missing)
│   │   ├── input/                # 可用模态 → 模型输入
│   │   │   └── BraTS20_Training_137/
│   │   │       └── BraTS20_Training_137_t2.nii.gz
│   │   ├── ground_truth/         # 缺失模态 → 原始真值
│   │   │   └── BraTS20_Training_137/
│   │   │       ├── BraTS20_Training_137_flair.nii.gz
│   │   │       ├── BraTS20_Training_137_t1.nii.gz
│   │   │       └── BraTS20_Training_137_t1ce.nii.gz
│   │   └── prediction/           # 缺失模态 → 模型生成
│   │       └── BraTS20_Training_137/
│   │           ├── BraTS20_Training_137_flair.nii.gz
│   │           ├── BraTS20_Training_137_t1.nii.gz
│   │           └── BraTS20_Training_137_t1ce.nii.gz
│   ├── 2/                        # mask=0010: 仅 t1ce 可用
│   ├── ...
│   └── 14/                       # mask=1110: 缺 t2 (1-missing)
│
└── prediction_metric_result/
    ├── 1/result.txt
    ├── 2/result.txt
    ├── ...
    └── 14/result.txt
```

### 7.4 result.txt 格式

```
ssim     psnr     mse      mae      fid     lpips
0.891234   28.456789  0.001234  0.012345  -1.000000   0.056789
```

> **注意**: `fid` 为 -1 表示未计算（3D 体数据无法逐样本计算 FID）。

---

## 8 查看与汇总结果

```bash
TASK_TS="XXXXXX"  # 替换为实际时间戳
METRIC_DIR="$COMPARE_ROOT/results/task_$TASK_TS/prediction_metric_result"

# ============================================================
# 逐 mask 查看
# ============================================================
for mask_id in $(seq 1 14); do
    f="$METRIC_DIR/$mask_id/result.txt"
    if [ -f "$f" ]; then
        printf "Mask %-2d: %s\n" "$mask_id" "$(tail -1 "$f")"
    fi
done

# ============================================================
# 按缺失数分类汇总
# ============================================================
echo ""
echo "=============================================="
echo " CoPeDiT BraTS2020 评估结果汇总"
echo "=============================================="
printf "%-8s %-8s %-10s %-10s %-12s %-12s %-10s\n" \
    "mask" "type" "ssim" "psnr" "mse" "mae" "lpips"
echo "--------------------------------------------------------------------"

ONE_MISS=(11 12 13 14)
TWO_MISS=(5 6 7 8 9 10)
THREE_MISS=(1 2 3 4)

for mask_id in "${ONE_MISS[@]}"; do
    f="$METRIC_DIR/$mask_id/result.txt"
    vals=$(tail -1 "$f" 2>/dev/null || echo "N/A")
    printf "%-8s %-8s %s\n" "$mask_id" "1-miss" "$vals"
done
for mask_id in "${TWO_MISS[@]}"; do
    f="$METRIC_DIR/$mask_id/result.txt"
    vals=$(tail -1 "$f" 2>/dev/null || echo "N/A")
    printf "%-8s %-8s %s\n" "$mask_id" "2-miss" "$vals"
done
for mask_id in "${THREE_MISS[@]}"; do
    f="$METRIC_DIR/$mask_id/result.txt"
    vals=$(tail -1 "$f" 2>/dev/null || echo "N/A")
    printf "%-8s %-8s %s\n" "$mask_id" "3-miss" "$vals"
done
echo "--------------------------------------------------------------------"
```

---

## 9 完整端到端流程

```bash
# ============================================================
# 从零开始的完整流程
# ============================================================
set -e

export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/CoPeDiT__arxiv2026
export DATA_ROOT=/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
export DATALIST_DIR=$COMPARE_ROOT/datalist/BraTS2020
cd $COMPARE_ROOT

# Step 1: 训练 Stage 1 (CoPeVAE) — ~12-24 小时
echo "================================"
echo " Step 1/4: 训练 CoPeVAE"
echo "================================"
python train_CoPeVAE_Brain_adapted.py \
    --data_root "$DATA_ROOT" --datalist_dir "$DATALIST_DIR" \
    --epochs 200 --batch_size 2 --lr 1e-4 \
    --val_interval 5 --ckpt_interval 50

# 获取 Stage 1 的 task 时间戳
TASK_TS=$(ls -t $COMPARE_ROOT/results/task_* 2>/dev/null | head -1 | grep -oP 'task_\K.*')
echo "Task timestamp: $TASK_TS"

# Step 2: 训练 Stage 2 (MDiT3D) — ~12-24 小时
echo ""
echo "================================"
echo " Step 2/4: 训练 MDiT3D"
echo "================================"
python train_MDiT3D_Brain_adapted.py \
    --data_root "$DATA_ROOT" --datalist_dir "$DATALIST_DIR" \
    --ae_ckpt "$COMPARE_ROOT/results/task_$TASK_TS/models/CoPeVAE/Autoencoder.pt" \
    --epochs 200 --batch_size 2 --missing_num 1 --lr 5e-5 \
    --val_interval 20 --ckpt_interval 50

# Step 3: 推理生成 — ~6-12 小时
echo ""
echo "================================"
echo " Step 3/4: 推理 (14 种 mask)"
echo "================================"
python eval.py \
    --ae_ckpt "$COMPARE_ROOT/results/task_$TASK_TS/models/CoPeVAE/Autoencoder.pt" \
    --dit_ckpt "$COMPARE_ROOT/results/task_$TASK_TS/models/MDiT3D/DiT.pt" \
    --data_root "$DATA_ROOT" --datalist_dir "$DATALIST_DIR" \
    --task_dir "$COMPARE_ROOT/results/task_$TASK_TS"

# Step 4: 查看结果
echo ""
echo "================================"
echo " Step 4/4: 查看指标"
echo "================================"
for mask_id in $(seq 1 14); do
    f="$COMPARE_ROOT/results/task_$TASK_TS/prediction_metric_result/$mask_id/result.txt"
    if [ -f "$f" ]; then
        printf "Mask %-2d: %s\n" "$mask_id" "$(tail -1 "$f")"
    fi
done

echo ""
echo "================================"
echo " ✅ 全部完成！"
echo " 模型: $COMPARE_ROOT/results/task_$TASK_TS/models/"
echo " 预测: $COMPARE_ROOT/results/task_$TASK_TS/prediction/"
echo " 指标: $COMPARE_ROOT/results/task_$TASK_TS/prediction_metric_result/"
echo "================================"
```

---

## 10 关键超参数说明

| 参数 | Stage 1 默认 | Stage 2 默认 | 说明 |
|------|-------------|-------------|------|
| `--epochs` | 200 | 200 | 训练轮数 |
| `--batch_size` | 2 | 2 | OOM 时降为 1 |
| `--lr` | 1e-4 | 5e-5 | 学习率 |
| `--lr_min` | 8e-5 | 2e-5 | Cosine 最低学习率 |
| `--warmup_steps` | 5 | 50 | LR 预热步数 |
| `--val_interval` | 5 | 20 | 每隔 N epoch 验证 |
| `--ckpt_interval` | 50 | 50 | 每隔 N epoch 保存 period ckpt |
| `--gradient_accumulation_steps` | 2 | 2 | 梯度累积 |
| `--missing_num` | — | 1 | Stage 2 固定缺失数 (1/2/3) |
| `--pred_type` | — | x | 预测目标: noise/x/v |
| `--diff_loss` | — | l2 | 扩散损失: l1/l2 |
| `--sample_steps` | — | 200 | DDIM 采样步数 (仅推理) |

---

## 11 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `CUDA out of memory` | batch_size=2 显存不足 | `--batch_size 1 --gradient_accumulation_steps 4` |
| `ModuleNotFoundError: No module named 'generative'` | monai-generative 未安装 | `pip install monai-generative==0.2.3` |
| `FileNotFoundError: Datalist not found` | datalist 路径不对 | 确认 `$DATALIST_DIR/{train,val,test}.list` 存在 |
| `RuntimeError: shape mismatch` | 训练/推理 noise 维度不一致 | 检查 --missing_num 与 checkpoint 训练设置匹配 |
| 训练 loss 不下降 | VQ-VAE codebook collapse | 增大 `--vq_weight` (如 2.0)，减小 `--lr` |
| `xformers not available` | 库未安装 | 自动回退 `F.scaled_dot_product_attention`，无需处理 |
| DDP 初始化失败 | 多 GPU 通信问题 | 默认单 GPU，加 `--distributed` 才开启 DDP |
| `ImportError: syn_metrics` | 外部指标脚本路径不对 | 检查 `$EVAL_SCRIPT` 环境变量或 `--eval_script` 参数 |
| eval.py 推理结果为空 | 数据加载失败 | 检查 `$DATA_ROOT/{sid}/{sid}_{mod}.nii` 文件命名 |
| 训练脚本加载了 test 数据 | 已修复 | train 脚本使用 `read_train_val_datalist`，不接触 test |
