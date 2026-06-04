#!/bin/bash
# ==============================================================================
# CoPeDiT Training Launch Script for BraTS2020
# ==============================================================================
#
# Two-stage training pipeline:
#   Stage 1: CoPeVAE — completeness-aware VQ-VAE tokenizer
#   Stage 2: MDiT3D — diffusion transformer for missing modality synthesis
#
# Usage:
#   bash run_train.sh                          # Train both stages
#   bash run_train.sh --stage1-only            # Train only Stage 1 (CoPeVAE)
#   bash run_train.sh --stage2-only            # Train only Stage 2 (MDiT3D)
#   bash run_train.sh --batch_size 1           # Override batch size (if OOM)
#   bash run_train.sh --epochs 100             # Override number of epochs
#
# Environment variables (optional):
#   DATA_ROOT       Path to BraTS2020 TrainingData
#   DATALIST_DIR    Path to datalist directory
#   CUDA_VISIBLE_DEVICES  GPU device ID (default: 0)
#
# Hardware: Quadro RTX 8000 48G
# Estimated training time (200 epochs, single GPU): ~24-48 hours total
# ==============================================================================

set -e  # Exit on error

# ---- Path Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Data paths
DATA_ROOT="${DATA_ROOT:-/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
DATALIST_DIR="${DATALIST_DIR:-$SCRIPT_DIR/datalist/BraTS2020}"

# GPU
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# ---- Default Hyperparameters ----
BATCH_SIZE_STAGE1=2      # Official value; reduce to 1 if OOM
BATCH_SIZE_STAGE2=2      # Official value; reduce to 1 if OOM
EPOCHS=200               # Unified 200 epochs
STAGE1_ONLY=false
STAGE2_ONLY=false

# ---- Parse CLI Arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage1-only)
            STAGE1_ONLY=true
            shift
            ;;
        --stage2-only)
            STAGE2_ONLY=true
            shift
            ;;
        --batch_size)
            BATCH_SIZE_STAGE1="$2"
            BATCH_SIZE_STAGE2="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --ae_ckpt)
            AE_CKPT="$2"
            shift 2
            ;;
        --missing_num)
            MISSING_NUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_train.sh [--stage1-only|--stage2-only] [--batch_size N] [--epochs N] [--ae_ckpt PATH] [--missing_num 1|2|3]"
            exit 1
            ;;
    esac
done

MISSING_NUM="${MISSING_NUM:-1}"

echo "=============================================="
echo " CoPeDiT Training on BraTS2020"
echo "=============================================="
echo "Data root:      $DATA_ROOT"
echo "Datalist dir:   $DATALIST_DIR"
echo "GPU device:     $CUDA_VISIBLE_DEVICES"
echo "Epochs:         $EPOCHS"
echo "Stage1 batch:   $BATCH_SIZE_STAGE1"
echo "Stage2 batch:   $BATCH_SIZE_STAGE2"
echo "Missing num:    $MISSING_NUM"
echo "=============================================="

# ---- Stage 1: CoPeVAE Training ----
if [ "$STAGE2_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo " Stage 1: Training CoPeVAE"
    echo "=============================================="

    python train_CoPeVAE_Brain_adapted.py \
        --data_root "$DATA_ROOT" \
        --datalist_dir "$DATALIST_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE_STAGE1" \
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

    STAGE1_EXIT_CODE=$?
    if [ $STAGE1_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Stage 1 training failed with exit code $STAGE1_EXIT_CODE"
        exit $STAGE1_EXIT_CODE
    fi
    echo "Stage 1 training completed successfully."
fi

# ---- Find latest CoPeVAE checkpoint ----
if [ -z "$AE_CKPT" ]; then
    LATEST_RESULTS=$(ls -dt "$SCRIPT_DIR"/results/task_*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RESULTS" ]; then
        echo "ERROR: No results directory found. Stage 1 training may have failed."
        exit 1
    fi
    AE_CKPT="$LATEST_RESULTS/models/CoPeVAE/Autoencoder.pt"
    if [ ! -f "$AE_CKPT" ]; then
        AE_CKPT=$(find "$SCRIPT_DIR"/results -name "Autoencoder.pt" -path "*/CoPeVAE/*" 2>/dev/null | sort | tail -1)
    fi
fi

echo "Using CoPeVAE checkpoint: $AE_CKPT"
if [ ! -f "$AE_CKPT" ]; then
    echo "WARNING: CoPeVAE checkpoint not found at $AE_CKPT"
    echo "Stage 2 will start with randomly initialized autoencoder."
    echo "This is not recommended - please train Stage 1 first."
fi

# ---- Stage 2: MDiT3D Training ----
if [ "$STAGE1_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo " Stage 2: Training MDiT3D"
    echo "=============================================="

    python train_MDiT3D_Brain_adapted.py \
        --data_root "$DATA_ROOT" \
        --datalist_dir "$DATALIST_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE_STAGE2" \
        --lr 5e-5 \
        --lr_min 2e-5 \
        --lr_schedule warmup_cosine \
        --opt adamw \
        --missing_num "$MISSING_NUM" \
        --ae_ckpt "$AE_CKPT" \
        --warmup_steps 50 \
        --val_interval 20 \
        --ckpt_interval 50 \
        --cache 0.2 \
        --gradient_accumulation_steps 2 \
        --pred_type x \
        --diff_loss l2 \
        --seed 2025

    STAGE2_EXIT_CODE=$?
    if [ $STAGE2_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Stage 2 training failed with exit code $STAGE2_EXIT_CODE"
        exit $STAGE2_EXIT_CODE
    fi
    echo "Stage 2 training completed successfully."
fi

echo ""
echo "=============================================="
echo " Training Complete!"
echo "=============================================="
echo ""
echo "Model checkpoints saved in:"
echo "  $SCRIPT_DIR/results/task_*/models/"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir $SCRIPT_DIR/results/task_*/tensorboard/"
echo ""
echo "To run evaluation:"
echo "  bash run_eval.sh --ae_ckpt <path> --dit_ckpt <path>"
echo ""
