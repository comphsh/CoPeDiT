#!/bin/bash
# ==============================================================================
# CoPeDiT Evaluation Launch Script for BraTS2020
# ==============================================================================
#
# Generates missing modalities for all 14 mask patterns on the test set,
# then computes image quality metrics (SSIM/PSNR/MSE/MAE/LPIPS).
#
# Usage:
#   bash run_eval.sh --ae_ckpt <CoPeVAE checkpoint> --dit_ckpt <MDiT3D checkpoint>
#   bash run_eval.sh --ae_ckpt results/task_20250101_120000/models/CoPeVAE/Autoencoder.pt \
#                    --dit_ckpt results/task_20250101_120000/models/MDiT3D/DiT.pt
#
# Options:
#   --ae_ckpt PATH         Path to trained CoPeVAE checkpoint (required)
#   --dit_ckpt PATH        Path to trained MDiT3D checkpoint (required)
#   --task_dir PATH        Output directory (auto: results/task_{timestamp}/)
#   --gpu ID               GPU device ID (default: 0)
#
# Environment variables:
#   DATA_ROOT              Path to BraTS2020 TrainingData
#   DATALIST_DIR           Path to datalist directory
#   EVAL_SCRIPT            Path to evaluation script
#   CUDA_VISIBLE_DEVICES   GPU device ID
# ==============================================================================

set -e

# ---- Path Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Data paths
DATA_ROOT="${DATA_ROOT:-/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
DATALIST_DIR="${DATALIST_DIR:-$SCRIPT_DIR/datalist/BraTS2020}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py}"

# GPU
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# ---- Defaults ----
AE_CKPT=""
DIT_CKPT=""
TASK_DIR=""

# ---- Parse CLI Arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --ae_ckpt)
            AE_CKPT="$2"
            shift 2
            ;;
        --dit_ckpt)
            DIT_CKPT="$2"
            shift 2
            ;;
        --task_dir)
            TASK_DIR="$2"
            shift 2
            ;;
        --gpu)
            CUDA_VISIBLE_DEVICES="$2"
            export CUDA_VISIBLE_DEVICES
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_eval.sh --ae_ckpt <CoPeVAE ckpt> --dit_ckpt <MDiT3D ckpt> [--task_dir PATH] [--gpu ID]"
            exit 1
            ;;
    esac
done

# ---- Validate inputs ----
if [ -z "$AE_CKPT" ]; then
    echo "ERROR: --ae_ckpt is required (CoPeVAE checkpoint path)"
    echo "Usage: bash run_eval.sh --ae_ckpt <path> --dit_ckpt <path>"
    exit 1
fi

if [ -z "$DIT_CKPT" ]; then
    echo "ERROR: --dit_ckpt is required (MDiT3D checkpoint path)"
    echo "Usage: bash run_eval.sh --ae_ckpt <path> --dit_ckpt <path>"
    exit 1
fi

if [ ! -f "$AE_CKPT" ]; then
    echo "ERROR: CoPeVAE checkpoint not found: $AE_CKPT"
    exit 1
fi

if [ ! -f "$DIT_CKPT" ]; then
    echo "ERROR: MDiT3D checkpoint not found: $DIT_CKPT"
    exit 1
fi

echo "=============================================="
echo " CoPeDiT Evaluation on BraTS2020"
echo "=============================================="
echo "Data root:      $DATA_ROOT"
echo "Datalist dir:   $DATALIST_DIR"
echo "Eval script:    $EVAL_SCRIPT"
echo "GPU device:     $CUDA_VISIBLE_DEVICES"
echo "AE checkpoint:  $AE_CKPT"
echo "DiT checkpoint: $DIT_CKPT"
echo "=============================================="

# ---- Determine task directory ----
if [ -z "$TASK_DIR" ]; then
    if [[ "$DIT_CKPT" == */results/task_*/models/* ]]; then
        TASK_DIR=$(echo "$DIT_CKPT" | sed 's|/models/.*||')
        echo "Auto-detected task directory: $TASK_DIR"
    else
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        TASK_DIR="$SCRIPT_DIR/results/task_${TIMESTAMP}_eval"
        echo "Task directory: $TASK_DIR"
    fi
fi

# ---- Run evaluation ----
echo ""
echo "Starting evaluation..."
echo ""

python eval.py \
    --ae_ckpt "$AE_CKPT" \
    --dit_ckpt "$DIT_CKPT" \
    --data_root "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
    --eval_script "$EVAL_SCRIPT" \
    --task_dir "$TASK_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=============================================="
echo " Evaluation Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Predictions: $TASK_DIR/prediction/"
echo "  Metrics:     $TASK_DIR/prediction_metric_result/"
echo ""
echo "To view per-mask results:"
for mask_id in $(seq 1 14); do
    result_file="$TASK_DIR/prediction_metric_result/$mask_id/result.txt"
    if [ -f "$result_file" ]; then
        echo "  Mask $mask_id: $(cat "$result_file" | tail -1)"
    fi
done
echo ""
