#!/usr/bin/env python
"""
Evaluation Script for CoPeDiT on BraTS2020.

For each test sample:
- Generate missing modalities for all 14 missing-modality masks
- Save inputs (available), ground truth (missing), predictions (generated)
- Compute image quality metrics using the unified EVAL_SCRIPT

Usage:
    python eval.py --ae_ckpt <CoPeVAE checkpoint> --dit_ckpt <MDiT3D checkpoint>

Directory structure per mask_id:
    results/task_{timestamp}/
        prediction/{mask_id}/
            input/           # Available modality images
            ground_truth/    # GT of missing modalities
            prediction/      # Generated missing modalities
        prediction_metric_result/{mask_id}/result.txt
"""

import argparse
import os
import sys
import timeit
import numpy as np
import torch
import nibabel as nib
from datetime import datetime
from tqdm import tqdm
from einops import rearrange

from torch.cuda.amp import autocast

# Add project root to path
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from LDM.model.MDiT3D_Brain import MDiT3D_models
from LDM.inferers import inferer_DiT_Brain
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.networks.schedulers.ddim import DDIMPredictionType

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, EnsureType,
    Orientation, ScaleIntensityRangePercentilesd,
    CenterSpatialCrop, Resize, ToTensor,
)
from monai.data import DataLoader, Dataset

from data.BraTS2020_data import (
    MODALITY_KEYS, MASK_LIST, mask_to_string, read_datalist,
    get_transforms, fixed_mask_sample,
)


def get_eval_transforms(args):
    """
    Build MONAI transforms for evaluation (same as test transforms but also
    usable for individual modality loading).
    """
    from monai.transforms import Compose as ComposeD
    keys = MODALITY_KEYS
    transform = ComposeD([
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        EnsureTyped(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        ScaleIntensityRangePercentilesd(
            keys=keys, lower=0, upper=99.5, b_min=0, b_max=1,
            allow_missing_keys=True,
        ),
        CenterSpatialCropd(
            keys=keys, roi_size=args.brain_roi, allow_missing_keys=True,
        ),
        Resized(
            keys=keys, spatial_size=args.brain_size,
            mode="trilinear", align_corners=True,
            allow_missing_keys=True,
        ),
        ToTensord(keys=keys, allow_missing_keys=True),
    ])
    return transform


def save_nifti(data, affine, filepath, is_tensor=True):
    """
    Save a 3D volume as NIfTI.

    Args:
        data: numpy array or torch tensor of shape (H, W, D)
        affine: affine matrix
        filepath: output path
    """
    if is_tensor:
        data = data.detach().cpu().numpy()
    data = np.squeeze(data).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nib.save(img, filepath)


def compute_metrics(args, test_files, prediction_root, metric_result_root, device):
    """Compute image quality metrics from saved predictions (no model needed)."""
    print("\n" + "=" * 60)
    print("Computing image quality metrics...")
    print("=" * 60)

    eval_script_path = os.environ.get("EVAL_SCRIPT", args.eval_script)
    eval_dir = os.path.dirname(eval_script_path)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    try:
        from syn_metrics import ImageQualityEvaluator
        evaluator = ImageQualityEvaluator(
            LPIPS_model_type='nomedical', device=str(device)
        )
    except ImportError as e:
        print(f"Warning: Could not import from EVAL_SCRIPT: {e}")
        print("Attempting direct import...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "syn_metrics", eval_script_path
        )
        syn_metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(syn_metrics)
        evaluator = syn_metrics.ImageQualityEvaluator(
            LPIPS_model_type='nomedical', device=str(device)
        )

    for mask_idx, mask in enumerate(MASK_LIST):
        mask_id = mask_idx + 1
        mask_str = mask_to_string(mask)
        missing_idx = [i for i, m in enumerate(mask) if m == 0]

        print(f"\n--- Mask {mask_id}/{len(MASK_LIST)}: {mask_str} ---")

        mask_pred_dir = os.path.join(prediction_root, str(mask_id), "prediction")
        mask_gt_dir = os.path.join(prediction_root, str(mask_id), "ground_truth")

        all_metrics = {"ssim": [], "psnr": [], "mse": [], "mae": [], "lpips": []}

        for item in test_files:
            subject_id = item["subject_id"]

            for mi in missing_idx:
                mod_name = MODALITY_KEYS[mi]

                pred_path = os.path.join(
                    mask_pred_dir, subject_id, f"{subject_id}_{mod_name}.nii.gz"
                )
                gt_path = os.path.join(
                    mask_gt_dir, subject_id, f"{subject_id}_{mod_name}.nii.gz"
                )

                if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                    continue

                try:
                    pred_nii = nib.load(pred_path)
                    gt_nii = nib.load(gt_path)

                    pred_data = pred_nii.get_fdata().astype(np.float32)
                    gt_data = gt_nii.get_fdata().astype(np.float32)

                    pred_data = (pred_data - pred_data.min()) / (
                        pred_data.max() - pred_data.min() + 1e-8
                    )
                    pred_data = 2 * pred_data - 1
                    gt_data = (gt_data - gt_data.min()) / (
                        gt_data.max() - gt_data.min() + 1e-8
                    )
                    gt_data = 2 * gt_data - 1

                    pred_tensor = torch.from_numpy(pred_data).unsqueeze(0).unsqueeze(0).to(device)
                    gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0).to(device)

                    metrics = evaluator.evaluate_all_metrics(pred_tensor, gt_tensor)
                    for k in all_metrics:
                        all_metrics[k].append(metrics[k])
                except Exception as e:
                    print(f"  Warning: Error processing {subject_id}/{mod_name}: {e}")

        avg_metrics = {
            k: np.mean(v) if v else float("nan") for k, v in all_metrics.items()
        }
        avg_metrics["fid"] = -1.0

        result_dir = os.path.join(metric_result_root, str(mask_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "result.txt")

        table_head = "ssim     psnr     mse      mae      fid     lpips"
        table = (
            f"{avg_metrics['ssim']:.6f}   {avg_metrics['psnr']:.6f}  "
            f"{avg_metrics['mse']:.6f}  {avg_metrics['mae']:.6f}  "
            f"{avg_metrics['fid']:.6f}   {avg_metrics['lpips']:.6f}"
        )

        with open(result_path, "w") as f:
            f.write(table_head + "\n")
            f.write(table + "\n")

        print(f"  Results: {table}")
        print(f"  Saved to: {result_path}")


def inference(args):
    """
    Main entry point: either run inference (generate missing modalities for all
    14 masks on all test samples) or --metrics_only (recompute metrics from
    existing predictions).
    """
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load test data (once, reused) ----
    _, _, test_files = read_datalist(args.datalist_dir, args.data_root)
    print(f"Loaded {len(test_files)} test samples")

    # ---- Output directories ----
    prediction_root = os.path.join(args.task_dir, "prediction")
    metric_result_root = os.path.join(args.task_dir, "prediction_metric_result")

    # ---- Metrics-only mode ----
    if args.metrics_only:
        os.makedirs(metric_result_root, exist_ok=True)
        compute_metrics(args, test_files, prediction_root, metric_result_root, device)
        print("\n" + "=" * 60)
        print("Metrics computation complete!")
        print(f"Metrics: {metric_result_root}")
        print("=" * 60)
        return

    # ---- Full inference mode ----
    os.makedirs(prediction_root, exist_ok=True)
    os.makedirs(metric_result_root, exist_ok=True)

    # ---- Load models ----
    print("Loading CoPeVAE model...")
    autoencoder = CopeVAE(args).to(device)
    if not os.path.exists(args.ae_ckpt):
        raise FileNotFoundError(f"CoPeVAE checkpoint not found: {args.ae_ckpt}")
    ae_state = torch.load(args.ae_ckpt, map_location=device)
    autoencoder.load_state_dict(ae_state["model"])
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    print(f"CoPeVAE loaded from {args.ae_ckpt} (epoch {ae_state.get('epoch', 'N/A')})")

    print("Loading MDiT3D model...")
    DiT = MDiT3D_models[args.DiT](num_modalities=args.modality_num).to(device)
    if not os.path.exists(args.dit_ckpt):
        raise FileNotFoundError(f"MDiT3D checkpoint not found: {args.dit_ckpt}")
    dit_state = torch.load(args.dit_ckpt, map_location=device)
    # Load EMA weights (preferred) or raw model weights
    if "ema" in dit_state:
        DiT.load_state_dict(dit_state["ema"])
        print("Loaded EMA weights")
    else:
        DiT.load_state_dict(dit_state["model"])
        print("Loaded raw model weights")
    DiT.eval()
    DiT.requires_grad_(False)
    print(f"MDiT3D loaded from {args.dit_ckpt} (epoch {dit_state.get('epoch', 'N/A')})")

    # ---- Setup scheduler and inferer ----
    if args.noise_scheduler == "linear":
        if args.pred_type == "noise":
            scheduler_ddim = DDIMScheduler(
                num_train_timesteps=args.sample_steps,
                schedule="scaled_linear_beta",
                beta_start=0.0015, beta_end=0.0195,
                clip_sample=False,
            )
        elif args.pred_type == "x":
            scheduler_ddim = DDIMScheduler(
                num_train_timesteps=args.sample_steps,
                schedule="scaled_linear_beta",
                beta_start=0.0015, beta_end=0.0195,
                clip_sample=False,
                prediction_type=DDIMPredictionType.SAMPLE,
            )
        elif args.pred_type == "v":
            scheduler_ddim = DDIMScheduler(
                num_train_timesteps=args.sample_steps,
                schedule="scaled_linear_beta",
                beta_start=0.0015, beta_end=0.0195,
                clip_sample=False,
                prediction_type=DDIMPredictionType.V_PREDICTION,
            )
    elif args.noise_scheduler == "cosine":
        if args.pred_type == "noise":
            scheduler_ddim = DDIMScheduler(
                num_train_timesteps=args.sample_steps,
                schedule="cosine", clip_sample=True,
                clip_sample_min=0, clip_sample_max=1,
            )
        elif args.pred_type == "x":
            scheduler_ddim = DDIMScheduler(
                num_train_timesteps=args.sample_steps,
                schedule="cosine", clip_sample=True,
                clip_sample_min=0, clip_sample_max=1,
                prediction_type=DDIMPredictionType.SAMPLE,
            )
        elif args.pred_type == "v":
            scheduler_ddim = DDIMScheduler(
                num_train_timesteps=args.sample_steps,
                schedule="cosine", clip_sample=True,
                clip_sample_min=0, clip_sample_max=1,
                prediction_type=DDIMPredictionType.V_PREDICTION,
            )

    scheduler_ddim.set_timesteps(num_inference_steps=args.sample_steps)
    scale_factor = 1.0
    inferer = inferer_DiT_Brain.LatentDiffusionInferer(
        scheduler_ddim, scale_factor=scale_factor
    )

    eval_transform = get_eval_transforms(args)

    # ---- Generate for each test sample and each mask ----
    start = timeit.default_timer()

    for sample_idx, item in enumerate(tqdm(test_files, desc="Processing samples")):
        subject_id = item["subject_id"]

        # Load and preprocess all 4 modalities (batch_size=1: one sample at a time)
        try:
            data = eval_transform(item)
        except Exception as e:
            print(f"Warning: Failed to load {subject_id}: {e}")
            continue

        # Read affine from original NIfTI for saving
        ref_path = item[MODALITY_KEYS[0]]
        ref_img = nib.load(ref_path)
        affine = ref_img.affine

        # Stack all modalities: [1, 4, H, W, D]
        full_img = torch.stack(
            [data[k] for k in MODALITY_KEYS], dim=1
        )  # [1, 4, H, W, D]

        # Process all 14 masks
        for mask_idx, mask in enumerate(MASK_LIST):
            mask_id = mask_idx + 1
            mask_str = mask_to_string(mask)

            available_idx = [i for i, m in enumerate(mask) if m == 1]
            missing_idx = [i for i, m in enumerate(mask) if m == 0]

            if len(available_idx) == 0 or len(missing_idx) == 0:
                continue  # Should not happen with our mask list

            x_available = full_img[:, available_idx, ...].to(device)
            x_missing_gt = full_img[:, missing_idx, ...].to(device)

            # Create noise for missing modalities
            with autocast(enabled=True):
                # Encode available modalities to get latent shape
                latent_avail = autoencoder.encode_stage_2_inputs(
                    x_available
                ) * scale_factor
                # latent_avail: [1, N_avail, latent_dim=8, h, w, d]
                _, _, latent_dim, h_lat, w_lat, d_lat = latent_avail.shape

                # Noise shape must match: [B, N_miss, latent_dim, h, w, d]
                noise = torch.randn(
                    (1, len(missing_idx), latent_dim, h_lat, w_lat, d_lat),
                    device=device
                )

                prompts = autoencoder.get_condition(x_available)

                generated = inferer.sample(
                    input_a=x_available,
                    input_noise=noise,
                    autoencoder_model=autoencoder,
                    diffusion_model=DiT,
                    scheduler=scheduler_ddim,
                    conditioning=prompts,
                    verbose=False,
                )
                # generated: [1, N_miss, H, W, D]

            # ---- Save outputs ----
            mask_pred_dir = os.path.join(
                prediction_root, str(mask_id), "prediction", subject_id
            )
            mask_input_dir = os.path.join(
                prediction_root, str(mask_id), "input", subject_id
            )
            mask_gt_dir = os.path.join(
                prediction_root, str(mask_id), "ground_truth", subject_id
            )

            for mi_idx, mi in enumerate(missing_idx):
                mod_name = MODALITY_KEYS[mi]
                gen_data = generated[0, mi_idx, ...]
                save_nifti(
                    gen_data, affine,
                    os.path.join(mask_pred_dir, f"{subject_id}_{mod_name}.nii.gz"),
                )

            for mi_idx, mi in enumerate(missing_idx):
                mod_name = MODALITY_KEYS[mi]
                gt_data = x_missing_gt[0, mi_idx, ...]
                save_nifti(
                    gt_data, affine,
                    os.path.join(mask_gt_dir, f"{subject_id}_{mod_name}.nii.gz"),
                )

            for ai_idx, ai in enumerate(available_idx):
                mod_name = MODALITY_KEYS[ai]
                in_data = x_available[0, ai_idx, ...]
                save_nifti(
                    in_data, affine,
                    os.path.join(mask_input_dir, f"{subject_id}_{mod_name}.nii.gz"),
                )

            torch.cuda.empty_cache()

    end = timeit.default_timer()
    elapsed = end - start
    print(f"\nInference completed in {elapsed:.1f}s")

    # ---- Compute metrics from saved predictions ----
    compute_metrics(args, test_files, prediction_root, metric_result_root, device)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Predictions:  {prediction_root}")
    print(f"Metrics:      {metric_result_root}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoPeDiT Evaluation on BraTS2020")

    parser.add_argument(
        "--ae_ckpt", required=True, type=str,
        help="Path to trained CoPeVAE checkpoint",
    )
    parser.add_argument(
        "--dit_ckpt", required=True, type=str,
        help="Path to trained MDiT3D checkpoint",
    )

    parser.add_argument(
        "--data_root",
        default=os.environ.get(
            "DATA_ROOT",
            "/devdata/hsh/datasets/seg_dataset/BraTS2020/"
            "brats20-dataset-training-validation/versions/1/"
            "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        ),
        type=str, help="Root path to BraTS2020 TrainingData (env: DATA_ROOT)",
    )
    parser.add_argument(
        "--datalist_dir",
        default=os.environ.get(
            "DATALIST_DIR",
            os.path.join(
                os.environ.get(
                    "COMPARE_ROOT",
                    os.path.dirname(os.path.abspath(__file__)),
                ),
                "datalist/BraTS2020",
            ),
        ),
        type=str, help="Directory containing test.list (env: DATALIST_DIR)",
    )
    parser.add_argument(
        "--eval_script",
        default=os.environ.get(
            "EVAL_SCRIPT",
            "/devdata2/hsh/program/python/methods/MySparseDiffusion/"
            "my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py",
        ),
        type=str, help="Path to syn_metrics.py (env: EVAL_SCRIPT)",
    )

    parser.add_argument(
        "--task_dir",
        default=None,
        type=str,
        help="Task directory for outputs (default: $COMPARE_ROOT/results/task_{timestamp})",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Inference batch size (fixed at 1 for per-sample generation)",
    )
    parser.add_argument(
        "--metrics_only",
        default=False,
        action="store_true",
        help="Skip inference, only re-compute metrics from existing predictions",
    )

    parser.add_argument("--DiT", default="MDiT3D-B/2", type=str)
    parser.add_argument("--noise_scheduler", default="linear", type=str)
    parser.add_argument(
        "--pred_type", default="x", type=str,
        choices=["noise", "x", "v"],
    )
    parser.add_argument("--diffusion_steps", default=500, type=int)
    parser.add_argument("--sample_steps", default=200, type=int)

    parser.add_argument("--spatial_dims", default=3, type=int)
    parser.add_argument("--vae_channel", default=(256, 384, 512))
    parser.add_argument("--res_channel", default=256, type=int)
    parser.add_argument("--code_num", default=8192, type=int)
    parser.add_argument("--code_dim", default=8, type=int)
    parser.add_argument("--recon_loss", default="l1", type=str)
    parser.add_argument("--brain_pad", default=(240, 240, 128))
    parser.add_argument("--brain_roi", default=(192, 192, 80))
    parser.add_argument("--brain_size", default=(192, 192, 64))
    parser.add_argument("--modality_num", default=4, type=int)
    parser.add_argument("--latent_dim", default=8, type=int)
    parser.add_argument("--proj_dim", default=512, type=int)
    parser.add_argument("--contrast_dim", default=128, type=int)
    parser.add_argument("--lambda1", default=1.0, type=float)
    parser.add_argument("--lambda2", default=0.1, type=float)
    parser.add_argument("--lambda3", default=1.0, type=float)
    parser.add_argument("--vq_weight", default=1.0, type=float)
    parser.add_argument("--per_weight", default=0.01, type=float)
    parser.add_argument("--adv_weight", default=0.01, type=float)
    parser.add_argument("--pretext_weight", default=0.1, type=float)
    parser.add_argument("--dataset", default="BraTS", type=str)
    parser.add_argument("--missing_num", default=1, type=int)

    args = parser.parse_args()

    for attr in ["vae_channel", "brain_pad", "brain_roi", "brain_size"]:
        val = getattr(args, attr)
        if isinstance(val, str):
            setattr(args, attr, tuple(map(int, val.strip("()").split(","))))

    if args.task_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.task_dir = os.path.join(PROJ_ROOT, f"results/task_{timestamp}")

    os.makedirs(args.task_dir, exist_ok=True)
    print(f"Task directory: {args.task_dir}")

    inference(args)
