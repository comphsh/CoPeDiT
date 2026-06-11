#!/usr/bin/env python
"""
CoPeDiT Inference Script for BraTS2020 Test Set.

For each test sample × 14 missing-modality masks:
- Generate missing modalities via DDIM sampling
- Save ONLY synthetic images (no GT — global dataset serves as GT)

Output structure (mask_str = 4-bit binary, 1=available, 0=missing):
    results/task_{timestamp}/prediction/
        {mask_str}/
            {subject_id}/
                {subject_id}_{missing_mod}.nii.gz    # synthetic image only

Usage:
    python eval.py --ae_ckpt <CoPeVAE ckpt> --dit_ckpt <MDiT3D ckpt>

Environment:
    COMPARE_ROOT   Project root (default: script dir)
    DATA_ROOT      BraTS2020 TrainingData path
    DATALIST_DIR   Directory containing test.list
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

from torch.cuda.amp import autocast

# Project root — respect COMPARE_ROOT env
PROJ_ROOT = os.environ.get("COMPARE_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from LDM.model.MDiT3D_Brain import MDiT3D_models
from LDM.inferers import inferer_DiT_Brain
from generative.networks.schedulers import DDIMScheduler
from generative.networks.schedulers.ddim import DDIMPredictionType

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, ScaleIntensityRangePercentilesd,
    CenterSpatialCropd, Resized, ToTensord,
)

from data.BraTS2020_data import (
    MODALITY_KEYS, MASK_LIST, mask_to_string, read_datalist,
)


def get_eval_transforms(args):
    """Build MONAI transforms for evaluation (batch_size=1 per-sample loading)."""
    keys = MODALITY_KEYS
    transform = Compose([
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
    """Save a 3D volume as NIfTI."""
    if is_tensor:
        data = data.detach().cpu().numpy()
    data = np.squeeze(data).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nib.save(img, filepath)


def inference(args):
    """
    Main inference: generate missing modalities for all 14 masks
    on every test sample. Saves only synthetic images.
    """
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load test data (test split only, never train/val) ----
    _, _, test_files = read_datalist(args.datalist_dir, args.data_root)
    print(f"Loaded {len(test_files)} test samples")

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
        extra = {}
        if args.pred_type == "x":
            extra["prediction_type"] = DDIMPredictionType.SAMPLE
        elif args.pred_type == "v":
            extra["prediction_type"] = DDIMPredictionType.V_PREDICTION
        scheduler_ddim = DDIMScheduler(
            num_train_timesteps=args.sample_steps,
            schedule="scaled_linear_beta",
            beta_start=0.0015, beta_end=0.0195,
            clip_sample=False,
            **extra,
        )
    elif args.noise_scheduler == "cosine":
        extra = {"clip_sample": True, "clip_sample_min": 0, "clip_sample_max": 1}
        if args.pred_type == "x":
            extra["prediction_type"] = DDIMPredictionType.SAMPLE
        elif args.pred_type == "v":
            extra["prediction_type"] = DDIMPredictionType.V_PREDICTION
        scheduler_ddim = DDIMScheduler(
            num_train_timesteps=args.sample_steps,
            schedule="cosine",
            **extra,
        )

    scheduler_ddim.set_timesteps(num_inference_steps=args.sample_steps)
    scale_factor = 1.0
    inferer = inferer_DiT_Brain.LatentDiffusionInferer(
        scheduler_ddim, scale_factor=scale_factor
    )

    eval_transform = get_eval_transforms(args)

    # ---- Output directory ----
    prediction_root = os.path.join(args.task_dir, "prediction")
    os.makedirs(prediction_root, exist_ok=True)

    total_tasks = len(test_files) * len(MASK_LIST)
    print(f"Starting inference: {len(test_files)} patients × {len(MASK_LIST)} masks = {total_tasks} generations")

    # ---- Generate for each test sample × each mask ----
    start = timeit.default_timer()

    for sample_idx, item in enumerate(tqdm(test_files, desc="Patients")):
        subject_id = item["subject_id"]

        # Load and preprocess all 4 modalities (batch_size=1)
        try:
            data = eval_transform(item)
        except Exception as e:
            print(f"Warning: Failed to load {subject_id}: {e}")
            continue

        # Read affine from original NIfTI for output
        ref_path = item[MODALITY_KEYS[0]]
        ref_img = nib.load(ref_path)
        affine = ref_img.affine

        # Stack all modalities: [1, 4, H, W, D]
        full_img = torch.stack(
            [data[k] for k in MODALITY_KEYS], dim=1
        )  # [1, 4, H, W, D]

        # Process all 14 masks
        for mask in MASK_LIST:
            mask_str = mask_to_string(mask)

            available_idx = [i for i, m in enumerate(mask) if m == 1]
            missing_idx = [i for i, m in enumerate(mask) if m == 0]

            if len(available_idx) == 0 or len(missing_idx) == 0:
                continue  # Should not happen

            x_available = full_img[:, available_idx, ...].to(device)

            with autocast(enabled=True):
                # Encode available modalities to get latent shape
                latent_avail = autoencoder.encode_stage_2_inputs(
                    x_available
                ) * scale_factor
                _, _, latent_dim, h_lat, w_lat, d_lat = latent_avail.shape

                # Noise for missing modalities [B, N_miss, latent_dim, h, w, d]
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

            # Save ONLY synthetic (missing) images — no GT, no input
            out_dir = os.path.join(prediction_root, mask_str, subject_id)
            for mi_idx, mi in enumerate(missing_idx):
                mod_name = MODALITY_KEYS[mi]
                gen_data = generated[0, mi_idx, ...]
                save_nifti(
                    gen_data, affine,
                    os.path.join(out_dir, f"{subject_id}_{mod_name}.nii.gz"),
                )

            torch.cuda.empty_cache()

    end = timeit.default_timer()
    elapsed = end - start
    print(f"\nInference completed in {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Output: {prediction_root}/")
    print("\nNext: use global eval script to compute metrics, e.g.:")
    print(f"  python syn_metric.py --pred_path {prediction_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoPeDiT Inference on BraTS2020 Test Set")

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
            os.path.join(PROJ_ROOT, "datalist/BraTS2020"),
        ),
        type=str, help="Directory containing test.list (env: DATALIST_DIR)",
    )

    parser.add_argument(
        "--task_dir",
        default=None,
        type=str,
        help="Output directory (default: $COMPARE_ROOT/results/task_{timestamp})",
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
        args.task_dir = os.path.join(PROJ_ROOT, "results", f"task_{timestamp}")

    os.makedirs(args.task_dir, exist_ok=True)
    print(f"Task directory: {args.task_dir}")

    inference(args)
