#!/usr/bin/env python
"""
MDiT3D (Stage 2) Training Script for BraTS2020 — Rule-compliant.

Rules satisfied:
  Rule 0  — Global path variables (COMPARE_ROOT/DATA_ROOT/DATALIST_DIR)
  Rule 1  — Train never loads test data (read_train_val_datalist)
  Rule 7a — Output dirs: models/, tensorboard/, logs/
  Rule 8  — Log format: per-step every log_interval + epoch-end summary
  Rule 9  — Checkpoints: 0.pt, latest.pt, checkpoint_epoch_N.pt, final_model.pt
"""

import argparse
import os
import sys
import timeit
import logging
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from torch.nn import L1Loss, MSELoss
from timm.scheduler.cosine_lr import CosineLRScheduler
from monai.utils import first
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from LDM.model.MDiT3D_Brain import *
from LDM.inferers import inferer_DiT_Brain
from LDM.utils import *
from generative.networks.schedulers import DDPMScheduler
from data.BraTS2020_data import (
    MODALITY_KEYS, read_train_val_datalist, get_transforms, get_loader_MDiT3D,
)


def setup_logger(task_dir):
    """Setup logging to BOTH console and logs/train.log (Rule 8)."""
    log_dir = os.path.join(task_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("MDiT3D")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_checkpoint(path, epoch, global_step, model, ema_model,
                    optimizer, scheduler, epoch_loss, lr, args):
    """Save checkpoint in Rule 9 format."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch_loss": epoch_loss,
        "lr": lr,
        "args": args,
    }, path)


def load_ae_ckpt(autoencoder, ckpt_path, device):
    """Load CoPeVAE checkpoint supporting both old and new Rule 9 format."""
    state = torch.load(str(ckpt_path), map_location=device)
    if "model_state_dict" in state:
        autoencoder.load_state_dict(state["model_state_dict"])
        print(f"Loaded AE via model_state_dict (Rule 9 format)")
    elif "model" in state:
        autoencoder.load_state_dict(state["model"])
        print(f"Loaded AE via model (legacy format)")
    else:
        raise KeyError(f"AE checkpoint has neither 'model_state_dict' nor 'model' key. "
                       f"Keys: {list(state.keys())}")
    return state.get("epoch", "?")


def train(args, autoencoder, DiT):
    """Main training loop for MDiT3D Stage 2."""

    # ---- Logger (Rule 8: console + file) ----
    logger = setup_logger(args.task_dir)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=os.path.join(args.task_dir, "tensorboard"))
    logger.info(f"TensorBoard: {args.task_dir}/tensorboard")

    # ---- Load data (Rule 1: train+val only, never test) ----
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Datalist dir: {args.datalist_dir}")
    train_files, val_files = read_train_val_datalist(args.datalist_dir, args.data_root)
    logger.info(f"Loaded {len(train_files)} train, {len(val_files)} val "
                f"(test data NOT loaded — use eval.py for test)")

    train_transform, _ = get_transforms(args, MODALITY_KEYS, is_train=True)
    _, val_transform = get_transforms(args, MODALITY_KEYS, is_train=False)
    dataloader_train, dataloader_val, train_sampler, val_sampler = (
        get_loader_MDiT3D(args, args.rank, args.world_size,
                          train_files, val_files,
                          train_transform, val_transform)
    )
    logger.info(f"Train batches/epoch: {len(dataloader_train)}")

    # ---- Loss ----
    diff_loss_fn = MSELoss() if args.diff_loss == "l2" else L1Loss(reduction="mean")
    logger.info(f"Diffusion loss: {args.diff_loss}, pred_type: {args.pred_type}")

    # ---- DDPM scheduler ----
    if args.noise_scheduler == "linear":
        scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            schedule="scaled_linear_beta",
            beta_start=0.0015, beta_end=0.0195, clip_sample=False)
    else:
        scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            schedule="cosine", clip_sample=False)

    accumulation_steps = args.gradient_accumulation_steps

    # ---- EMA model ----
    ema = deepcopy(DiT).to(args.device)
    requires_grad(ema, False)
    DiT = DiT.to(args.device)

    # ---- Load frozen CoPeVAE ----
    autoencoder = autoencoder.to(args.device)
    if args.ae_ckpt and os.path.exists(args.ae_ckpt):
        ae_epoch = load_ae_ckpt(autoencoder, args.ae_ckpt, args.device)
        logger.info(f"CoPeVAE loaded from {args.ae_ckpt} (epoch {ae_epoch})")
    else:
        logger.warning(f"AE checkpoint not found at {args.ae_ckpt}")

    autoencoder.eval()
    autoencoder.requires_grad_(False)

    # ---- Scale factor from first batch ----
    check_data = first(dataloader_train)
    with torch.no_grad(), autocast(enabled=True):
        z0 = autoencoder.encode_stage_2_inputs(check_data[0].to(args.device))
        z1 = autoencoder.encode_stage_2_inputs(check_data[1].to(args.device))
    scale_factor = 1.0
    logger.info(f"Latent scale factor: {scale_factor}")
    torch.cuda.empty_cache()

    inferer = inferer_DiT_Brain.LatentDiffusionInferer(scheduler_ddpm, scale_factor=scale_factor)

    # ---- Optimizer ----
    if args.opt == "sgd":
        optimizer = optim.SGD(DiT.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.decay)
    elif args.opt == "adam":
        optimizer = optim.Adam(DiT.parameters(), lr=args.lr)
    else:
        optimizer = optim.AdamW(DiT.parameters(), lr=args.lr, weight_decay=args.decay)
    logger.info(f"Optimizer: {args.opt}, lr={args.lr}")

    # ---- Scheduler ----
    scheduler = None
    if args.lr_schedule == "warmup_cosine":
        scheduler = CosineLRScheduler(
            optimizer, warmup_t=args.warmup_steps, warmup_lr_init=1e-6,
            t_initial=args.num_steps, lr_min=args.lr_min, cycle_limit=1)
    elif args.lr_schedule == "poly":
        def lambdas(e):
            return (1 - float(e) / float(args.epochs)) ** 0.9
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    scaler = GradScaler() if args.amp else None

    logger.info(f"AE params: {sum(p.numel() for p in autoencoder.parameters()):,}")
    logger.info(f"DiT params: {sum(p.numel() for p in DiT.parameters()):,}")

    # ---- Setup ----
    DiT.train()
    ema.eval()

    if args.distributed:
        DiT = DistributedDataParallel(DiT, device_ids=[args.rank])

    # Resume support
    start_epoch = 0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(str(args.resume_ckpt), map_location="cpu")
        if args.distributed:
            DiT.module.load_state_dict(ckpt["model_state_dict"])
        else:
            DiT.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"]
        logger.info(f"Resumed from {args.resume_ckpt}, epoch={start_epoch}")
    else:
        if args.distributed:
            update_ema(ema, DiT.module, decay=0)
        else:
            update_ema(ema, DiT, decay=0)

    max_epochs = args.epochs
    total_steps_per_epoch = len(dataloader_train)
    total_global_steps = max_epochs * total_steps_per_epoch
    global_step = start_epoch * total_steps_per_epoch
    start_time = timeit.default_timer()

    # ---- Rule 9: save 0.pt before training ----
    save_checkpoint(
        os.path.join(args.task_dir, "models", "0.pt"),
        0, global_step, DiT, ema, optimizer, scheduler,
        float("nan"), args.lr, args)
    logger.info("Saved 0.pt (initial weights)")

    logger.info(f"Start training: {max_epochs} epochs, "
                f"{total_steps_per_epoch} steps/epoch, "
                f"{total_global_steps} total steps")

    # ---- Training loop ----
    for epoch in range(start_epoch, max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        DiT.train()
        epoch_loss_sum = 0.0

        progress_bar = tqdm(enumerate(dataloader_train),
                            total=total_steps_per_epoch, ncols=130)
        progress_bar.set_description(f"Epoch {epoch+1}/{max_epochs}")

        for i, batch in progress_bar:
            global_step += 1
            x_available, x_missing, missing_condition = batch
            x_available = x_available.to(args.device)
            x_missing = x_missing.to(args.device)
            missing_condition = missing_condition.to(args.device)

            with torch.no_grad(), autocast(enabled=args.amp):
                prompts = autoencoder.get_condition(x_available)

            with autocast(enabled=args.amp):
                noise, noise_missing = get_noise(z0, z1)
                noise = noise.to(args.device)
                noise_missing = noise_missing.to(args.device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps,
                    (x_available.shape[0],), device=x_available.device).long()

                pred, latent = inferer(
                    inputs=[x_available, x_missing],
                    autoencoder_model=autoencoder,
                    diffusion_model=DiT, noise=noise_missing,
                    timesteps=timesteps, condition=prompts)

                if args.pred_type == "noise":
                    l_diff = diff_loss_fn(pred, noise_missing)
                elif args.pred_type == "x":
                    l_diff = diff_loss_fn(pred, latent)
                elif args.pred_type == "v":
                    v = scheduler_ddpm.get_velocity(latent, noise, timesteps)
                    l_diff = diff_loss_fn(pred, v)

                l_diff = l_diff / accumulation_steps

            epoch_loss_sum += l_diff.item() * accumulation_steps

            # ---- Backward ----
            if args.amp:
                scaler.scale(l_diff).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(DiT.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                l_diff.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(DiT.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # ---- EMA update ----
            if args.distributed:
                update_ema(ema, DiT.module)
            else:
                update_ema(ema, DiT)

            # ---- Rule 8: per-step log every log_interval ----
            if global_step % args.log_interval == 0:
                current_lr = (scheduler._get_lr(epoch)[0] if scheduler
                              and args.lr_schedule == "warmup_cosine"
                              else scheduler.get_last_lr()[0] if scheduler
                              else optimizer.param_groups[0]["lr"])
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} | "
                    f"Step {i+1}/{total_steps_per_epoch} "
                    f"[global {global_step}/{total_global_steps}] | "
                    f"Loss: {l_diff.item() * accumulation_steps:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

        # ---- End of epoch ----
        if scheduler and args.lrdecay:
            scheduler.step(epoch)

        avg_epoch_loss = epoch_loss_sum / total_steps_per_epoch
        if args.lr_schedule == "warmup_cosine":
            current_lr = scheduler._get_lr(epoch)[0]
        else:
            current_lr = (scheduler.get_last_lr()[0] if scheduler
                          else optimizer.param_groups[0]["lr"])
        elapsed = timeit.default_timer() - start_time

        # ---- Rule 8: epoch-end summary ----
        logger.info(f"Epoch [{epoch+1}/{max_epochs}] | "
                    f"Loss: {avg_epoch_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s")

        # ---- TensorBoard ----
        writer.add_scalar("Loss/train_diff", avg_epoch_loss, epoch + 1)
        writer.add_scalar("LR/lr", current_lr, epoch + 1)

        # ---- Rule 9: save latest.pt (every epoch, overwrites) ----
        save_checkpoint(
            os.path.join(args.task_dir, "models", "latest.pt"),
            epoch + 1, global_step, DiT, ema, optimizer, scheduler,
            avg_epoch_loss, current_lr, args)

        # ---- Rule 9: checkpoint_epoch_N.pt (milestone) ----
        if (epoch + 1) % args.ckpt_interval == 0:
            save_checkpoint(
                os.path.join(args.task_dir, "models",
                             f"checkpoint_epoch_{epoch+1}.pt"),
                epoch + 1, global_step, DiT, ema, optimizer, scheduler,
                avg_epoch_loss, current_lr, args)
            logger.info(f"Saved checkpoint_epoch_{epoch+1}.pt")

        # ---- Validation ----
        if (epoch + 1) % val_interval == 0 and epoch > 0:
            DiT.eval()
            val_epoch_losses = {"diff_loss": 0}
            for batch in dataloader_val:
                with torch.no_grad(), autocast(enabled=args.amp):
                    x_available, x_missing, missing_condition = batch
                    x_available = x_available.to(args.device)
                    x_missing = x_missing.to(args.device)
                    missing_condition = missing_condition.to(args.device)
                    prompts = autoencoder.get_condition(x_available)
                    noise, noise_missing = get_noise(z0, z1)
                    noise = noise.to(args.device)
                    noise_missing = noise_missing.to(args.device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps,
                        (x_available.shape[0],), device=x_available.device).long()
                    pred, latent = inferer(
                        inputs=[x_available, x_missing],
                        autoencoder_model=autoencoder, diffusion_model=DiT,
                        noise=noise_missing, timesteps=timesteps,
                        condition=prompts)
                    if args.pred_type == "noise":
                        l = diff_loss_fn(pred, noise_missing)
                    elif args.pred_type == "x":
                        l = diff_loss_fn(pred, latent)
                    else:
                        l = diff_loss_fn(pred,
                                         scheduler_ddpm.get_velocity(latent, noise, timesteps))
                    val_epoch_losses["diff_loss"] += l.item()

            val_diff = val_epoch_losses["diff_loss"] / len(dataloader_val)
            writer.add_scalar("Loss/val_diff", val_diff, epoch + 1)
            logger.info(f"Val [{epoch+1}/{max_epochs}] | Loss: {val_diff:.6f}")

    # ---- Rule 9: final_model.pt ----
    save_checkpoint(
        os.path.join(args.task_dir, "models", "final_model.pt"),
        max_epochs, global_step, DiT, ema, optimizer, scheduler,
        avg_epoch_loss, current_lr, args)
    logger.info("Saved final_model.pt")

    writer.close()
    logger.info(f"Training complete. Total time: {timeit.default_timer() - start_time:.1f}s")
    if args.distributed:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDiT3D Stage 2 Training (Rule-compliant)")
    parser.add_argument("--data_root",
                        default=os.environ.get(
                            "DATA_ROOT",
                            "/devdata/hsh/datasets/seg_dataset/BraTS2020/"
                            "brats20-dataset-training-validation/versions/1/"
                            "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"),
                        type=str, help="Root path to BraTS2020 TrainingData (env: DATA_ROOT)")
    parser.add_argument("--datalist_dir",
                        default=os.environ.get(
                            "DATALIST_DIR",
                            os.path.join(os.environ.get(
                                "COMPARE_ROOT",
                                os.path.dirname(os.path.abspath(__file__))),
                                "datalist/BraTS2020")),
                        type=str, help="Directory with train.list, val.list (env: DATALIST_DIR)")
    parser.add_argument("--ae_ckpt", default=None, type=str,
                        help="Path to pre-trained CoPeVAE checkpoint")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--num_steps", default=8000, type=int)
    parser.add_argument("--warmup_steps", default=50, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lrdecay", default=True)
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--lr_min", default=2e-5, type=float)
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--val_interval", default=20, type=int)
    parser.add_argument("--ckpt_interval", default=20, type=int)
    parser.add_argument("--log_interval", default=10, type=int,
                        help="Log every N steps (Rule 8)")
    parser.add_argument("--missing_num", default=1, type=int, choices=[1, 2, 3])
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--cache", default=0.2, type=float)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--gpu_ids", default=[0])
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--DiT", default="MDiT3D-B/2", type=str)
    parser.add_argument("--noise_scheduler", default="linear", type=str)
    parser.add_argument("--pred_type", default="x", type=str,
                        choices=["noise", "x", "v"])
    parser.add_argument("--diffusion_steps", default=500, type=int)
    parser.add_argument("--sample_steps", default=200, type=int)
    parser.add_argument("--diff_loss", default="l2", type=str)
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

    args = parser.parse_args()
    for attr in ["vae_channel", "brain_pad", "brain_roi", "brain_size"]:
        val = getattr(args, attr)
        if isinstance(val, str):
            setattr(args, attr, tuple(map(int, val.strip("()").split(","))))

    # ---- Task directory (Rule 7a) ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.task_dir = os.path.join(
        os.environ.get("COMPARE_ROOT", os.path.dirname(os.path.abspath(__file__))),
        "results", f"task_{timestamp}")
    os.makedirs(os.path.join(args.task_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.task_dir, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(args.task_dir, "logs"), exist_ok=True)

    args.resume_ckpt = None
    args.amp = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.device)
    else:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert args.rank >= 0

    autoencoder = CopeVAE(args)
    DiT = MDiT3D_models[args.DiT](num_modalities=args.modality_num)

    print(f"Task directory: {args.task_dir}")
    print(f"Models:        {args.task_dir}/models/")
    print(f"TensorBoard:   {args.task_dir}/tensorboard/")
    print(f"Logs:          {args.task_dir}/logs/train.log")
    print(f"Starting MDiT3D Stage 2: {args.epochs} epochs, {args.DiT}, "
          f"missing_num={args.missing_num}")
    print(f"AE checkpoint: {args.ae_ckpt}")
    train(args, autoencoder, DiT)
