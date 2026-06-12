#!/usr/bin/env python
"""
CoPeVAE (Stage 1) Training Script for BraTS2020 — Rule-compliant.

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
from datetime import datetime
from time import time
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from torch.nn import L1Loss, MSELoss
from monai.networks.nets import PatchDiscriminator
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.optimizers import WarmupCosineSchedule
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import mean_absolute_error as mae

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from AutoEncoder.utils_brain import *
from data.BraTS2020_data import (
    MODALITY_KEYS,
    read_train_val_datalist,
    get_transforms,
    get_loader_CoPeVAE,
)


def setup_logger(task_dir):
    """Setup logging to BOTH console and logs/train.log (Rule 8)."""
    log_dir = os.path.join(task_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("CoPeVAE")
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


def save_checkpoint(path, epoch, global_step, autoencoder, discriminator,
                    optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                    epoch_loss, lr, args):
    """Save checkpoint in Rule 9 format."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": autoencoder.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "scheduler_g_state_dict": scheduler_g.state_dict() if scheduler_g else None,
        "scheduler_d_state_dict": scheduler_d.state_dict() if scheduler_d else None,
        "epoch_loss": epoch_loss,
        "lr": lr,
        "args": args,
    }, path)


def train(args, autoencoder, discriminator):
    """Main training loop for CoPeVAE Stage 1."""

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
        get_loader_CoPeVAE(args, args.rank, args.world_size,
                           train_files, val_files,
                           train_transform, val_transform)
    )
    logger.info(f"Train batches/epoch: {len(dataloader_train)}")

    # ---- Losses ----
    intensity_loss = MSELoss() if args.recon_loss == "l2" else L1Loss(reduction="mean")
    logger.info(f"Reconstruction loss: {args.recon_loss}")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(
        spatial_dims=3, network_type="squeeze",
        is_fake_3d=True, fake_3d_ratio=0.2).eval().to(args.device)

    accumulation_steps = args.gradient_accumulation_steps
    autoencoder = autoencoder.to(args.device)
    discriminator = discriminator.to(args.device)

    if args.distributed:
        autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
        autoencoder = DistributedDataParallel(autoencoder, device_ids=[args.rank])
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DistributedDataParallel(discriminator, device_ids=[args.rank])

    logger.info(f"AE params: {sum(p.numel() for p in autoencoder.parameters()):,}")

    # ---- Optimizers ----
    if args.opt == "adamw":
        optimizer_g = optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=0)
        optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=0)
    else:
        optimizer_g = optim.Adam(autoencoder.parameters(), lr=args.lr)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)

    # ---- Schedulers ----
    scheduler_g, scheduler_d = None, None
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler_g = WarmupCosineSchedule(
                optimizer_g, warmup_steps=args.warmup_steps,
                t_total=args.num_steps, end_lr=args.lr_min,
                warmup_multiplier=1e-3)
            scheduler_d = WarmupCosineSchedule(
                optimizer_d, warmup_steps=args.warmup_steps,
                t_total=args.num_steps, end_lr=args.lr_min,
                warmup_multiplier=1e-3)
        elif args.lr_schedule == "poly":
            def lambdas(e):
                return (1 - float(e) / float(args.epochs)) ** 0.9
            scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambdas)
            scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lambdas)

    scaler_g = GradScaler() if args.amp else None
    scaler_d = GradScaler() if args.amp else None

    # ---- Setup ----
    max_epochs = args.epochs
    total_steps_per_epoch = len(dataloader_train)
    total_global_steps = max_epochs * total_steps_per_epoch
    global_step = 0
    val_interval = args.val_interval
    start_time = timeit.default_timer()

    # ---- Rule 9: save 0.pt before training ----
    save_checkpoint(
        os.path.join(args.task_dir, "models", "0.pt"),
        0, 0, autoencoder, discriminator,
        optimizer_g, optimizer_d, scheduler_g, scheduler_d,
        float("nan"), args.lr, args)
    logger.info("Saved 0.pt (initial weights)")

    logger.info(f"Start training: {max_epochs} epochs, "
                f"{total_steps_per_epoch} steps/epoch, "
                f"{total_global_steps} total steps")

    # ---- Training loop ----
    for epoch in range(max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        autoencoder.train()
        discriminator.train()

        epoch_loss_sum = 0.0
        progress_bar = tqdm(enumerate(dataloader_train),
                            total=total_steps_per_epoch, ncols=130)
        progress_bar.set_description(f"Epoch {epoch+1}/{max_epochs}")

        for i, batch in progress_bar:
            global_step += 1
            x_incomp, x_missing, missing_length, missing_label = batch
            x_incomp = x_incomp.to(args.device)
            x_missing = x_missing.to(args.device)
            missing_length = missing_length.to(args.device)
            missing_label = missing_label.to(args.device)

            with autocast(enabled=args.amp):
                x_in, x_rec, l_vq, l_pretext = autoencoder(
                    x_incomp, x_missing, missing_length, missing_label)
                generator_loss = get_adv_loss(x_rec, discriminator, adv_loss)
                loss_pretext = pretext_loss_weighted_sum(args, l_pretext)
                rec_loss, per_loss = get_train_loss(x_in, x_rec, intensity_loss,
                                                    loss_perceptual)
                losses = {
                    "rec_loss": rec_loss, "vq_loss": l_vq,
                    "per_loss": per_loss, "adv_loss": generator_loss,
                    "pretext_loss": loss_pretext,
                }
                for k in losses:
                    if torch.isnan(losses[k]) or torch.isinf(losses[k]):
                        losses[k] = torch.tensor(0.001, device=losses[k].device,
                                                 requires_grad=True)
                loss_g = train_loss_weighted_sum(args, losses) / accumulation_steps

            epoch_loss_sum += loss_g.item() * accumulation_steps

            # ---- Generator backward ----
            if args.amp:
                scaler_g.scale(loss_g).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler_g.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
                                                   max_norm=1.0)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad(set_to_none=True)
            else:
                loss_g.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
                                                   max_norm=1.0)
                    optimizer_g.step()
                    optimizer_g.zero_grad(set_to_none=True)

            # ---- Discriminator backward ----
            with autocast(enabled=args.amp):
                disc_loss = get_discriminator_loss(x_in, x_rec, discriminator, adv_loss)
                loss_d = args.adv_weight * disc_loss / accumulation_steps
                if torch.isnan(loss_d) or torch.isinf(loss_d):
                    loss_d = torch.tensor(0.001, device=loss_d.device,
                                          requires_grad=True)

            if args.amp:
                scaler_d.scale(loss_d).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                                   max_norm=1.0)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad(set_to_none=True)
            else:
                loss_d.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                                   max_norm=1.0)
                    optimizer_d.step()
                    optimizer_d.zero_grad(set_to_none=True)

            # ---- Rule 8: per-step log every log_interval ----
            if global_step % args.log_interval == 0:
                current_lr = (scheduler_g.get_lr()[0] if scheduler_g
                              else optimizer_g.param_groups[0]["lr"])
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} | "
                    f"Step {i+1}/{total_steps_per_epoch} "
                    f"[global {global_step}/{total_global_steps}] | "
                    f"Loss: {loss_g.item() * accumulation_steps:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

        # ---- End of epoch ----
        if scheduler_g:
            scheduler_g.step()
            scheduler_d.step()

        avg_epoch_loss = epoch_loss_sum / total_steps_per_epoch
        current_lr = (scheduler_g.get_lr()[0] if scheduler_g
                      else optimizer_g.param_groups[0]["lr"])
        elapsed = timeit.default_timer() - start_time

        # ---- Rule 8: epoch-end summary ----
        logger.info(f"Epoch [{epoch+1}/{max_epochs}] | "
                    f"Loss: {avg_epoch_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s")

        # ---- TensorBoard ----
        writer.add_scalar("Loss/train_total", avg_epoch_loss, epoch + 1)
        writer.add_scalar("LR/lr", current_lr, epoch + 1)

        # ---- Rule 9: save latest.pt (every epoch, overwrites) ----
        save_checkpoint(
            os.path.join(args.task_dir, "models", "latest.pt"),
            epoch + 1, global_step, autoencoder, discriminator,
            optimizer_g, optimizer_d, scheduler_g, scheduler_d,
            avg_epoch_loss, current_lr, args)

        # ---- Rule 9: checkpoint_epoch_N.pt (milestone, never overwritten) ----
        if (epoch + 1) % args.ckpt_interval == 0:
            save_checkpoint(
                os.path.join(args.task_dir, "models",
                             f"checkpoint_epoch_{epoch+1}.pt"),
                epoch + 1, global_step, autoencoder, discriminator,
                optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                avg_epoch_loss, current_lr, args)
            logger.info(f"Saved checkpoint_epoch_{epoch+1}.pt")

        # ---- Validation ----
        if (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            metric_epoch = {"psnr": 0, "ssim": 0, "mae": 0, "lpips": 0}
            test_epoch_losses = {"rec_loss": 0, "per_loss": 0}
            for batch in dataloader_val:
                with torch.no_grad(), autocast(enabled=args.amp):
                    x_incomp, x_missing, _, _ = batch
                    x_incomp = x_incomp.to(args.device)
                    x_missing = x_missing.to(args.device)
                    if args.distributed:
                        x_incomp, rec_incomp = autoencoder.module.val(x_incomp)
                        x_missing, rec_missing = autoencoder.module.val(x_missing)
                    else:
                        x_incomp, rec_incomp = autoencoder.val(x_incomp)
                        x_missing, rec_missing = autoencoder.val(x_missing)
                    x_rec0 = torch.cat([rec_incomp, rec_missing], dim=0)
                    x_in0 = torch.cat([x_incomp, x_missing], dim=0)
                    xi = {"x_incomp": x_incomp, "x_missing": x_missing}
                    xr = {"x_incomp": rec_incomp, "x_missing": rec_missing}
                    rec_loss, per_loss = get_train_loss(xi, xr, intensity_loss,
                                                        loss_perceptual)
                    test_epoch_losses["rec_loss"] += rec_loss.item()
                    test_epoch_losses["per_loss"] += per_loss.item()
                    metric_epoch["psnr"] += psnr(x_rec0, x_in0)
                    metric_epoch["ssim"] += ssim(x_rec0, x_in0)
                    metric_epoch["mae"] += mae(x_rec0, x_in0)
                    metric_epoch["lpips"] += per_loss.item()

            for k in test_epoch_losses:
                test_epoch_losses[k] /= len(dataloader_val)
            for k in metric_epoch:
                metric_epoch[k] /= len(dataloader_val)

            writer.add_scalar("Loss/val_total",
                              test_loss_weighted_sum(args, test_epoch_losses),
                              epoch + 1)
            writer.add_scalar("Metrics/val_psnr", metric_epoch["psnr"], epoch + 1)
            writer.add_scalar("Metrics/val_ssim", metric_epoch["ssim"], epoch + 1)
            writer.add_scalar("Metrics/val_mae", metric_epoch["mae"], epoch + 1)
            writer.add_scalar("Metrics/val_lpips", metric_epoch["lpips"], epoch + 1)

            logger.info(
                f"Val [{epoch+1}/{max_epochs}] | "
                f"PSNR: {metric_epoch['psnr']:.3f} | "
                f"SSIM: {metric_epoch['ssim']:.3f} | "
                f"MAE: {metric_epoch['mae']:.3f} | "
                f"LPIPS: {metric_epoch['lpips']:.3f}"
            )

    # ---- Rule 9: final_model.pt ----
    save_checkpoint(
        os.path.join(args.task_dir, "models", "final_model.pt"),
        max_epochs, global_step, autoencoder, discriminator,
        optimizer_g, optimizer_d, scheduler_g, scheduler_d,
        avg_epoch_loss, current_lr, args)
    logger.info("Saved final_model.pt")

    writer.close()
    logger.info(f"Training complete. Total time: {timeit.default_timer() - start_time:.1f}s")
    if args.distributed:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoPeVAE Stage 1 Training (Rule-compliant)")
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
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--num_steps", default=8000, type=int)
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lrdecay", default=True)
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--lr_min", default=8e-5, type=float)
    parser.add_argument("--opt", default="adam", type=str)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--ckpt_interval", default=20, type=int)
    parser.add_argument("--log_interval", default=10, type=int,
                        help="Log every N steps (Rule 8)")
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--cache", default=0.2, type=float)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--gpu_ids", default=[0])
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
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
    parser.add_argument("--lambda2", default=0.5, type=float)
    parser.add_argument("--lambda3", default=0.5, type=float)
    parser.add_argument("--vq_weight", default=1.0, type=float)
    parser.add_argument("--per_weight", default=0.05, type=float)
    parser.add_argument("--adv_weight", default=0.01, type=float)
    parser.add_argument("--pretext_weight", default=0.01, type=float)

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

    print(f"Task directory: {args.task_dir}")
    print(f"Models:        {args.task_dir}/models/")
    print(f"TensorBoard:   {args.task_dir}/tensorboard/")
    print(f"Logs:          {args.task_dir}/logs/train.log")

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
    discriminator = PatchDiscriminator(spatial_dims=args.spatial_dims, num_layers_d=3,
                                       channels=32, in_channels=1, out_channels=1)

    print(f"Starting CoPeVAE Stage 1 training for {args.epochs} epochs "
          f"on {args.device}")
    train(args, autoencoder, discriminator)
