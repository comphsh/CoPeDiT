#!/usr/bin/env python
"""
Adapted CoPeVAE (Stage 1) Training Script for BraTS2020.
Changes from original:
- Uses data/BraTS2020_data.py for MONAI-based BraTS2020 data loading
- TensorBoard logging for epoch loss/lr
- Model checkpoints saved to results/task_{timestamp}/models/
- Single GPU mode by default
- 200 epochs
Original code: train_CoPeVAE_Brain.py
"""

import argparse
import os
import timeit
from datetime import datetime
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.nn import L1Loss, MSELoss
from monai.networks.nets import PatchDiscriminator
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.optimizers import WarmupCosineSchedule
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import mean_absolute_error as mae

from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AutoEncoder.model.CoPeVAE_BrainMRI import CopeVAE
from AutoEncoder.utils_brain import *
from data.BraTS2020_data import (
    MODALITY_KEYS,
    read_datalist,
    get_transforms,
    get_loader_CoPeVAE,
)


def train(args, autoencoder, discriminator):
    """Main training loop for CoPeVAE Stage 1."""
    if args.rank == 0:
        logger = create_logger(args.log_dir, args.distributed)
    else:
        logger = create_logger(args.log_dir, args.distributed)
    if args.rank == 0:
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        logger.info(f"TensorBoard log directory: {args.tb_log_dir}")
    else:
        writer = None
    logger.info(f"Rank {args.rank}: Starting to load BraTS2020 data")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Datalist dir: {args.datalist_dir}")
    train_files, val_files, test_files = read_datalist(
        args.datalist_dir, args.data_root
    )
    logger.info(
        f"Loaded {len(train_files)} train, {len(val_files)} val, "
        f"{len(test_files)} test samples"
    )
    train_files_combined = train_files
    test_files_combined = val_files
    train_transform, test_transform = get_transforms(
        args, MODALITY_KEYS, is_train=True
    )
    _, test_transform_eval = get_transforms(
        args, MODALITY_KEYS, is_train=False
    )
    logger.info(f"Rank {args.rank}: Transforms created")
    dataloader_train, dataloader_test, train_sampler, test_sampler = (
        get_loader_CoPeVAE(
            args, args.rank, args.world_size,
            train_files_combined, test_files_combined,
            train_transform, test_transform_eval,
        )
    )
    logger.info(f"Rank {args.rank}: DataLoaders created")
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
        if args.rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        if args.rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = (
        PerceptualLoss(
            spatial_dims=3, network_type="squeeze",
            is_fake_3d=True, fake_3d_ratio=0.2,
        )
        .eval()
        .to(args.device)
    )
    accumulation_steps = args.gradient_accumulation_steps
    autoencoder = autoencoder.to(args.device)
    discriminator = discriminator.to(args.device)
    if args.distributed:
        autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
        autoencoder = DistributedDataParallel(
            autoencoder, device_ids=[args.rank]
        )
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[args.rank]
        )
    logger.info(
        f"AE Parameters: {sum(p.numel() for p in autoencoder.parameters()):,}"
    )
    if args.opt == "adam":
        optimizer_g = optim.Adam(params=autoencoder.parameters(), lr=args.lr)
        optimizer_d = optim.Adam(params=discriminator.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer_g = optim.AdamW(
            params=autoencoder.parameters(), lr=args.lr, weight_decay=0
        )
        optimizer_d = optim.AdamW(
            params=discriminator.parameters(), lr=args.lr, weight_decay=0
        )
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler_g = WarmupCosineSchedule(
                optimizer_g, warmup_steps=args.warmup_steps,
                t_total=args.num_steps, end_lr=args.lr_min,
                warmup_multiplier=1e-3,
            )
            scheduler_d = WarmupCosineSchedule(
                optimizer_d, warmup_steps=args.warmup_steps,
                t_total=args.num_steps, end_lr=args.lr_min,
                warmup_multiplier=1e-3,
            )
        elif args.lr_schedule == "poly":
            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(
                optimizer_g, lr_lambda=lambdas
            )
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(
                optimizer_d, lr_lambda=lambdas
            )
    if args.amp:
        scaler_g = GradScaler()
        scaler_d = GradScaler()
    val_interval = args.val_interval
    best_train_loss = 1000
    start_epoch = 0
    max_epochs = args.epochs
    start = timeit.default_timer()
    logger.info(f"Rank {args.rank}: Start Training (CoPeVAE Stage 1)")
    for epoch in range(start_epoch, max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        autoencoder.train()
        discriminator.train()
        train_epoch_losses = {
            "rec_loss": 0, "vq_loss": 0, "per_loss": 0,
            "disc_loss": 0, "adv_loss": 0, "pretext_loss": 0,
            "len_loss": 0, "loc_loss": 0, "con_loss": 0,
        }
        progress_bar = tqdm(
            enumerate(dataloader_train), total=len(dataloader_train), ncols=150
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for i, batch in progress_bar:
            x_incomp, x_missing, missing_length, missing_label = batch
            x_incomp, x_missing, missing_length, missing_label = (
                x_incomp.to(args.device), x_missing.to(args.device),
                missing_length.to(args.device), missing_label.to(args.device),
            )
            with autocast(enabled=args.amp):
                x_in, x_rec, l_vq, l_pretext = autoencoder(
                    x_incomp, x_missing, missing_length, missing_label
                )
                generator_loss = get_adv_loss(x_rec, discriminator, adv_loss)
                loss_pretext = pretext_loss_weighted_sum(args, l_pretext)
                rec_loss, per_loss = get_train_loss(
                    x_in, x_rec, intensity_loss, loss_perceptual
                )
                losses = {
                    "rec_loss": rec_loss, "vq_loss": l_vq,
                    "per_loss": per_loss, "adv_loss": generator_loss,
                    "pretext_loss": loss_pretext,
                    "len_loss": l_pretext["l_len"],
                    "loc_loss": l_pretext["l_loc"],
                    "con_loss": l_pretext["l_con"],
                }
                for key in losses:
                    if torch.isnan(losses[key]) or torch.isinf(losses[key]):
                        print(f"WARNING: NaN/Inf detected in {key}")
                        losses[key] = torch.tensor(
                            0.001, device=losses[key].device, requires_grad=True
                        )
                loss_g = train_loss_weighted_sum(args, losses) / accumulation_steps
            for loss_name, loss_value in losses.items():
                train_epoch_losses[loss_name] += loss_value.item()
            if args.amp:
                scaler_g.scale(loss_g).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler_g.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(
                        autoencoder.parameters(), max_norm=1.0
                    )
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad(set_to_none=True)
            else:
                loss_g.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        autoencoder.parameters(), max_norm=1.0
                    )
                    optimizer_g.step()
                    optimizer_g.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                disc_loss = get_discriminator_loss(
                    x_in, x_rec, discriminator, adv_loss
                )
                loss_d = args.adv_weight * disc_loss / accumulation_steps
                if torch.isnan(loss_d) or torch.isinf(loss_d):
                    print(f"WARNING: NaN/Inf detected in dis loss")
                    loss_d = torch.tensor(
                        0.001, device=loss_d.device, requires_grad=True
                    )
                train_epoch_losses["disc_loss"] += disc_loss.item()
            if args.amp:
                scaler_d.scale(loss_d).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), max_norm=1.0
                    )
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad(set_to_none=True)
            else:
                loss_d.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), max_norm=1.0
                    )
                    optimizer_d.step()
                    optimizer_d.zero_grad(set_to_none=True)
        scheduler_g.step()
        scheduler_d.step()
        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)
        loss_g_total = train_loss_weighted_sum(args, train_epoch_losses)
        current_lr = scheduler_g.get_lr()[0]
        end = timeit.default_timer()
        elapsed = end - start
        if args.rank == 0 and writer is not None:
            writer.add_scalar("Loss/train_total", loss_g_total, epoch)
            writer.add_scalar("Loss/train_rec", train_epoch_losses["rec_loss"], epoch)
            writer.add_scalar("Loss/train_vq", train_epoch_losses["vq_loss"], epoch)
            writer.add_scalar("Loss/train_per", train_epoch_losses["per_loss"], epoch)
            writer.add_scalar("Loss/train_disc", train_epoch_losses["disc_loss"], epoch)
            writer.add_scalar("Loss/train_pretext", train_epoch_losses["pretext_loss"], epoch)
            writer.add_scalar("Loss/train_len", train_epoch_losses["len_loss"], epoch)
            writer.add_scalar("Loss/train_loc", train_epoch_losses["loc_loss"], epoch)
            writer.add_scalar("Loss/train_con", train_epoch_losses["con_loss"], epoch)
            writer.add_scalar("LR/lr", current_lr, epoch)
        if args.rank == 0:
            print(
                "[Epoch: %d][Time: %d][L: %.4f][L_rec: %.4f]"
                "[L_vq: %.4f][L_per: %.4f][L_disc: %.4f][L_pretext: %.4f]"
                "[l_len: %.4f][l_loc: %.4f][l_con: %.4f][lr: %.6f]"
                % (epoch, elapsed, loss_g_total,
                   train_epoch_losses["rec_loss"], train_epoch_losses["vq_loss"],
                   train_epoch_losses["per_loss"], train_epoch_losses["disc_loss"],
                   train_epoch_losses["pretext_loss"], train_epoch_losses["len_loss"],
                   train_epoch_losses["loc_loss"], train_epoch_losses["con_loss"],
                   current_lr)
            )
        logger.info(
            "[Epoch: %d][Time: %d][L: %.4f][L_rec: %.4f][L_vq: %.4f]"
            "[L_per: %.4f][L_disc: %.4f][L_pretext: %.4f]"
            "[l_len: %.4f][l_loc: %.4f][l_con: %.4f][lr: %.6f]"
            % (epoch, elapsed, loss_g_total,
               train_epoch_losses["rec_loss"], train_epoch_losses["vq_loss"],
               train_epoch_losses["per_loss"], train_epoch_losses["disc_loss"],
               train_epoch_losses["pretext_loss"], train_epoch_losses["len_loss"],
               train_epoch_losses["loc_loss"], train_epoch_losses["con_loss"],
               current_lr)
        )
        if args.distributed:
            if args.rank == 0:
                checkpoint = {
                    "model": autoencoder.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "args": args, "epoch": epoch,
                }
                checkpoint_path = os.path.join(args.model_dir, "Autoencoder.pt")
                torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint = {
                "model": autoencoder.state_dict(),
                "discriminator": discriminator.state_dict(),
                "args": args, "epoch": epoch,
            }
            checkpoint_path = os.path.join(args.model_dir, "Autoencoder.pt")
            torch.save(checkpoint, checkpoint_path)
        if loss_g_total < best_train_loss:
            best_train_loss = loss_g_total
            if args.distributed:
                if args.rank == 0:
                    checkpoint = {
                        "model": autoencoder.module.state_dict(),
                        "discriminator": discriminator.module.state_dict(),
                        "args": args, "epoch": epoch,
                    }
                    checkpoint_path = os.path.join(
                        args.model_dir, f"Autoencoder_best_epoch{epoch}.pt"
                    )
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved best checkpoint to {checkpoint_path}")
            else:
                checkpoint = {
                    "model": autoencoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "args": args, "epoch": epoch,
                }
                checkpoint_path = os.path.join(
                    args.model_dir, f"Autoencoder_best_epoch{epoch}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
        if epoch % args.ckpt_interval == 0 and epoch > 0:
            if args.distributed:
                if args.rank == 0:
                    checkpoint = {
                        "model": autoencoder.module.state_dict(),
                        "discriminator": discriminator.module.state_dict(),
                        "args": args, "epoch": epoch,
                    }
                    checkpoint_path = os.path.join(
                        args.model_dir, f"Autoencoder_epoch{epoch}.pt"
                    )
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved periodic checkpoint to {checkpoint_path}")
            else:
                checkpoint = {
                    "model": autoencoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "args": args, "epoch": epoch,
                }
                checkpoint_path = os.path.join(
                    args.model_dir, f"Autoencoder_epoch{epoch}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved periodic checkpoint to {checkpoint_path}")
        if epoch % val_interval == 0:
            autoencoder.eval()
            metric_epoch = {"psnr": 0, "ssim": 0, "mae": 0, "lpips": 0}
            test_epoch_losses = {"rec_loss": 0, "per_loss": 0}
            for batch in dataloader_test:
                with torch.no_grad():
                    with autocast(enabled=args.amp):
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
                        x_in = {"x_incomp": x_incomp, "x_missing": x_missing}
                        x_rec = {"x_incomp": rec_incomp, "x_missing": rec_missing}
                        rec_loss, per_loss = get_train_loss(
                            x_in, x_rec, intensity_loss, loss_perceptual
                        )
                        losses = {"rec_loss": rec_loss, "per_loss": per_loss}
                        metrics = {
                            "psnr": psnr(x_rec0, x_in0),
                            "ssim": ssim(x_rec0, x_in0),
                            "mae": mae(x_rec0, x_in0),
                            "lpips": per_loss,
                        }
                    for metric_name, metric_value in metrics.items():
                        metric_epoch[metric_name] += metric_value
                    for loss_name, loss_value in losses.items():
                        test_epoch_losses[loss_name] += loss_value.item()
            for key in test_epoch_losses:
                test_epoch_losses[key] /= len(dataloader_test)
            loss_g_total = test_loss_weighted_sum(args, test_epoch_losses)
            for key in metric_epoch:
                metric_epoch[key] /= len(dataloader_test)
            end_val = timeit.default_timer()
            time_val = end_val - start
            if args.rank == 0 and writer is not None:
                writer.add_scalar("Loss/val_total", loss_g_total, epoch)
                writer.add_scalar("Loss/val_rec", test_epoch_losses["rec_loss"], epoch)
                writer.add_scalar("Metrics/val_psnr", metric_epoch["psnr"], epoch)
                writer.add_scalar("Metrics/val_ssim", metric_epoch["ssim"], epoch)
                writer.add_scalar("Metrics/val_mae", metric_epoch["mae"], epoch)
                writer.add_scalar("Metrics/val_lpips", metric_epoch["lpips"], epoch)
            if args.rank == 0:
                print(
                    "[Val Epoch: %d][Time: %d][L: %.4f][L_rec: %.4f][L_per: %.4f]"
                    % (epoch, time_val, loss_g_total,
                       test_epoch_losses["rec_loss"], test_epoch_losses["per_loss"])
                )
                print(
                    "[Val Epoch: %d][PSNR: %.3f][SSIM: %.3f][MAE: %.3f][LPIPS: %.3f]"
                    % (epoch, metric_epoch["psnr"], metric_epoch["ssim"],
                       metric_epoch["mae"], metric_epoch["lpips"])
                )
            logger.info(
                "[Val Epoch: %d][Time: %d][L: %.4f][L_rec: %.4f][L_per: %.4f]"
                % (epoch, time_val, loss_g_total,
                   test_epoch_losses["rec_loss"], test_epoch_losses["per_loss"])
            )
            logger.info(
                "[Val Epoch: %d][PSNR: %.3f][SSIM: %.3f][MAE: %.3f][LPIPS: %.3f]"
                % (epoch, metric_epoch["psnr"], metric_epoch["ssim"],
                   metric_epoch["mae"], metric_epoch["lpips"])
            )
    if args.rank == 0 and writer is not None:
        writer.close()
    if args.distributed:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoPeVAE Stage 1 Training (Adapted)")
    parser.add_argument(
        "--data_root",
        default=os.environ.get(
            "DATA_ROOT",
            "/devdata/hsh/datasets/seg_dataset/BraTS2020/"
            "brats20-dataset-training-validation/versions/1/"
            "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        ),
        type=str, help="Root path to BraTS2020 TrainingData",
    )
    parser.add_argument(
        "--datalist_dir",
        default=os.environ.get(
            "DATALIST_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "datalist/BraTS2020"),
        ),
        type=str, help="Directory containing train.list, val.list, test.list",
    )
    parser.add_argument("--dataset", default="BraTS", type=str)
    parser.add_argument("--epochs", default=200, type=int,
                        help="number of training epochs (default: 200)")
    parser.add_argument("--num_steps", default=8000, type=int)
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--batch_size", default=2, type=int,
                        help="batch size (default: 2; reduce to 1 if OOM)")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--lrdecay", default=True, help="enable LR decay")
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--lr_min", default=8e-5, type=float)
    parser.add_argument("--opt", default="adam", type=str)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--ckpt_interval", default=50, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--cache", default=0.2, type=float)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--gpu_ids", default=[0])
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--diffusion_model", default="DiT", type=str,
                        choices=["UNet", "DiT"])
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results/task_{timestamp}"
    )
    args.result_dir = os.path.join(task_dir, "stage1_CoPeVAE")
    os.makedirs(args.result_dir, exist_ok=True)
    args.log_dir = os.path.join(args.result_dir, "log")
    os.makedirs(args.log_dir, exist_ok=True)
    args.model_dir = os.path.join(task_dir, "models", "CoPeVAE")
    os.makedirs(args.model_dir, exist_ok=True)
    args.tb_log_dir = os.path.join(task_dir, "tensorboard", "stage1")
    os.makedirs(args.tb_log_dir, exist_ok=True)
    print(f"Task directory: {task_dir}")
    print(f"Model save dir: {args.model_dir}")
    print(f"TensorBoard dir: {args.tb_log_dir}")
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
        print(
            f"Distributed training with {torch.cuda.device_count()} GPUs. "
            f"Process {args.rank}/{args.world_size}"
        )
    else:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Single process training on {args.device}")
    assert args.rank >= 0
    autoencoder = CopeVAE(args)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3, channels=32,
        in_channels=1, out_channels=1,
    )
    print(f"Starting CoPeVAE Stage 1 training for {args.epochs} epochs")
    train(args, autoencoder, discriminator)
