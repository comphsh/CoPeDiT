#!/usr/bin/env python
"""
Adapted MDiT3D (Stage 2) Training Script for BraTS2020.
Changes from original:
- Uses data/BraTS2020_data.py for MONAI-based BraTS2020 data loading
- TensorBoard logging for epoch loss/lr
- Model checkpoints saved to results/task_{timestamp}/models/
- Single GPU mode by default, 200 epochs
Requires: Pre-trained CoPeVAE checkpoint from Stage 1.
Original code: train_MDiT3D_Brain.py
"""

import argparse, os, timeit, logging, sys
from copy import deepcopy
from datetime import datetime
from time import time
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
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
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from data.BraTS2020_data import (
    MODALITY_KEYS, read_datalist, get_transforms, get_loader_MDiT3D,
)


def train(args, autoencoder, DiT):
    """Main training loop for MDiT3D Stage 2."""
    if args.rank == 0:
        logger = create_logger(args.log_dir, args.distributed)
    else:
        logger = create_logger(args.log_dir, args.distributed)
    if args.rank == 0:
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        logger.info(f"TensorBoard log directory: {args.tb_log_dir}")
    else:
        writer = None
    train_files, val_files, test_files = read_datalist(
        args.datalist_dir, args.data_root
    )
    logger.info(
        f"Loaded {len(train_files)} train, {len(val_files)} val, "
        f"{len(test_files)} test samples"
    )
    train_transform, test_transform = get_transforms(
        args, MODALITY_KEYS, is_train=True
    )
    _, val_transform = get_transforms(args, MODALITY_KEYS, is_train=False)
    dataloader_train, dataloader_test, train_sampler, test_sampler = (
        get_loader_MDiT3D(
            args, args.rank, args.world_size,
            train_files, val_files,
            train_transform, val_transform,
        )
    )
    if args.diff_loss == "l2":
        diff_loss = MSELoss()
        if args.rank == 0:
            print("Use l2 loss")
    else:
        diff_loss = L1Loss(reduction="mean")
        if args.rank == 0:
            print("Use l1 loss")
    accumulation_steps = args.gradient_accumulation_steps
    if args.noise_scheduler == "linear":
        scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            schedule="scaled_linear_beta",
            beta_start=0.0015, beta_end=0.0195,
            clip_sample=False,
        )
    elif args.noise_scheduler == "cosine":
        scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            schedule="cosine", clip_sample=False,
        )
    ema = deepcopy(DiT).to(args.device)
    requires_grad(ema, False)
    DiT = DiT.to(args.device)
    autoencoder = autoencoder.to(args.device)
    if args.ae_ckpt is not None and os.path.exists(args.ae_ckpt):
        state_dict = torch.load(str(args.ae_ckpt), map_location=args.device)
        autoencoder.load_state_dict(state_dict["model"])
        if args.rank == 0:
            print(f"AE checkpoint loaded from {args.ae_ckpt}")
    else:
        if args.rank == 0:
            print(
                f"WARNING: AE checkpoint not found at {args.ae_ckpt}. "
                "Training with randomly initialized autoencoder!"
            )
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    check_data = first(dataloader_train)
    with torch.no_grad():
        with autocast(enabled=True):
            z0 = autoencoder.encode_stage_2_inputs(
                check_data[0].to(args.device)
            )
            z1 = autoencoder.encode_stage_2_inputs(
                check_data[1].to(args.device)
            )
    scale_factor = 1.0
    print(f"scale_factor -> {scale_factor}.")
    torch.cuda.empty_cache()
    inferer = inferer_DiT_Brain.LatentDiffusionInferer(
        scheduler_ddpm, scale_factor=scale_factor
    )
    logger.info(
        f"AE Parameters: {sum(p.numel() for p in autoencoder.parameters()):,}"
    )
    logger.info(
        f"DiT Parameters: {sum(p.numel() for p in DiT.parameters()):,}"
    )
    if args.opt == "adam":
        optimizer = optim.Adam(params=DiT.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(
            params=DiT.parameters(), lr=args.lr, weight_decay=args.decay
        )
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            params=DiT.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.decay,
        )
    if args.lr_schedule == "warmup_cosine":
        scheduler = CosineLRScheduler(
            optimizer, warmup_t=args.warmup_steps, warmup_lr_init=1e-6,
            t_initial=args.num_steps, lr_min=args.lr_min, cycle_limit=1,
        )
    elif args.lr_schedule == "poly":
        def lambdas(epoch):
            return (1 - float(epoch) / float(args.epochs)) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambdas
        )
    if args.amp:
        scaler = GradScaler()
    DiT.train()
    ema.eval()
    val_interval = args.val_interval
    start_epoch = 0
    max_epochs = args.epochs
    start = timeit.default_timer()
    if args.distributed:
        DiT = DistributedDataParallel(DiT, device_ids=[args.rank])
    if args.resume_ckpt is not None and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(str(args.resume_ckpt), map_location="cpu")
        if args.distributed:
            DiT.module.load_state_dict(ckpt["model"])
        else:
            DiT.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"]
        logger.info(f"Resuming training from checkpoint, epoch: {start_epoch}")
    if args.resume_ckpt is None:
        if args.distributed:
            update_ema(ema, DiT.module, decay=0)
        else:
            update_ema(ema, DiT, decay=0)
    for epoch in range(start_epoch, max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        DiT.train()
        train_epoch_losses = {"diff_loss": 0}
        progress_bar = tqdm(
            enumerate(dataloader_train), total=len(dataloader_train), ncols=150
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for i, batch in progress_bar:
            x_available, x_missing, missing_condition = batch
            x_available = x_available.to(args.device)
            x_missing = x_missing.to(args.device)
            missing_condition = missing_condition.to(args.device)
            with torch.no_grad():
                with autocast(enabled=args.amp):
                    prompts = autoencoder.get_condition(x_available)
            with autocast(enabled=args.amp):
                noise, noise_missing = get_noise(z0, z1)
                noise = noise.to(args.device)
                noise_missing = noise_missing.to(args.device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps,
                    (x_available.shape[0],), device=x_available.device,
                ).long()
                pred, latent = inferer(
                    inputs=[x_available, x_missing],
                    autoencoder_model=autoencoder,
                    diffusion_model=DiT,
                    noise=noise_missing,
                    timesteps=timesteps,
                    condition=prompts,
                )
                if args.pred_type == "noise":
                    l_diff = diff_loss(pred, noise_missing)
                elif args.pred_type == "x":
                    l_diff = diff_loss(pred, latent)
                elif args.pred_type == "v":
                    v = scheduler_ddpm.get_velocity(latent, noise, timesteps)
                    l_diff = diff_loss(pred, v)
                l_diff = l_diff / accumulation_steps
                losses = {"diff_loss": l_diff * accumulation_steps}
            for loss_name, loss_value in losses.items():
                train_epoch_losses[loss_name] += loss_value.item()
            if args.amp:
                scaler.scale(l_diff).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        DiT.parameters(), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                l_diff.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        DiT.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if args.distributed:
                update_ema(ema, DiT.module)
            else:
                update_ema(ema, DiT)
        if args.lrdecay:
            scheduler.step(epoch)
        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)
        diff_loss_epoch = train_epoch_losses["diff_loss"]
        end = timeit.default_timer()
        elapsed = end - start
        if args.lr_schedule == "warmup_cosine":
            current_lr = scheduler._get_lr(epoch)[0]
        else:
            current_lr = scheduler.get_last_lr()[0]
        if args.rank == 0 and writer is not None:
            writer.add_scalar("Loss/train_diff", diff_loss_epoch, epoch)
            writer.add_scalar("LR/lr", current_lr, epoch)
        if args.rank == 0:
            print(
                "[Train Epoch: %d][Time: %d][L: %.4f][lr: %.6f]"
                % (epoch, elapsed, diff_loss_epoch, current_lr)
            )
        logger.info(
            "[Train Epoch: %d][Time: %d][L: %.4f][lr: %.6f]"
            % (epoch, elapsed, diff_loss_epoch, current_lr)
        )
        if args.distributed:
            if args.rank == 0:
                checkpoint = {
                    "model": DiT.module.state_dict(),
                    "ema": ema.state_dict(),
                    "args": args, "epoch": epoch,
                }
                checkpoint_path = os.path.join(args.model_dir, "DiT.pt")
                torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint = {
                "model": DiT.state_dict(),
                "ema": ema.state_dict(),
                "args": args, "epoch": epoch,
            }
            checkpoint_path = os.path.join(args.model_dir, "DiT.pt")
            torch.save(checkpoint, checkpoint_path)
        if epoch % args.ckpt_interval == 0 and epoch > 0:
            if args.distributed:
                if args.rank == 0:
                    checkpoint = {
                        "model": DiT.module.state_dict(),
                        "ema": ema.state_dict(),
                        "args": args, "epoch": epoch,
                    }
                    checkpoint_path = os.path.join(
                        args.model_dir, f"DiT_epoch{epoch}.pt"
                    )
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                checkpoint = {
                    "model": DiT.state_dict(),
                    "ema": ema.state_dict(),
                    "args": args, "epoch": epoch,
                }
                checkpoint_path = os.path.join(
                    args.model_dir, f"DiT_epoch{epoch}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        if epoch % val_interval == 0 and epoch > 0:
            DiT.eval()
            val_epoch_losses = {"diff_loss": 0}
            for batch in dataloader_test:
                with torch.no_grad():
                    with autocast(enabled=args.amp):
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
                            (x_available.shape[0],),
                            device=x_available.device,
                        ).long()
                        pred, latent = inferer(
                            inputs=[x_available, x_missing],
                            autoencoder_model=autoencoder,
                            diffusion_model=DiT,
                            noise=noise_missing,
                            timesteps=timesteps,
                            condition=prompts,
                        )
                        if args.pred_type == "noise":
                            l_diff = diff_loss(pred, noise_missing)
                        elif args.pred_type == "x":
                            l_diff = diff_loss(pred, latent)
                        elif args.pred_type == "v":
                            v = scheduler_ddpm.get_velocity(
                                latent, noise, timesteps
                            )
                            l_diff = diff_loss(pred, v)
                        losses = {"diff_loss": l_diff}
                    for loss_name, loss_value in losses.items():
                        val_epoch_losses[loss_name] += loss_value.item()
            for key in val_epoch_losses:
                val_epoch_losses[key] /= len(dataloader_test)
            val_diff_loss = val_epoch_losses["diff_loss"]
            end_val = timeit.default_timer()
            time_val = end_val - start
            if args.rank == 0 and writer is not None:
                writer.add_scalar("Loss/val_diff", val_diff_loss, epoch)
            if args.rank == 0:
                print(
                    "[Val Epoch: %d][Time: %d][L: %.4f]"
                    % (epoch, time_val, val_diff_loss)
                )
            logger.info(
                "[Val Epoch: %d][Time: %d][L: %.4f]"
                % (epoch, time_val, val_diff_loss)
            )
    if args.rank == 0 and writer is not None:
        writer.close()
    if args.distributed:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDiT3D Stage 2 Training (Adapted)")
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
    parser.add_argument(
        "--ae_ckpt", default=None, type=str,
        help="Path to pre-trained CoPeVAE checkpoint (required for Stage 2)",
    )
    parser.add_argument("--dataset", default="BraTS", type=str)
    parser.add_argument(
        "--missing_num", default=1, type=int,
        help="number of missing modalities (1, 2, or 3)",
        choices=[1, 2, 3],
    )
    parser.add_argument(
        "--epochs", default=200, type=int,
        help="number of training epochs (default: 200)",
    )
    parser.add_argument("--num_steps", default=8000, type=int)
    parser.add_argument("--warmup_steps", default=50, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int,
        help="batch size (default: 2; reduce to 1 if OOM)",
    )
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--lrdecay", default=True, help="enable LR decay")
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--lr_min", default=2e-5, type=float)
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--val_interval", default=20, type=int)
    parser.add_argument("--ckpt_interval", default=50, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--cache", default=0.2, type=float)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--gpu_ids", default=[0])
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--DiT", default="MDiT3D-B/2", type=str)
    parser.add_argument("--noise_scheduler", default="linear", type=str)
    parser.add_argument(
        "--pred_type", default="x", type=str,
        choices=["noise", "x", "v"],
    )
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results/task_{timestamp}"
    )
    args.result_dir = os.path.join(task_dir, "stage2_MDiT3D")
    os.makedirs(args.result_dir, exist_ok=True)
    args.log_dir = os.path.join(args.result_dir, "log")
    os.makedirs(args.log_dir, exist_ok=True)
    args.model_dir = os.path.join(task_dir, "models", "MDiT3D")
    os.makedirs(args.model_dir, exist_ok=True)
    args.tb_log_dir = os.path.join(task_dir, "tensorboard", "stage2")
    os.makedirs(args.tb_log_dir, exist_ok=True)
    print(f"Task directory: {task_dir}")
    print(f"Model save dir: {args.model_dir}")
    print(f"TensorBoard dir: {args.tb_log_dir}")
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
        print(
            f"Distributed training: {torch.cuda.device_count()} GPUs. "
            f"Process {args.rank}/{args.world_size}"
        )
    else:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Single process training on {args.device}")
    assert args.rank >= 0
    autoencoder = CopeVAE(args)
    DiT = MDiT3D_models[args.DiT](num_modalities=args.modality_num)
    print(f"Starting MDiT3D Stage 2 training for {args.epochs} epochs")
    print(f"Model: {args.DiT}, Missing modalities: {args.missing_num}")
    print(f"AE checkpoint: {args.ae_ckpt}")
    train(args, autoencoder, DiT)
