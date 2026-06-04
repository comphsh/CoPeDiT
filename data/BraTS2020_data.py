"""
Unified BraTS2020 Data Loading Module for CoPeDiT Adaptation.

Replaces the original BraTS2021/IXI hardcoded data loading with:
- MONAI-based DataLoader
- BraTS2020 dataset with datalist-based train/val/test split
- Modality order: [flair, t1, t1ce, t2]
- Intensity normalization: ScaleIntensityRangePercentilesd(lower=0, upper=99.5, b_min=0, b_max=1)
- Random missing modality masking for training (excludes all-0 and all-1 masks)
- Fixed 14-mask enumeration for evaluation

Reference: $MY_ROOT/scripts/_01_vae/maisi_train_vae_brats.py
"""

import os
import argparse
import random
import numpy as np
import torch
from collections.abc import Sequence
from functools import partial

from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    DistributedSampler,
    load_decathlon_datalist,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ToTensord,
)

# Modality order: [flair, t1, t1ce, t2]
MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]
MODALITY_COUNT = 4

# 14 missing modality masks (1=available, 0=missing), excluding all-0 and all-1
# Ordered by increasing number of available modalities
MASK_LIST = [
    # 3 missing (1 available) - mask_id 1-4
    [0, 0, 0, 1],  # mask_id 1:  only t2
    [0, 0, 1, 0],  # mask_id 2:  only t1ce
    [0, 1, 0, 0],  # mask_id 3:  only t1
    [1, 0, 0, 0],  # mask_id 4:  only flair
    # 2 missing (2 available) - mask_id 5-10
    [0, 0, 1, 1],  # mask_id 5:  t1ce + t2
    [0, 1, 0, 1],  # mask_id 6:  t1 + t2
    [0, 1, 1, 0],  # mask_id 7:  t1 + t1ce
    [1, 0, 0, 1],  # mask_id 8:  flair + t2
    [1, 0, 1, 0],  # mask_id 9:  flair + t1ce
    [1, 1, 0, 0],  # mask_id 10: flair + t1
    # 1 missing (3 available) - mask_id 11-14
    [0, 1, 1, 1],  # mask_id 11: t1 + t1ce + t2 (missing flair)
    [1, 0, 1, 1],  # mask_id 12: flair + t1ce + t2 (missing t1)
    [1, 1, 0, 1],  # mask_id 13: flair + t1 + t2 (missing t1ce)
    [1, 1, 1, 0],  # mask_id 14: flair + t1 + t1ce (missing t2)
]


def mask_to_string(mask):
    """Convert mask list to string like '0111'."""
    return ''.join(str(int(m)) for m in mask)


def read_datalist(datalist_dir, data_root):
    """
    Read datalist files and create MONAI-compatible data dicts.

    Args:
        datalist_dir: Path to directory containing train.list, val.list, test.list
        data_root: Root path to BraTS2020 TrainingData

    Returns:
        train_files, val_files, test_files: lists of dicts for MONAI DataLoader
    """
    splits = {}
    for split_name in ["train", "val", "test"]:
        list_path = os.path.join(datalist_dir, f"{split_name}.list")
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Datalist not found: {list_path}")

        with open(list_path, "r") as f:
            subject_ids = [line.strip() for line in f if line.strip()]

        data_dicts = []
        for sid in subject_ids:
            item = {}
            for mod in MODALITY_KEYS:
                # BraTS2020 file naming: BraTS20_Training_XXX_{modality}.nii
                item[mod] = os.path.join(data_root, sid, f"{sid}_{mod}.nii")
            item["subject_id"] = sid
            data_dicts.append(item)

        splits[split_name] = data_dicts

    return splits["train"], splits["val"], splits["test"]


def get_transforms(args, keys, is_train=True):
    """
    Build MONAI transform pipelines for BraTS2020.

    The transform chain follows the original CoPeDiT design but adapted for BraTS2020:
    - LoadImaged: Load NIfTI files
    - EnsureChannelFirstd: Add channel dimension (C=1)
    - EnsureTyped: Convert to float32
    - Orientationd: RAS orientation
    - ScaleIntensityRangePercentilesd: 0-99.5 percentile → [0, 1]
    - Crop / Resize
    - Data augmentation (training only)

    Args:
        args: Arguments with brain_size, brain_roi attributes
        keys: Modality key names
        is_train: Whether to add training augmentations

    Returns:
        MONAI Compose transform
    """
    # Common transforms for both train and test
    common_transform = [
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        EnsureTyped(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        ScaleIntensityRangePercentilesd(
            keys=keys, lower=0, upper=99.5, b_min=0, b_max=1,
            allow_missing_keys=True
        ),
        CenterSpatialCropd(
            keys=keys, roi_size=args.brain_roi, allow_missing_keys=True
        ),
    ]

    if is_train:
        transform = Compose(
            common_transform
            + [
                RandSpatialCropd(
                    keys=keys, roi_size=args.brain_roi,
                    allow_missing_keys=True,
                    random_size=False, random_center=True,
                ),
                RandFlipd(keys=keys, prob=0.3, spatial_axis=0, allow_missing_keys=True),
                RandFlipd(keys=keys, prob=0.3, spatial_axis=1, allow_missing_keys=True),
                RandFlipd(keys=keys, prob=0.3, spatial_axis=2, allow_missing_keys=True),
                RandRotate90d(keys=keys, prob=0.3, max_k=3, allow_missing_keys=True),
                RandScaleIntensityd(keys=keys, prob=0.2, factors=(0.9, 1.1), allow_missing_keys=True),
                RandShiftIntensityd(keys=keys, prob=0.2, offsets=0.05, allow_missing_keys=True),
                Resized(
                    keys=keys, spatial_size=args.brain_size,
                    mode="trilinear", align_corners=True,
                    allow_missing_keys=True,
                ),
                ToTensord(keys=keys, allow_missing_keys=True),
            ]
        )
    else:
        transform = Compose(
            common_transform
            + [
                Resized(
                    keys=keys, spatial_size=args.brain_size,
                    mode="trilinear", align_corners=True,
                    allow_missing_keys=True,
                ),
                ToTensord(keys=keys, allow_missing_keys=True),
            ]
        )

    return transform


def random_mask_sample(x_in, seed):
    """
    Randomly mask modalities for training.
    Excludes all-0 (no input) and all-1 (nothing to generate) cases.

    Args:
        x_in: Full modality tensor [1, 4, H, W, D]
        seed: Random seed for reproducibility

    Returns:
        x_available: Available modalities [1, N_avail, H, W, D]
        x_missing: Missing modalities [1, N_miss, H, W, D]
        missing_mask: Binary mask tensor [4] (1=available, 0=missing)
    """
    torch.manual_seed(seed)
    modality = x_in.shape[1]  # 4

    # Random number of missing modalities: 1, 2, or 3
    missing_num = int(torch.randint(1, modality, (1,)).item())

    comp_list = list(range(modality))
    indices = torch.randperm(modality)[:missing_num]
    missing_idx = sorted([comp_list[i] for i in indices])
    available_idx = sorted(list(set(comp_list) - set(missing_idx)))

    x_missing = x_in[:, missing_idx, ...]
    x_available = x_in[:, available_idx, ...]

    missing_mask = torch.zeros(modality, dtype=torch.float32)
    missing_mask[available_idx] = 1.0

    return x_available, x_missing, missing_mask


def fixed_mask_sample(x_in, mask):
    """
    Apply a fixed mask to split modalities into available/missing.

    Args:
        x_in: Full modality tensor [1, 4, H, W, D]
        mask: List of 4 ints (1=available, 0=missing)

    Returns:
        x_available: Available modalities [1, N_avail, H, W, D]
        x_missing: Missing modalities [1, N_miss, H, W, D]
    """
    available_idx = [i for i, m in enumerate(mask) if m == 1]
    missing_idx = [i for i, m in enumerate(mask) if m == 0]

    x_available = x_in[:, available_idx, ...]
    x_missing = x_in[:, missing_idx, ...]

    return x_available, x_missing


def get_missing_label(number, missing_idx):
    """Create a binary label vector: 1 at missing_idx positions."""
    missing_label = np.zeros(number, dtype=int)
    for i in missing_idx:
        missing_label[i] = 1
    return torch.FloatTensor(missing_label)


# ==============================================================================
# Collate functions for different training stages
# ==============================================================================

def collate_fn_CoPeVAE(batch, missing_num=None):
    """
    Collate function for CoPeVAE (Stage 1) training.

    Returns:
        x_incomp: Incomplete (available) modalities [B, N_avail, H, W, D]
        x_missing: Missing modalities [B, N_miss, H, W, D]
        missing_length: One-hot for number of missing modalities
        missing_label: Binary mask [B, 4] (1=missing position)
    """
    data, idx = zip(*batch)
    seed0 = sum(idx) + 2025

    x_incomp_list, x_missing_list = [], []
    missing_length_list, missing_label_list = [], []

    for i, (sample_idx, item) in enumerate(zip(idx, data)):
        seed1 = sample_idx + 2025
        img = torch.stack(
            [item[k] for k in MODALITY_KEYS], dim=1
        )  # [1, 4, H, W, D]

        # Randomly determine missing count
        torch.manual_seed(seed0)
        modality = img.shape[1]
        missing_values = list(range(1, modality))
        midx = int(torch.randint(0, len(missing_values), (1,)).item())
        missing_num = missing_values[midx]
        missing_num_list = np.zeros(len(missing_values), dtype=int)
        missing_num_list[midx] = 1

        # Randomly select which modalities to mask
        torch.manual_seed(seed1)
        comp_list = list(range(modality))
        indices = torch.randperm(modality)[:missing_num]
        missing_idx = sorted([comp_list[i] for i in indices])
        incomp_idx = sorted(list(set(comp_list) - set(missing_idx)))

        x_incomp = img[:, incomp_idx, ...]
        x_missing = img[:, missing_idx, ...]
        missing_label = get_missing_label(modality, missing_idx)

        x_incomp_list.append(x_incomp)
        x_missing_list.append(x_missing)
        missing_length_list.append(
            torch.tensor(missing_num_list, dtype=torch.float32)
        )
        missing_label_list.append(missing_label)

    return (
        torch.cat(x_incomp_list),
        torch.cat(x_missing_list),
        torch.stack(missing_length_list),
        torch.stack(missing_label_list),
    )


def collate_fn_MDiT3D(batch, missing_num=1):
    """
    Collate function for MDiT3D (Stage 2) training.

    Uses fixed missing_num for training consistency.

    Returns:
        x_available: Available modalities [B, N_avail, H, W, D]
        x_missing: Missing modalities [B, N_miss, H, W, D]
        missing_condition: Binary mask [B, 4] (1=available, 0=missing)
    """
    data, idx = zip(*batch)
    x_available_list, x_missing_list = [], []
    missing_cond_list = []

    for i, (sample_idx, item) in enumerate(zip(idx, data)):
        seed = sample_idx + 2025
        img = torch.stack(
            [item[k] for k in MODALITY_KEYS], dim=1
        )  # [1, 4, H, W, D]

        x_avail, x_miss, mask = random_mask_sample(img, seed)
        x_available_list.append(x_avail)
        x_missing_list.append(x_miss)
        missing_cond_list.append(mask)

    return (
        torch.cat(x_available_list),
        torch.cat(x_missing_list),
        torch.stack(missing_cond_list),
    )


# ==============================================================================
# MonaiDataset wrapper (compatible with original CoPeDiT pattern)
# ==============================================================================

class MonaiDataset(CacheDataset):
    """CacheDataset wrapper that returns (data, index) for collate functions."""

    def __getitem__(self, index):
        if isinstance(index, slice):
            from torch.utils.data import Subset as TorchSubset
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return TorchSubset(dataset=self, indices=indices)
        if isinstance(index, Sequence):
            from torch.utils.data import Subset as TorchSubset
            return TorchSubset(dataset=self, indices=index)
        return self._transform(index), index


# ==============================================================================
# DataLoader factory functions
# ==============================================================================

def get_loader_CoPeVAE(args, rank, world_size, train_files, test_files, train_transform, test_transform):
    """
    Create DataLoaders for CoPeVAE (Stage 1) training.

    Returns:
        dataloader_train, dataloader_test, train_sampler, test_sampler
    """
    dataset_train = MonaiDataset(
        data=train_files, transform=train_transform,
        cache_rate=args.cache, num_workers=8,
    )
    dataset_test = MonaiDataset(
        data=test_files, transform=test_transform,
        cache_rate=args.cache, num_workers=8,
    )

    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=dataset_train, even_divisible=True, shuffle=True,
            rank=rank, num_replicas=world_size,
        )
        test_sampler = DistributedSampler(
            dataset=dataset_test, even_divisible=True, shuffle=False,
            rank=rank, num_replicas=world_size,
        )
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size,
            collate_fn=collate_fn_CoPeVAE, num_workers=8,
            shuffle=False, drop_last=True, sampler=train_sampler,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size,
            collate_fn=collate_fn_CoPeVAE, num_workers=8,
            shuffle=False, drop_last=True, sampler=test_sampler,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )
    else:
        train_sampler = None
        test_sampler = None
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size,
            collate_fn=collate_fn_CoPeVAE, num_workers=4,
            shuffle=True, drop_last=True,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size,
            collate_fn=collate_fn_CoPeVAE, num_workers=4,
            shuffle=False, drop_last=True,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )

    return dataloader_train, dataloader_test, train_sampler, test_sampler


def get_loader_MDiT3D(args, rank, world_size, train_files, test_files, train_transform, test_transform):
    """
    Create DataLoaders for MDiT3D (Stage 2) training.

    Returns:
        dataloader_train, dataloader_test, train_sampler, test_sampler
    """
    inf_collate_fn = partial(collate_fn_MDiT3D, missing_num=args.missing_num)

    dataset_train = MonaiDataset(
        data=train_files, transform=train_transform,
        cache_rate=args.cache, num_workers=8,
    )
    dataset_test = MonaiDataset(
        data=test_files, transform=test_transform,
        cache_rate=args.cache, num_workers=8,
    )

    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=dataset_train, even_divisible=True, shuffle=True,
            rank=rank, num_replicas=world_size,
        )
        test_sampler = DistributedSampler(
            dataset=dataset_test, even_divisible=True, shuffle=False,
            rank=rank, num_replicas=world_size,
        )
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size,
            collate_fn=inf_collate_fn, num_workers=8,
            shuffle=False, drop_last=True, sampler=train_sampler,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size,
            collate_fn=inf_collate_fn, num_workers=8,
            shuffle=False, drop_last=True, sampler=test_sampler,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )
    else:
        train_sampler = None
        test_sampler = None
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size,
            collate_fn=inf_collate_fn, num_workers=4,
            shuffle=True, drop_last=True,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size,
            collate_fn=inf_collate_fn, num_workers=4,
            shuffle=False, drop_last=True,
            persistent_workers=True, pin_memory=True, prefetch_factor=2,
        )

    return dataloader_train, dataloader_test, train_sampler, test_sampler


def get_loader_eval(args, test_files, test_transform):
    """
    Create DataLoader for evaluation (returns subject_id, no random masking).

    Each batch item returns the full 4-modality tensor and subject_id.
    """
    class EvalDataset(Dataset):
        def __getitem__(self, index):
            data = self._transform(index)
            return data, index

    dataset_test = EvalDataset(data=test_files, transform=test_transform)
    dataloader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=4, persistent_workers=True, pin_memory=True,
    )
    return dataloader_test
