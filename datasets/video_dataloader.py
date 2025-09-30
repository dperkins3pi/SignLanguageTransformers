import functools
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader


def collate_batch(batch, label_to_idx=None):
    """Collate function that pads variable-length clips along T (time) with zeros.

    This is provided here so DataLoader workers can pickle and import it from
    the `datasets.collate` module. See training.py for expected usage.
    """
    tensors = []
    filenames = []
    masks_in = []
    labels_in = []

    for item in batch:
        if len(item) == 2:
            t, fn = item
            tensors.append(t)
            filenames.append(fn)
            masks_in.append(None)
            labels_in.append(None)
        elif len(item) == 3:
            t, fn, third = item
            tensors.append(t)
            filenames.append(fn)
            # disambiguate: assume string => label, else mask
            if isinstance(third, str):
                masks_in.append(None)
                labels_in.append(third)
            else:
                masks_in.append(third)
                labels_in.append(None)
        elif len(item) == 4:
            t, fn, m, lab = item
            tensors.append(t)
            filenames.append(fn)
            masks_in.append(m)
            labels_in.append(lab)
        else:
            raise RuntimeError("Unexpected item format from dataset")

    # convert tensors to torch.Tensor if needed and collect temporal lengths
    ts = []
    T_list = []
    HW_shape = None
    HW_list = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        # t shape: (C, T, H, W)
        if t.ndim != 4:
            raise RuntimeError(f"Expected tensor with 4 dims (C,T,H,W), got {t.shape}")
        ts.append(t)
        T_list.append(t.shape[1])
        if HW_shape is None:
            HW_shape = t.shape[2:]
        HW_list.append(t.shape[2:])

    B = len(ts)
    C = ts[0].shape[0]
    T_max = max(T_list)
    H, W = HW_shape

    # Allocate batch tensor and pad along T with zeros
    batch_t = torch.zeros((B, C, T_max, H, W), dtype=ts[0].dtype)
    batch_mask = torch.zeros((B, T_max), dtype=torch.uint8)
    for i, t in enumerate(ts):
        Ti = t.shape[1]
        batch_t[i, :, :Ti, :, :] = t
        batch_mask[i, :Ti] = 1

    out = [batch_t, filenames]

    # Attach mask only if any dataset requested/returned a mask
    if any(m is not None for m in masks_in):
        provided_masks = [m for m in masks_in if m is not None]
        if len(provided_masks) == B:
            mask_tensor = torch.zeros((B, T_max), dtype=torch.uint8)
            for i, m in enumerate(masks_in):
                mi = m
                if not isinstance(mi, torch.Tensor):
                    mi = torch.as_tensor(mi)
                mask_tensor[i, : mi.shape[0]] = mi.to(torch.uint8)
            out.append(mask_tensor)
        else:
            out.append(batch_mask)
    # Attach labels if present
    if any(l is not None for l in labels_in):
        labs = [l for l in labels_in if l is not None]
        if label_to_idx is None:
            uniq = sorted(set(labs))
            label_to_idx = {l: i for i, l in enumerate(uniq)}
        label_idxs = torch.tensor([label_to_idx[l] if l is not None else -1 for l in labels_in], dtype=torch.long)
        out.append(label_idxs)

    return tuple(out)


def get_data_loaders(
    videos_dir: str,
    splits_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: Optional[int] = None,
    pad_mode: str = "repeat",
    return_mask: bool = True,
):
    """Helper to construct train/val/test VideoDatasets and DataLoaders.

    Returns: (train_loader, val_loader, test_loader, label_to_idx)
    """
    from .video_dataset import VideoDataset

    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_ds = VideoDataset.from_split_csv(train_csv, videos_dir, num_frames=num_frames, return_mask=return_mask, pad_mode=pad_mode)
    val_ds = VideoDataset.from_split_csv(val_csv, videos_dir, num_frames=num_frames, return_mask=return_mask, pad_mode=pad_mode)
    test_ds = VideoDataset.from_split_csv(test_csv, videos_dir, num_frames=num_frames, return_mask=return_mask, pad_mode=pad_mode)

    # build label->idx
    label_to_idx = None
    if getattr(train_ds, 'labels', None):
        uniq = sorted(set(train_ds.labels))
        if uniq:
            label_to_idx = {l: i for i, l in enumerate(uniq)}
    else:
        # fallback: inspect a few items
        example = train_ds[0]
        if len(example) == 4:
            all_labels = []
            for i in range(min(500, len(train_ds))):
                it = train_ds[i]
                if len(it) == 4:
                    all_labels.append(it[3])
            uniq = sorted(set(all_labels))
            if uniq:
                label_to_idx = {l: i for i, l in enumerate(uniq)}

    collate = functools.partial(collate_batch, label_to_idx=label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), collate_fn=collate)

    return train_loader, val_loader, test_loader, label_to_idx
