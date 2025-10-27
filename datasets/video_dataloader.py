import functools
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
from .video_dataset import VideoDataset

def collate_batch(batch, label_to_idx=None, use_original_videos=True):
    """Collate function that dynamically pads variable-length video tensors along T (time)."""
    if use_original_videos: all_frames, all_segmented_frames, all_joints, all_filenames, all_labels, lengths = [], [], [], [], [], []
    else: all_segmented_frames, all_joints, all_filenames, all_labels, lengths = [], [], [], [], []

    # Extract dimensions
    if use_original_videos: _, H, W = batch[0][1].shape
    else: _, H, W = batch[0][0].shape
    if use_original_videos: _, D = batch[0][2].shape
    else: _, D = batch[0][1].shape

    # Split items and record lengths
    if use_original_videos:
        for frames, segmented_frames, joints, filename, label in batch:
            all_frames.append(frames)
            all_segmented_frames.append(segmented_frames)
            all_joints.append(joints)
            all_filenames.append(filename)
            all_labels.append(label)
            lengths.append(frames.shape[0])
    else:
        for segmented_frames, joints, filename, label in batch:
            all_segmented_frames.append(segmented_frames)
            all_joints.append(joints)
            all_filenames.append(filename)
            all_labels.append(label)
            lengths.append(segmented_frames.shape[0])

    max_length = max(lengths)
    batch_size = len(batch)

    # Initialize masks and padded containers
    masks = torch.zeros((batch_size, max_length), dtype=torch.float32)
    padded_frames, padded_segmented_frames, padded_joints = [], [], []

    for i in range(batch_size):
        if use_original_videos: t, h, w, c = all_frames[i].shape
        else: t, h, w = all_segmented_frames[i].shape
        tdiff = max_length - t

        # Pad along the temporal dimension (dim=0)
        if tdiff > 0:
            if use_original_videos: frame_pad = torch.zeros((tdiff, h, w, c), dtype=all_frames[i].dtype)
            seg_pad = torch.zeros((tdiff, h, w), dtype=all_segmented_frames[i].dtype)
            joint_pad = torch.zeros((tdiff, D), dtype=all_joints[i].dtype)
            if use_original_videos: frames_padded = torch.cat([all_frames[i], frame_pad], dim=0)
            seg_padded = torch.cat([all_segmented_frames[i], seg_pad], dim=0)
            joints_padded = torch.cat([all_joints[i], joint_pad], dim=0)
        else:
            if use_original_videos: frames_padded, seg_padded, joints_padded = all_frames[i], all_segmented_frames[i], all_joints[i]
            else: seg_padded, joints_padded = all_segmented_frames[i], all_joints[i]

        if use_original_videos: padded_frames.append(frames_padded)
        padded_segmented_frames.append(seg_padded)
        padded_joints.append(joints_padded)
        masks[i, :t] = 1

    # Stack all tensors along batch dimension
    if use_original_videos: all_frames = torch.stack(padded_frames).float()
    all_segmented_frames = torch.stack(padded_segmented_frames).float()
    all_joints = torch.stack(padded_joints).float()

    # Map labels to indices
    all_labels = torch.tensor([label_to_idx[l] if label_to_idx and l is not None else -1 for l in all_labels], dtype=torch.long)

    if use_original_videos: return all_filenames, all_frames, all_segmented_frames, all_joints, masks, all_labels
    else: return all_filenames, all_segmented_frames, all_joints, masks, all_labels


def get_data_loaders(
    videos_dir: str,
    splits_dir: str,
    segmented_dir: str, 
    joint_dir: str,
    batch_size: int = 8,
    stride: int=1,
    num_workers: int = 4,
    num_frames: Optional[int] = None,
    pad_mode: str = "repeat",
    return_mask: bool = True,
    debugging: bool = False,
    use_original_videos: bool = True,
):
    """Helper to construct train/val/test VideoDatasets and DataLoaders.

    Returns: (train_loader, val_loader, test_loader, label_to_idx)
    """
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_ds = VideoDataset.from_split_csv(train_csv, videos_dir, segmented_dir, joint_dir, num_frames=num_frames, return_mask=return_mask, pad_mode=pad_mode, stride=stride, use_original_videos=use_original_videos)
    val_ds = VideoDataset.from_split_csv(val_csv, videos_dir, segmented_dir, joint_dir, num_frames=num_frames, return_mask=return_mask, pad_mode=pad_mode, stride=stride, use_original_videos=use_original_videos)
    test_ds = VideoDataset.from_split_csv(test_csv, videos_dir, segmented_dir, joint_dir, num_frames=num_frames, return_mask=return_mask, pad_mode=pad_mode, stride=stride, use_original_videos=use_original_videos)

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

    if debugging:
        # use small subsets for debugging
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(20, len(train_ds)))))
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(20, len(val_ds)))))
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(20, len(test_ds)))))

    collate = functools.partial(collate_batch, label_to_idx=label_to_idx, use_original_videos=use_original_videos)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), collate_fn=collate)

    return train_loader, val_loader, test_loader, label_to_idx
