import os
import cv2
import torch
import numpy as np
from typing import List, Optional, Callable
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Lightweight video dataset using OpenCV.

    Expects a list of file paths (or a directory) and will sample `num_frames`
    evenly across each video and return a tensor of shape
    (C, T, H, W) where T is num_frames.
    """
    def __init__(
        self,
        files: List[str],
        num_frames: Optional[int] = 16,
        transform: Optional[Callable] = None,
        *,
        pad_mode: str = "zero",
        return_mask: bool = False,
        labels: Optional[List[str]] = None,
        frame_size: Optional[tuple[int, int]] = (480, 640),
    ):
        """Initialize VideoDataset.

        Args:
            files: list of video file paths.
        num_frames: number of frames to sample per clip. If None, return all frames.
            transform: optional transform applied to the returned tensor.
            pad_mode: how to pad when video is shorter than num_frames. One of
                - "zero": pad with zeros (default)
                - "repeat": repeat last frame
            return_mask: if True, __getitem__ will return a mask (1=real frame, 0=padded).
            labels: optional list of labels aligned with `files`.
        """
        self.files = files
        self.num_frames = num_frames
        self.transform = transform
        assert pad_mode in ("repeat", "zero"), "pad_mode must be 'repeat' or 'zero'"
        self.pad_mode = pad_mode
        self.return_mask = return_mask
        self.labels = labels
        # frame_size is (H, W) to which frames will be resized using cv2.resize
        self.frame_size = frame_size

    def __len__(self):
        return len(self.files)

    def _read_video(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError(f"Could not open video: {path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0: raise RuntimeError(f"No frames extracted from: {path}")
        return np.stack(frames)

    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        # frames shape: (T_full, H, W, C)
        T_full = frames.shape[0]
        T_req = self.num_frames
        # If num_frames is None, return all frames (no sampling/padding)
        if T_req is None:
            if T_full == 0:
                raise RuntimeError("Video has no frames to sample")
            sampled = frames
            mask = np.ones(T_full, dtype=np.uint8)
            return sampled, mask

        # Existing behavior when a target number of frames is requested
        if T_full >= T_req:
            # uniform sampling (no padding)
            idx = np.linspace(0, T_full - 1, T_req).astype(int)
            sampled = frames[idx]
            mask = np.ones(T_req, dtype=np.uint8)
        else:
            # need to pad
            if self.pad_mode == "repeat":
                # repeat last frame to reach requested length
                if T_full == 0:
                    raise RuntimeError("Video has no frames to sample")
                idx = list(range(T_full)) + [T_full - 1] * (T_req - T_full)
                sampled = frames[idx]
                mask = np.array([1] * T_full + [0] * (T_req - T_full), dtype=np.uint8)
            else:  # zero padding
                H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]
                sampled = np.zeros((T_req, H, W, C), dtype=frames.dtype)
                sampled[:T_full] = frames
                mask = np.array([1] * T_full + [0] * (T_req - T_full), dtype=np.uint8)
        return sampled, mask

    def __getitem__(self, idx: int):
        path = self.files[idx]
        frames = self._read_video(path)
        sampled, mask = self._sample_frames(frames)
        # sampled shape: (T, H, W, C)
        # optionally resize frames to fixed frame_size (H, W)
        if self.frame_size is not None:
            target_h, target_w = self.frame_size
            resized_list = []
            for f in sampled:
                # f is HxWxC numpy array in RGB
                f_resized = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
                resized_list.append(f_resized)
            sampled = np.stack(resized_list)

        # convert to (C, T, H, W) and to float tensor in [0,1]
        sampled = sampled.astype(np.float32) / 255.0
        # transpose
        sampled = np.transpose(sampled, (3, 0, 1, 2))
        tensor = torch.from_numpy(sampled)  # (C, T, H, W)
        if self.transform is not None:
            # user-provided transform should accept a tensor or numpy array
            tensor = self.transform(tensor)
        # return (tensor, filename) or (tensor, filename, mask)
        filename = os.path.basename(path)
        if self.return_mask:
            mask_tensor = torch.from_numpy(mask.astype(np.uint8))
            # if labels are available, return (tensor, filename, mask, label)
            if self.labels is not None:
                return tensor, filename, mask_tensor, self.labels[idx]
            return tensor, filename, mask_tensor
        # optionally return label if provided
        if self.labels is not None:
            return tensor, filename, self.labels[idx]
        return tensor, filename

    @classmethod
    def from_split_csv(
        cls,
        csv_path: str,
        videos_dir: str,
        num_frames: Optional[int] = 16,
        transform: Optional[Callable] = None,
        frame_size: Optional[tuple[int, int]] = (480, 640),
        *,
        pad_mode: str = "zero",
        return_mask: bool = False,
    ) -> "VideoDataset":
        """Create a VideoDataset from a split CSV (Participant ID,Video file,Gloss,...).

        The CSV is expected to have a header containing a "Video file" column.
        The filenames will be joined with `videos_dir` to form full paths.
        """
        import csv

        files: List[str] = []
        labels: List[str] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "Video file" not in reader.fieldnames:
                # fallback: assume second column is filename
                reader = csv.reader(open(csv_path, newline="", encoding="utf-8"))
                next(reader)  # skip header
                for row in reader:
                    if len(row) < 2:
                        continue
                    fname = row[1].strip()
                    files.append(os.path.join(videos_dir, fname))
                    # try to get gloss/label if present
                    if len(row) >= 3:
                        labels.append(row[2].strip())
            else:
                for row in reader:
                    fname = row["Video file"].strip()
                    files.append(os.path.join(videos_dir, fname))
                    # optional label
                    labels.append(row.get("Gloss", ""))

        # filter files that don't exist (warn)
        existing_files = []
        existing_labels: List[str] = []
        for i, f in enumerate(files):
            if os.path.exists(f):
                existing_files.append(f)
                if labels:
                    existing_labels.append(labels[i])
            else:
                print(f"Warning: file not found, skipping: {f}")

        if labels and len(existing_labels) == len(existing_files):
            # warn if any label is empty
            missing = [existing_files[i] for i, lab in enumerate(existing_labels) if lab == ""]
            if missing:
                for f in missing:
                    print(f"Warning: file has no label (empty Gloss) in CSV: {f}")
            return cls(existing_files, num_frames, transform, pad_mode=pad_mode, return_mask=return_mask, labels=existing_labels, frame_size=frame_size)
        # if labels were provided but didn't align, warn about files with no label
        if labels and len(existing_labels) != len(existing_files):
            print("Warning: Some files referenced in CSV were missing on disk; labels may not align exactly.")
        return cls(existing_files, num_frames, transform, pad_mode=pad_mode, return_mask=return_mask, frame_size=frame_size)

