import os
import csv
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
        video_files: List[str],
        segmented_files: List[str],
        joint_files: List[str],
        num_frames: Optional[int] = 16,
        transform: Optional[Callable] = None,
        *,
        pad_mode: str = "zero",
        return_mask: bool = False,
        labels: Optional[List[str]] = None,
        frame_size: Optional[tuple[int, int]] = (224, 224),
        stride: int=1
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
        self.video_files = video_files
        self.segmented_files = segmented_files
        self.joint_files = joint_files
        self.num_frames = num_frames
        self.transform = transform
        assert pad_mode in ("repeat", "zero"), "pad_mode must be 'repeat' or 'zero'"
        self.pad_mode = pad_mode
        self.return_mask = return_mask
        self.labels = labels
        # frame_size is (H, W) to which frames will be resized using cv2.resize
        self.frame_size = frame_size
        self.stride = stride

    def __len__(self):
        return len(self.video_files)

    def _read_video(self, path: str, frame_numbers: np.ndarray) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError(f"Could not open video: {path}")

        frames = []
        if frame_numbers is None: frame_numbers = np.arange(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.stride)
        for frame_idx in frame_numbers:  # Only keep frames where the joints were foung
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        if len(frames) == 0: raise RuntimeError(f"No frames extracted from: {path}")
        return np.stack(frames)/255.0   # Normalize it
    
    def _read_coordinates(self, path: str) -> np.ndarray:
        rows = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                rows.append([float(x) for x in row])  # convert all to float
        return np.array(rows)  # shape: (num_frames, num_features)

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
        video_path = self.video_files[idx]
        segmented_path = self.segmented_files[idx]
        joint_path = self.joint_files[idx]
        joints = self._read_coordinates(joint_path)
        joints = joints[::self.stride]   
        frame_numbers = joints[:,0].astype(int)
        joints = joints[:, 1:]   # Remove the frame number
        frames = self._read_video(video_path, frame_numbers)
        segmented_frames = self._read_video(segmented_path, None)   # Set frame_numbers to None for segmented

        frames = torch.from_numpy(frames)  # (T, H, W)
        segmented_frames = torch.from_numpy(segmented_frames)
        joints = torch.from_numpy(joints)
        if self.transform is not None:
            frames = self.transform(frames)
            segmented_frames = self.transform(segmented_frames)

        filename = os.path.basename(video_path)
        if self.labels is not None:
            return frames, segmented_frames, joints, filename, self.labels[idx]
        return frames, segmented_frames, joints, filename

    @classmethod
    def from_split_csv(
        cls,
        csv_path: str,
        videos_dir: str,
        segmented_dir: str, 
        joint_dir: str,
        num_frames: Optional[int] = 16,
        transform: Optional[Callable] = None,
        frame_size: Optional[tuple[int, int]] = (224, 224),
        stride: int=1,
        *,
        pad_mode: str = "zero",
        return_mask: bool = False,
    ) -> "VideoDataset":
        """Create a VideoDataset from a split CSV (Participant ID,Video file,Gloss,...).

        The CSV is expected to have a header containing a "Video file" column.
        The filenames will be joined with `videos_dir` to form full paths.
        """
        video_files: List[str] = []
        segmented_files: List[str] = []
        joint_files: List[str] = []
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
                    video_files.append(os.path.join(videos_dir, fname))
                    segmented_files.append(os.path.join(segmented_dir, f"segmented-{fname}"))
                    joint_files.append(os.path.join(joint_dir, f"{fname[:-4]}.csv"))
                    # try to get gloss/label if present
                    if len(row) >= 3:
                        labels.append(row[2].strip())
            else:
                for row in reader:
                    fname = row["Video file"].strip()
                    video_files.append(os.path.join(videos_dir, fname))
                    segmented_files.append(os.path.join(segmented_dir, f"segmented-{fname}"))
                    joint_files.append(os.path.join(joint_dir, f"{fname[:-4]}.csv"))
                    # optional label
                    labels.append(row.get("Gloss", ""))

        # filter files that don't exist (warn)
        existing_files = []
        existing_segmented_files = []
        existing_joint_files = []
        existing_labels: List[str] = []
        for i, (f1, f2, f3) in enumerate(zip(video_files, segmented_files, joint_files)):
            if os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3):
                existing_files.append(f1)
                existing_segmented_files.append(f2)
                existing_joint_files.append(f3)
                if labels: existing_labels.append(labels[i])
            elif os.path.exists(f1) and os.path.exists(f2): print(f"Warning: joint file not found, skipping: {f3}")
            elif os.path.exists(f1): print(f"Warning: segmented file not found, skipping: {f2}")
            else: print(f"Warning: video file not found, skipping: {f1}")

        if labels and len(existing_labels) == len(existing_files):
            # warn if any label is empty
            missing = [existing_files[i] for i, lab in enumerate(existing_labels) if lab == ""]
            if missing:
                for f in missing: print(f"Warning: file has no label (empty Gloss) in CSV: {f}")

            return cls(existing_files, existing_segmented_files, existing_joint_files, num_frames, transform, pad_mode=pad_mode, return_mask=return_mask, labels=existing_labels, frame_size=frame_size, stride=stride)
        # if labels were provided but didn't align, warn about files with no label
        if labels and len(existing_labels) != len(existing_files):
            print("Warning: Some files referenced in CSV were missing on disk; labels may not align exactly.")
        
        return cls(existing_files, existing_segmented_files, existing_joint_files, num_frames, transform, pad_mode=pad_mode, return_mask=return_mask, frame_size=frame_size, stride=stride)

