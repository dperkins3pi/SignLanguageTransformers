import torch
import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from datasets.video_dataloader import get_data_loaders


DATA_DIR = 'SignEase/ASL_Citizen'   # Replace with your data directory
VIDEO_DIR = DATA_DIR + '/videos'
SPLIT_DIR = DATA_DIR + '/splits'
BATCH_SIZE = 8

if __name__ == '__main__':

	# Build datasets and loaders using helper in datasets.video_dataset
	print('Loading datasets')
	train_loader, val_loader, test_loader, label_to_idx = get_data_loaders(VIDEO_DIR, SPLIT_DIR, batch_size=BATCH_SIZE)
	print('Train / Val / Test sizes:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

	print(len(train_loader))
	for batch_idx, batch in enumerate(train_loader):
		inputs = batch[0]           # (B, C, T, H, W)   (Batch, Channels, Time, Height, Width)
		filenames = batch[1]        # list[str]
		mask = batch[2]             # (B, T) or None if no masks
		labels = batch[3]           # (B,) or None if no labels
		print('inputs.shape =', inputs.shape)
		print('filenames =', filenames)
		if mask is not None:
			print('mask.shape =', mask.shape)
		if labels is not None:
			print('labels =', labels)
        
		assert 1==0

	# quick model: global average pool over frames and a linear head
	# class TinyModel(torch.nn.Module):
	#     def __init__(self, in_channels=3, num_frames=16, num_classes=10):
	#         super().__init__()
	#         self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))  # pool C,T,H,W -> C,1,1,1
	#         self.fc = torch.nn.Linear(in_channels, num_classes)

	#     def forward(self, x, mask=None):
	#         # x: (B, C, T, H, W)
	#         # If mask is provided, zero out padded frames before pooling
	#         if mask is not None:
	#             # mask: (B, T) -> (B, 1, T, 1, 1)
	#             m = mask[:, None, :, None, None].to(x.dtype)
	#             x = x * m
	#         pooled = self.pool(x).reshape(x.shape[0], -1)  # (B, C)
	#         return self.fc(pooled)

	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# model = TinyModel(in_channels=3, num_frames=16, num_classes=max(2, len(label_to_idx) if label_to_idx else 2)).to(device)
	# optim = torch.optim.Adam(model.parameters(), lr=1e-4)
	# loss_fn = torch.nn.CrossEntropyLoss()

	# # Run a single training epoch over a few batches as an example
	# model.train()
	# for batch_idx, batch in enumerate(train_loader):
	#     tensors, filenames = batch[0], batch[1]
	#     masks = None
	#     labels = None
	#     if len(batch) >= 3:
	#         masks = batch[2]
	#     if len(batch) == 4:
	#         labels = batch[3]

	#     tensors = tensors.to(device)
	#     if masks is not None:
	#         masks = masks.to(device)
	#     if labels is not None:
	#         labels = labels.to(device)

	#     logits = model(tensors, mask=masks)
	#     if labels is None:
	#         # dummy target (all zeros) to show loss computation
	#         target = torch.zeros(tensors.shape[0], dtype=torch.long, device=device)
	#     else:
	#         target = labels
	#     loss = loss_fn(logits, target)
	#     optim.zero_grad()
	#     loss.backward()
	#     optim.step()

	#     print(f'Batch {batch_idx} loss: {loss.item():.4f}')
	#     if batch_idx >= 2:
	#         break