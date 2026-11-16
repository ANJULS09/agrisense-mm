"""
AgriSense-MM: Multimodal Dataset and DataLoader.
Loads paired samples containing:
- Leaf image (RGB, 224x224, normalized)
- Environmental 14-day sequence [14, 7]
- IMU 6-channel waveform [6, 100]
- Static geographic and edaphic features [5]
- Multi-task targets (cls, severity, forecast_3d, forecast_7d, quality, platform_state)
- Field zone coordinates & grid graph edge index
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from agrisense_mm.models.gnn import build_grid_edge_index
from agrisense_mm.data.constants import FIELD_GRID_ROWS, FIELD_GRID_COLS


def get_default_transforms(is_train: bool = True) -> T.Compose:
    """Standard image transforms for 224x224 EfficientNet/ViT."""
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            normalize,
        ])


class MultiModalCropDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        transform: Optional[T.Compose] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or get_default_transforms(is_train=(split == "train"))

        # Load metadata and arrays
        meta_path = os.path.join(data_dir, "metadata.json")
        with open(meta_path, "r") as f:
            all_meta = json.load(f)

        self.env_seqs = np.load(os.path.join(data_dir, "env_sequences.npy"))
        self.imu_windows = np.load(os.path.join(data_dir, "imu_windows.npy"))
        self.static_feats = np.load(os.path.join(data_dir, "static_features.npy"))
        self.targets_cls = np.load(os.path.join(data_dir, "targets_cls.npy"))
        self.targets_sev = np.load(os.path.join(data_dir, "targets_sev.npy"))
        self.targets_fc3 = np.load(os.path.join(data_dir, "targets_fc3.npy"))
        self.targets_fc7 = np.load(os.path.join(data_dir, "targets_fc7.npy"))
        self.targets_qual = np.load(os.path.join(data_dir, "targets_qual.npy"))
        self.targets_state = np.load(os.path.join(data_dir, "targets_state.npy"))

        num_samples = len(all_meta)
        indices = np.arange(num_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)

        split_idx = int(num_samples * train_ratio)
        if split == "train":
            self.selected_indices = indices[:split_idx]
        else:
            self.selected_indices = indices[split_idx:]

        self.metadata = [all_meta[i] for i in self.selected_indices]
        self.images_dir = os.path.join(data_dir, "images")

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.selected_indices[idx]
        meta = self.metadata[idx]

        # 1. Image
        img_path = os.path.join(self.images_dir, meta["image_file"])
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(raw_img)

        # 2. Sensor arrays
        env_tensor = torch.tensor(self.env_seqs[global_idx], dtype=torch.float32)
        imu_tensor = torch.tensor(self.imu_windows[global_idx], dtype=torch.float32)
        static_tensor = torch.tensor(self.static_feats[global_idx], dtype=torch.float32)

        # 3. Targets
        target_cls = torch.tensor(self.targets_cls[global_idx], dtype=torch.long)
        target_sev = torch.tensor(self.targets_sev[global_idx], dtype=torch.float32).unsqueeze(-1)
        target_fc3 = torch.tensor(self.targets_fc3[global_idx], dtype=torch.long)
        target_fc7 = torch.tensor(self.targets_fc7[global_idx], dtype=torch.long)
        target_qual = torch.tensor(self.targets_qual[global_idx], dtype=torch.float32).unsqueeze(-1)
        target_state = torch.tensor(self.targets_state[global_idx], dtype=torch.long)

        return {
            "image": img_tensor,
            "env_seq": env_tensor,
            "imu_window": imu_tensor,
            "static_feat": static_tensor,
            "target_cls": target_cls,
            "target_sev": target_sev,
            "target_fc3": target_fc3,
            "target_fc7": target_fc7,
            "target_qual": target_qual,
            "target_state": target_state,
            "grid_row": meta["grid_row"],
            "grid_col": meta["grid_col"],
            "zone_id": meta["zone_id"],
            "image_file": meta["image_file"],
        }


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that batches multimodal tensors and builds
    dynamic grid edge indices for the batch when zones are present.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    env_seqs = torch.stack([b["env_seq"] for b in batch], dim=0)
    imu_windows = torch.stack([b["imu_window"] for b in batch], dim=0)
    static_feats = torch.stack([b["static_feat"] for b in batch], dim=0)

    targets_cls = torch.stack([b["target_cls"] for b in batch], dim=0)
    targets_sev = torch.stack([b["target_sev"] for b in batch], dim=0)
    targets_fc3 = torch.stack([b["target_fc3"] for b in batch], dim=0)
    targets_fc7 = torch.stack([b["target_fc7"] for b in batch], dim=0)
    targets_qual = torch.stack([b["target_qual"] for b in batch], dim=0)
    targets_state = torch.stack([b["target_state"] for b in batch], dim=0)

    # Build k-NN or fully connected spatial edges within the batch if <= 16
    B = len(batch)
    rows = int(np.sqrt(B))
    cols = B // rows if rows > 0 else 1
    if rows * cols == B and rows > 1 and cols > 1:
        edge_index = build_grid_edge_index(rows, cols)
    else:
        # Full bidirectional adjacency among batch items
        edge_list = []
        for i in range(B):
            for j in range(B):
                if i != j:
                    edge_list.append((i, j))
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "images": images,
        "env_seqs": env_seqs,
        "imu_windows": imu_windows,
        "static_feats": static_feats,
        "targets_cls": targets_cls,
        "targets_sev": targets_sev,
        "targets_fc3": targets_fc3,
        "targets_fc7": targets_fc7,
        "targets_qual": targets_qual,
        "targets_state": targets_state,
        "edge_index": edge_index,
        "metadata": [{k: b[k] for k in ["zone_id", "grid_row", "grid_col", "image_file"]} for b in batch],
    }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Helper to instantiate Train and Validation DataLoaders."""
    train_ds = MultiModalCropDataset(data_dir=data_dir, split="train")
    val_ds = MultiModalCropDataset(data_dir=data_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=num_workers,
    )
    return train_loader, val_loader

