"""
AgriSense-MM: Complete Training Script.
Features:
- AdamW optimizer with differential learning rates (1e-5 for backbone, 1e-4 for fusion/heads)
- Cosine Annealing learning rate schedule
- Early stopping & best model checkpoint saving
- Multi-task validation evaluation (Accuracy, Macro-F1, Severity MAE)
"""

import os
import argparse
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from typing import Dict, Optional, Tuple, Any, List

from agrisense_mm.models.agrisense_net import AgriSenseMMNet
from agrisense_mm.training.dataset import create_dataloaders
from agrisense_mm.training.losses import AgriSenseMultiTaskLoss
from agrisense_mm.data.generator import create_full_paired_dataset


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: AgriSenseMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Executes one training epoch across the dataset."""
    model.train()
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []

    for batch in dataloader:
        images = batch["images"].to(device)
        env_seqs = batch["env_seqs"].to(device)
        imu_windows = batch["imu_windows"].to(device)
        static_feats = batch["static_feats"].to(device)
        edge_index = batch["edge_index"].to(device)

        targets = {
            "targets_cls": batch["targets_cls"].to(device),
            "targets_sev": batch["targets_sev"].to(device),
            "targets_fc3": batch["targets_fc3"].to(device),
            "targets_fc7": batch["targets_fc7"].to(device),
            "targets_qual": batch["targets_qual"].to(device),
            "targets_state": batch["targets_state"].to(device),
        }

        optimizer.zero_grad()
        predictions = model(images, env_seqs, imu_windows, static_feats, edge_index)

        loss, loss_dict = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(predictions["cls_logits"], dim=-1).detach().cpu().numpy()
        all_preds_cls.extend(preds)
        all_targets_cls.extend(targets["targets_cls"].detach().cpu().numpy())

    avg_loss = running_loss / max(1, len(dataloader))
    train_acc = float(accuracy_score(all_targets_cls, all_preds_cls))
    train_f1 = float(f1_score(all_targets_cls, all_preds_cls, average="macro", zero_division=0))

    return {
        "train_loss": avg_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }


def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: AgriSenseMultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validates the model on unseen field data."""
    model.eval()
    running_loss = 0.0
    all_preds_cls = []
    all_targets_cls = []
    all_preds_sev = []
    all_targets_sev = []
    all_preds_fc3 = []
    all_targets_fc3 = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            env_seqs = batch["env_seqs"].to(device)
            imu_windows = batch["imu_windows"].to(device)
            static_feats = batch["static_feats"].to(device)
            edge_index = batch["edge_index"].to(device)

            targets = {
                "targets_cls": batch["targets_cls"].to(device),
                "targets_sev": batch["targets_sev"].to(device),
                "targets_fc3": batch["targets_fc3"].to(device),
                "targets_fc7": batch["targets_fc7"].to(device),
                "targets_qual": batch["targets_qual"].to(device),
                "targets_state": batch["targets_state"].to(device),
            }

            predictions = model(images, env_seqs, imu_windows, static_feats, edge_index)
            loss, _ = criterion(predictions, targets)
            running_loss += loss.item()

            preds_cls = torch.argmax(predictions["cls_logits"], dim=-1).cpu().numpy()
            all_preds_cls.extend(preds_cls)
            all_targets_cls.extend(targets["targets_cls"].cpu().numpy())

            all_preds_sev.extend(predictions["severity"].cpu().numpy().flatten())
            all_targets_sev.extend(targets["targets_sev"].cpu().numpy().flatten())

            preds_fc3 = torch.argmax(predictions["forecast_3d_logits"], dim=-1).cpu().numpy()
            all_preds_fc3.extend(preds_fc3)
            all_targets_fc3.extend(targets["targets_fc3"].cpu().numpy())

    avg_loss = running_loss / max(1, len(dataloader))
    val_acc = float(accuracy_score(all_targets_cls, all_preds_cls))
    val_f1 = float(f1_score(all_targets_cls, all_preds_cls, average="macro", zero_division=0))
    val_sev_mae = float(mean_absolute_error(all_targets_sev, all_preds_sev))
    val_fc3_acc = float(accuracy_score(all_targets_fc3, all_preds_fc3))

    return {
        "val_loss": avg_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_sev_mae": val_sev_mae,
        "val_fc3_acc": val_fc3_acc,
    }


def train_agrisense_model(
    data_dir: str,
    output_dir: str = "checkpoints",
    epochs: int = 15,
    batch_size: int = 16,
    lr_base: float = 1e-4,
    lr_backbone: float = 1e-5,
    enable_env: bool = True,
    enable_motion: bool = True,
    enable_gnn: bool = True,
    device_str: Optional[str] = None,
) -> Tuple[AgriSenseMMNet, Dict[str, Any]]:
    """
    Main training function orchestrating dataset loading, optimizer setup,
    epoch loop, evaluation, and saving best checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Device selection: MPS (Apple Silicon), CUDA, or CPU
    if device_str:
        device = torch.device(device_str)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"--> Training AgriSense-MM on Device: {device}")

    # Verify or generate dataset
    if not os.path.exists(os.path.join(data_dir, "metadata.json")):
        print(f"Data not found in {data_dir}. Generating synthetic field dataset...")
        create_full_paired_dataset(data_dir, num_samples=160)

    train_loader, val_loader = create_dataloaders(data_dir, batch_size=batch_size)

    # Initialize model
    model = AgriSenseMMNet(
        enable_env=enable_env,
        enable_motion=enable_motion,
        enable_gnn=enable_gnn,
    ).to(device)

    # Differential learning rates: smaller for pretrained visual backbone
    backbone_params = list(model.visual_branch.features.parameters())
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("visual_branch.features")
    ]

    optimizer = AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_base, "weight_decay": 1e-3},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = AgriSenseMultiTaskLoss(use_focal=True)

    best_val_f1 = -1.0
    history = []

    checkpoint_path = os.path.join(output_dir, "best_agrisense_model.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_metrics['train_loss']:.3f} | Train F1: {train_metrics['train_f1']:.3f} | "
            f"Val Loss: {val_metrics['val_loss']:.3f} | Val Acc: {val_metrics['val_acc']:.3f} | "
            f"Val F1: {val_metrics['val_f1']:.3f} | Sev MAE: {val_metrics['val_sev_mae']:.1f}"
        )

        # Save checkpoint if best macro-F1
        if val_metrics["val_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": val_metrics,
                "config": {
                    "enable_env": enable_env,
                    "enable_motion": enable_motion,
                    "enable_gnn": enable_gnn,
                }
            }, checkpoint_path)
            print(f"  [+] Saved new best model checkpoint to {checkpoint_path} (F1: {best_val_f1:.4f})")

    # Load best checkpoint weights
    if os.path.exists(checkpoint_path):
        chk = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        print(f"Training complete. Reloaded best weights from epoch {chk['epoch']} with Val F1: {chk['val_f1']:.4f}")

    return model, {"history": history, "best_val_f1": best_val_f1, "checkpoint": checkpoint_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AgriSense-MM")
    parser.add_argument("--data_dir", type=str, default="agrisense_mm/data/synthetic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_agrisense_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
