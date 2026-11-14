"""Lightweight, CPU-friendly training and evaluation pipeline for TerraSpectra.

Provides configurable training loops, validation routines, checkpoint persistence,
and metric logging using PyTorch and AdamW.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.metrics import evaluate_segmentation


@dataclass
class TrainConfig:
    """Hyperparameters and configuration settings for model training.

    Attributes:
        epochs: Number of complete passes over the training dataset (default: 5).
        batch_size: Number of spatial-spectral patches per batch (default: 4).
        learning_rate: Optimizer learning rate (default: 1e-3).
        weight_decay: L2 regularization penalty factor (default: 1e-4).
        device: Target execution device ('cpu' or 'cuda', default: 'cpu').
        checkpoint_dir: Directory path where model weights are stored (default: 'checkpoints').
    """

    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
) -> float:
    """Run a single training epoch across all batches in the dataloader.

    Args:
        model: PyTorch segmentation neural network.
        dataloader: DataLoader delivering (patch_tensor, mask_tensor) batches.
        optimizer: PyTorch optimizer (e.g., AdamW).
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Compute device ('cpu' or 'cuda').

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    target_device = torch.device(device)

    for patches, masks in dataloader:
        patches = patches.to(target_device)
        masks = masks.to(target_device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return float(total_loss / max(num_batches, 1))


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model loss and segmentation metrics across an evaluation dataset.

    Args:
        model: PyTorch segmentation neural network.
        dataloader: DataLoader delivering evaluation batches.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (average_loss, metrics_dictionary).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_logits = []
    all_targets = []

    target_device = torch.device(device)

    with torch.no_grad():
        for patches, masks in dataloader:
            patches = patches.to(target_device)
            masks = masks.to(target_device)

            logits = model(patches)
            loss = criterion(logits, masks)

            total_loss += float(loss.item())
            num_batches += 1

            all_logits.append(logits.detach().cpu())
            all_targets.append(masks.detach().cpu())

    avg_loss = float(total_loss / max(num_batches, 1))

    if all_logits and all_targets:
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        metrics = evaluate_segmentation(cat_logits, cat_targets)
    else:
        metrics = {
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pixel_accuracy": 0.0,
        }

    return avg_loss, metrics


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainConfig] = None,
    save_name: str = "best_model.pt",
) -> Dict[str, Any]:
    """Execute end-to-end model training, validation, metric logging, and checkpointing.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader supplying training patches.
        val_loader: Optional DataLoader supplying validation patches.
        config: Training configuration dataclass. Defaults to TrainConfig() if None.
        save_name: Filename under checkpoint_dir for saving the best model weights.

    Returns:
        Dictionary containing training history:
            - 'train_loss': List[float] per epoch
            - 'val_loss': List[float] per epoch (if val_loader provided)
            - 'val_metrics': List[Dict[str, float]] per epoch (if val_loader provided)
    """
    cfg = config or TrainConfig()
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, save_name)

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
    }

    best_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    for _ in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                torch.save(best_state_dict, checkpoint_path)
        else:
            if train_loss < best_loss:
                best_loss = train_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                torch.save(best_state_dict, checkpoint_path)

    # Ensure checkpoint file exists on disk
    if not os.path.exists(checkpoint_path):
        torch.save(best_state_dict, checkpoint_path)

    return history
