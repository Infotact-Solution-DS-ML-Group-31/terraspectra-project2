"""Segmentation evaluation metrics for binary hyperspectral crop disease detection.

Provides standard computer vision and medical imaging segmentation metrics:
Intersection over Union (IoU / Jaccard index), Dice similarity coefficient (F1 score),
Precision, Recall, and overall Pixel Accuracy.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union
import torch


def compute_confusion_counts(
    preds: torch.Tensor, targets: torch.Tensor
) -> Tuple[int, int, int, int]:
    """Compute True Positives, False Positives, True Negatives, and False Negatives.

    Accepts raw logits (B, 2, H, W) or binary predictions (B, H, W) against
    ground truth binary masks (B, H, W) for disease class (1).

    Args:
        preds: Tensor of raw class logits (B, 2, H, W) or predicted classes (B, H, W) / (H, W).
        targets: Ground truth binary target masks (B, H, W) or (H, W).

    Returns:
        Tuple of (tp, fp, tn, fn) as integers.

    Raises:
        ValueError: If tensor ranks or spatial dimensions do not match.
    """
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"preds must be a torch.Tensor, got {type(preds)}")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets must be a torch.Tensor, got {type(targets)}")

    # Standardize predictions to (B, H, W)
    if preds.ndim == 4:
        if preds.shape[1] == 2:
            preds_cls = torch.argmax(preds, dim=1)
        elif preds.shape[1] == 1:
            preds_cls = (preds.squeeze(1) > 0.5).long()
        else:
            raise ValueError(
                f"Expected 4D logits tensor to have 1 or 2 channels, got shape {tuple(preds.shape)}"
            )
    elif preds.ndim == 3:
        preds_cls = (preds > 0).long()
    elif preds.ndim == 2:
        preds_cls = (preds > 0).long().unsqueeze(0)
    else:
        raise ValueError(
            f"Expected preds to have 2, 3, or 4 dimensions, got shape {tuple(preds.shape)}"
        )

    # Standardize targets to (B, H, W)
    if targets.ndim == 2:
        targets_cls = (targets > 0).long().unsqueeze(0)
    elif targets.ndim == 3:
        targets_cls = (targets > 0).long()
    elif targets.ndim == 4 and targets.shape[1] == 1:
        targets_cls = (targets.squeeze(1) > 0).long()
    else:
        raise ValueError(
            f"Expected targets to have 2 or 3 dimensions (or shape (B, 1, H, W)), "
            f"got shape {tuple(targets.shape)}"
        )

    # Validate shape compatibility
    if preds_cls.shape != targets_cls.shape:
        raise ValueError(
            f"Shape mismatch: resolved predictions have shape {tuple(preds_cls.shape)}, "
            f"but targets have shape {tuple(targets_cls.shape)}"
        )

    tp = int(torch.sum((preds_cls == 1) & (targets_cls == 1)).item())
    fp = int(torch.sum((preds_cls == 1) & (targets_cls == 0)).item())
    tn = int(torch.sum((preds_cls == 0) & (targets_cls == 0)).item())
    fn = int(torch.sum((preds_cls == 0) & (targets_cls == 1)).item())

    return tp, fp, tn, fn


def iou_score(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7
) -> float:
    """Compute Intersection over Union (IoU / Jaccard Index) for the disease class (1).

    Args:
        preds: Raw logits (B, 2, H, W) or predicted classes (B, H, W).
        targets: Ground truth binary masks (B, H, W).
        eps: Small epsilon constant for numerical stability.

    Returns:
        IoU score as a Python float in [0.0, 1.0].
    """
    tp, fp, tn, fn = compute_confusion_counts(preds, targets)
    if fp == 0 and fn == 0:
        return 1.0
    union = tp + fp + fn
    if union == 0:
        return 1.0
    return float(tp / (union + eps))


def dice_score(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7
) -> float:
    """Compute Dice Similarity Coefficient (F1 score) for the disease class (1).

    Args:
        preds: Raw logits (B, 2, H, W) or predicted classes (B, H, W).
        targets: Ground truth binary masks (B, H, W).
        eps: Small epsilon constant for numerical stability.

    Returns:
        Dice score as a Python float in [0.0, 1.0].
    """
    tp, fp, tn, fn = compute_confusion_counts(preds, targets)
    if fp == 0 and fn == 0:
        return 1.0
    denom = 2.0 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2.0 * tp) / (denom + eps))


def evaluate_segmentation(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7
) -> Dict[str, float]:
    """Compute comprehensive segmentation evaluation metrics for crop disease detection.

    Args:
        logits: Raw class logits of shape (B, 2, H, W) or predicted classes (B, H, W).
        targets: Ground truth binary masks of shape (B, H, W).
        eps: Small epsilon constant for numerical stability.

    Returns:
        Dictionary mapping metric names ('iou', 'dice', 'precision', 'recall',
        'pixel_accuracy') to Python floats.
    """
    tp, fp, tn, fn = compute_confusion_counts(logits, targets)

    if fp == 0 and fn == 0:
        iou = 1.0
        dice = 1.0
        precision = 1.0
        recall = 1.0
    else:
        union = tp + fp + fn
        iou = float(tp / (union + eps)) if union > 0 else 1.0
        denom = 2.0 * tp + fp + fn
        dice = float((2.0 * tp) / (denom + eps)) if denom > 0 else 1.0
        precision = float(tp / (tp + fp + eps)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn + eps)) if (tp + fn) > 0 else 0.0

    total_pixels = tp + fp + tn + fn
    pixel_accuracy = float((tp + tn) / total_pixels) if total_pixels > 0 else 1.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
    }
