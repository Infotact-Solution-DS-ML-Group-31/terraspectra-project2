"""Demo model training script for TerraSpectra.

Trains a lightweight HybridSpectralSpatialViT or SpectralSpatial3DCNN on synthetic
hyperspectral datacubes, saves the checkpoint via save_model_checkpoint, and
evaluates segmentation metrics on a held-out synthetic cube.

Usage:
    python scripts/train_demo_model.py --model hybrid --epochs 3
    python scripts/train_demo_model.py --model cnn3d --epochs 3 --output metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.api.tiling import TilingConfig, predict_full_raster
from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.dataset import HyperspectralPatchDataset
from src.data.pca_reducer import SpectralPCAReducer
from src.models.checkpoint import save_model_checkpoint
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT
from src.models.metrics import evaluate_segmentation
from src.models.train import TrainConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for training the demo model."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a lightweight TerraSpectra demo model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["hybrid", "cnn3d"],
        default="hybrid",
        help="Neural model architecture to train ('hybrid' or 'cnn3d').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=15,
        help="Number of spectral PCA components.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training DataLoader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write evaluation metrics JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training synthetic cube generation.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=99,
        help="Random seed for held-out evaluation synthetic cube generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Blight classification probability threshold.",
    )
    return parser


def train_and_evaluate(
    model_type: str = "hybrid",
    epochs: int = 3,
    pca_components: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
    eval_seed: int = 99,
    threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Execute training pipeline and held-out evaluation.

    Returns:
        Dictionary of segmentation evaluation metrics on held-out cube.
    """
    # 1. Generate synthetic training cube (64x64, 224 bands)
    train_cube, train_mask = generate_synthetic_hyperspectral_cube(
        height=64,
        width=64,
        num_bands=224,
        seed=seed,
    )

    # 2. PCA dimensionality reduction
    reducer = SpectralPCAReducer(n_components=pca_components, random_state=seed)
    train_reduced = reducer.fit_transform(train_cube)

    # 3. Create dataset and DataLoader
    train_dataset = HyperspectralPatchDataset(
        cube=train_reduced,
        mask=train_mask,
        patch_size=16,
        stride=8,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # 4. Instantiate model
    if model_type == "hybrid":
        model = HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
        )
    elif model_type == "cnn3d":
        model = SpectralSpatial3DCNN(
            in_channels=1,
            num_classes=2,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    # 5. Train model
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )
    history = run_training(
        model=model,
        train_loader=train_loader,
        config=config,
        save_name=f"{model_type}_raw_best.pt",
    )

    # 6. Save checkpoint via save_model_checkpoint
    ckpt_dest = Path(checkpoint_dir) / f"{model_type}_demo.pt"
    metadata = {
        "model_type": model_type,
        "epochs": epochs,
        "pca_components": pca_components,
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else 0.0,
    }
    save_model_checkpoint(
        model=model,
        path=ckpt_dest,
        model_type=model_type,
        metadata=metadata,
    )

    # 7. Evaluate on held-out synthetic cube (seed=eval_seed)
    eval_cube, eval_mask = generate_synthetic_hyperspectral_cube(
        height=64,
        width=64,
        num_bands=224,
        seed=eval_seed,
    )
    eval_reduced = reducer.transform(eval_cube)
    tiling_cfg = TilingConfig(patch_size=16, stride=8, batch_size=8, device="cpu")
    prob_map = predict_full_raster(model, eval_reduced, config=tiling_cfg)

    pred_binary = (torch.from_numpy(prob_map) >= threshold).long()
    target_binary = torch.from_numpy(eval_mask).long()
    metrics = evaluate_segmentation(pred_binary, target_binary)

    # Convert metrics to clean Python floats
    metrics_summary = {
        "iou": float(metrics["iou"]),
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
    }

    if output_path:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for training demo model."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        metrics = train_and_evaluate(
            model_type=args.model,
            epochs=args.epochs,
            pca_components=args.pca_components,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            eval_seed=args.eval_seed,
            threshold=args.threshold,
            output_path=args.output,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    except Exception as exc:
        print(f"[error] Failed to train demo model: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
