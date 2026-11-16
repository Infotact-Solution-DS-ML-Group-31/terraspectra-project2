"""Tiled spatial inference pipeline for full-field hyperspectral rasters.

Provides overlapping patch slicing, reflection/edge boundary padding, batched
neural inference, and seamless blend stitching for arbitrary raster extents.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.data.cube_loader import HyperspectralCube


@dataclass
class TilingConfig:
    """Configuration hyperparameters for overlapping spatial raster inference.

    Attributes:
        patch_size: Square tile dimension in pixels (default: 16).
        stride: Step size between adjacent tiles in pixels (default: 8).
        batch_size: Number of tiles processed per inference forward pass (default: 8).
        device: Compute execution device ('cpu' or 'cuda', default: 'cpu').
    """

    patch_size: int = 16
    stride: int = 8
    batch_size: int = 8
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate tiling parameters."""
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.stride <= 0 or self.stride > self.patch_size:
            raise ValueError(
                f"stride must satisfy 0 < stride <= patch_size, "
                f"got stride={self.stride} with patch_size={self.patch_size}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


def _compute_target_dim(dim: int, patch_size: int, stride: int) -> int:
    """Compute the minimum padded dimension required to cleanly cover all pixels."""
    if dim <= patch_size:
        return patch_size
    num_steps = math.ceil((dim - patch_size) / stride)
    return num_steps * stride + patch_size


def predict_full_raster(
    model: nn.Module,
    cube: Union[HyperspectralCube, np.ndarray, torch.Tensor],
    config: Optional[TilingConfig] = None,
) -> np.ndarray:
    """Generate full-extent pixel-level disease probability map via overlapping tiling.

    Pads spatial boundaries with reflection/edge padding to accommodate arbitrary
    or sub-patch dimensions, slices overlapping tiles, performs batched inference,
    averages overlapping predictions, and crops back to the exact original shape.

    Args:
        model: Trained segmentation model implementing `predict_probability(...)`.
        cube: HyperspectralCube or 3D tensor/array of shape (num_channels, height, width).
        config: Tiling configuration settings. Defaults to TilingConfig() if None.

    Returns:
        2D numpy array of shape (height, width) with float32 disease probabilities in [0.0, 1.0].

    Raises:
        TypeError: If input cube or model type is invalid.
        ValueError: If input tensor rank is not 3D or dimensions are non-positive.
    """
    cfg = config or TilingConfig()

    # 1. Extract data array
    if isinstance(cube, HyperspectralCube):
        data = cube.data
    elif isinstance(cube, torch.Tensor):
        data = cube.detach().cpu().numpy()
    elif isinstance(cube, np.ndarray):
        data = cube
    else:
        raise TypeError(
            f"Expected HyperspectralCube, np.ndarray, or torch.Tensor, got {type(cube)}"
        )

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D raster array of shape (C, H, W), got {data.ndim}D tensor of shape {data.shape}"
        )

    channels, orig_h, orig_w = data.shape
    if channels <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"All raster dimensions must be strictly positive, got {data.shape}")

    # 2. Determine target padded dimensions and compute padding
    target_h = _compute_target_dim(orig_h, cfg.patch_size, cfg.stride)
    target_w = _compute_target_dim(orig_w, cfg.patch_size, cfg.stride)

    pad_bottom = target_h - orig_h
    pad_right = target_w - orig_w

    if pad_bottom > 0 or pad_right > 0:
        # Use reflection padding when feasible, otherwise fall back to edge replication
        try:
            padded_data = np.pad(
                data,
                ((0, 0), (0, pad_bottom), (0, pad_right)),
                mode="reflect",
            )
        except ValueError:
            padded_data = np.pad(
                data,
                ((0, 0), (0, pad_bottom), (0, pad_right)),
                mode="edge",
            )
    else:
        padded_data = data

    # 3. Generate patch coordinates
    patch_coords = []
    for y in range(0, target_h - cfg.patch_size + 1, cfg.stride):
        for x in range(0, target_w - cfg.patch_size + 1, cfg.stride):
            patch_coords.append((y, x))

    # 4. Prepare model and accumulators
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    prob_accumulator = np.zeros((target_h, target_w), dtype=np.float32)
    count_accumulator = np.zeros((target_h, target_w), dtype=np.float32)

    # 5. Batched inference loop
    with torch.no_grad():
        for i in range(0, len(patch_coords), cfg.batch_size):
            batch_slice_coords = patch_coords[i : i + cfg.batch_size]

            # Extract patches: shape (B, C, patch_size, patch_size)
            patches = [
                padded_data[:, y : y + cfg.patch_size, x : x + cfg.patch_size]
                for y, x in batch_slice_coords
            ]
            batch_np = np.stack(patches, axis=0).astype(np.float32)

            # Model expects 5D tensor: (B, in_channels=1, C, patch_size, patch_size)
            batch_tensor = torch.from_numpy(batch_np).unsqueeze(1).to(device)

            batch_probs = model.predict_probability(batch_tensor)

            if hasattr(batch_probs, "cpu"):
                batch_probs_np = batch_probs.cpu().numpy()
            else:
                batch_probs_np = np.asarray(batch_probs)

            # Accumulate probabilities
            for (y, x), patch_prob in zip(batch_slice_coords, batch_probs_np):
                prob_accumulator[y : y + cfg.patch_size, x : x + cfg.patch_size] += patch_prob
                count_accumulator[y : y + cfg.patch_size, x : x + cfg.patch_size] += 1.0

    # 6. Normalize overlapping predictions and crop back to original extent
    avg_prob_map = prob_accumulator / np.maximum(count_accumulator, 1e-8)
    final_prob_map = avg_prob_map[:orig_h, :orig_w]
    return np.clip(final_prob_map, 0.0, 1.0).astype(np.float32)
