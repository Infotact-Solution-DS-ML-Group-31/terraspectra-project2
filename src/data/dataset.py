"""PyTorch Dataset for spatial-spectral hyperspectral patch extraction.

Slices 3D spatial-spectral patches from full or dimensionality-reduced
hyperspectral datacubes along with corresponding ground-truth disease masks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cube_loader import HyperspectralCube


class HyperspectralPatchDataset(Dataset):
    """PyTorch Dataset for sampling 3D spatial-spectral patches.

    Extracts spatial-spectral patches of shape (1, C, patch_size, patch_size)
    along with corresponding 2D spatial segmentation masks of shape
    (patch_size, patch_size).

    Attributes:
        data: Spectral tensor of shape (C, H, W) as torch.float32.
        mask: Disease label mask of shape (H, W) as torch.long.
        patch_size: Spatial dimension (height and width) of extracted patches.
        stride: Step size between patch extraction origins.
        coordinates: List of (top_y, left_x) coordinates for each patch.
    """

    def __init__(
        self,
        cube: Union[HyperspectralCube, np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
        patch_size: int = 16,
        stride: Optional[int] = None,
    ) -> None:
        """Initialize HyperspectralPatchDataset.

        Args:
            cube: HyperspectralCube, numpy array, or torch tensor of shape (C, H, W).
            mask: Binary or multi-class segmentation mask of shape (H, W).
            patch_size: Spatial width and height of each patch (default: 16).
            stride: Stride between successive patches (default: patch_size for non-overlapping).
        """
        if patch_size <= 0:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size}")

        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be a positive integer, got {stride}")

        # Extract tensor data from cube
        if isinstance(cube, HyperspectralCube):
            raw_data = cube.data
        elif isinstance(cube, (np.ndarray, torch.Tensor)):
            raw_data = cube
        else:
            raise TypeError(
                f"Expected HyperspectralCube, np.ndarray, or torch.Tensor, got {type(cube)}"
            )

        # Convert to torch.Tensor
        if isinstance(raw_data, np.ndarray):
            self.data = torch.from_numpy(raw_data.copy()).to(dtype=torch.float32)
        else:
            self.data = raw_data.clone().detach().to(dtype=torch.float32)

        if isinstance(mask, np.ndarray):
            self.mask = torch.from_numpy(mask.copy()).to(dtype=torch.long)
        elif isinstance(mask, torch.Tensor):
            self.mask = mask.clone().detach().to(dtype=torch.long)
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor for mask, got {type(mask)}")

        if self.data.ndim != 3:
            raise ValueError(
                f"Spectral data must have shape (C, H, W), got shape {tuple(self.data.shape)}"
            )

        if self.mask.ndim != 2:
            raise ValueError(
                f"Mask must have shape (H, W), got shape {tuple(self.mask.shape)}"
            )

        _, height, width = self.data.shape
        if (height, width) != tuple(self.mask.shape):
            raise ValueError(
                f"Spatial dimensions mismatch: data has ({height}, {width}), "
                f"but mask has {tuple(self.mask.shape)}"
            )

        if patch_size > height or patch_size > width:
            raise ValueError(
                f"patch_size ({patch_size}) cannot exceed spatial dimensions ({height}, {width})"
            )

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        # Precompute valid patch top-left coordinates
        self.coordinates: List[Tuple[int, int]] = []
        for y in range(0, height - patch_size + 1, self.stride):
            for x in range(0, width - patch_size + 1, self.stride):
                self.coordinates.append((y, x))

        if not self.coordinates:
            raise ValueError(
                f"No valid patches could be extracted with patch_size={patch_size} and stride={self.stride} "
                f"from dimensions ({height}, {width})."
            )

    def __len__(self) -> int:
        """Return total number of extractable patches."""
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract a single 3D patch and corresponding mask patch.

        Args:
            idx: Patch index.

        Returns:
            Tuple of:
                - patch_tensor: Shape (1, C, patch_size, patch_size) as torch.float32.
                - mask_tensor: Shape (patch_size, patch_size) as torch.long.
        """
        y, x = self.coordinates[idx]
        patch = self.data[:, y : y + self.patch_size, x : x + self.patch_size]
        # Shape: (1, C, patch_size, patch_size)
        patch_tensor = patch.unsqueeze(0)

        mask_tensor = self.mask[y : y + self.patch_size, x : x + self.patch_size]
        return patch_tensor, mask_tensor

