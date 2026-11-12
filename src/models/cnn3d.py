"""Spectral-Spatial 3D Convolutional Neural Network for Hyperspectral Crop Disease Forecasting.

Implements an end-to-end 3D-2D CNN architecture for pixel-level pre-symptomatic
fungal blight detection from continuous hyperspectral data cubes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralSpatial3DCNN(nn.Module):
    """3D-2D Spectral-Spatial CNN for pixel-level crop disease segmentation.

    Processes 3D hyperspectral patches of shape:
        (batch_size, in_channels, spectral_depth, height, width)
    where in_channels is typically 1.

    Architecture:
        1. 3D Conv Layer 1: Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1)
           + BatchNorm3d(16) + GELU()
        2. 3D Conv Layer 2: Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
           + BatchNorm3d(32) + GELU()
        3. Spectral Depth Collapse: Adaptive spectral pooling collapsing spectral
           depth (dim=2) to produce 2D spatial feature maps (batch_size, 32, height, width).
        4. Classifier Head: Conv2d(32, 2, kernel_size=1) outputting raw logits
           for healthy (0) vs. fungal blight (1) per pixel (batch_size, 2, height, width).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        """Initialize SpectralSpatial3DCNN.

        Args:
            in_channels: Number of input channels (default: 1).
            num_classes: Number of target segmentation classes (default: 2, healthy vs blight).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Layer 1: Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1) + BatchNorm3d(16) + GELU()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
        )

        # Layer 2: Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1) + BatchNorm3d(32) + GELU()
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
        )

        # Classifier head: 1x1 2D Conv mapping 32 spatial features to raw class logits
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D-2D network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, spectral_depth, height, width).

        Returns:
            Raw logits tensor of shape (batch_size, num_classes, height, width).
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input tensor (B, C, D, H, W), got tensor of shape {tuple(x.shape)}"
            )

        # Layer 1: 3D spatial-spectral convolution -> (B, 16, D, H, W)
        out = self.layer1(x)

        # Layer 2: 3D spatial-spectral convolution -> (B, 32, D, H, W)
        out = self.layer2(out)

        # Collapse spectral depth via adaptive pooling -> (B, 32, H, W)
        spatial_features = torch.mean(out, dim=2)

        # Pixel-level classification head -> (B, num_classes, H, W)
        logits = self.classifier(spatial_features)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-level disease probability map for fungal blight (class 1).

        Args:
            x: Input tensor of shape (batch_size, in_channels, spectral_depth, height, width).

        Returns:
            Disease probability map of shape (batch_size, height, width) with values in [0.0, 1.0].
        """
        logits = self.forward(x)
        # Compute softmax probabilities across classes (dim=1)
        probs = torch.softmax(logits, dim=1)
        # Class index 1 corresponds to fungal blight
        disease_prob = probs[:, 1, :, :]
        return disease_prob

