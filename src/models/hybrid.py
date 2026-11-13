"""Hybrid Spectral-Spatial Vision Transformer for Crop Disease Forecasting.

Combines a 3D-CNN local spectral-spatial feature extractor with a global
Spatial Vision Transformer branch for high-fidelity pixel-level segmentation
and pre-symptomatic blight forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.vit import SpatialVisionTransformer


class HybridSpectralSpatialViT(nn.Module):
    """Hybrid 3D-CNN + ViT architecture for hyperspectral crop disease segmentation.

    Pipeline:
        1. Local 3D-CNN Backbone: Extracts fine-grained spectral-spatial features
           and collapses spectral depth via adaptive pooling -> (B, 32, H, W).
        2. Global ViT Branch: Tokenizes local 2D feature maps and models long-range
           spatial dependencies via multi-head self-attention -> (B, embed_dim, H, W).
        3. Feature Fusion: Fuses local (CNN) and global (ViT) feature maps via
           concatenation or residual connection + 2D convolution projection.
        4. Segmentation Head: 1x1 2D convolution outputting raw logits
           (B, num_classes, H, W) for healthy (0) vs. fungal blight (1).
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_features: int = 32,
        vit_embed_dim: int = 64,
        patch_size: int = 4,
        num_classes: int = 2,
        num_vit_layers: int = 2,
        num_vit_heads: int = 4,
        fusion_mode: str = "concat",
        upsample_mode: str = "conv_transpose",
        dropout: float = 0.0,
    ) -> None:
        """Initialize HybridSpectralSpatialViT.

        Args:
            in_channels: Spectral input channels (default: 1).
            cnn_features: Number of spatial feature maps produced by 3D-CNN (default: 32).
            vit_embed_dim: ViT embedding dimension (default: 64).
            patch_size: ViT token patch size (default: 4).
            num_classes: Target segmentation classes (default: 2, healthy vs blight).
            num_vit_layers: Number of ViT encoder layers (default: 2).
            num_vit_heads: Number of attention heads (default: 4).
            fusion_mode: Feature fusion strategy ('concat' or 'residual', default: 'concat').
            upsample_mode: Spatial upsampling method for ViT ('conv_transpose' or 'bilinear').
            dropout: ViT dropout rate (default: 0.0).
        """
        super().__init__()

        if fusion_mode not in ("concat", "residual"):
            raise ValueError(
                f"fusion_mode must be 'concat' or 'residual', got '{fusion_mode}'"
            )

        self.in_channels = in_channels
        self.cnn_features = cnn_features
        self.vit_embed_dim = vit_embed_dim
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode

        # 1. Local 3D-CNN backbone
        self.cnn = SpectralSpatial3DCNN(in_channels=in_channels, num_classes=num_classes)

        # 2. Global Vision Transformer branch
        self.vit = SpatialVisionTransformer(
            in_channels=cnn_features,
            embed_dim=vit_embed_dim,
            patch_size=patch_size,
            num_layers=num_vit_layers,
            num_heads=num_vit_heads,
            dropout=dropout,
            upsample_mode=upsample_mode,
        )

        # 3. Feature fusion module
        if fusion_mode == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Conv2d(
                    cnn_features + vit_embed_dim,
                    vit_embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(vit_embed_dim),
                nn.GELU(),
            )
        else:
            self.residual_proj = (
                nn.Conv2d(cnn_features, vit_embed_dim, kernel_size=1)
                if cnn_features != vit_embed_dim
                else nn.Identity()
            )
            self.fusion_proj = nn.Sequential(
                nn.Conv2d(vit_embed_dim, vit_embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(vit_embed_dim),
                nn.GELU(),
            )

        # 4. Final pixel-level classification head
        self.classifier = nn.Conv2d(vit_embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input 5D hyperspectral tensor of shape
               (batch_size, in_channels, spectral_depth, height, width).

        Returns:
            Raw logits tensor of shape (batch_size, num_classes, height, width).
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input tensor (batch_size, in_channels, spectral_depth, height, width), "
                f"got shape {tuple(x.shape)}"
            )

        batch_size, channels, depth, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Channel mismatch: expected in_channels={self.in_channels}, "
                f"got {channels} (shape: {tuple(x.shape)})"
            )

        # 1. Local spectral-spatial feature extraction via 3D-CNN -> (B, 32, H, W)
        local_features = self.cnn.extract_features(x)

        # 2. Global spatial contextualization via ViT -> (B, vit_embed_dim, H, W)
        global_features = self.vit(local_features)

        # 3. Fusion of local and global representations
        if self.fusion_mode == "concat":
            fused = torch.cat([local_features, global_features], dim=1)
            fused = self.fusion_proj(fused)
        else:
            fused = self.residual_proj(local_features) + global_features
            fused = self.fusion_proj(fused)

        # 4. Final classification head -> (B, num_classes, H, W)
        logits = self.classifier(fused)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-level disease probability map for fungal blight (class 1).

        Args:
            x: Input 5D hyperspectral tensor of shape
               (batch_size, in_channels, spectral_depth, height, width).

        Returns:
            Disease probability map of shape (batch_size, height, width) with values in [0.0, 1.0].
        """
        logits = self.forward(x)
        # Compute softmax probabilities along class dimension (dim=1)
        probs = torch.softmax(logits, dim=1)
        # Fungal blight corresponds to class index 1
        return probs[:, 1, :, :]
