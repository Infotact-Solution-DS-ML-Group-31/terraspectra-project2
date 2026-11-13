"""Spatial Vision Transformer (ViT) for Hyperspectral Feature Maps.

Captures long-range spatial context and field-level dependencies from 2D
feature maps extracted by local 3D convolutional spectral backbones.
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialVisionTransformer(nn.Module):
    """Spatial Vision Transformer operating on 2D feature representations.

    Ingests spatial feature maps of shape:
        (batch_size, in_channels, height, width)
    such as those output by SpectralSpatial3DCNN (in_channels=32).

    Architecture:
        1. Non-overlapping patch projection (kernel_size=patch_size, stride=patch_size)
           mapping patches to embed_dim.
        2. Addition of learnable 2D positional embeddings (with bicubic interpolation
           fallback for arbitrary spatial resolutions).
        3. Multi-layer Transformer Encoder with Multi-Head Self-Attention (MHSA).
        4. Spatial reconstruction back to (batch_size, embed_dim, height, width)
           via transposed convolution or bilinear interpolation.
    """

    def __init__(
        self,
        in_channels: int = 32,
        embed_dim: int = 64,
        patch_size: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        base_img_size: int = 16,
        upsample_mode: str = "conv_transpose",
    ) -> None:
        """Initialize SpatialVisionTransformer.

        Args:
            in_channels: Number of channels in the incoming 2D feature map (default: 32).
            embed_dim: Transformer embedding dimension (default: 64).
            patch_size: Spatial patch size for tokenization (default: 4).
            num_layers: Number of Transformer encoder layers (default: 2).
            num_heads: Number of attention heads (default: 4).
            mlp_ratio: Feedforward dimension expansion factor (default: 4.0).
            dropout: Dropout probability (default: 0.0).
            base_img_size: Canonical image size for base positional embeddings (default: 16).
            upsample_mode: Method for expanding patch tokens back to (H, W).
                           Options: 'conv_transpose' (default) or 'bilinear'.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        if upsample_mode not in ("conv_transpose", "bilinear"):
            raise ValueError(
                f"upsample_mode must be 'conv_transpose' or 'bilinear', got '{upsample_mode}'"
            )

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.upsample_mode = upsample_mode

        # Base patch grid dimensions for positional embeddings
        self.base_h_p = max(1, base_img_size // patch_size)
        self.base_w_p = max(1, base_img_size // patch_size)
        num_base_patches = self.base_h_p * self.base_w_p

        # 1. Patch projection
        self.patch_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_base_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Spatial reconstruction head
        if upsample_mode == "conv_transpose":
            self.reconstruct = nn.Sequential(
                nn.ConvTranspose2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        else:
            self.reconstruct = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )

    def _get_pos_embedding(self, h_p: int, w_p: int) -> torch.Tensor:
        """Retrieve or interpolate positional embeddings for target patch grid (h_p, w_p)."""
        num_patches = h_p * w_p
        if num_patches == self.pos_embedding.shape[1] and h_p == self.base_h_p and w_p == self.base_w_p:
            return self.pos_embedding

        # Reshape to (1, embed_dim, base_h_p, base_w_p) for 2D spatial interpolation
        pos_2d = self.pos_embedding.reshape(
            1, self.base_h_p, self.base_w_p, self.embed_dim
        ).permute(0, 3, 1, 2)

        # Interpolate to match target patch grid
        interpolated = F.interpolate(
            pos_2d,
            size=(h_p, w_p),
            mode="bicubic",
            align_corners=False,
        )

        # Permute back to (1, num_patches, embed_dim)
        pos_res = interpolated.permute(0, 2, 3, 1).reshape(1, num_patches, self.embed_dim)
        return pos_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: 2D feature map of shape (batch_size, in_channels, height, width).

        Returns:
            Spatially contextualized feature map of shape (batch_size, embed_dim, height, width).
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor of shape (batch_size, in_channels, height, width), "
                f"got tensor of shape {tuple(x.shape)}"
            )

        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected in_channels={self.in_channels}, "
                f"got {channels} channels (tensor shape: {tuple(x.shape)})"
            )

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Spatial dimensions ({height}, {width}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

        # 1. Patch projection -> (B, embed_dim, H_p, W_p)
        patches = self.patch_proj(x)
        h_p, w_p = patches.shape[2], patches.shape[3]

        # Flatten spatial dimensions to token sequence -> (B, N_patches, embed_dim)
        tokens = patches.flatten(2).transpose(1, 2)

        # 2. Add learnable positional embeddings
        pos_embed = self._get_pos_embedding(h_p, w_p)
        tokens = self.pos_drop(tokens + pos_embed)

        # 3. Transformer Encoder
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # 4. Reshape tokens back to 2D patch grid -> (B, embed_dim, H_p, W_p)
        tokens_grid = tokens.transpose(1, 2).reshape(batch_size, self.embed_dim, h_p, w_p)

        # 5. Reconstruct back to original spatial dimensions (H, W)
        if self.upsample_mode == "conv_transpose":
            out = self.reconstruct(tokens_grid)
        else:
            upsampled = F.interpolate(
                tokens_grid,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            out = self.reconstruct(upsampled)

        return out
