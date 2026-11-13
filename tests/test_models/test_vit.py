"""Unit tests for SpatialVisionTransformer and HybridSpectralSpatialViT."""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.pca_reducer import SpectralPCAReducer
from src.data.dataset import HyperspectralPatchDataset
from src.models.vit import SpatialVisionTransformer
from src.models.hybrid import HybridSpectralSpatialViT


class TestSpatialVisionTransformer(unittest.TestCase):
    """Test suite for SpatialVisionTransformer."""

    def test_spatial_vit_forward_shape(self) -> None:
        """Verifies forward pass output shape for various spatial configurations and upsample modes."""
        batch_size = 2
        in_channels = 32
        embed_dim = 64
        patch_size = 4
        height = 16
        width = 16

        # 1. Default transposed convolution upsampler
        vit = SpatialVisionTransformer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=2,
            num_heads=4,
            upsample_mode="conv_transpose",
        )
        vit.eval()

        x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32)
        out = vit(x)

        self.assertEqual(out.shape, (batch_size, embed_dim, height, width))
        self.assertEqual(out.dtype, torch.float32)

        # 2. Bilinear interpolation upsampler
        vit_bilinear = SpatialVisionTransformer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=2,
            num_heads=4,
            upsample_mode="bilinear",
        )
        vit_bilinear.eval()
        out_bilinear = vit_bilinear(x)
        self.assertEqual(out_bilinear.shape, (batch_size, embed_dim, height, width))

        # 3. Dynamic positional embedding interpolation on larger resolution (32x32)
        x_32 = torch.randn(1, in_channels, 32, 32, dtype=torch.float32)
        out_32 = vit(x_32)
        self.assertEqual(out_32.shape, (1, embed_dim, 32, 32))

    def test_spatial_vit_validation_errors(self) -> None:
        """Verifies dimension validation and error reporting."""
        vit = SpatialVisionTransformer(in_channels=32, embed_dim=64, patch_size=4)

        # Non-4D input
        with self.assertRaises(ValueError):
            vit(torch.randn(2, 32, 16))

        with self.assertRaises(ValueError):
            vit(torch.randn(2, 1, 32, 16, 16))

        # Channel mismatch
        with self.assertRaises(ValueError):
            vit(torch.randn(2, 16, 16, 16))  # 16 instead of 32

        # Non-divisible spatial dimensions
        with self.assertRaises(ValueError):
            vit(torch.randn(2, 32, 15, 15))  # 15 not divisible by patch_size 4

        # Invalid init parameters
        with self.assertRaises(ValueError):
            SpatialVisionTransformer(embed_dim=65, num_heads=4)  # 65 not divisible by 4

        with self.assertRaises(ValueError):
            SpatialVisionTransformer(upsample_mode="invalid_mode")


class TestHybridSpectralSpatialViT(unittest.TestCase):
    """Test suite for HybridSpectralSpatialViT."""

    def test_hybrid_vit_forward_logits_and_probabilities(self) -> None:
        """Tests forward logits (2, 2, 16, 16) and predict_probability bounds [0.0, 1.0]."""
        batch_size = 2
        in_channels = 1
        spectral_depth = 15
        height = 16
        width = 16

        # 1. Concat fusion mode
        model = HybridSpectralSpatialViT(
            in_channels=in_channels,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
            num_vit_layers=2,
            num_vit_heads=4,
            fusion_mode="concat",
        )
        model.eval()

        x = torch.randn(batch_size, in_channels, spectral_depth, height, width, dtype=torch.float32)

        logits = model(x)
        self.assertEqual(logits.shape, (batch_size, 2, height, width))
        self.assertEqual(logits.dtype, torch.float32)

        probs = model.predict_probability(x)
        self.assertEqual(probs.shape, (batch_size, height, width))
        self.assertEqual(probs.dtype, torch.float32)
        self.assertTrue(torch.all(probs >= 0.0), "Probabilities must be >= 0.0")
        self.assertTrue(torch.all(probs <= 1.0), "Probabilities must be <= 1.0")

        # Softmax consistency check
        expected_probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        self.assertTrue(torch.allclose(probs, expected_probs, atol=1e-6))

        # 2. Residual fusion mode
        model_residual = HybridSpectralSpatialViT(
            in_channels=in_channels,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
            fusion_mode="residual",
        )
        model_residual.eval()
        logits_res = model_residual(x)
        self.assertEqual(logits_res.shape, (batch_size, 2, height, width))

    def test_hybrid_vit_gradient_backpropagation(self) -> None:
        """Verifies end-to-end gradient flow through both the 3D-CNN and ViT layers."""
        model = HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
        )
        model.train()

        x = torch.randn(2, 1, 15, 16, 16, dtype=torch.float32)
        targets = torch.randint(0, 2, (2, 16, 16), dtype=torch.long)

        logits = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        loss.backward()

        # 1. 3D-CNN Backbone gradients
        cnn_conv1 = model.cnn.layer1[0]
        self.assertIsNotNone(cnn_conv1.weight.grad)
        self.assertTrue(torch.any(cnn_conv1.weight.grad != 0.0), "3D-CNN Conv1 must have gradients.")

        cnn_conv2 = model.cnn.layer2[0]
        self.assertIsNotNone(cnn_conv2.weight.grad)
        self.assertTrue(torch.any(cnn_conv2.weight.grad != 0.0), "3D-CNN Conv2 must have gradients.")

        # 2. Vision Transformer branch gradients
        vit_patch_proj = model.vit.patch_proj
        self.assertIsNotNone(vit_patch_proj.weight.grad)
        self.assertTrue(torch.any(vit_patch_proj.weight.grad != 0.0), "ViT patch_proj must have gradients.")

        self.assertIsNotNone(model.vit.pos_embedding.grad)
        self.assertTrue(torch.any(model.vit.pos_embedding.grad != 0.0), "ViT pos_embedding must have gradients.")

        # 3. Classifier head gradients
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertTrue(torch.any(model.classifier.weight.grad != 0.0), "Classifier must have gradients.")

    def test_hybrid_vit_validation_errors(self) -> None:
        """Verifies input shape and dimension validation."""
        model = HybridSpectralSpatialViT()

        # Non-5D input
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 16, 16))

        # Channel mismatch
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 15, 16, 16))

        # Invalid fusion mode
        with self.assertRaises(ValueError):
            HybridSpectralSpatialViT(fusion_mode="invalid_fusion")

    def test_hybrid_vit_dataset_batching(self) -> None:
        """Verifies end-to-end integration with HyperspectralPatchDataset and DataLoader."""
        # 1. Generate synthetic cube and reduce with PCA
        cube, mask = generate_synthetic_hyperspectral_cube(
            height=64,
            width=64,
            num_bands=224,
            disease_zone_radius=8,
            seed=42,
        )
        reducer = SpectralPCAReducer(n_components=15, random_state=42)
        reduced_cube = reducer.fit_transform(cube)  # shape: (15, 64, 64)

        # 2. Patch dataset
        patch_size = 16
        dataset = HyperspectralPatchDataset(
            cube=reduced_cube,
            mask=mask,
            patch_size=patch_size,
        )

        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
        )
        model.eval()

        batches_processed = 0
        for batch_patches, batch_masks in loader:
            self.assertEqual(batch_patches.shape, (batch_size, 1, 15, patch_size, patch_size))
            self.assertEqual(batch_masks.shape, (batch_size, patch_size, patch_size))

            with torch.no_grad():
                logits = model(batch_patches)
                probs = model.predict_probability(batch_patches)

            self.assertEqual(logits.shape, (batch_size, 2, patch_size, patch_size))
            self.assertEqual(probs.shape, (batch_size, patch_size, patch_size))
            self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))
            batches_processed += 1

        self.assertEqual(batches_processed, len(dataset) // batch_size)


if __name__ == "__main__":
    unittest.main()
