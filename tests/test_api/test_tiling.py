"""Unit tests for tiled spatial inference pipeline."""

import unittest
import numpy as np
import torch

from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT
from src.api.tiling import TilingConfig, predict_full_raster


class TestTilingInference(unittest.TestCase):
    """Test suite for overlapping patch tiling and full raster inference."""

    def setUp(self) -> None:
        """Initialize lightweight models for testing."""
        self.cnn_model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
        self.vit_model = HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=32,
            patch_size=4,
            num_classes=2,
            num_vit_layers=1,
            num_vit_heads=2,
        )
        self.config = TilingConfig(patch_size=16, stride=8, batch_size=4, device="cpu")

    def test_predict_full_raster_non_divisible_shape(self) -> None:
        """Verifies full raster inference on non-divisible dimensions (15, 37, 45)."""
        num_channels = 15
        height = 37
        width = 45
        raster = np.random.uniform(0.0, 1.0, size=(num_channels, height, width)).astype(np.float32)

        # 1. Test with SpectralSpatial3DCNN
        prob_cnn = predict_full_raster(self.cnn_model, raster, config=self.config)
        self.assertEqual(prob_cnn.shape, (height, width))
        self.assertEqual(prob_cnn.dtype, np.float32)
        self.assertTrue(np.all(prob_cnn >= 0.0), "Probabilities must be >= 0.0")
        self.assertTrue(np.all(prob_cnn <= 1.0), "Probabilities must be <= 1.0")

        # 2. Test with HybridSpectralSpatialViT
        prob_vit = predict_full_raster(self.vit_model, raster, config=self.config)
        self.assertEqual(prob_vit.shape, (height, width))
        self.assertEqual(prob_vit.dtype, np.float32)
        self.assertTrue(np.all(prob_vit >= 0.0), "Probabilities must be >= 0.0")
        self.assertTrue(np.all(prob_vit <= 1.0), "Probabilities must be <= 1.0")

    def test_predict_full_raster_small_extent(self) -> None:
        """Verifies reflection/edge padding on a raster strictly smaller than patch_size (15, 10, 12)."""
        num_channels = 15
        height = 10
        width = 12
        small_raster = np.random.uniform(0.0, 1.0, size=(num_channels, height, width)).astype(np.float32)

        prob_small_cnn = predict_full_raster(self.cnn_model, small_raster, config=self.config)
        self.assertEqual(prob_small_cnn.shape, (height, width))
        self.assertEqual(prob_small_cnn.dtype, np.float32)
        self.assertTrue(np.all(prob_small_cnn >= 0.0) and np.all(prob_small_cnn <= 1.0))

        prob_small_vit = predict_full_raster(self.vit_model, small_raster, config=self.config)
        self.assertEqual(prob_small_vit.shape, (height, width))
        self.assertEqual(prob_small_vit.dtype, np.float32)
        self.assertTrue(np.all(prob_small_vit >= 0.0) and np.all(prob_small_vit <= 1.0))

    def test_predict_full_raster_hyperspectral_cube_input(self) -> None:
        """Verifies seamless inference when passing a HyperspectralCube instance directly."""
        cube, _ = generate_synthetic_hyperspectral_cube(
            height=32,
            width=32,
            num_bands=15,
            disease_zone_radius=4,
            seed=42,
        )
        prob = predict_full_raster(self.cnn_model, cube, config=self.config)
        self.assertEqual(prob.shape, (32, 32))
        self.assertEqual(prob.dtype, np.float32)

    def test_predict_full_raster_tensor_input(self) -> None:
        """Verifies seamless inference when passing a torch.Tensor directly."""
        tensor = torch.rand(15, 20, 20, dtype=torch.float32)
        prob = predict_full_raster(self.cnn_model, tensor, config=self.config)
        self.assertEqual(prob.shape, (20, 20))
        self.assertEqual(prob.dtype, np.float32)

    def test_validation_errors(self) -> None:
        """Verifies validation errors for configuration parameters and input shapes."""
        # TilingConfig validation
        with self.assertRaises(ValueError):
            TilingConfig(patch_size=0)

        with self.assertRaises(ValueError):
            TilingConfig(stride=0)

        with self.assertRaises(ValueError):
            TilingConfig(patch_size=16, stride=20)  # stride > patch_size

        with self.assertRaises(ValueError):
            TilingConfig(batch_size=0)

        # Input rank errors
        with self.assertRaises(ValueError):
            predict_full_raster(self.cnn_model, np.zeros((32, 32), dtype=np.float32))

        with self.assertRaises(ValueError):
            predict_full_raster(self.cnn_model, np.zeros((1, 15, 32, 32), dtype=np.float32))

        with self.assertRaises(TypeError):
            predict_full_raster(self.cnn_model, "invalid_cube")  # type: ignore


if __name__ == "__main__":
    unittest.main()
