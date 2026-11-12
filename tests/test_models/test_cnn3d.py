"""Unit tests for SpectralSpatial3DCNN and HyperspectralPatchDataset."""

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.pca_reducer import SpectralPCAReducer
from src.data.dataset import HyperspectralPatchDataset
from src.models.cnn3d import SpectralSpatial3DCNN


class TestSpectralSpatial3DCNN(unittest.TestCase):
    """Test suite for SpectralSpatial3DCNN and HyperspectralPatchDataset."""

    def test_3d_cnn_forward_shape(self) -> None:
        """Passes (2, 1, 15, 32, 32) and asserts output logit shape (2, 2, 32, 32)

        and probability bounds in [0.0, 1.0].
        """
        model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
        model.eval()

        batch_size = 2
        in_channels = 1
        spectral_depth = 15
        height = 32
        width = 32

        # 1. Forward pass with raw logits
        x = torch.randn(batch_size, in_channels, spectral_depth, height, width, dtype=torch.float32)
        logits = model(x)

        self.assertEqual(logits.shape, (batch_size, 2, height, width))
        self.assertEqual(logits.dtype, torch.float32)

        # 2. Probability prediction
        probs = model.predict_probability(x)
        self.assertEqual(probs.shape, (batch_size, height, width))
        self.assertEqual(probs.dtype, torch.float32)
        self.assertTrue(torch.all(probs >= 0.0), "Probabilities must be >= 0.0")
        self.assertTrue(torch.all(probs <= 1.0), "Probabilities must be <= 1.0")

        # Verify probability equals softmax class 1
        expected_probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        self.assertTrue(torch.allclose(probs, expected_probs, atol=1e-6))

    def test_gradient_backpropagation(self) -> None:
        """Runs a dummy loss backward pass confirming non-zero gradients on convolutional weights."""
        model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
        model.train()

        x = torch.randn(2, 1, 15, 32, 32, dtype=torch.float32)
        targets = torch.randint(0, 2, (2, 32, 32), dtype=torch.long)

        logits = model(x)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Check gradients exist and are non-zero on Layer 1 Conv3d
        conv1 = model.layer1[0]
        self.assertIsNotNone(conv1.weight.grad)
        self.assertTrue(torch.any(conv1.weight.grad != 0.0), "Layer 1 weights must have non-zero gradients.")

        # Check gradients exist and are non-zero on Layer 2 Conv3d
        conv2 = model.layer2[0]
        self.assertIsNotNone(conv2.weight.grad)
        self.assertTrue(torch.any(conv2.weight.grad != 0.0), "Layer 2 weights must have non-zero gradients.")

        # Check gradients exist and are non-zero on 2D Classifier Conv
        classifier = model.classifier
        self.assertIsNotNone(classifier.weight.grad)
        self.assertTrue(torch.any(classifier.weight.grad != 0.0), "Classifier weights must have non-zero gradients.")

    def test_patch_dataset_batching(self) -> None:
        """Verifies HyperspectralPatchDataset batches cleanly with PyTorch DataLoader."""
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

        # 2. Instantiate dataset with patch_size=16
        patch_size = 16
        dataset = HyperspectralPatchDataset(
            cube=reduced_cube,
            mask=mask,
            patch_size=patch_size,
        )

        expected_patches = (64 // patch_size) * (64 // patch_size)  # 16 patches
        self.assertEqual(len(dataset), expected_patches)

        # Single patch check
        patch_0, mask_0 = dataset[0]
        self.assertEqual(patch_0.shape, (1, 15, patch_size, patch_size))
        self.assertEqual(mask_0.shape, (patch_size, patch_size))
        self.assertEqual(patch_0.dtype, torch.float32)
        self.assertEqual(mask_0.dtype, torch.long)

        # 3. Batching with DataLoader
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        batches_checked = 0
        model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
        model.eval()

        for batch_patches, batch_masks in loader:
            self.assertEqual(batch_patches.shape, (batch_size, 1, 15, patch_size, patch_size))
            self.assertEqual(batch_masks.shape, (batch_size, patch_size, patch_size))
            self.assertEqual(batch_patches.dtype, torch.float32)
            self.assertEqual(batch_masks.dtype, torch.long)

            # End-to-end model execution on batched data
            with torch.no_grad():
                logits = model(batch_patches)
                probs = model.predict_probability(batch_patches)

            self.assertEqual(logits.shape, (batch_size, 2, patch_size, patch_size))
            self.assertEqual(probs.shape, (batch_size, patch_size, patch_size))
            batches_checked += 1

        self.assertEqual(batches_checked, expected_patches // batch_size)

    def test_cnn_validation_and_dataset_errors(self) -> None:
        """Verifies dimension validation and error handling in CNN and Dataset."""
        model = SpectralSpatial3DCNN()

        # Non-5D input raises ValueError
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 16, 16))

        # Dataset dimension mismatch raises ValueError
        with self.assertRaises(ValueError):
            HyperspectralPatchDataset(
                cube=np.zeros((15, 64, 64)),
                mask=np.zeros((32, 32)),  # mismatch with 64x64
            )

        # Patch size exceeding dimensions raises ValueError
        with self.assertRaises(ValueError):
            HyperspectralPatchDataset(
                cube=np.zeros((15, 32, 32)),
                mask=np.zeros((32, 32)),
                patch_size=64,
            )


if __name__ == "__main__":
    unittest.main()

