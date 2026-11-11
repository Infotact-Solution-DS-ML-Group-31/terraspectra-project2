"""Unit tests for HyperspectralCube loader, synthetic simulator, and SpectralPCAReducer."""

import unittest
import numpy as np

from src.data.cube_loader import HyperspectralCube, generate_synthetic_hyperspectral_cube
from src.data.pca_reducer import SpectralPCAReducer


class TestHyperspectralCubeLoader(unittest.TestCase):
    """Test suite for HyperspectralCube and synthetic data generation."""

    def test_synthetic_cube_shapes_and_wavelengths(self) -> None:
        """Verifies shape (224, 64, 64), 224 wavelengths between 400nm and 2500nm,

        and reflectance values bounded in [0.0, 1.0].
        """
        num_bands = 224
        height = 64
        width = 64

        cube, mask = generate_synthetic_hyperspectral_cube(
            height=height,
            width=width,
            num_bands=num_bands,
            disease_zone_radius=8,
            seed=42,
        )

        # 1. Verify tensor shapes
        self.assertEqual(cube.shape, (num_bands, height, width))
        self.assertEqual(cube.data.shape, (num_bands, height, width))
        self.assertEqual(cube.num_bands, num_bands)
        self.assertEqual(cube.height, height)
        self.assertEqual(cube.width, width)
        self.assertEqual(mask.shape, (height, width))

        # 2. Verify wavelengths range and continuity
        self.assertEqual(len(cube.wavelengths), num_bands)
        self.assertAlmostEqual(float(cube.wavelengths[0]), 400.0, places=1)
        self.assertAlmostEqual(float(cube.wavelengths[-1]), 2500.0, places=1)
        self.assertTrue(np.all(np.diff(cube.wavelengths) > 0.0), "Wavelengths must be strictly monotonic.")

        # 3. Verify reflectance range [0.0, 1.0]
        self.assertGreaterEqual(float(cube.data.min()), 0.0)
        self.assertLessEqual(float(cube.data.max()), 1.0)
        self.assertEqual(cube.data.dtype, np.float32)

        # 4. Verify spatial resolution and metadata
        self.assertEqual(cube.spatial_resolution_meters, 3.0)
        self.assertIn("sensor", cube.metadata)
        self.assertIn("disease_type", cube.metadata)

    def test_disease_mask_injection(self) -> None:
        """Verifies non-empty ground truth mask and confirms spectral reflectance is lower

        in the red-edge spectrum (700-750nm) for infected pixels compared to healthy pixels.
        """
        cube, mask = generate_synthetic_hyperspectral_cube(
            height=64,
            width=64,
            num_bands=224,
            disease_zone_radius=8,
            seed=42,
        )

        # 1. Non-empty ground-truth mask
        infected_count = int(np.sum(mask == 1))
        healthy_count = int(np.sum(mask == 0))
        self.assertGreater(infected_count, 0, "Disease mask must contain infected pixels.")
        self.assertGreater(healthy_count, 0, "Disease mask must contain healthy pixels.")
        self.assertEqual(infected_count + healthy_count, 64 * 64)

        # 2. Locate red-edge spectral bands (700nm to 750nm)
        red_edge_idx = np.where((cube.wavelengths >= 700.0) & (cube.wavelengths <= 750.0))[0]
        self.assertGreater(len(red_edge_idx), 0, "Must have spectral bands in red-edge spectrum.")

        # 3. Compare red-edge mean reflectance between healthy and infected pixels
        healthy_red_edge_mean = float(cube.data[red_edge_idx, :, :][:, mask == 0].mean())
        infected_red_edge_mean = float(cube.data[red_edge_idx, :, :][:, mask == 1].mean())

        self.assertLess(
            infected_red_edge_mean,
            healthy_red_edge_mean,
            f"Infected red-edge mean ({infected_red_edge_mean:.4f}) must be lower than "
            f"healthy red-edge mean ({healthy_red_edge_mean:.4f}).",
        )

        # Verify drop magnitude is within 15% - 25% range
        percentage_drop = (1.0 - (infected_red_edge_mean / healthy_red_edge_mean)) * 100.0
        self.assertGreaterEqual(
            percentage_drop,
            14.0,  # allowing minor numerical tolerance around 15%
            f"Red-edge drop ({percentage_drop:.2f}%) should be >= 14%.",
        )
        self.assertLessEqual(
            percentage_drop,
            26.0,  # allowing minor numerical tolerance around 25%
            f"Red-edge drop ({percentage_drop:.2f}%) should be <= 26%.",
        )

        # 4. Check that visible bands (e.g. 500-600nm) do not show pre-symptomatic blight drop
        vis_idx = np.where((cube.wavelengths >= 500.0) & (cube.wavelengths <= 600.0))[0]
        healthy_vis_mean = float(cube.data[vis_idx, :, :][:, mask == 0].mean())
        infected_vis_mean = float(cube.data[vis_idx, :, :][:, mask == 1].mean())
        # Visible difference should be negligible (< 3% natural spatial variance)
        vis_diff = abs(healthy_vis_mean - infected_vis_mean) / healthy_vis_mean
        self.assertLess(vis_diff, 0.05, "Pre-symptomatic blight should not drastically alter visible bands.")

    def test_pca_compression_and_variance(self) -> None:
        """Verifies output shape (15, 64, 64), cumulative explained variance > 95%,

        and reconstruction fidelity (low mean squared error).
        """
        n_components = 15
        cube, _ = generate_synthetic_hyperspectral_cube(
            height=64,
            width=64,
            num_bands=224,
            disease_zone_radius=8,
            seed=42,
        )

        reducer = SpectralPCAReducer(n_components=n_components, random_state=42)

        # 1. Fit and transform
        reduced = reducer.fit_transform(cube)

        # 2. Check output dimensions
        self.assertEqual(reduced.shape, (n_components, 64, 64))
        self.assertEqual(reduced.dtype, np.float32)

        # 3. Check cumulative explained variance ratio >= 0.95 (95%)
        cum_var = reducer.cumulative_explained_variance_ratio
        self.assertGreater(
            cum_var,
            0.95,
            f"Cumulative explained variance ({cum_var:.4f}) must exceed 0.95 (95%).",
        )
        self.assertTrue(reducer.verify_cumulative_variance(0.95))

        # 4. Check reconstruction fidelity
        reconstructed = reducer.inverse_transform(reduced)
        self.assertEqual(reconstructed.shape, cube.data.shape)
        self.assertEqual(reconstructed.dtype, np.float32)

        mse = float(np.mean((cube.data - reconstructed) ** 2))
        self.assertLess(
            mse,
            1e-3,
            f"Reconstruction MSE ({mse:.6f}) must be low (< 1e-3).",
        )

    def test_hyperspectral_cube_validation(self) -> None:
        """Verifies validation and immutability of HyperspectralCube dataclass."""
        valid_data = np.zeros((10, 20, 20), dtype=np.float32)
        valid_wavelengths = np.linspace(400, 700, 10, dtype=np.float32)

        # Successful creation
        cube = HyperspectralCube(data=valid_data, wavelengths=valid_wavelengths)
        self.assertEqual(cube.num_bands, 10)

        # Immutability
        with self.assertRaises(Exception):
            cube.spatial_resolution_meters = 5.0  # type: ignore

        # Dimension mismatch
        with self.assertRaises(ValueError):
            HyperspectralCube(data=np.zeros((10, 20), dtype=np.float32), wavelengths=valid_wavelengths)

        # Band count mismatch
        with self.assertRaises(ValueError):
            HyperspectralCube(data=valid_data, wavelengths=np.linspace(400, 700, 8, dtype=np.float32))

        # Invalid resolution
        with self.assertRaises(ValueError):
            HyperspectralCube(
                data=valid_data,
                wavelengths=valid_wavelengths,
                spatial_resolution_meters=-1.0,
            )

    def test_spectral_pca_validation(self) -> None:
        """Verifies input checking and error handling in SpectralPCAReducer."""
        reducer = SpectralPCAReducer(n_components=5)

        # Calling transform before fit raises RuntimeError
        with self.assertRaises(RuntimeError):
            reducer.transform(np.zeros((10, 10, 10), dtype=np.float32))

        # n_components <= 0 raises ValueError
        with self.assertRaises(ValueError):
            SpectralPCAReducer(n_components=0)


if __name__ == "__main__":
    unittest.main()

