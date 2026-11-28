"""Unit tests for compute_spectral_signatures in src.analysis.spectral_index."""

from __future__ import annotations

import unittest
import numpy as np

from src.analysis.spectral_index import compute_spectral_signatures
from src.data.cube_loader import generate_synthetic_hyperspectral_cube


class TestSpectralProfile(unittest.TestCase):
    """Tests for hyperspectral signature extraction."""

    def test_spectral_signatures_lengths_and_values(self) -> None:
        """Returns matching list lengths equal to cube.num_bands for wavelengths and means."""
        cube, mask = generate_synthetic_hyperspectral_cube(
            height=32,
            width=32,
            num_bands=224,
            disease_zone_radius=6,
            seed=42,
        )
        prob_map = mask.astype(np.float32)

        sig = compute_spectral_signatures(cube, prob_map, threshold=0.5)

        self.assertIn("wavelengths_nm", sig)
        self.assertIn("healthy_mean", sig)
        self.assertIn("infected_mean", sig)

        self.assertEqual(len(sig["wavelengths_nm"]), cube.num_bands)
        self.assertEqual(len(sig["healthy_mean"]), cube.num_bands)
        self.assertEqual(len(sig["infected_mean"]), cube.num_bands)

        # Red-edge (~715 nm) reflectance should be lower in infected pixels
        wl = np.array(sig["wavelengths_nm"])
        re_idx = int(np.argmin(np.abs(wl - 715.0)))
        self.assertLess(sig["infected_mean"][re_idx], sig["healthy_mean"][re_idx])

    def test_spectral_signatures_mismatched_shape_raises_value_error(self) -> None:
        """Raises ValueError when prob_map spatial shape does not match cube."""
        cube, _ = generate_synthetic_hyperspectral_cube(
            height=32,
            width=32,
            num_bands=30,
            seed=42,
        )
        bad_prob_map = np.zeros((16, 16), dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_spectral_signatures(cube, bad_prob_map, threshold=0.5)

    def test_spectral_signatures_all_healthy_fallback(self) -> None:
        """Returns zeros for infected_mean when no pixels exceed threshold."""
        cube, _ = generate_synthetic_hyperspectral_cube(
            height=16,
            width=16,
            num_bands=20,
            seed=42,
        )
        all_healthy = np.zeros((16, 16), dtype=np.float32)

        sig = compute_spectral_signatures(cube, all_healthy, threshold=0.5)
        self.assertEqual(sig["infected_mean"], [0.0] * 20)
        self.assertTrue(any(v > 0.0 for v in sig["healthy_mean"]))


if __name__ == "__main__":
    unittest.main()
