"""Unit tests for spectral index calculations and field health summarization."""

import unittest
import numpy as np

from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.analysis.spectral_index import (
    compute_ndvi,
    compute_red_edge_stress,
    summarize_field_health,
)


class TestSpectralIndex(unittest.TestCase):
    """Test suite for spectral indices and field health summary functions."""

    def test_compute_ndvi_and_red_edge_stress(self) -> None:
        """Verifies NDVI and Red-Edge Stress output shapes (H, W) and bounds in [-1.0, 1.0]."""
        height = 64
        width = 64
        cube, mask = generate_synthetic_hyperspectral_cube(
            height=height,
            width=width,
            num_bands=224,
            disease_zone_radius=8,
            seed=42,
        )

        # 1. Compute NDVI
        ndvi = compute_ndvi(cube, red_nm=670.0, nir_nm=800.0)
        self.assertEqual(ndvi.shape, (height, width))
        self.assertEqual(ndvi.dtype, np.float32)
        self.assertTrue(np.all(ndvi >= -1.0), "NDVI must be >= -1.0")
        self.assertTrue(np.all(ndvi <= 1.0), "NDVI must be <= 1.0")

        # Healthy vegetation canopy has high positive NDVI (typically > 0.70)
        self.assertGreater(float(np.mean(ndvi)), 0.70)

        # 2. Compute Red-Edge Stress Index
        stress = compute_red_edge_stress(cube, red_edge_nm=710.0, nir_nm=800.0)
        self.assertEqual(stress.shape, (height, width))
        self.assertEqual(stress.dtype, np.float32)
        self.assertTrue(np.all(stress >= -1.0), "Stress index must be >= -1.0")
        self.assertTrue(np.all(stress <= 1.0), "Stress index must be <= 1.0")

        # Early blight lowers red-edge reflectance, elevating (NIR - RedEdge) stress contrast
        infected_stress_mean = float(stress[mask == 1].mean())
        healthy_stress_mean = float(stress[mask == 0].mean())
        self.assertGreater(
            infected_stress_mean,
            healthy_stress_mean,
            "Infected pixels must exhibit higher red-edge stress contrast than healthy pixels.",
        )

    def test_summarize_field_health_known_values(self) -> None:
        """Verifies summary metrics on controlled probability and NDVI maps."""
        # Setup 10x10 field (100 pixels), each pixel = 100 m^2 (10,000 m^2 total = 1 ha)
        prob_map = np.full((10, 10), 0.1, dtype=np.float32)
        ndvi_map = np.full((10, 10), 0.85, dtype=np.float32)
        pixel_area = 100.0  # m^2

        # 1. Low Severity: 5% infected pixels
        prob_map[0, 0:5] = 0.80  # 5 infected pixels
        ndvi_map[0, 0:5] = 0.60

        summary_low = summarize_field_health(
            prob_map=prob_map,
            ndvi_map=ndvi_map,
            threshold=0.5,
            pixel_area_sq_m=pixel_area,
        )

        self.assertEqual(summary_low["total_pixels"], 100)
        self.assertEqual(summary_low["infected_pixels"], 5)
        self.assertAlmostEqual(summary_low["infected_percentage"], 5.0, places=2)
        # 5 pixels * 100 m^2 = 500 m^2 = 0.05 ha
        self.assertAlmostEqual(summary_low["infected_area_hectares"], 0.05, places=4)
        self.assertAlmostEqual(summary_low["mean_ndvi_healthy"], 0.85, places=2)
        self.assertAlmostEqual(summary_low["mean_ndvi_infected"], 0.60, places=2)
        self.assertEqual(summary_low["severity_level"], "Low")

        # 2. Moderate Severity: 15% infected pixels
        prob_map_mod = np.full((10, 10), 0.1, dtype=np.float32)
        prob_map_mod[0:3, 0:5] = 0.85  # 15 infected pixels
        summary_mod = summarize_field_health(
            prob_map=prob_map_mod,
            ndvi_map=ndvi_map,
            threshold=0.5,
            pixel_area_sq_m=pixel_area,
        )
        self.assertEqual(summary_mod["infected_pixels"], 15)
        self.assertAlmostEqual(summary_mod["infected_percentage"], 15.0, places=2)
        self.assertAlmostEqual(summary_mod["infected_area_hectares"], 0.15, places=4)
        self.assertEqual(summary_mod["severity_level"], "Moderate")

        # 3. High Severity: 35% infected pixels
        prob_map_high = np.full((10, 10), 0.1, dtype=np.float32)
        prob_map_high[0:7, 0:5] = 0.90  # 35 infected pixels
        summary_high = summarize_field_health(
            prob_map=prob_map_high,
            ndvi_map=ndvi_map,
            threshold=0.5,
            pixel_area_sq_m=pixel_area,
        )
        self.assertEqual(summary_high["infected_pixels"], 35)
        self.assertAlmostEqual(summary_high["infected_percentage"], 35.0, places=2)
        self.assertAlmostEqual(summary_high["infected_area_hectares"], 0.35, places=4)
        self.assertEqual(summary_high["severity_level"], "High")

    def test_summarize_field_health_validation_errors(self) -> None:
        """Verifies ValueError is raised for shape mismatches and invalid parameters."""
        prob_map = np.zeros((10, 10), dtype=np.float32)
        ndvi_map = np.zeros((10, 10), dtype=np.float32)

        # Spatial shape mismatch
        with self.assertRaises(ValueError):
            summarize_field_health(prob_map, np.zeros((10, 8), dtype=np.float32))

        # 1D array instead of 2D
        with self.assertRaises(ValueError):
            summarize_field_health(np.zeros(100), ndvi_map)

        # Invalid threshold
        with self.assertRaises(ValueError):
            summarize_field_health(prob_map, ndvi_map, threshold=1.5)

        with self.assertRaises(ValueError):
            summarize_field_health(prob_map, ndvi_map, threshold=-0.1)

        # Invalid pixel area
        with self.assertRaises(ValueError):
            summarize_field_health(prob_map, ndvi_map, pixel_area_sq_m=0.0)

        # Type errors for compute_ndvi
        with self.assertRaises(TypeError):
            compute_ndvi("not_a_cube")  # type: ignore


if __name__ == "__main__":
    unittest.main()
