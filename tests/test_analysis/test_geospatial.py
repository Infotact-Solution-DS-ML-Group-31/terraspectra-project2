"""Unit tests for geospatial raster transformation and temporal spread simulation."""

import unittest
import numpy as np

from src.analysis.geospatial import raster_to_geogrid, simulate_temporal_spread


class TestGeospatialAnalysis(unittest.TestCase):
    """Test suite for geospatial grid conversions and temporal disease spread."""

    def test_raster_to_geogrid_8x8(self) -> None:
        """Verifies raster_to_geogrid on an 8x8 grid for 64 records, coordinates, colors, and elevations."""
        h, w = 8, 8
        prob_map = np.linspace(0.0, 1.0, h * w).reshape(h, w).astype(np.float32)
        ndvi_map = np.full((h, w), 0.75, dtype=np.float32)
        stress_map = np.full((h, w), 0.35, dtype=np.float32)

        origin_lat = 18.5204
        origin_lon = 73.8567
        records = raster_to_geogrid(
            prob_map=prob_map,
            ndvi_map=ndvi_map,
            stress_map=stress_map,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            threshold=0.5,
        )

        # 1. Total records check
        self.assertEqual(len(records), 64)

        # 2. Inspect record contents
        labels_found = set()
        for rec in records:
            self.assertIn("row", rec)
            self.assertIn("col", rec)
            self.assertIn("lat", rec)
            self.assertIn("lon", rec)
            self.assertIn("probability", rec)
            self.assertIn("ndvi", rec)
            self.assertIn("stress", rec)
            self.assertIn("elevation", rec)
            self.assertIn("risk_label", rec)
            self.assertIn("color_rgba", rec)

            # Check lat/lon proximity to origin
            self.assertAlmostEqual(rec["lat"], origin_lat - rec["row"] * 0.00015, places=5)
            self.assertAlmostEqual(rec["lon"], origin_lon + rec["col"] * 0.00015, places=5)

            # Check elevation bounds (prob * 80.0 -> [0.0, 80.0])
            self.assertGreaterEqual(rec["elevation"], 0.0)
            self.assertLessEqual(rec["elevation"], 80.0)

            # Check color_rgba format
            self.assertEqual(len(rec["color_rgba"]), 4)
            for c in rec["color_rgba"]:
                self.assertGreaterEqual(c, 0)
                self.assertLessEqual(c, 255)

            labels_found.add(rec["risk_label"])

        # With linspace from 0 to 1, we should encounter all 3 risk categories
        self.assertIn("Healthy", labels_found)
        self.assertIn("Watch", labels_found)
        self.assertIn("Infected", labels_found)

    def test_raster_to_geogrid_without_stress_map(self) -> None:
        """Verifies raster_to_geogrid operates correctly when stress_map is None."""
        prob_map = np.zeros((4, 4), dtype=np.float32)
        ndvi_map = np.ones((4, 4), dtype=np.float32)
        records = raster_to_geogrid(prob_map, ndvi_map, stress_map=None)
        self.assertEqual(len(records), 16)
        self.assertEqual(records[0]["stress"], 0.0)

    def test_raster_to_geogrid_validation_errors(self) -> None:
        """Verifies ValueError is raised for dimension mismatches and invalid ranks."""
        prob_map = np.zeros((8, 8), dtype=np.float32)
        ndvi_mismatch = np.zeros((8, 6), dtype=np.float32)
        stress_mismatch = np.zeros((6, 8), dtype=np.float32)

        # Mismatched NDVI shape
        with self.assertRaises(ValueError):
            raster_to_geogrid(prob_map, ndvi_mismatch)

        # Mismatched Stress shape
        with self.assertRaises(ValueError):
            raster_to_geogrid(prob_map, prob_map, stress_map=stress_mismatch)

        # 1D array instead of 2D
        with self.assertRaises(ValueError):
            raster_to_geogrid(np.zeros(16), np.zeros(16))

    def test_simulate_temporal_spread(self) -> None:
        """Verifies temporal spread keys [0, 3, 7, 14], shapes, value ranges, and spread monotonicity."""
        h, w = 16, 16
        base_map = np.zeros((h, w), dtype=np.float32)
        # Seed an infection patch in the center
        base_map[6:10, 6:10] = 0.85

        days = [0, 3, 7, 14]
        spread_dict = simulate_temporal_spread(base_map, days=days, growth_rate=0.06)

        # 1. Verify keys
        self.assertEqual(list(spread_dict.keys()), days)

        # 2. Verify shapes and value bounds
        mean_probs = []
        for d in days:
            grid = spread_dict[d]
            self.assertEqual(grid.shape, (h, w))
            self.assertEqual(grid.dtype, np.float32)
            self.assertTrue(np.all(grid >= 0.0), f"Day {d} values must be >= 0.0")
            self.assertTrue(np.all(grid <= 1.0), f"Day {d} values must be <= 1.0")
            mean_probs.append(float(np.mean(grid)))

        # 3. Verify monotonic epidemic spread over time
        for i in range(1, len(mean_probs)):
            self.assertGreaterEqual(
                mean_probs[i],
                mean_probs[i - 1],
                f"Mean disease probability at day {days[i]} ({mean_probs[i]:.4f}) "
                f"must be >= day {days[i-1]} ({mean_probs[i-1]:.4f})",
            )

    def test_simulate_temporal_spread_validation_errors(self) -> None:
        """Verifies error handling on invalid temporal simulation inputs."""
        # Non-2D input
        with self.assertRaises(ValueError):
            simulate_temporal_spread(np.zeros((4, 4, 4), dtype=np.float32))

        # Non-positive growth rate
        with self.assertRaises(ValueError):
            simulate_temporal_spread(np.zeros((8, 8), dtype=np.float32), growth_rate=0.0)

        # Negative forecast days
        with self.assertRaises(ValueError):
            simulate_temporal_spread(np.zeros((8, 8), dtype=np.float32), days=[-1, 5])


if __name__ == "__main__":
    unittest.main()
