"""Smoke tests for the Streamlit + PyDeck pure-Python dashboard pipeline."""

import unittest
import numpy as np

from app import run_dashboard_pipeline


class TestAppSmoke(unittest.TestCase):
    """Smoke test suite for the dashboard end-to-end data and modeling pipeline."""

    def test_run_dashboard_pipeline_hybrid(self) -> None:
        """Verifies run_dashboard_pipeline executes end-to-end with Hybrid model without errors."""
        size = 32
        threshold = 0.5
        seed = 42

        results = run_dashboard_pipeline(
            model_type="hybrid",
            size=size,
            threshold=threshold,
            seed=seed,
        )

        # 1. Structure check
        self.assertIn("prob_map", results)
        self.assertIn("ndvi_map", results)
        self.assertIn("stress_map", results)
        self.assertIn("temporal_frames", results)
        self.assertIn("health_summary", results)
        self.assertIn("geogrid_records", results)

        # 2. Raster map dimensions
        prob_map = results["prob_map"]
        ndvi_map = results["ndvi_map"]
        stress_map = results["stress_map"]

        self.assertEqual(prob_map.shape, (size, size))
        self.assertEqual(ndvi_map.shape, (size, size))
        self.assertEqual(stress_map.shape, (size, size))

        self.assertTrue(np.all(prob_map >= 0.0) and np.all(prob_map <= 1.0))
        self.assertTrue(np.all(ndvi_map >= -1.0) and np.all(ndvi_map <= 1.0))
        self.assertTrue(np.all(stress_map >= -1.0) and np.all(stress_map <= 1.0))

        # 3. Temporal frames check
        temporal_frames = results["temporal_frames"]
        expected_days = [0, 3, 7, 14]
        self.assertEqual(list(temporal_frames.keys()), expected_days)
        for day in expected_days:
            frame = temporal_frames[day]
            self.assertEqual(frame.shape, (size, size))
            self.assertTrue(np.all(frame >= 0.0) and np.all(frame <= 1.0))

        # 4. Summary metrics
        summary = results["health_summary"]
        self.assertEqual(summary["total_pixels"], size * size)
        self.assertIn("infected_pixels", summary)
        self.assertIn("infected_percentage", summary)
        self.assertIn("infected_area_hectares", summary)
        self.assertIn("severity_level", summary)
        self.assertIn(summary["severity_level"], ["Low", "Moderate", "High"])

        # 5. Geo-grid records
        geogrid = results["geogrid_records"]
        self.assertEqual(len(geogrid), size * size)
        sample = geogrid[0]
        self.assertIn("lat", sample)
        self.assertIn("lon", sample)
        self.assertIn("elevation", sample)
        self.assertIn("risk_label", sample)
        self.assertIn("color_rgba", sample)
        self.assertEqual(len(sample["color_rgba"]), 4)

    def test_run_dashboard_pipeline_cnn3d(self) -> None:
        """Verifies run_dashboard_pipeline operates cleanly with 3D-CNN architecture."""
        results = run_dashboard_pipeline(
            model_type="cnn3d",
            size=32,
            threshold=0.5,
            seed=42,
        )
        self.assertEqual(results["prob_map"].shape, (32, 32))
        self.assertEqual(len(results["geogrid_records"]), 1024)


if __name__ == "__main__":
    unittest.main()
