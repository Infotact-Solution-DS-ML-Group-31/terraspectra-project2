"""Unit and integration tests for TerraSpectra FastAPI endpoints."""

import unittest
from fastapi.testclient import TestClient

from src.api.main import app


class TestMainAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def setUp(self) -> None:
        """Create TestClient instance."""
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        """Verifies GET /health returns status 200 and expected schema keys."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertIn("hybrid", data["available_models"])
        self.assertIn("cnn3d", data["available_models"])

    def test_forecast_hybrid_model(self) -> None:
        """Verifies POST /api/v1/forecast with Hybrid model returns 200 and (32, 32) maps."""
        payload = {
            "height": 32,
            "width": 32,
            "bands": 224,
            "pca_components": 15,
            "model_type": "hybrid",
            "disease_threshold": 0.5,
            "pixel_area_sq_m": 1.0,
            "seed": 42,
        }
        response = self.client.post("/api/v1/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_type"], "hybrid")
        self.assertEqual(data["raster_shape"], [32, 32])

        # Verify map dimensions
        prob_map = data["probability_map"]
        ndvi_map = data["ndvi_map"]
        stress_map = data["red_edge_stress_map"]

        self.assertEqual(len(prob_map), 32)
        self.assertEqual(len(prob_map[0]), 32)
        self.assertEqual(len(ndvi_map), 32)
        self.assertEqual(len(ndvi_map[0]), 32)
        self.assertEqual(len(stress_map), 32)
        self.assertEqual(len(stress_map[0]), 32)

        # Verify summary keys
        summary = data["health_summary"]
        self.assertEqual(summary["total_pixels"], 32 * 32)
        self.assertIn("infected_pixels", summary)
        self.assertIn("infected_percentage", summary)
        self.assertIn("infected_area_hectares", summary)
        self.assertIn("severity_level", summary)

    def test_forecast_cnn3d_model(self) -> None:
        """Verifies POST /api/v1/forecast with 3D-CNN model returns 200."""
        payload = {
            "height": 32,
            "width": 32,
            "bands": 224,
            "pca_components": 15,
            "model_type": "cnn3d",
            "disease_threshold": 0.5,
            "pixel_area_sq_m": 1.0,
            "seed": 42,
        }
        response = self.client.post("/api/v1/forecast", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_type"], "cnn3d")
        self.assertEqual(data["raster_shape"], [32, 32])

    def test_forecast_invalid_payload(self) -> None:
        """Verifies invalid parameters return 422 Unprocessable Entity."""
        # Height too small (ge=16)
        res_height = self.client.post("/api/v1/forecast", json={"height": 4})
        self.assertEqual(res_height.status_code, 422)

        # Disease threshold too high (le=1.0)
        res_thresh = self.client.post("/api/v1/forecast", json={"disease_threshold": 1.5})
        self.assertEqual(res_thresh.status_code, 422)

        # Invalid model type
        res_model = self.client.post("/api/v1/forecast", json={"model_type": "invalid_model"})
        self.assertEqual(res_model.status_code, 422)

        # PCA components exceed bands
        res_pca = self.client.post("/api/v1/forecast", json={"bands": 20, "pca_components": 25})
        self.assertEqual(res_pca.status_code, 422)


if __name__ == "__main__":
    unittest.main()
