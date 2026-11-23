"""Integration tests for GET /api/v1/benchmark endpoint."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestBenchmarkEndpointBasic(unittest.TestCase):
    """Basic smoke tests for GET /api/v1/benchmark."""

    def test_default_params_returns_200(self):
        """GET /api/v1/benchmark with defaults returns HTTP 200."""
        response = client.get("/api/v1/benchmark?height=32&width=32")
        self.assertEqual(response.status_code, 200, response.text)

    def test_response_has_required_top_level_keys(self):
        """Response body contains height, width, pca_components, and results."""
        response = client.get("/api/v1/benchmark?height=32&width=32")
        data = response.json()
        for key in ("height", "width", "pca_components", "results"):
            self.assertIn(key, data, f"Missing top-level key: '{key}'")

    def test_response_height_width_echo(self):
        """Response echoes back the requested height and width."""
        response = client.get("/api/v1/benchmark?height=32&width=32&pca_components=15")
        data = response.json()
        self.assertEqual(data["height"], 32)
        self.assertEqual(data["width"], 32)
        self.assertEqual(data["pca_components"], 15)

    def test_returns_two_model_results(self):
        """results list must contain exactly 2 entries (cnn3d + hybrid)."""
        response = client.get("/api/v1/benchmark?height=32&width=32")
        data = response.json()
        self.assertEqual(len(data["results"]), 2)

    def test_model_names_in_results(self):
        """results must include both 'cnn3d' and 'hybrid' model names."""
        response = client.get("/api/v1/benchmark?height=32&width=32")
        names = {r["model_name"] for r in response.json()["results"]}
        self.assertIn("cnn3d", names)
        self.assertIn("hybrid", names)


class TestBenchmarkEndpointValues(unittest.TestCase):
    """Value-level assertions on the benchmark response."""

    @classmethod
    def setUpClass(cls):
        """Fetch results once for all value tests."""
        response = client.get("/api/v1/benchmark?height=32&width=32&seed=7")
        cls.data = response.json()
        cls.results_by_name = {r["model_name"]: r for r in cls.data["results"]}

    def test_trainable_params_positive(self):
        """Both models must report positive trainable_params."""
        for name, r in self.results_by_name.items():
            with self.subTest(model=name):
                self.assertGreater(r["trainable_params"], 0)

    def test_latencies_positive(self):
        """mean / min / max latency must all be strictly positive."""
        for name, r in self.results_by_name.items():
            with self.subTest(model=name):
                self.assertGreater(r["mean_latency_ms"], 0.0)
                self.assertGreater(r["min_latency_ms"], 0.0)
                self.assertGreater(r["max_latency_ms"], 0.0)

    def test_latency_ordering(self):
        """min_latency_ms ≤ mean_latency_ms ≤ max_latency_ms."""
        for name, r in self.results_by_name.items():
            with self.subTest(model=name):
                self.assertLessEqual(r["min_latency_ms"], r["mean_latency_ms"])
                self.assertLessEqual(r["mean_latency_ms"], r["max_latency_ms"])

    def test_metrics_present_and_valid(self):
        """metrics dict is present with valid keys and [0,1]-bounded values."""
        expected_keys = {"iou", "dice", "precision", "recall", "pixel_accuracy"}
        for name, r in self.results_by_name.items():
            with self.subTest(model=name):
                self.assertIsNotNone(r["metrics"])
                for key in expected_keys:
                    self.assertIn(key, r["metrics"])
                    val = r["metrics"][key]
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

    def test_hybrid_has_more_params_than_cnn3d(self):
        """Hybrid model must have more trainable params than 3D-CNN."""
        n_cnn = self.results_by_name["cnn3d"]["trainable_params"]
        n_hybrid = self.results_by_name["hybrid"]["trainable_params"]
        self.assertGreater(n_hybrid, n_cnn)


class TestBenchmarkEndpointValidation(unittest.TestCase):
    """Input validation / 422 error tests for GET /api/v1/benchmark."""

    def test_height_too_small_returns_422(self):
        """height=4 (< 16) must return HTTP 422 Unprocessable Entity."""
        response = client.get("/api/v1/benchmark?height=4&width=32")
        self.assertEqual(response.status_code, 422, response.text)

    def test_width_too_small_returns_422(self):
        """width=4 (< 16) must return HTTP 422."""
        response = client.get("/api/v1/benchmark?height=32&width=4")
        self.assertEqual(response.status_code, 422, response.text)

    def test_height_too_large_returns_422(self):
        """height=256 (> 128) must return HTTP 422."""
        response = client.get("/api/v1/benchmark?height=256&width=32")
        self.assertEqual(response.status_code, 422, response.text)

    def test_pca_components_too_small_returns_422(self):
        """pca_components=1 (< 3) must return HTTP 422."""
        response = client.get("/api/v1/benchmark?height=32&width=32&pca_components=1")
        self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
