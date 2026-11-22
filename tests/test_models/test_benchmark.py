"""Unit tests for src.models.benchmark — architecture benchmarking utilities."""

from __future__ import annotations

import unittest

import torch

from src.models.benchmark import (
    benchmark_model,
    compare_architectures,
    count_trainable_parameters,
)
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cnn3d() -> SpectralSpatial3DCNN:
    m = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
    m.eval()
    return m


def _make_hybrid() -> HybridSpectralSpatialViT:
    m = HybridSpectralSpatialViT(
        in_channels=1,
        cnn_features=32,
        vit_embed_dim=64,
        patch_size=4,
        num_classes=2,
    )
    m.eval()
    return m


def _dummy_input(spectral: int = 15, h: int = 16, w: int = 16) -> torch.Tensor:
    return torch.randn(1, 1, spectral, h, w)


# ---------------------------------------------------------------------------
# count_trainable_parameters
# ---------------------------------------------------------------------------

class TestCountTrainableParameters(unittest.TestCase):
    """Tests for count_trainable_parameters."""

    def test_cnn3d_returns_positive_integer(self):
        """count_trainable_parameters returns a positive int for SpectralSpatial3DCNN."""
        n = count_trainable_parameters(_make_cnn3d())
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_hybrid_returns_positive_integer(self):
        """count_trainable_parameters returns a positive int for HybridSpectralSpatialViT."""
        n = count_trainable_parameters(_make_hybrid())
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_hybrid_has_more_params_than_cnn3d(self):
        """HybridSpectralSpatialViT should have more trainable params than SpectralSpatial3DCNN."""
        n_cnn = count_trainable_parameters(_make_cnn3d())
        n_hybrid = count_trainable_parameters(_make_hybrid())
        self.assertGreater(
            n_hybrid,
            n_cnn,
            f"Expected hybrid ({n_hybrid}) > cnn3d ({n_cnn}).",
        )

    def test_frozen_model_returns_zero(self):
        """count_trainable_parameters returns 0 for a fully frozen model."""
        model = _make_cnn3d()
        for p in model.parameters():
            p.requires_grad_(False)
        self.assertEqual(count_trainable_parameters(model), 0)


# ---------------------------------------------------------------------------
# benchmark_model
# ---------------------------------------------------------------------------

class TestBenchmarkModel(unittest.TestCase):
    """Tests for benchmark_model."""

    def test_cnn3d_result_keys(self):
        """benchmark_model result for cnn3d contains all required keys."""
        result = benchmark_model(
            model=_make_cnn3d(),
            model_name="cnn3d",
            input_tensor=_dummy_input(),
            runs=2,
        )
        for key in ("model_name", "trainable_params", "mean_latency_ms",
                    "min_latency_ms", "max_latency_ms"):
            self.assertIn(key, result, f"Missing key '{key}' in benchmark result.")

    def test_latencies_are_positive(self):
        """All latency fields must be strictly positive."""
        result = benchmark_model(
            model=_make_cnn3d(),
            model_name="cnn3d",
            input_tensor=_dummy_input(),
            runs=2,
        )
        self.assertGreater(result["mean_latency_ms"], 0.0)
        self.assertGreater(result["min_latency_ms"], 0.0)
        self.assertGreater(result["max_latency_ms"], 0.0)

    def test_min_leq_mean_leq_max(self):
        """min_latency_ms ≤ mean_latency_ms ≤ max_latency_ms."""
        result = benchmark_model(
            model=_make_cnn3d(),
            model_name="cnn3d",
            input_tensor=_dummy_input(),
            runs=3,
        )
        self.assertLessEqual(result["min_latency_ms"], result["mean_latency_ms"])
        self.assertLessEqual(result["mean_latency_ms"], result["max_latency_ms"])

    def test_metrics_none_without_target(self):
        """metrics key must be None when target_mask is not provided."""
        result = benchmark_model(
            model=_make_cnn3d(),
            model_name="cnn3d",
            input_tensor=_dummy_input(),
        )
        self.assertIsNone(result["metrics"])

    def test_metrics_populated_with_target(self):
        """metrics dict is populated when target_mask is provided."""
        h, w = 16, 16
        target = torch.zeros(1, h, w, dtype=torch.long)
        result = benchmark_model(
            model=_make_cnn3d(),
            model_name="cnn3d",
            input_tensor=_dummy_input(h=h, w=w),
            target_mask=target,
            runs=2,
        )
        self.assertIsNotNone(result["metrics"])
        for key in ("iou", "dice", "precision", "recall", "pixel_accuracy"):
            self.assertIn(key, result["metrics"])

    def test_invalid_runs_raises_value_error(self):
        """benchmark_model raises ValueError when runs < 1."""
        with self.assertRaises(ValueError):
            benchmark_model(
                model=_make_cnn3d(),
                model_name="cnn3d",
                input_tensor=_dummy_input(),
                runs=0,
            )

    def test_model_name_stored_correctly(self):
        """model_name in result matches the string passed to benchmark_model."""
        result = benchmark_model(
            model=_make_hybrid(),
            model_name="hybrid",
            input_tensor=_dummy_input(),
            runs=1,
        )
        self.assertEqual(result["model_name"], "hybrid")


# ---------------------------------------------------------------------------
# compare_architectures
# ---------------------------------------------------------------------------

class TestCompareArchitectures(unittest.TestCase):
    """Tests for compare_architectures end-to-end comparison."""

    @classmethod
    def setUpClass(cls):
        """Run compare_architectures once for the whole test class."""
        cls.results = compare_architectures(
            height=32, width=32, pca_components=15, seed=42
        )

    def test_returns_two_entries(self):
        """compare_architectures must return exactly 2 benchmark dicts."""
        self.assertEqual(len(self.results), 2)

    def test_model_names(self):
        """Results list must contain one 'cnn3d' and one 'hybrid' entry."""
        names = {r["model_name"] for r in self.results}
        self.assertIn("cnn3d", names)
        self.assertIn("hybrid", names)

    def test_all_latencies_positive(self):
        """All latency fields must be strictly positive for both architectures."""
        for r in self.results:
            with self.subTest(model=r["model_name"]):
                self.assertGreater(r["mean_latency_ms"], 0.0,
                                   f"{r['model_name']}: mean_latency_ms must be > 0")
                self.assertGreater(r["min_latency_ms"], 0.0,
                                   f"{r['model_name']}: min_latency_ms must be > 0")
                self.assertGreater(r["max_latency_ms"], 0.0,
                                   f"{r['model_name']}: max_latency_ms must be > 0")

    def test_trainable_params_positive(self):
        """Both models must have at least one trainable parameter."""
        for r in self.results:
            with self.subTest(model=r["model_name"]):
                self.assertGreater(r["trainable_params"], 0)

    def test_metrics_present_and_valid(self):
        """metrics dict must exist and contain all five standard metric keys."""
        expected_keys = {"iou", "dice", "precision", "recall", "pixel_accuracy"}
        for r in self.results:
            with self.subTest(model=r["model_name"]):
                self.assertIsNotNone(r["metrics"],
                                     f"{r['model_name']}: metrics should not be None")
                for key in expected_keys:
                    self.assertIn(key, r["metrics"],
                                  f"{r['model_name']}: missing metric key '{key}'")

    def test_metrics_in_valid_range(self):
        """All metric values must be floats in [0.0, 1.0]."""
        for r in self.results:
            for key, val in r["metrics"].items():
                with self.subTest(model=r["model_name"], metric=key):
                    self.assertIsInstance(val, float)
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

    def test_hybrid_has_more_params_than_cnn3d(self):
        """HybridSpectralSpatialViT must have more trainable params than SpectralSpatial3DCNN."""
        by_name = {r["model_name"]: r for r in self.results}
        n_cnn = by_name["cnn3d"]["trainable_params"]
        n_hybrid = by_name["hybrid"]["trainable_params"]
        self.assertGreater(
            n_hybrid,
            n_cnn,
            f"Expected hybrid ({n_hybrid}) > cnn3d ({n_cnn}).",
        )


if __name__ == "__main__":
    unittest.main()
