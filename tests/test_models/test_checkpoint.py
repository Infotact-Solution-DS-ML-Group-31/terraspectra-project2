"""Unit and integration tests for src.models.checkpoint.

Covers:
- Round-trip save / load for HybridSpectralSpatialViT and SpectralSpatial3DCNN.
- Prediction determinism: outputs match torch.allclose after reload.
- Error handling: FileNotFoundError for missing files, ValueError for bad types.
- FastAPI GET /api/v1/models endpoint returning expected schema keys.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.checkpoint import load_model_checkpoint, save_model_checkpoint
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_cnn3d() -> SpectralSpatial3DCNN:
    m = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
    m.eval()
    return m


def _dummy_input(spectral_depth: int = 15, h: int = 16, w: int = 16) -> torch.Tensor:
    """Return a batch-1 input tensor suitable for both architectures."""
    return torch.randn(1, 1, spectral_depth, h, w)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip(unittest.TestCase):
    """Verifies that save → load preserves weights and predictions exactly."""

    def _assert_round_trip(self, model: torch.nn.Module, model_type: str) -> None:
        """Save, reload, and confirm predictions match via torch.allclose."""
        x = _dummy_input()

        with torch.no_grad():
            probs_before = model.predict_probability(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            meta = {"epoch": 3, "loss": 0.142, "pca_components": 15}

            returned_path = save_model_checkpoint(
                model, ckpt_path, model_type=model_type, metadata=meta
            )

            # Returned path must be the resolved checkpoint file
            self.assertTrue(returned_path.is_file(), "Checkpoint file should exist after save.")
            self.assertEqual(returned_path.suffix, ".pt")

            # Reload
            loaded_model, loaded_meta = load_model_checkpoint(
                ckpt_path, in_channels=15, device="cpu"
            )

        # Metadata round-trips cleanly
        self.assertEqual(loaded_meta["epoch"], 3)
        self.assertAlmostEqual(loaded_meta["loss"], 0.142, places=5)
        self.assertEqual(loaded_meta["pca_components"], 15)

        # Predictions must be numerically identical
        with torch.no_grad():
            probs_after = loaded_model.predict_probability(x)

        self.assertTrue(
            torch.allclose(probs_before, probs_after, atol=1e-6),
            f"[{model_type}] Predictions diverged after checkpoint reload.\n"
            f"  max_diff={( probs_before - probs_after).abs().max().item():.2e}",
        )

    def test_hybrid_round_trip(self):
        """Verifies HybridSpectralSpatialViT round-trip checkpoint preserves predictions."""
        self._assert_round_trip(_make_hybrid(), "hybrid")

    def test_cnn3d_round_trip(self):
        """Verifies SpectralSpatial3DCNN round-trip checkpoint preserves predictions."""
        self._assert_round_trip(_make_cnn3d(), "cnn3d")


class TestCheckpointMetadataDefaults(unittest.TestCase):
    """Verifies that omitting metadata stores an empty dict."""

    def test_no_metadata_stores_empty_dict(self):
        """When metadata=None, loaded metadata must be an empty dict."""
        model = _make_cnn3d()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "no_meta.pt"
            save_model_checkpoint(model, path, model_type="cnn3d")
            _, meta = load_model_checkpoint(path)

        self.assertIsInstance(meta, dict)
        self.assertEqual(len(meta), 0, "Metadata should be empty when not provided.")

    def test_nested_subdirectory_creation(self):
        """Verifies parent directories are auto-created if missing."""
        model = _make_cnn3d()

        with tempfile.TemporaryDirectory() as tmp_dir:
            deep_path = Path(tmp_dir) / "deep" / "nested" / "dir" / "ckpt.pt"
            save_model_checkpoint(model, deep_path, model_type="cnn3d")
            self.assertTrue(deep_path.is_file(), "Checkpoint should be created in nested dir.")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestCheckpointErrorHandling(unittest.TestCase):
    """Verifies correct exception types and messages on bad inputs."""

    def test_load_missing_file_raises_file_not_found(self):
        """Verifies FileNotFoundError for a non-existent checkpoint path."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_model_checkpoint("/nonexistent/path/model.pt")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_save_invalid_model_type_raises_value_error(self):
        """Verifies ValueError when saving with an unknown model_type."""
        model = _make_cnn3d()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.pt"
            with self.assertRaises(ValueError) as ctx:
                save_model_checkpoint(model, path, model_type="transformer_xl")
        self.assertIn("transformer_xl", str(ctx.exception))

    def test_load_invalid_model_type_in_checkpoint_raises_value_error(self):
        """Verifies ValueError when checkpoint contains an unknown model_type."""
        import torch as _torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_ckpt = Path(tmp_dir) / "corrupt.pt"
            # Manually craft a checkpoint with a bogus model_type
            _torch.save(
                {
                    "model_type": "megaformer",
                    "state_dict": {},
                    "metadata": {},
                },
                bad_ckpt,
            )
            with self.assertRaises(ValueError) as ctx:
                load_model_checkpoint(bad_ckpt)
        self.assertIn("megaformer", str(ctx.exception))


# ---------------------------------------------------------------------------
# FastAPI /api/v1/models endpoint test
# ---------------------------------------------------------------------------

class TestModelsEndpoint(unittest.TestCase):
    """Integration test for GET /api/v1/models."""

    def setUp(self):
        """Initialise FastAPI TestClient."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        self.client = TestClient(app)

    def test_models_endpoint_returns_200(self):
        """Verifies GET /api/v1/models returns HTTP 200."""
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200, response.text)

    def test_models_endpoint_contains_expected_keys(self):
        """Verifies the response contains architectures, default_pca_components, and checkpoint_dir."""
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("architectures", data, "Response must include 'architectures'.")
        self.assertIn("default_pca_components", data, "Response must include 'default_pca_components'.")
        self.assertIn("checkpoint_dir_exists", data, "Response must include 'checkpoint_dir_exists'.")

    def test_models_endpoint_architectures_values(self):
        """Verifies architectures list contains 'hybrid' and 'cnn3d'."""
        response = self.client.get("/api/v1/models")
        data = response.json()
        archs = data["architectures"]
        self.assertIn("hybrid", archs)
        self.assertIn("cnn3d", archs)

    def test_models_endpoint_pca_components_is_int(self):
        """Verifies default_pca_components is a positive integer."""
        response = self.client.get("/api/v1/models")
        data = response.json()
        self.assertIsInstance(data["default_pca_components"], int)
        self.assertGreater(data["default_pca_components"], 0)


if __name__ == "__main__":
    unittest.main()
