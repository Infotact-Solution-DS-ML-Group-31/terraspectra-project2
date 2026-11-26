"""Unit tests for checkpoint auto-loading in src.api.main.get_model_for_inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.api.main import get_model_for_inference
from src.models.checkpoint import save_model_checkpoint
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT


class TestCheckpointLoadingHelper(unittest.TestCase):
    """Tests for get_model_for_inference checkpoint discovery and fallback."""

    def test_returns_loaded_true_when_hybrid_checkpoint_exists(self) -> None:
        """Returns (model, True) when checkpoints/hybrid_demo.pt exists and pca_components == 15."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "hybrid_demo.pt"
            model = HybridSpectralSpatialViT(
                in_channels=1,
                cnn_features=32,
                vit_embed_dim=64,
                patch_size=4,
                num_classes=2,
            )
            save_model_checkpoint(model, ckpt_path, model_type="hybrid", metadata={"epoch": 1})

            loaded_model, loaded_from_ckpt = get_model_for_inference(
                model_type="hybrid",
                pca_components=15,
                checkpoint_dir=tmp_dir,
            )
            self.assertTrue(loaded_from_ckpt)
            self.assertIsInstance(loaded_model, HybridSpectralSpatialViT)
            self.assertFalse(loaded_model.training)

    def test_returns_loaded_true_when_cnn3d_checkpoint_exists(self) -> None:
        """Returns (model, True) when checkpoints/cnn3d_demo.pt exists and pca_components == 15."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "cnn3d_demo.pt"
            model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
            save_model_checkpoint(model, ckpt_path, model_type="cnn3d", metadata={"epoch": 1})

            loaded_model, loaded_from_ckpt = get_model_for_inference(
                model_type="cnn3d",
                pca_components=15,
                checkpoint_dir=tmp_dir,
            )
            self.assertTrue(loaded_from_ckpt)
            self.assertIsInstance(loaded_model, SpectralSpatial3DCNN)
            self.assertFalse(loaded_model.training)

    def test_returns_loaded_false_when_checkpoint_dir_empty(self) -> None:
        """Returns (fresh_model, False) when no checkpoint file is present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for model_type, expected_cls in [
                ("hybrid", HybridSpectralSpatialViT),
                ("cnn3d", SpectralSpatial3DCNN),
            ]:
                with self.subTest(model_type=model_type):
                    model, loaded_from_ckpt = get_model_for_inference(
                        model_type=model_type,
                        pca_components=15,
                        checkpoint_dir=tmp_dir,
                    )
                    self.assertFalse(loaded_from_ckpt)
                    self.assertIsInstance(model, expected_cls)
                    self.assertFalse(model.training)

    def test_returns_loaded_false_when_pca_components_mismatch(self) -> None:
        """Returns (fresh_model, False) when pca_components != 15 even if checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "hybrid_demo.pt"
            model = HybridSpectralSpatialViT(
                in_channels=1,
                cnn_features=32,
                vit_embed_dim=64,
                patch_size=4,
                num_classes=2,
            )
            save_model_checkpoint(model, ckpt_path, model_type="hybrid")

            loaded_model, loaded_from_ckpt = get_model_for_inference(
                model_type="hybrid",
                pca_components=10,
                checkpoint_dir=tmp_dir,
            )
            self.assertFalse(loaded_from_ckpt)
            self.assertIsInstance(loaded_model, HybridSpectralSpatialViT)

    def test_invalid_model_type_raises_value_error(self) -> None:
        """Raises ValueError for unsupported model_type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                get_model_for_inference(
                    model_type="unknown_arch",
                    pca_components=15,
                    checkpoint_dir=tmp_dir,
                )


if __name__ == "__main__":
    unittest.main()
