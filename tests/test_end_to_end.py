"""End-to-end integration tests for the full TerraSpectra pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_demo_model import main as train_demo_main
from src.analysis.geospatial import raster_to_geogrid, simulate_temporal_spread
from src.analysis.spectral_index import (
    compute_ndvi,
    compute_red_edge_stress,
    summarize_field_health,
)
from src.api.tiling import TilingConfig, predict_full_raster
from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.dataset import HyperspectralPatchDataset
from src.data.pca_reducer import SpectralPCAReducer
from src.models.checkpoint import load_model_checkpoint, save_model_checkpoint
from src.models.hybrid import HybridSpectralSpatialViT
from src.models.train import TrainConfig, run_training


class TestEndToEndPipeline(unittest.TestCase):
    """Comprehensive test validating the complete TerraSpectra lifecycle end-to-end."""

    def test_full_terraspectra_pipeline(self) -> None:
        """Execute all six stages of the TerraSpectra forecasting pipeline."""
        height = 64
        width = 64
        num_bands = 224
        pca_components = 15
        seed = 42

        # -------------------------------------------------------------
        # Stage 1: Generate 224-band synthetic hyperspectral cube (64x64)
        # -------------------------------------------------------------
        cube, disease_mask = generate_synthetic_hyperspectral_cube(
            height=height,
            width=width,
            num_bands=num_bands,
            disease_zone_radius=8,
            seed=seed,
        )
        self.assertEqual(cube.data.shape, (num_bands, height, width))
        self.assertEqual(disease_mask.shape, (height, width))
        self.assertEqual(len(cube.wavelengths), num_bands)
        self.assertTrue(np.all(cube.data >= 0.0) and np.all(cube.data <= 1.0))
        self.assertGreater(int(np.sum(disease_mask)), 0)

        # -------------------------------------------------------------
        # Stage 2: Biophysical indices (NDVI & Red-Edge Stress)
        # -------------------------------------------------------------
        ndvi_map = compute_ndvi(cube)
        stress_map = compute_red_edge_stress(cube)
        self.assertEqual(ndvi_map.shape, (height, width))
        self.assertEqual(stress_map.shape, (height, width))
        self.assertTrue(np.all(ndvi_map >= -1.0) and np.all(ndvi_map <= 1.0))
        self.assertTrue(np.all(stress_map >= -1.0) and np.all(stress_map <= 1.0))

        # -------------------------------------------------------------
        # Stage 3: Spectral PCA compression (224 bands -> 15 components)
        # -------------------------------------------------------------
        reducer = SpectralPCAReducer(n_components=pca_components, random_state=seed)
        reduced_cube = reducer.fit_transform(cube)
        self.assertEqual(reduced_cube.shape, (pca_components, height, width))
        self.assertGreaterEqual(
            reducer.cumulative_explained_variance_ratio,
            0.95,
            "15 PCA components must preserve >= 95% cumulative explained variance.",
        )
        self.assertTrue(reducer.verify_cumulative_variance(0.95))

        # -------------------------------------------------------------
        # Stage 4: Train Hybrid ViT for 1 epoch & checkpoint save/reload
        # -------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = HyperspectralPatchDataset(
                cube=reduced_cube,
                mask=disease_mask,
                patch_size=16,
                stride=16,
            )
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            model = HybridSpectralSpatialViT(
                in_channels=1,
                cnn_features=32,
                vit_embed_dim=64,
                patch_size=4,
                num_classes=2,
            )

            train_cfg = TrainConfig(
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                device="cpu",
                checkpoint_dir=tmp_dir,
            )
            history = run_training(model, dataloader, config=train_cfg)
            self.assertEqual(len(history["train_loss"]), 1)
            self.assertGreater(history["train_loss"][0], 0.0)

            # Save model checkpoint
            ckpt_path = Path(tmp_dir) / "hybrid_pipeline.pt"
            save_model_checkpoint(
                model=model,
                path=ckpt_path,
                model_type="hybrid",
                metadata={"test_run": True, "epoch": 1, "loss": history["train_loss"][0]},
            )
            self.assertTrue(ckpt_path.is_file())

            # Reload model checkpoint
            loaded_model, metadata = load_model_checkpoint(
                path=ckpt_path,
                in_channels=pca_components,
                device="cpu",
            )
            self.assertIsInstance(loaded_model, HybridSpectralSpatialViT)
            self.assertEqual(metadata.get("epoch"), 1)
            self.assertIn("loss", metadata)

        # -------------------------------------------------------------
        # Stage 5: Overlapping spatial tiled raster inference
        # -------------------------------------------------------------
        tiling_cfg = TilingConfig(patch_size=16, stride=8, batch_size=8, device="cpu")
        prob_map = predict_full_raster(loaded_model, reduced_cube, config=tiling_cfg)
        self.assertEqual(prob_map.shape, (height, width))
        self.assertTrue(np.all(prob_map >= 0.0) and np.all(prob_map <= 1.0))

        # -------------------------------------------------------------
        # Stage 6: Field health summarization & 3D geospatial / temporal
        # -------------------------------------------------------------
        health_summary = summarize_field_health(
            prob_map=prob_map,
            ndvi_map=ndvi_map,
            threshold=0.5,
            pixel_area_sq_m=1.0,
        )
        self.assertEqual(health_summary["total_pixels"], height * width)
        self.assertIn("severity_level", health_summary)
        self.assertIn(health_summary["severity_level"], ["Low", "Moderate", "High"])
        self.assertIn("infected_area_hectares", health_summary)

        # Geogrid conversion for 3D WebGL column rendering
        geogrid = raster_to_geogrid(
            prob_map=prob_map,
            ndvi_map=ndvi_map,
            stress_map=stress_map,
            threshold=0.5,
        )
        self.assertEqual(len(geogrid), height * width)
        first_record = geogrid[0]
        for key in ("lat", "lon", "probability", "ndvi", "stress", "elevation", "risk_label", "color_rgba"):
            self.assertIn(key, first_record, f"Missing key '{key}' in geogrid record.")
        self.assertEqual(len(first_record["color_rgba"]), 4)

        # Temporal disease spread simulation across forecast horizons
        temporal_frames = simulate_temporal_spread(
            base_prob_map=prob_map,
            days=[0, 3, 7, 14],
            growth_rate=0.06,
        )
        self.assertEqual(list(temporal_frames.keys()), [0, 3, 7, 14])
        for day in [0, 3, 7, 14]:
            self.assertEqual(temporal_frames[day].shape, (height, width))
            self.assertTrue(np.all(temporal_frames[day] >= 0.0) and np.all(temporal_frames[day] <= 1.0))

    def test_train_demo_script_execution(self) -> None:
        """Verify train_demo_model script executes and outputs clean metrics JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = Path(tmp_dir) / "demo_metrics.json"
            exit_code = train_demo_main(
                [
                    "--model", "hybrid",
                    "--epochs", "1",
                    "--pca-components", "15",
                    "--batch-size", "4",
                    "--checkpoint-dir", tmp_dir,
                    "--output", str(out_json),
                    "--seed", "42",
                    "--eval-seed", "99",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_json.is_file(), "Metrics output file should exist.")

            with open(out_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            for key in ("iou", "dice", "precision", "recall", "pixel_accuracy"):
                self.assertIn(key, metrics)
                self.assertIsInstance(metrics[key], float)
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)


if __name__ == "__main__":
    unittest.main()
