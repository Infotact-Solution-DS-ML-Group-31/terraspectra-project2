"""Unit tests for segmentation evaluation metrics and training pipeline."""

import math
import os
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.dataset import HyperspectralPatchDataset
from src.models.hybrid import HybridSpectralSpatialViT
from src.models.metrics import (
    compute_confusion_counts,
    dice_score,
    evaluate_segmentation,
    iou_score,
)
from src.models.train import TrainConfig, evaluate_model, run_training, train_one_epoch


class TestSegmentationMetrics(unittest.TestCase):
    """Test suite for segmentation evaluation metrics."""

    def test_perfect_match_metrics(self) -> None:
        """Verifies IoU, Dice, and evaluation metrics return 1.0 for perfect matches."""
        # 1. Partial positive mask matching exactly
        targets = torch.zeros(2, 16, 16, dtype=torch.long)
        targets[:, 4:12, 4:12] = 1
        preds = targets.clone()

        self.assertEqual(iou_score(preds, targets), 1.0)
        self.assertEqual(dice_score(preds, targets), 1.0)

        metrics = evaluate_segmentation(preds, targets)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["pixel_accuracy"], 1.0)

        # 2. All zeros mask matching exactly
        zeros_mask = torch.zeros(2, 16, 16, dtype=torch.long)
        self.assertEqual(iou_score(zeros_mask, zeros_mask), 1.0)
        self.assertEqual(dice_score(zeros_mask, zeros_mask), 1.0)

    def test_partial_overlap_metrics(self) -> None:
        """Verifies exact analytical values for partial overlaps."""
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        preds = torch.zeros(1, 4, 4, dtype=torch.long)

        # Targets has 4 positives: (0,0), (0,1), (1,0), (1,1)
        targets[0, 0:2, 0:2] = 1
        # Preds has 4 positives: (0,1), (0,2), (1,1), (1,2)
        preds[0, 0:2, 1:3] = 1

        # Intersection (TP) = 2, FP = 2, FN = 2, TN = 10
        tp, fp, tn, fn = compute_confusion_counts(preds, targets)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 2)
        self.assertEqual(tn, 10)
        self.assertEqual(fn, 2)

        # Expected IoU: 2 / (2 + 2 + 2) = 2/6 = 1/3
        expected_iou = 2.0 / 6.0
        self.assertAlmostEqual(iou_score(preds, targets), expected_iou, places=4)

        # Expected Dice: (2 * 2) / (2 * 2 + 2 + 2) = 4/8 = 0.5
        expected_dice = 0.5
        self.assertAlmostEqual(dice_score(preds, targets), expected_dice, places=4)

        metrics = evaluate_segmentation(preds, targets)
        self.assertAlmostEqual(metrics["iou"], expected_iou, places=4)
        self.assertAlmostEqual(metrics["dice"], expected_dice, places=4)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 12.0 / 16.0, places=4)

    def test_logits_input_evaluation(self) -> None:
        """Verifies evaluate_segmentation operates seamlessly on raw (B, 2, H, W) logits."""
        batch_size = 2
        height = 8
        width = 8
        logits = torch.randn(batch_size, 2, height, width, dtype=torch.float32)
        targets = torch.randint(0, 2, (batch_size, height, width), dtype=torch.long)

        metrics = evaluate_segmentation(logits, targets)
        for key in ["iou", "dice", "precision", "recall", "pixel_accuracy"]:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_shape_mismatch_errors(self) -> None:
        """Verifies ValueError is raised when spatial or batch dimensions mismatch."""
        targets = torch.zeros(2, 16, 16, dtype=torch.long)

        # Spatial mismatch
        with self.assertRaises(ValueError):
            iou_score(torch.zeros(2, 8, 8), targets)

        # Batch size mismatch
        with self.assertRaises(ValueError):
            dice_score(torch.zeros(1, 16, 16), targets)

        # Unsupported tensor rank (1D)
        with self.assertRaises(ValueError):
            evaluate_segmentation(torch.zeros(16), targets)


class TestTrainingPipeline(unittest.TestCase):
    """Test suite for training pipeline, epoch routines, and checkpoint persistence."""

    def test_run_training_checkpoint_and_history(self) -> None:
        """Runs 2 training epochs with validation on a synthetic dataset and verifies checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Create a lightweight synthetic dataset
            cube, mask = generate_synthetic_hyperspectral_cube(
                height=32,
                width=32,
                num_bands=15,
                disease_zone_radius=4,
                seed=42,
            )
            dataset = HyperspectralPatchDataset(
                cube=cube.data,
                mask=mask,
                patch_size=16,
            )
            self.assertEqual(len(dataset), 4)

            train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

            # 2. Lightweight model for fast testing
            model = HybridSpectralSpatialViT(
                in_channels=1,
                cnn_features=32,
                vit_embed_dim=32,
                patch_size=4,
                num_classes=2,
                num_vit_layers=1,
                num_vit_heads=2,
            )

            # 3. Training configuration
            config = TrainConfig(
                epochs=2,
                batch_size=2,
                learning_rate=1e-3,
                checkpoint_dir=tmp_dir,
            )

            checkpoint_filename = "test_best_model.pt"
            history = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                save_name=checkpoint_filename,
            )

            # 4. Verify history structure
            self.assertIn("train_loss", history)
            self.assertIn("val_loss", history)
            self.assertIn("val_metrics", history)

            self.assertEqual(len(history["train_loss"]), 2)
            self.assertEqual(len(history["val_loss"]), 2)
            self.assertEqual(len(history["val_metrics"]), 2)

            for loss_val in history["train_loss"]:
                self.assertTrue(math.isfinite(loss_val))
                self.assertGreater(loss_val, 0.0)

            for metric_dict in history["val_metrics"]:
                self.assertIn("iou", metric_dict)
                self.assertIn("dice", metric_dict)
                self.assertIn("pixel_accuracy", metric_dict)

            # 5. Verify checkpoint file on disk and reloadability
            checkpoint_path = os.path.join(tmp_dir, checkpoint_filename)
            self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint file must exist.")

            state_dict = torch.load(checkpoint_path, map_location="cpu")
            eval_model = HybridSpectralSpatialViT(
                in_channels=1,
                cnn_features=32,
                vit_embed_dim=32,
                patch_size=4,
                num_classes=2,
                num_vit_layers=1,
                num_vit_heads=2,
            )
            eval_model.load_state_dict(state_dict)
            eval_model.eval()

            # Execute inference with restored model
            test_x = torch.randn(1, 1, 15, 16, 16, dtype=torch.float32)
            with torch.no_grad():
                out_logits = eval_model(test_x)
                out_probs = eval_model.predict_probability(test_x)

            self.assertEqual(out_logits.shape, (1, 2, 16, 16))
            self.assertEqual(out_probs.shape, (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
