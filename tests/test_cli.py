"""Unit tests for src.cli — TerraSpectra command-line interface."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.cli import main


class TestCLIHybridModel(unittest.TestCase):
    """Tests for the Hybrid (3D-CNN + ViT) model CLI execution."""

    def test_hybrid_exit_code_zero(self):
        """Verifies that the CLI exits cleanly (code 0) with valid hybrid args."""
        code = main(["--height", "32", "--width", "32", "--model", "hybrid"])
        self.assertEqual(code, 0, "CLI should return exit code 0 on success.")

    def test_hybrid_json_output_keys(self):
        """Verifies JSON output file is written with all expected health_summary keys."""
        expected_keys = {
            "total_pixels",
            "infected_pixels",
            "infected_percentage",
            "infected_area_hectares",
            "mean_disease_probability",
            "mean_ndvi_healthy",
            "mean_ndvi_infected",
            "severity_level",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            code = main(
                [
                    "--height", "32",
                    "--width", "32",
                    "--bands", "30",
                    "--pca-components", "10",
                    "--model", "hybrid",
                    "--threshold", "0.5",
                    "--seed", "7",
                    "--output", tmp_path,
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(tmp_path), "Output JSON file should exist.")

            with open(tmp_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            self.assertIsInstance(data, dict, "JSON output must be a dict.")
            for key in expected_keys:
                self.assertIn(key, data, f"Missing expected key: '{key}'")

            # Sanity-check value ranges
            self.assertGreaterEqual(data["total_pixels"], 1)
            self.assertGreaterEqual(data["infected_pixels"], 0)
            self.assertGreaterEqual(data["infected_percentage"], 0.0)
            self.assertLessEqual(data["infected_percentage"], 100.0)
            self.assertIn(
                data["severity_level"],
                ["Low", "Moderate", "High"],
                "severity_level must be one of Low / Moderate / High.",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_hybrid_total_pixels_match_dimensions(self):
        """Verifies that total_pixels == height * width in the JSON output."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            code = main(
                [
                    "--height", "32",
                    "--width", "32",
                    "--bands", "30",
                    "--pca-components", "10",
                    "--model", "hybrid",
                    "--output", tmp_path,
                ]
            )
            self.assertEqual(code, 0)
            with open(tmp_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(
                data["total_pixels"],
                32 * 32,
                "total_pixels must equal height * width.",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCLICNN3DModel(unittest.TestCase):
    """Tests for the SpectralSpatial3DCNN model CLI execution."""

    def test_cnn3d_exit_code_zero(self):
        """Verifies the CLI exits cleanly (code 0) with cnn3d model."""
        code = main(["--height", "32", "--width", "32", "--model", "cnn3d"])
        self.assertEqual(code, 0, "CLI should return exit code 0 for cnn3d model.")

    def test_cnn3d_json_output(self):
        """Verifies JSON output file for cnn3d model contains valid health summary."""
        expected_keys = {
            "total_pixels",
            "infected_pixels",
            "infected_percentage",
            "infected_area_hectares",
            "mean_disease_probability",
            "severity_level",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            code = main(
                [
                    "--height", "32",
                    "--width", "32",
                    "--bands", "30",
                    "--pca-components", "10",
                    "--model", "cnn3d",
                    "--threshold", "0.4",
                    "--seed", "123",
                    "--output", tmp_path,
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(tmp_path))

            with open(tmp_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            for key in expected_keys:
                self.assertIn(key, data, f"Missing key: '{key}'")

            self.assertGreaterEqual(data["total_pixels"], 1)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCLIValidation(unittest.TestCase):
    """Tests for CLI argument validation."""

    def test_invalid_height_returns_1(self):
        """Verifies exit code 1 when --height is below minimum (16)."""
        code = main(["--height", "8", "--width", "32"])
        self.assertEqual(code, 1, "Should reject height < 16.")

    def test_invalid_width_returns_1(self):
        """Verifies exit code 1 when --width is below minimum (16)."""
        code = main(["--height", "32", "--width", "8"])
        self.assertEqual(code, 1, "Should reject width < 16.")

    def test_invalid_threshold_returns_1(self):
        """Verifies exit code 1 when --threshold is out of [0.0, 1.0]."""
        code = main(["--height", "32", "--width", "32", "--threshold", "1.5"])
        self.assertEqual(code, 1, "Should reject threshold > 1.0.")

    def test_invalid_pca_components_too_large(self):
        """Verifies exit code 1 when --pca-components exceeds --bands."""
        code = main(
            [
                "--height", "32",
                "--width", "32",
                "--bands", "20",
                "--pca-components", "25",
            ]
        )
        self.assertEqual(code, 1, "Should reject pca_components > bands.")


if __name__ == "__main__":
    unittest.main()
