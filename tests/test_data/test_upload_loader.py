"""Unit tests for upload byte-buffer loading in src.data.cube_loader."""

from __future__ import annotations

import io
import unittest

import numpy as np

from src.data.cube_loader import HyperspectralCube, load_cube_from_bytes


class TestUploadCubeLoader(unittest.TestCase):
    """Tests for load_cube_from_bytes supporting .npy and .npz payloads."""

    def test_load_valid_npy_cube(self) -> None:
        """Loads a valid (15, 32, 32) .npy byte buffer into HyperspectralCube."""
        arr = np.random.default_rng(42).uniform(0.0, 1.0, size=(15, 32, 32)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)

        cube = load_cube_from_bytes(buf.getvalue(), filename="field_sample.npy")
        self.assertIsInstance(cube, HyperspectralCube)
        self.assertEqual(cube.shape, (15, 32, 32))
        self.assertEqual(len(cube.wavelengths), 15)
        self.assertAlmostEqual(float(cube.wavelengths[0]), 400.0, places=3)
        self.assertAlmostEqual(float(cube.wavelengths[-1]), 2500.0, places=3)
        np.testing.assert_allclose(cube.data, arr, rtol=1e-6)

    def test_load_valid_npz_with_wavelengths(self) -> None:
        """Loads a .npz buffer containing 'data' and 'wavelengths_nm'."""
        arr = np.random.default_rng(7).uniform(0.0, 1.0, size=(20, 32, 32)).astype(np.float32)
        wl = np.linspace(450.0, 2400.0, 20, dtype=np.float32)
        buf = io.BytesIO()
        np.savez(buf, data=arr, wavelengths_nm=wl)

        cube = load_cube_from_bytes(buf.getvalue(), filename="hyperion_tile.npz")
        self.assertIsInstance(cube, HyperspectralCube)
        self.assertEqual(cube.shape, (20, 32, 32))
        np.testing.assert_allclose(cube.wavelengths, wl, rtol=1e-6)
        np.testing.assert_allclose(cube.data, arr, rtol=1e-6)

    def test_load_valid_npz_without_wavelengths_uses_defaults(self) -> None:
        """Loads a .npz buffer without wavelengths_nm and generates default [400, 2500] nm."""
        arr = np.zeros((16, 24, 24), dtype=np.float32)
        buf = io.BytesIO()
        np.savez(buf, data=arr)

        cube = load_cube_from_bytes(buf.getvalue(), filename="no_wl.npz")
        self.assertEqual(cube.shape, (16, 24, 24))
        self.assertEqual(len(cube.wavelengths), 16)
        self.assertAlmostEqual(float(cube.wavelengths[0]), 400.0, places=3)
        self.assertAlmostEqual(float(cube.wavelengths[-1]), 2500.0, places=3)

    def test_invalid_extension_raises_value_error(self) -> None:
        """Raises ValueError on unsupported file extension such as .txt."""
        with self.assertRaises(ValueError):
            load_cube_from_bytes(b"not a numpy file", filename="notes.txt")

    def test_wrong_tensor_rank_raises_value_error(self) -> None:
        """Raises ValueError when uploaded .npy array is 2D instead of 3D."""
        arr_2d = np.zeros((32, 32), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr_2d, allow_pickle=False)

        with self.assertRaises(ValueError):
            load_cube_from_bytes(buf.getvalue(), filename="flat.npy")

    def test_npz_missing_data_key_raises_value_error(self) -> None:
        """Raises ValueError when .npz archive lacks 'data' key."""
        buf = io.BytesIO()
        np.savez(buf, other=np.zeros((15, 32, 32), dtype=np.float32))

        with self.assertRaises(ValueError):
            load_cube_from_bytes(buf.getvalue(), filename="missing_data.npz")


if __name__ == "__main__":
    unittest.main()
