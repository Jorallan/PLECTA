"""Numerical contracts for image-derived width measurement."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from skimage import io as skio

from plecta.image.measurement import CutParams, measure_cut, read_gray


class ImageMeasurementTests(unittest.TestCase):
    def test_read_gray_normalizes_uint8(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        image[:, 4:] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            skio.imsave(path, image, check_contrast=False)
            result = read_gray(path)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(result.min()), 0.0)
        self.assertEqual(float(result.max()), 1.0)

    def test_measure_cut_recovers_flat_ridge_fwhm(self) -> None:
        image = np.zeros((33, 33), dtype=np.float32)
        image[12:21, :] = 1.0
        params = CutParams(half_len=12.0, min_reach=5.0)
        result = measure_cut(
            image,
            background=0.0,
            noise=0.0,
            centre=np.array([16.0, 16.0]),
            normal=np.array([1.0, 0.0]),
            params=params,
        )
        self.assertTrue(result.ok, result.reason)
        self.assertAlmostEqual(result.width, 9.0, places=6)
        self.assertAlmostEqual(result.offset, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
