"""Contract tests for the versioned PLECTA core."""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict

import numpy as np

from plecta.predict import PARAMS_PATH, load_params, predict


class CoreContractTests(unittest.TestCase):
    def test_json_materializes_every_effective_parameter(self) -> None:
        recorded = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(recorded, asdict(load_params()))

    def test_prediction_is_deterministic(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[16, 6:26] = True
        first = predict(mask)
        second = predict(mask)
        self.assertEqual(list(first), list(second))
        self.assertGreaterEqual(len(first), 1)
        for key in first:
            self.assertEqual(first[key].dtype, np.bool_)
            np.testing.assert_array_equal(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
