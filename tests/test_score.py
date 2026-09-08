"""The scoring contract: `plecta.score` must mean what the release says it means.

Two kinds of check:

1. Constructed cases where the right answer is known by hand, so the pairwise
   definition itself is pinned -- a perfect grouping scores 1.0, merging two
   filaments costs precision, splitting one costs recall.
2. Reproduction of `evidence/heldout_per_scene.json`. Those rows were produced
   by the benchmark evaluation. Skipped when evaluation data is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plecta.score import common_fragments, score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".local" / "evaluation" / "data"


def _cross(shape=(140, 140)):
    """Two straight rods crossing near-perpendicularly."""
    mask = np.zeros(shape, dtype=bool)
    rod_a = np.zeros(shape, dtype=bool)
    rod_b = np.zeros(shape, dtype=bool)
    rod_a[68:70, 10:130] = True
    rod_b[10:130, 68:70] = True
    mask |= rod_a | rod_b
    return mask, {1: rod_a, 2: rod_b}


def test_perfect_grouping_scores_one():
    mask, gt = _cross()
    s = score(mask, gt, gt)
    assert s["f1"] == pytest.approx(1.0)
    assert s["precision"] == pytest.approx(1.0)
    assert s["recall"] == pytest.approx(1.0)
    assert s["adjusted_rand_index"] == pytest.approx(1.0)


def test_merging_two_filaments_costs_precision_not_recall():
    """One instance covering both rods keeps every true pair and invents more."""
    mask, gt = _cross()
    merged = {1: gt[1] | gt[2]}
    s = score(mask, gt, merged)
    assert s["recall"] == pytest.approx(1.0)     # nothing true was lost
    assert s["precision"] < 1.0                  # but pairs were invented
    assert s["fp"] > 0 and s["fn"] == 0


def test_splitting_a_filament_costs_recall_not_precision():
    """Cutting one rod in two keeps every predicted pair and drops true ones."""
    mask, gt = _cross()
    half = gt[1].copy()
    half[:, 70:] = False
    other = gt[1] & ~half
    s = score(mask, gt, {1: half, 2: other, 3: gt[2]})
    assert s["precision"] == pytest.approx(1.0)  # nothing wrong was joined
    assert s["recall"] < 1.0                     # but true pairs were missed
    assert s["fn"] > 0 and s["fp"] == 0


def test_fragments_depend_only_on_the_mask():
    """The unit of comparison must not depend on any method's output."""
    mask, gt = _cross()
    a = common_fragments(mask)
    b = common_fragments(mask)
    assert np.array_equal(a, b)
    assert int(a.max()) >= 4      # four arms around one crossing


def _recorded_rows():
    path = ROOT / "evidence" / "heldout_per_scene.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["per_scene"]


@pytest.mark.parametrize("row", _recorded_rows()[:6],
                         ids=lambda r: r["scene"].replace("/", "_"))
def test_reproduces_the_published_per_scene_scores(row):
    """This port must agree with the code that produced the release table."""
    scene = DATA / row["scene"]
    mask_path = scene / "mask_w1.png"
    if not mask_path.is_file():
        pytest.skip(f"local evaluation data not present: {scene}")

    from plecta.graph import read_mask
    from plecta.predict import load_params, predict
    from plecta.tune import _load_gt

    mask = read_mask(mask_path)
    gt = _load_gt(scene / "gt_multilabel.npz", mask.shape)
    s = score(mask, gt, predict(mask, load_params()))
    for key in ("f1", "precision", "recall", "adjusted_rand_index"):
        assert s[key] == pytest.approx(row[key], abs=1e-9), key
