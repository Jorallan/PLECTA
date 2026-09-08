"""Common-fragment scoring: the metric behind every number PLECTA reports.

Used for benchmark evaluation and configuration tuning with `plecta.tune`.

Why fragments rather than pixels
--------------------------------
The unit of comparison is derived from the INPUT MASK alone, never from any
method's output: skeletonize, prune skeletonization spurs, delete junction
pixels, and label what remains. Each surviving piece is an "atomic branch" that
no method is allowed to split, so two methods are compared on the same units and
the score cannot be gamed by carving the mask differently.

The score is then permutation-invariant and pairwise: for every pair of
fragments, do the ground truth and the prediction agree about whether they
belong to the same instance? That question needs no correspondence between
instance ids, which is what makes it usable when a method invents its own
numbering and when instances overlap.

    TP = sum_ij C(n_ij, 2)            n_ij = fragments in gt-group i AND pred-group j
    FP = sum_j C(b_j, 2) - TP         b_j  = pred-group sizes
    FN = sum_i C(a_i, 2) - TP         a_i  = gt-group sizes

A fragment either side leaves unassigned becomes its own singleton cluster
rather than joining a shared "cluster 0" -- lumping them would manufacture
agreement between fragments that nothing actually joined.

Validation
----------
`tests/test_score.py` checks constructed cases and, when evaluation data is
available, the per-scene values recorded in `evidence/heldout_per_scene.json`.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

Instances = Union[Mapping[int, np.ndarray], np.ndarray]

__all__ = ["common_fragments", "assign_fragments", "pairwise_scores", "score"]


# ── fragments: derived from the mask only ─────────────────────────────────


def _degree_map(skel: np.ndarray) -> np.ndarray:
    """8-neighbour count for every skeleton pixel (0 elsewhere)."""
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0) * skel


def _prune_spurs(skel: np.ndarray, prune_spur_px: int) -> np.ndarray:
    """Drop short dead-end hairs so they cannot fake a junction.

    A spur is a piece of skeleton (junction pixels excluded) that touches an
    endpoint at one end, is 8-adjacent to a junction at the other, and is no
    longer than `prune_spur_px`. Iterated, because removing one spur can drop a
    junction's degree and expose a new endpoint.
    """
    from scipy.ndimage import binary_dilation
    from skimage.measure import label as cc_label

    skel = np.asarray(skel).copy()
    if prune_spur_px <= 0:
        return skel
    struct = np.ones((3, 3), dtype=bool)
    for _ in range(20):
        deg = _degree_map(skel)
        junctions = skel & (deg >= 3)
        endpoints = skel & (deg == 1)
        if not junctions.any():
            break
        lbl = np.asarray(cc_label(skel & ~junctions, connectivity=2))
        n = int(lbl.max())
        if n == 0:
            break
        junc_dilated = binary_dilation(junctions, structure=struct)
        removed = False
        for k in range(1, n + 1):
            piece = lbl == k
            if int(piece.sum()) > prune_spur_px:
                continue
            if (piece & junc_dilated).any() and (piece & endpoints).any():
                skel[piece] = False
                removed = True
        if not removed:
            break
    return skel


def common_fragments(mask: np.ndarray, min_len_px: int = 6,
                     prune_spur_px: int = 3) -> np.ndarray:
    """Atomic reference fragments, from the mask and nothing else.

    Depends solely on `mask`, so every method being compared is measured
    against identical units. Returns an int32 label image: 0 background,
    1..K fragment ids.
    """
    from skimage.measure import label as cc_label
    from skimage.morphology import skeletonize

    mask = np.asarray(mask).astype(bool)
    skel = _prune_spurs(skeletonize(mask), prune_spur_px)
    deg = _degree_map(skel)
    lbl = np.asarray(cc_label(skel & ~(skel & (deg >= 3)), connectivity=2))
    out = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for k in range(1, int(lbl.max()) + 1):
        piece = lbl == k
        if int(piece.sum()) >= min_len_px:
            out[piece] = next_id
            next_id += 1
    return out


# ── assignment ────────────────────────────────────────────────────────────


def _as_masks(instances: Instances) -> Dict[int, np.ndarray]:
    """Normalise any accepted instance representation to {id: bool mask}."""
    if isinstance(instances, Mapping):
        return {int(k): np.asarray(v).astype(bool) for k, v in instances.items()}
    if hasattr(instances, "masks"):
        return {int(k): np.asarray(v).astype(bool)
                for k, v in getattr(instances, "masks").items()}
    arr = np.asarray(instances)
    if arr.ndim == 3:                      # (K, H, W) stack: id := index + 1
        return {i + 1: arr[i].astype(bool) for i in range(arr.shape[0])}
    return {int(v): (arr == int(v)) for v in np.unique(arr) if int(v) != 0}


def assign_fragments(frag_labels: np.ndarray, instances: Instances,
                     min_overlap_frac: float = 0.3,
                     max_dist_px: float = 6.0) -> Dict[int, int]:
    """Assign each fragment to exactly one instance; 0 means unassigned.

    Voting is against each instance's FULL mask rather than a flattened label
    image, so instances that overlap at a crossing both keep their claim on the
    pixels there -- flattening first would silently hand every crossing to
    whichever instance happened to win. Only the distance fallback, used when no
    instance overlaps the fragment enough, needs a flattened field.
    """
    from scipy.ndimage import distance_transform_edt

    frag_labels = np.asarray(frag_labels)
    ids = [int(i) for i in np.unique(frag_labels) if int(i) != 0]
    if not ids:
        return {}
    masks = _as_masks(instances)
    if any(m.shape != frag_labels.shape for m in masks.values()):
        raise ValueError("instance masks must match frag_labels in shape")
    if not masks:
        return {i: 0 for i in ids}

    flat = np.zeros(frag_labels.shape, dtype=np.int64)
    for iid in sorted(masks, key=lambda k: (-int(masks[k].sum()), k)):
        flat[masks[iid] & (flat == 0)] = iid
    occupied = flat != 0
    if occupied.any() and not occupied.all():
        dist, nearest_idx = cast(
            "Tuple[np.ndarray, np.ndarray]",
            distance_transform_edt(~occupied, return_indices=True))
        nearest = flat[tuple(nearest_idx)]
    else:
        dist = np.zeros(flat.shape, dtype=float)
        nearest = flat

    flat_masks = {k: m.ravel() for k, m in masks.items()}
    out: Dict[int, int] = {}
    for fid in ids:
        fragment = frag_labels == fid
        idx = np.flatnonzero(fragment.ravel())
        best_id, best_count = 0, 0
        for iid in sorted(masks):
            count = int(flat_masks[iid][idx].sum())
            if count > best_count:
                best_id, best_count = iid, count
        if best_count >= min_overlap_frac * idx.size:
            out[fid] = best_id
            continue
        cand = nearest[fragment & (dist <= max_dist_px)]
        cand = cand[cand != 0]
        if cand.size:
            labels, counts = np.unique(cand, return_counts=True)
            out[fid] = int(labels[np.argmax(counts)])
        else:
            out[fid] = 0
    return out


# ── pairwise co-assignment ────────────────────────────────────────────────


def _singletonize(vals: Sequence[int]) -> np.ndarray:
    """Unassigned fragments get a private negative id, never a shared 0."""
    out = np.empty(len(vals), dtype=np.int64)
    nxt = -1
    for i, v in enumerate(vals):
        if v == 0:
            out[i] = nxt
            nxt -= 1
        else:
            out[i] = v
    return out


def pairwise_scores(gt_assign: Dict[int, int], pred_assign: Dict[int, int]) -> dict:
    """Permutation-invariant pairwise agreement over the common fragments."""
    frag_ids = sorted(set(gt_assign) | set(pred_assign))
    raw_gt = [int(gt_assign.get(k, 0)) for k in frag_ids]
    raw_pred = [int(pred_assign.get(k, 0)) for k in frag_ids]
    n = len(frag_ids)
    nan = float("nan")
    base = {
        "n_fragments": n,
        "n_gt_groups": len({v for v in raw_gt if v != 0}),
        "n_pred_groups": len({v for v in raw_pred if v != 0}),
    }
    if n < 2:
        return {**base, "precision": nan, "recall": nan, "f1": nan,
                "adjusted_rand_index": 1.0 if n == 1 else nan,
                "vi_split_bits": 0.0 if n == 1 else nan,
                "vi_merge_bits": 0.0 if n == 1 else nan,
                "n_gt_pairs": 0.0, "n_pred_pairs": 0.0,
                "tp": 0.0, "fp": 0.0, "fn": 0.0}

    _, gi = np.unique(_singletonize(raw_gt), return_inverse=True)
    _, pi = np.unique(_singletonize(raw_pred), return_inverse=True)
    mat = np.zeros((gi.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(mat, (gi, pi), 1)
    a, b = mat.sum(axis=1), mat.sum(axis=0)

    def comb2(x):
        x = np.asarray(x, dtype=float)
        return x * (x - 1.0) / 2.0

    tp = float(comb2(mat).sum())
    same_gt, same_pred = float(comb2(a).sum()), float(comb2(b).sum())
    fp, fn = same_pred - tp, same_gt - tp
    precision = tp / same_pred if same_pred > 0 else nan
    recall = tp / same_gt if same_gt > 0 else nan
    if np.isfinite(precision) and np.isfinite(recall):
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    else:
        f1 = nan

    total = comb2(n)
    expected = (same_gt * same_pred / total) if total > 0 else 0.0
    denom = 0.5 * (same_gt + same_pred) - expected
    ari = ((tp - expected) / denom) if denom != 0 else 1.0

    nf = float(n)
    p_ij = mat.astype(float) / nf
    p_gt, p_pred = a.astype(float) / nf, b.astype(float) / nf
    with np.errstate(divide="ignore", invalid="ignore"):
        # H(pred | gt) is over-splitting; H(gt | pred) is over-merging.
        split = -np.nansum(np.where(p_ij > 0,
                                    p_ij * np.log2(p_ij / p_gt[:, None]), 0.0))
        merge = -np.nansum(np.where(p_ij > 0,
                                    p_ij * np.log2(p_ij / p_pred[None, :]), 0.0))
    return {**base, "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "adjusted_rand_index": float(ari),
            "vi_split_bits": float(split), "vi_merge_bits": float(merge),
            "n_gt_pairs": same_gt, "n_pred_pairs": same_pred,
            "tp": tp, "fp": fp, "fn": fn}


def score(mask: np.ndarray, gt: Instances, pred: Instances,
          min_len_px: int = 6, prune_spur_px: int = 3,
          min_overlap_frac: float = 0.3, max_dist_px: float = 6.0,
          frag_labels: Optional[np.ndarray] = None) -> dict:
    """Score one scene: build fragments from `mask`, assign both sides, compare.

    `frag_labels` may be passed in when the same mask is scored repeatedly (the
    tuner does this for every candidate configuration): fragments depend only on
    the mask, so recomputing them per candidate is wasted work.
    """
    if frag_labels is None:
        frag_labels = common_fragments(mask, min_len_px, prune_spur_px)
    ga = assign_fragments(frag_labels, gt, min_overlap_frac, max_dist_px)
    pa = assign_fragments(frag_labels, pred, min_overlap_frac, max_dist_px)
    out = pairwise_scores(ga, pa)
    out["n_common_fragments"] = int(frag_labels.max())
    return out
