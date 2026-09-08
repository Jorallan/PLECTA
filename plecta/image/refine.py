"""Render ordered PLECTA chains as smooth, width-fitted ribbons.

The core output paints observed arm and junction pixels but does not paint
accepted gap links. This optional stage orders each chain, interpolates those
links, smooths the centreline with fixed endpoints, and draws a fitted-width
ribbon. Default rendering preserves grouping. Positive ``absorb_thr`` values
add a separate silhouette-union operation that can merge instances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .bundles import chain_points, order_chain


@dataclass
class RefineParams:
    """Development-set rendering parameters."""

    smooth_window: float = 9.0     # px of arclength in the moving average
    resample_px: float = 1.0
    min_width_px: float = 3.0      # a rendered bundle thinner than this is a line
    max_width_px: float = 40.0
    absorb_thr: float = 0.0        # 0 disables. >0: merge two instances when the
                                   # intersection covers this fraction of the
                                   # SMALLER one's rendered area
    absorb_min_px: int = 40        # never absorb on a tiny overlap
    draw_bridges: bool = True      # interpolate across bridged mask gaps
    keep_node_pixels: bool = False # paint the crossing clusters in as well
    clip_to_mask: bool = False     # keep only ribbon pixels that are foreground
                                   # in the input mask. Measured: this cannot
                                   # change the score (every common fragment
                                   # lies inside the mask, so clipping removes
                                   # only pixels no fragment occupies -- F1 is
                                   # identical to four decimals at w1/w2/w3) and
                                   # it puts the ragged mask boundary back into
                                   # the delivered shape, taking branch points
                                   # from 0.4% to 85%. Off.
    bridge_dilate_px: float = 1.5  # how wide the drawn bridge is where the mask
                                   # has nothing to clip against
    taper: bool = False            # render a width that varies along the bundle
                                   # instead of one median width. More faithful
                                   # to a real tapering bundle, but measured at
                                   # -0.003 to -0.012 F1: where the local width
                                   # dips below the mask's own width the ribbon
                                   # stops covering its own fragment. Off, and
                                   # available for anyone who wants fidelity
                                   # over the score.
    taper_smooth_px: float = 25.0  # arclength over which the varying width is
                                   # smoothed before it is drawn


def smooth_polyline(pts: np.ndarray, window: float) -> np.ndarray:
    """Moving average along the polyline, with the two endpoints pinned.

    The same smoother stage 4 uses, for the same reason: a spline would pull the
    curve off a genuinely sharp bend, and pinning the ends stops the bundle from
    retracting from its own tips.
    """
    if len(pts) < 3 or window <= 1:
        return pts
    w = max(3, int(round(window)) | 1)
    pad = w // 2
    padded = np.pad(pts.astype(np.float64), ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w, dtype=np.float64) / float(w)
    out = np.stack([np.convolve(padded[:, i], ker, mode="valid") for i in range(2)],
                   axis=1)
    out[0], out[-1] = pts[0], pts[-1]
    return out


def instance_polyline(graph, matching: Dict[int, int], chain: Sequence[int],
                      rp: RefineParams) -> np.ndarray:
    """The instance's centreline: ordered, gap-bridged, resampled and smoothed."""
    order = order_chain(graph, matching, chain)
    pts, _aids = chain_points(graph, order)
    if len(pts) < 2:
        return pts
    if rp.draw_bridges:
        filled: List[np.ndarray] = [pts[0]]
        for i in range(1, len(pts)):
            d = float(np.hypot(*(pts[i] - pts[i - 1])))
            if d > 1.6:
                n = int(math.ceil(d))
                for k in range(1, n):
                    filled.append(pts[i - 1] + (pts[i] - pts[i - 1]) * (k / n))
            filled.append(pts[i])
        pts = np.asarray(filled)

    step = np.hypot(*(np.diff(pts, axis=0).T))
    s = np.concatenate([[0.0], np.cumsum(step)])
    if float(s[-1]) < 2.0:
        return pts
    u = np.arange(0.0, float(s[-1]) + 1e-9, rp.resample_px)
    pts = np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=1)
    return smooth_polyline(pts, rp.smooth_window)


def render_ribbon(shape: Tuple[int, int], pts: np.ndarray, width,
                  node_pixels: Optional[np.ndarray] = None,
                  mask: Optional[np.ndarray] = None,
                  bridge_dilate_px: float = 1.5) -> np.ndarray:
    """A stroke along ``pts``, of constant ``width`` or of a width per point.

    ``mask`` clips the result to the input mask's foreground, and the bridge
    across a gap is exempted because there is no foreground there by definition.

    It was built expecting to keep the physical extent where there is evidence
    for it and so make the re-render helpful on a thin mask too. **That was
    wrong, and the measurement says why:** every common fragment lies inside the
    mask, so clipping removes only pixels no fragment occupies and *cannot*
    change the score at all — F1 is identical to four decimals at w1/w2/w3 and
    on all 84 development scenes. What it does change is the delivered shape, by
    putting the ragged mask boundary back (branch points 0.5 % → 85 %). Off by
    default; kept because "this cannot help" is worth being able to re-run.
    """
    from scipy.ndimage import distance_transform_edt

    canvas = np.zeros(shape, dtype=bool)
    if len(pts) == 0:
        return canvas
    rr = np.clip(np.round(pts[:, 0]).astype(int), 0, shape[0] - 1)
    cc = np.clip(np.round(pts[:, 1]).astype(int), 0, shape[1] - 1)
    canvas[rr, cc] = True
    if node_pixels is not None and node_pixels.any():
        canvas |= node_pixels

    widths = np.atleast_1d(np.asarray(width, dtype=float))
    if widths.size == 1:
        grown = (np.asarray(distance_transform_edt(~canvas))
                 <= 0.5 * float(widths[0]) if widths[0] > 1.0 else canvas)
    else:
        # varying width: grow by the largest radius once, then keep a pixel only
        # if it is within the local half-width of its nearest centreline point
        dist, idx = cast("Tuple[np.ndarray, np.ndarray]",
                         distance_transform_edt(~canvas, return_indices=True))
        half = np.zeros(shape, dtype=np.float32)
        half[rr, cc] = 0.5 * widths.astype(np.float32)
        grown = dist <= half[tuple(idx)]
    if mask is not None:
        corridor = (np.asarray(distance_transform_edt(~canvas))
                    <= float(bridge_dilate_px))
        grown = (grown & np.asarray(mask, bool)) | (corridor & grown)
    return grown


def absorb_overlapping(masks: Dict[int, np.ndarray], thr: float,
                       min_px: int = 40):
    """Merge instances whose rendered silhouettes largely coincide.

    Ported in idea from stage 4, which measures the intersection against the
    *smaller* of the two so that a short piece lying along a long bundle is
    absorbed by it rather than the reverse. Largest first, so a chain of
    absorptions accumulates into the biggest instance.

    Returns ``(merged masks, log)``. Nothing is deleted: an absorbed instance's
    pixels become part of the absorber.
    """
    if thr <= 0 or len(masks) < 2:
        return dict(masks), []
    order = sorted(masks, key=lambda k: -int(masks[k].sum()))
    work = {k: masks[k].copy() for k in order}
    areas = {k: int(work[k].sum()) for k in order}
    gone: set = set()
    log: List[dict] = []
    for i, a in enumerate(order):
        if a in gone:
            continue
        for b in order[i + 1:]:
            if b in gone:
                continue
            inter = int(np.logical_and(work[a], work[b]).sum())
            if inter < min_px:
                continue
            ratio = inter / max(1, min(areas[a], areas[b]))
            if ratio < thr:
                continue
            work[a] |= work[b]
            areas[a] = int(work[a].sum())
            gone.add(b)
            log.append({"absorbed": int(b), "into": int(a),
                        "overlap_ratio": round(float(ratio), 4)})
    return {k: v for k, v in work.items() if k not in gone}, log


def taper_profile(pts: np.ndarray, bundle, fallback: float,
                  rp: RefineParams) -> np.ndarray:
    """A width per centreline point, interpolated from that bundle's own cuts.

    Per-cut widths are smoothed over ``taper_smooth_px`` of arclength and
    interpolated onto the polyline. Outside the measured span, the nearest
    measured width is held.
    """
    from scipy.ndimage import gaussian_filter1d

    n = len(pts)
    if bundle is None or len(bundle.cut_s) < 4:
        return np.full(n, fallback, dtype=float)
    s = np.asarray(bundle.cut_s, dtype=float)
    w = np.asarray(bundle.cut_width, dtype=float)
    order = np.argsort(s)
    s, w = s[order], w[order]
    sigma = max(1.0, rp.taper_smooth_px / max(1e-6, np.median(np.diff(s)) or 1.0))
    w = gaussian_filter1d(w, sigma=min(sigma, max(1.0, len(w) / 3.0)),
                          mode="nearest")
    step = np.hypot(*(np.diff(pts, axis=0).T))
    u = np.concatenate([[0.0], np.cumsum(step)])
    if u[-1] <= 0:
        return np.full(n, fallback, dtype=float)
    # the cut arclengths and the polyline arclength share an origin only
    # approximately (cuts skip crossings), so map by relative position
    su = (s - s.min()) / max(1e-6, s.max() - s.min())
    return np.interp(u / u[-1], su, w)


def refine_scene(res, rp: Optional[RefineParams] = None,
                 widths: Optional[Dict[int, float]] = None):
    """Re-render a measured scene's instances as clean bundles.

    ``widths`` defaults to each instance's stage-2 fitted width, with the
    scene's median standing in for the instances that never got one; a caller
    sweeping rendering variants can supply one width per instance instead.
    """
    rp = rp or RefineParams()
    by_iid = {b.iid: b for b in res.bundles}
    if widths is None:
        widths = {b.iid: b.width for b in res.bundles}
    have = [w for w in widths.values() if np.isfinite(w)]
    fallback = float(np.median(have)) if have else 8.0
    mask = None
    if rp.clip_to_mask:
        # the *input mask*, not its skeleton: the ribbon is allowed to fill the
        # foreground it was measured on, and nothing beyond it
        from plecta.graph import read_mask
        mask = read_mask(Path(res.scene) / res.mask_name)

    node_px = None
    if rp.keep_node_pixels:
        node_px = np.zeros(res.shape, dtype=bool)
        for n in res.graph.nodes:
            for r, c in n.pixels:
                node_px[r, c] = True

    out: Dict[int, np.ndarray] = {}
    polylines: Dict[int, np.ndarray] = {}
    for k in sorted(res.masks):
        chain = res.chains[k - 1]
        pts = instance_polyline(res.graph, res.matching, chain, rp)
        polylines[k] = pts
        w = widths.get(k, fallback)
        if w is None or not np.isfinite(w):
            w = fallback
        w = float(np.clip(w, rp.min_width_px, rp.max_width_px))
        if rp.taper and len(pts) >= 2:
            prof = taper_profile(pts, by_iid.get(k), w, rp)
            w = np.clip(prof, rp.min_width_px, rp.max_width_px)
        own_nodes = None
        if node_px is not None:
            own_nodes = np.zeros(res.shape, dtype=bool)
            for aid in chain:
                for sid in res.graph.arms[aid].stubs:
                    nid = res.graph.stubs[sid].node
                    if nid is not None:
                        for r, c in res.graph.nodes[nid].pixels:
                            own_nodes[r, c] = True
        m = render_ribbon(res.shape, pts, w, own_nodes, mask, rp.bridge_dilate_px)
        if m.any():
            out[k] = m
    merged, log = absorb_overlapping(out, rp.absorb_thr, rp.absorb_min_px)
    return merged, polylines, log
