"""Measure width and brightness for a PLECTA instance.

A PLECTA instance contains skeleton pixels but no physical width. This module
pairs the instance with the grayscale SEM and estimates width and brightness,
with uncertainty.

The centreline is taken from the chain of arms the linker actually built, not by
re-skeletonising the instance mask: the linker already knows the order the arms
run in and which end joins which, and a matching gives every arm at most one
partner per stub, so the arms of one instance always form a simple path (or, at
worst, a ring). That ordering defines the longitudinal bundle profile.

Cuts are refused near crossings.  A cut through a crossing measures two bundles
stacked on top of each other and reports the union; excluding a small
neighbourhood of every node pixel costs sample size and buys accuracy.  The
exclusion uses the *graph's* nodes, which are built from the mask before any
matching decision is made, so it does not depend on the grouping being right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .measurement import (CutParams, CutResult, SemImage, robust_sigma,
                          measure_cut)


#: Per-bundle systematic width error, in px, by quality class, measured on the
#: 84 development scenes.
#: Successive cuts on one bundle do not have independent errors: the axis
#: offset, the local background and the neighbouring bundles are shared along
#: its whole length, so much of the residual is a constant offset for that
#: bundle that no amount of resampling its own cuts can reveal.  A
#: sampling-only interval is consequently far too narrow -- it covers the truth
#: 36% of the time at a nominal 90% -- and this term is what makes the reported
#: interval mean what it says.  Cross-validated by scene, so the quoted
#: coverage is out-of-sample.
SYSTEMATIC_SIGMA_PX = {"good": 0.119, "weak": 1.740}
Z90 = 1.6448536269514722


@dataclass
class SampleParams:
    """Development-set choices controlling where cuts are placed."""

    cut_step: float = 2.0        # px of arclength between successive cuts
    node_clear_px: float = 6.0   # keep cuts this far from any crossing pixel
    tangent_sigma: float = 3.0   # px, smoothing of the axis before differencing
    join_px: float = 8.0         # consecutive arms closer than this are treated
                                 # as one continuous run (so the tangent is
                                 # continuous across a crossing)
    min_cuts: int = 3            # fewer valid cuts than this -> no width at all
    block: int = 5               # moving-block bootstrap block length, in cuts
    n_boot: int = 2000
    boot_min_cuts: int = 8       # below this the bootstrap is not reported
    scene_width_cap: float = 2.0 # drop a cut wider than this multiple of the
                                 # scene's own median cut width (0 disables).
                                 # Scale-free on purpose: a fixed pixel cap
                                 # would be assuming the answer on a real crop
                                 # whose bundle widths are not known in advance.
                                 # 1.6 scores better on the development set but
                                 # only because 1.6 x the median (17.6 px) sits
                                 # just above the generator's widest bundle
                                 # (16 px) -- that is the synthetic width range
                                 # leaking in, not a property of the method.
    weak_min_cuts: int = 12      # below this, the width is reported but flagged
    weak_max_mad: float = 0.60   # px; above this, likewise


@dataclass
class Bundle:
    """One instance, plus what the SEM says it physically is."""

    iid: int
    n_arms: int
    n_pixels: int
    axis_length: float            # px of arclength actually available to sample
    n_cuts_attempted: int
    n_cuts_valid: int
    width: float = float("nan")          # px, median FWHM
    width_mad: float = float("nan")      # px, robust sigma across cuts
    width_p16: float = float("nan")
    width_p84: float = float("nan")
    n_eff: float = float("nan")          # cuts, corrected for autocorrelation
    sigma_noise: float = float("nan")    # px, per-cut measurement noise
    sigma_real: float = float("nan")     # px, along-bundle width variation that
                                         #   is NOT measurement noise
    width_se: float = float("nan")       # standard error of the median
    width_se_noise: float = float("nan") # ... counting only measurement noise,
                                         #   i.e. treating a real taper as
                                         #   signal rather than as error
    width_ci_lo: float = float("nan")    # 90% block-bootstrap CI of the median
    width_ci_hi: float = float("nan")    #   -- sampling error only
    width_unc_lo: float = float("nan")   # 90% interval INCLUDING the calibrated
    width_unc_hi: float = float("nan")   #   per-bundle systematic term
    brightness: float = float("nan")     # median (peak - background), 0..1
    brightness_p90: float = float("nan")
    brightness_mad: float = float("nan")
    step_abs: float = float("nan")       # largest along-bundle width step, px
    step_rel: float = float("nan")       # ... as a fraction of the median width
    step_at: float = float("nan")        # arclength fraction where it sits
    excursion_z: float = float("nan")    # largest local departure of the width
    excursion_at: float = float("nan")   #   profile from the bundle's own level
    excursion_len: int = 0               #   (window length, in cuts)
    quality: str = "none"                # "good" | "weak" | "none"
    refusals: Dict[str, int] = field(default_factory=dict)
    cut_s: List[float] = field(default_factory=list)
    cut_width: List[float] = field(default_factory=list)
    cut_rc: List[Tuple[float, float]] = field(default_factory=list)
    # the unit normal each cut was taken along, and where the midpoint of its two
    # half-height crossings sat relative to the mask axis.  Kept so a figure can
    # draw the measured width where it was actually measured, at its actual
    # length, instead of re-deriving a direction and getting a different answer.
    cut_n: List[Tuple[float, float]] = field(default_factory=list)
    cut_off: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return math.isfinite(self.width)

    def row(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k not in ("cut_s", "cut_width", "cut_rc", "cut_n", "cut_off",
                             "refusals")}


# ── instance -> ordered centreline ─────────────────────────────────────────


def order_chain(graph, matching: Dict[int, int],
                chain: Sequence[int]) -> List[Tuple[int, int]]:
    """``[(arm id, end it is entered from), ...]`` along the chain.

    A stub matching gives each arm at most two partners (one per end), so the
    arms of a chain form a path or a ring; both are walked the same way.  Any
    arm the walk cannot reach -- which should not happen, and is asserted
    against in the tests -- is appended so that no pixels are silently dropped.
    """
    order: List[Tuple[int, int]] = []
    if not chain:
        return order
    start: Optional[Tuple[int, int]] = None
    for aid in chain:
        s0, s1 = graph.arms[aid].stubs
        if matching.get(s0) is None:
            start = (aid, 0)
            break
        if matching.get(s1) is None:
            start = (aid, 1)
            break
    if start is None:
        start = (int(chain[0]), 0)

    visited = set()
    aid, entry = start
    while aid is not None and aid not in visited:
        visited.add(aid)
        order.append((aid, entry))
        exit_stub = graph.arms[aid].stubs[1 - entry]
        nxt = matching.get(exit_stub)
        if nxt is None:
            break
        stub = graph.stubs[nxt]
        aid, entry = stub.aid, stub.end
    for aid in chain:
        if aid not in visited:
            order.append((int(aid), 0))
    return order


def chain_points(graph, order: Sequence[Tuple[int, int]]):
    """Ordered ``(N, 2)`` float axis points plus the arm id of each."""
    pts: List[Tuple[float, float]] = []
    aids: List[int] = []
    for aid, entry in order:
        path = graph.arms[aid].path
        seq = path if entry == 0 else path[::-1]
        for r, c in seq:
            pts.append((float(r), float(c)))
            aids.append(int(aid))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2), np.asarray(aids, dtype=np.int64)


def split_runs(pts: np.ndarray, join_px: float):
    """Split the ordered points where the axis jumps further than ``join_px``.

    A short jump is a crossing the arms pass through (the node's pixels belong
    to no arm) and is bridged by interpolation, so the tangent stays continuous
    across it.  A long jump is a bridged mask gap, where the real trajectory
    between the two ends is unknown; those start a new run rather than inventing
    an axis through empty mask.
    """
    runs: List[np.ndarray] = []
    if len(pts) == 0:
        return runs
    cur_p: List[np.ndarray] = [pts[0]]
    for i in range(1, len(pts)):
        d = float(np.hypot(*(pts[i] - pts[i - 1])))
        if d > join_px:
            runs.append(np.asarray(cur_p))
            cur_p = [pts[i]]
            continue
        if d > 1.6:                      # bridge a crossing with 1-px steps
            n = int(math.ceil(d))
            for k in range(1, n):
                cur_p.append(pts[i - 1] + (pts[i] - pts[i - 1]) * (k / n))
        cur_p.append(pts[i])
    runs.append(np.asarray(cur_p))
    return runs


def resample_run(pts: np.ndarray, tangent_sigma: float):
    """Uniform 1-px resampling of one run, with a smoothed tangent and normal."""
    from scipy.ndimage import gaussian_filter1d

    if len(pts) < 2:
        return None
    step = np.hypot(*(np.diff(pts, axis=0).T))
    s = np.concatenate([[0.0], np.cumsum(step)])
    total = float(s[-1])
    if total < 2.0:
        return None
    u = np.arange(0.0, total + 1e-9, 1.0)
    r = np.interp(u, s, pts[:, 0])
    c = np.interp(u, s, pts[:, 1])

    sigma = max(0.6, tangent_sigma)
    rs = gaussian_filter1d(r, sigma=sigma, mode="nearest")
    cs = gaussian_filter1d(c, sigma=sigma, mode="nearest")
    dr = np.gradient(rs)
    dc = np.gradient(cs)
    norm = np.hypot(dr, dc)
    good = norm > 1e-9
    dr = np.where(good, dr / np.maximum(norm, 1e-9), 0.0)
    dc = np.where(good, dc / np.maximum(norm, 1e-9), 0.0)
    # normal = tangent rotated by 90 deg
    nr, nc = -dc, dr
    return dict(u=u, r=r, c=c, nr=nr, nc=nc, valid=good)


# ── measurement ────────────────────────────────────────────────────────────


def _block_bootstrap_median_ci(x: np.ndarray, block: int, n_boot: int,
                               rng: np.random.Generator, lo: float = 5.0,
                               hi: float = 95.0) -> Tuple[float, float]:
    """Moving-block bootstrap CI for the median of a *correlated* sample.

    Successive cuts are 2 px apart on the same bundle, so they are anything but
    independent: an i.i.d. bootstrap would report an interval several times too
    narrow.  Resampling contiguous blocks keeps the local correlation intact.
    """
    n = len(x)
    if n < block or block < 1:
        return float("nan"), float("nan")
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, lo)), float(np.percentile(meds, hi))


def _n_eff(x: np.ndarray, max_lag: int = 40) -> float:
    """Effective sample size after the along-bundle autocorrelation.

    Cuts 2 px apart on the same bundle are not independent draws.  The
    integrated autocorrelation time is summed up to the first non-positive
    term (Geyer's initial-positive-sequence rule), which is the standard way to
    truncate the estimator without letting its own noise accumulate.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return float(n)
    y = x - x.mean()
    denom = float((y * y).sum())
    if denom <= 0:
        return float(n)
    total = 0.0
    for k in range(1, min(n // 2, max_lag) + 1):
        c = float((y[:-k] * y[k:]).sum()) / denom
        if c <= 0.0:
            break
        total += c
    tau = 1.0 + 2.0 * total
    return float(max(1.0, n / max(1.0, tau)))


def _split_scatter(s: np.ndarray, w: np.ndarray, max_gap: float = 3.0):
    """Separate per-cut measurement noise from real along-bundle variation.

    Consecutive cuts sit ``cut_step`` px apart, and no physical bundle changes
    width appreciably over 2 px, so the scatter of the *difference* between
    neighbouring cuts is almost entirely measurement noise.  The scatter of the
    whole profile is that noise plus whatever the bundle really does, so
    ``sigma_real^2 = sigma_total^2 - sigma_noise^2``.

    The distinction is not cosmetic.  On the synthetic scenes every bundle has a
    constant width by construction, so any real variation means the measurement
    or the grouping is wrong -- which is why the scatter predicts error so well
    there.  On the real crop the same statistic is mostly genuine taper, and
    reading it as a quality flag would be a mistake.
    """
    if w.size < 8:
        return float("nan"), float("nan")
    d = np.diff(w)
    near = np.diff(s) <= max_gap
    if near.sum() < 6:
        return float("nan"), float("nan")
    dd = d[near]
    sigma_noise = float(1.4826 * np.median(np.abs(dd - np.median(dd)))
                        / math.sqrt(2.0))
    sigma_total = robust_sigma(w)
    if not math.isfinite(sigma_total):
        return sigma_noise, float("nan")
    return sigma_noise, float(math.sqrt(max(0.0, sigma_total ** 2 - sigma_noise ** 2)))


def _max_excursion(w: np.ndarray, sigma_noise: float,
                   windows: Sequence[int] = (3, 5, 9, 15, 25)):
    """Largest local departure of the width profile from the bundle's own level.

    ``_largest_step`` splits the profile in two and compares the halves, which
    only sees a wrong merge that contributes a comparable *share* of the
    instance.  On the development set, a minority bundle contributing under 10% of the fragments is
    detected 21% of the time against 74% for one contributing over 40%, because
    a median is by construction insensitive to a minority.

    This statistic instead slides a window of several lengths along the profile
    and scores how far each window's median sits from the global median, in
    units of that window's own standard error.  A short stretch measured on a
    different bundle shows up as a large z even though it barely moves the
    median.  Returns ``(z, arclength index of the window, window length)``.
    """
    from scipy.ndimage import median_filter

    n = w.size
    if n < 6 or not math.isfinite(sigma_noise) or sigma_noise <= 1e-6:
        return float("nan"), float("nan"), 0
    level = float(np.median(w))
    best = (0.0, float("nan"), 0)
    for L in windows:
        if L > n:
            break
        roll = median_filter(w, size=L, mode="nearest")
        se = 1.2533 * sigma_noise / math.sqrt(L)
        z = np.abs(roll - level) / max(se, 1e-6)
        # ignore the mode="nearest" padding at the two ends of the series
        # (the windows are odd and L <= n here, so this slice is never empty)
        half = L // 2
        core = z[half:n - half]
        i = int(np.argmax(core)) + half
        if float(z[i]) > best[0]:
            best = (float(z[i]), float(i), int(L))
    return best


def _largest_step(s: np.ndarray, w: np.ndarray, min_side: int = 4):
    """Largest median-vs-median width step along the bundle.

    A real bundle's width drifts slowly; a step is what a wrong merge looks
    like, because the trace jumped onto a different bundle partway along.
    """
    n = len(w)
    if n < 2 * min_side:
        return float("nan"), float("nan")
    best, at = 0.0, float("nan")
    for k in range(min_side, n - min_side + 1):
        d = abs(float(np.median(w[:k])) - float(np.median(w[k:])))
        if d > best:
            best, at = d, float(s[k])
    return best, at


@dataclass
class CutSet:
    """Every cut that survived, before any scene-level judgement is applied."""

    iid: int
    n_arms: int
    n_pixels: int
    axis_length: float
    attempted: int
    refusals: Dict[str, int]
    s: np.ndarray
    width: np.ndarray
    amp: np.ndarray
    rc: List[Tuple[float, float]]
    normal: List[Tuple[float, float]]
    offset: List[float]


def collect_cuts(sem: SemImage, node_dist: np.ndarray, graph,
                 matching: Dict[int, int], chain: Sequence[int], iid: int,
                 n_pixels: int, sp: SampleParams = SampleParams(),
                 cp: CutParams = CutParams()) -> CutSet:
    """Every perpendicular cut along one instance that yielded a width."""
    order = order_chain(graph, matching, chain)
    pts, _aids = chain_points(graph, order)
    runs = split_runs(pts, sp.join_px)

    widths: List[float] = []
    amps: List[float] = []
    arcs: List[float] = []
    rcs: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float]] = []
    offsets: List[float] = []
    refusals: Dict[str, int] = {}
    attempted = 0
    axis_len = 0.0
    s_base = 0.0

    for run_pts in runs:
        rs = resample_run(run_pts, sp.tangent_sigma)
        if rs is None:
            continue
        axis_len += float(rs["u"][-1])
        stride = max(1, int(round(sp.cut_step)))
        for i in range(0, len(rs["u"]), stride):
            if not rs["valid"][i]:
                continue
            rr, cc = float(rs["r"][i]), float(rs["c"][i])
            ri, ci = int(round(rr)), int(round(cc))
            if not (0 <= ri < node_dist.shape[0] and 0 <= ci < node_dist.shape[1]):
                continue
            if node_dist[ri, ci] < sp.node_clear_px:
                refusals["near_crossing"] = refusals.get("near_crossing", 0) + 1
                continue
            attempted += 1
            res: CutResult = measure_cut(
                sem.image, float(sem.background[ri, ci]), sem.noise,
                np.array([rr, cc]), np.array([rs["nr"][i], rs["nc"][i]]),
                cp, bool(sem.bg_confident[ri, ci]))
            if res.ok:
                widths.append(res.width)
                amps.append(res.amplitude)
                arcs.append(s_base + float(rs["u"][i]))
                rcs.append((rr, cc))
                normals.append((float(rs["nr"][i]), float(rs["nc"][i])))
                offsets.append(res.offset)
            else:
                refusals[res.reason] = refusals.get(res.reason, 0) + 1
        # Separate consecutive runs by more than `_split_scatter`'s adjacency
        # window, so a pair of cuts either side of a bridged mask gap is never
        # mistaken for a pair 2 px apart on the same stretch of bundle -- that
        # pair's difference is not measurement noise.
        s_base += float(rs["u"][-1]) + 10.0

    return CutSet(iid=iid, n_arms=len(chain), n_pixels=int(n_pixels),
                  axis_length=float(axis_len), attempted=attempted,
                  refusals=refusals, s=np.asarray(arcs, dtype=float),
                  width=np.asarray(widths, dtype=float),
                  amp=np.asarray(amps, dtype=float), rc=rcs,
                  normal=normals, offset=offsets)


def aggregate(cs: CutSet, sp: SampleParams = SampleParams(),
              max_cut_width: Optional[float] = None,
              rng: Optional[np.random.Generator] = None) -> Bundle:
    """Turn one instance's cuts into a width, a brightness and a spread.

    ``max_cut_width`` is the scene-level guard: a cut claiming a width several
    times the typical bundle in this image has not measured one bundle, it has
    measured a bundle lying along another one.  Applying it here rather than
    inside ``measure_cut`` is what makes it scale-free -- the threshold comes
    from the image itself, not from a number that happens to suit the synthetic
    generator's 7-16 px range.
    """
    rng = rng or np.random.default_rng(12345 + cs.iid)
    refusals = dict(cs.refusals)
    w, a, s, rcs = cs.width, cs.amp, cs.s, cs.rc
    nrm, off = cs.normal, cs.offset
    if max_cut_width is not None and w.size:
        keep = w <= max_cut_width
        dropped = int((~keep).sum())
        if dropped:
            refusals["wider_than_scene"] = refusals.get("wider_than_scene", 0) + dropped
        w, a, s = w[keep], a[keep], s[keep]
        flags = keep.tolist()
        rcs = [p for p, k in zip(rcs, flags) if k]
        nrm = [p for p, k in zip(nrm, flags) if k]
        off = [p for p, k in zip(off, flags) if k]

    b = Bundle(iid=cs.iid, n_arms=cs.n_arms, n_pixels=cs.n_pixels,
               axis_length=cs.axis_length, n_cuts_attempted=cs.attempted,
               n_cuts_valid=int(w.size), refusals=refusals)
    if w.size < sp.min_cuts:
        return b

    b.width = float(np.median(w))
    b.width_mad = robust_sigma(w)
    b.width_p16 = float(np.percentile(w, 16))
    b.width_p84 = float(np.percentile(w, 84))
    b.n_eff = _n_eff(w)
    b.sigma_noise, b.sigma_real = _split_scatter(s, w)
    if math.isfinite(b.width_mad) and b.n_eff >= 1.0:
        # 1.2533 = sqrt(pi/2), the median's standard error relative to the mean's
        b.width_se = float(1.2533 * b.width_mad / math.sqrt(b.n_eff))
        if math.isfinite(b.sigma_noise):
            b.width_se_noise = float(1.2533 * b.sigma_noise / math.sqrt(b.n_eff))
    if len(w) >= sp.boot_min_cuts:
        b.width_ci_lo, b.width_ci_hi = _block_bootstrap_median_ci(
            w, sp.block, sp.n_boot, rng)
    b.quality = ("weak" if (b.n_cuts_valid < sp.weak_min_cuts
                            or not math.isfinite(b.width_mad)
                            or b.width_mad > sp.weak_max_mad)
                 else "good")
    if math.isfinite(b.width_se):
        half = Z90 * math.hypot(b.width_se,
                                SYSTEMATIC_SIGMA_PX.get(b.quality, 0.0))
        b.width_unc_lo, b.width_unc_hi = b.width - half, b.width + half
    b.brightness = float(np.median(a))
    b.brightness_p90 = float(np.percentile(a, 90))
    b.brightness_mad = robust_sigma(a)
    step, at = _largest_step(s, w)
    b.step_abs = step
    b.step_rel = step / b.width if b.width > 0 else float("nan")
    b.step_at = at
    z, zi, zl = _max_excursion(w, b.sigma_noise)
    b.excursion_z = z
    b.excursion_at = float(s[int(zi)]) if math.isfinite(zi) else float("nan")
    b.excursion_len = zl
    b.cut_s = [float(x) for x in s]
    b.cut_width = [float(x) for x in w]
    b.cut_rc = [(float(r), float(c)) for r, c in rcs]
    b.cut_n = [(float(r), float(c)) for r, c in nrm]
    b.cut_off = [float(x) for x in off]
    return b


def node_distance(graph) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest crossing pixel."""
    from scipy.ndimage import distance_transform_edt

    node = np.zeros(graph.shape, dtype=bool)
    for n in graph.nodes:
        for r, c in n.pixels:
            node[r, c] = True
    if not node.any():
        return np.full(graph.shape, 1e6, dtype=np.float32)
    return np.asarray(distance_transform_edt(~node)).astype(np.float32)
