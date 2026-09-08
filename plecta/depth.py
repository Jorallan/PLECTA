"""Secondary post-hoc depth stage: crossing order, layers, and planar z.

This module never changes the 2-D grouping. `DEPTH_MODE = off` is literally
"do not run this file"; `plecta/predict.py` and `plecta/linking.py` are not
modified by it. The stage consumes reconstructed 2-D instances (either
PLECTA's own chains or oracle centrelines supplied by a caller), a registered
grayscale image, and produces -- one identifiability level at a time:

1. projected crossings between instance centrelines;
2. local over/under evidence per crossing, from ONE interpretable image
   feature -- occlusion continuity through the crossing core, normalised by
   a NOISE floor rather than by the two rods' contrast -- with a signed
   score and an abstention band; an unidentifiable crossing is reported as
   abstained, never invented. The shipped rule reads a single channel and
   consults no radius (see `crossing_evidence`);
3. a weighted precedence graph and a globally consistent linear order
   (exact maximum-weight acyclic orientation for small components, greedy
   insertion with local improvement above that); local relations the global
   solver had to flip are recorded, both raw and corrected are kept;
4. a minimal-K discrete layer assignment (longest path in the consistent
   DAG; non-conflicting instances share layers; K is inferred, never set to
   the instance/crossing/junction count);
5. optionally metric z under an explicit *compact-stack assumption*: nothing
   floats higher than it must, subject to non-interpenetration
   (z_i - z_j >= r_i + r_j). A projected crossing does not imply physical
   contact, so metric z is only meaningful under this stated prior and is
   reported as such.

Pairs whose STROKES overlap without their centrelines ever meeting are
enumerated separately (`identify_contacts`). By default they carry clearance
and no opinion; `DepthParams.grazing_evidence` puts the same occlusion
question to them at their point of closest approach, which is what recovers a
real crossing that 2-D fragmentation broke apart. Both switches are measured
in `parameters.yaml`, which is where the numbers behind their defaults live.

An abstained crossing still has to clear, but nothing says which way round.
`DepthParams.undecided_order` chooses how those pairs are stacked:

  "id_based" the order the components happen to fall in, which for singleton
           components is instance-id rank. Historical behaviour.
  "compact"  colour the overlap graph so filaments that do not overlap each
             other can share a level, subject to every decided relation
             (default). Never thicker than "id_based": both are solved and the
           thinner film is kept.

Only the abstained pairs move: the decided over/unders, the layer count K and
every reported probability are identical either way.

Every tuned value lives in `parameters.yaml` under `depth_3d`, and the command
line builds `DepthParams` through `parameters.build` so that reading that file
tells you the configuration. The dataclass defaults exist only to keep the class
constructible on its own -- a caller that constructs `DepthParams()` directly,
as a library user or another application may, gets those defaults and not the
file.

The physical model is planar instances (dz/ds = 0) with circular
cross-sections; z is in pixels, the same unit as radii.

Output: a `pred_depth.json`-shaped dict (see `run_scene`), consumed by the
evaluation side (`eval/core/depth_metric.py` in filaments_quantification).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

# Polyline geometry lives in `geometry`; `_densify` is re-exported here
# because that is the name the downstream app binds.
from .geometry import _densify, _unit_normals

# The width measurement lives with the cut machinery it is built on, and is
# re-exported so `plecta.depth.measure_diameters` keeps working. Importing it
# costs nothing: `plecta.image.measurement` pulls in numpy and no more, and
# its scikit-image and scipy uses are all inside functions.
from .image.measurement import measure_diameters, width_to_diameter


# ── parameters ─────────────────────────────────────────────────────────────


@dataclass
class DepthParams:
    # crossing identification
    merge_px: float = 8.0          # merge intersection hits closer than this
    # evidence sampling
    window_px: float = 22.0        # arclength window either side of a crossing
    core_px: float = 7.5           # half-size of the shared crossing core
    min_flank_px: int = 6          # fewer flank samples than this -> abstain
    # How the core half-size is set; see `crossing_evidence`.
    #
    # "flat"           core = core_px, a constant. NO radius enters the
    #                  evidence path at all, which is the point: the radius it
    #                  used to need is an oracle quantity on synthetic data and
    #                  an estimate everywhere else.
    # "radius_scaled"  core = max(core_px, 1.1*(r_i + r_j)/2 + 2), the
    #                  historical rule. Retained so stored records reproduce;
    #                  reproducing them needs core_px = 6.0 as well, since
    #                  core_px is the floor of that max.
    #
    # Measured on the clean held-out optical set (20 scenes, 1355 crossings,
    # seeds disjoint from every tuning set): the oracle per-rod core is 8.30 px
    # median, range 6.00-10.69. A flat core is never significantly worse than
    # the oracle anywhere in 6..14 px, and 7.5 px matches the oracle's decision
    # rate (88.0% vs 87.5%) while edging its accuracy (+0.0017 at matched
    # coverage). 10.0 px, picked earlier on the three TUNING domains, does not
    # replicate here (-0.0051 [-0.0141, +0.0050] against the oracle, and
    # 1.8 points less coverage), so it is not the default. 7.5 px is also
    # exactly what the radius-free fallback already computed,
    # max(6.0, 1.1*5 + 2).
    #
    # PROVENANCE: that 6..14 px sweep was run ON THE CLEAN HELD-OUT SET, so
    # core_px = 7.5 is a held-out-SELECTED constant, not a pre-frozen one --
    # the one exception to "no parameter was chosen on that set", which holds
    # for the rule (scoring, w_intensity, the noise floor, abstain_score) and
    # not for this. Mitigating, not excusing: the sweep is flat and 7.5 is the
    # value the radius-free fallback already produced. See METHOD.md,
    # "A flat core, and where 7.5 px comes from".
    core_mode: str = "flat"        # "flat" | "radius_scaled"
    # Channel weights. `w_intensity` is the weight on the intensity channel
    # under all three rules -- for "noise_floored" the whole score IS
    # w_intensity * F_inf, and the measurement pinned that weight at 2. The
    # (2.446, 1.576) pair is a logistic fit on the 12-scene development
    # nucleus (synthetic_depth_dev, 830 matched crossings, 2026-08-18),
    # monotone-calibrated on that set (accuracy 0.59 -> 0.98 across confidence
    # bins). Development data only -- never refit on held-out or locked scenes.
    w_intensity: float = 2.0       # weight on the intensity channel
    w_sharpness: float = 1.0       # weight on the edge-sharpness channel
    # How the evidence becomes one signed score; see `combine_channels`.
    #
    # "noise_floored" (default, shipped)  s = w_I * F_inf, ONE channel, where
    #     F_inf = (|m_c - m_j| - |m_c - m_i|) / max(|m_i - m_j|, 2*sigma_hat)
    #     and sigma_hat is the within-rod flank noise. Dividing by the rods'
    #     own contrast alone (what F_I does) saturates at +/-1 whenever the two
    #     rods differ at all, including by less than the noise; flooring the
    #     denominator at 2*sigma_hat stops it claiming certainty it cannot
    #     have. Measured on the clean held-out optical set (20 scenes, 1355
    #     crossings, no calibration has seen any of it): the SAME accuracy as
    #     the previous rule at the same coverage (+0.0025, 95% CI
    #     [-0.0042, +0.0079] -- no difference), with a much better-ordered
    #     confidence (AUC of correctness against |s| 0.7772 -> 0.8698,
    #     +0.0926 [+0.0563, +0.1321]), which turns into accuracy as soon as
    #     more abstention is acceptable: +0.0188 at 82.5% coverage, +0.0240 at
    #     80% -- coverage-matched comparison points below where the default
    #     runs, which on that set is 87.1% (see `abstain_score`). The honest
    #     headline is "same accuracy, better-ordered confidence, simpler
    #     evidence path", not "more accurate".
    # "winsorized_linear"   s = w_I F_I + w_S clip(F_S, -1, 1). The previous
    #     shipped rule: linear in F_I (already bounded by construction) and
    #     clipping the one unbounded channel.
    # "weight_of_evidence"  s = w_I tanh(F_I) + w_S tanh(F_S), w = (2.446,
    #     1.576). The smooth-bounded variant the recorded held-out aggregates
    #     were produced under; it and "winsorized_linear" are measured to
    #     change no crossing's direction.
    #
    # An axial second channel (each rod's core arc read on its OWN axis) was
    # built and measured on the same clean set: -0.0008 [-0.0059, +0.0048] at
    # the shipped operating point, and NONE of the 26 directions it changes
    # clears any deployable threshold. It is deliberately not implemented.
    scoring: str = "noise_floored"   # | "winsorized_linear"
                                     # | "weight_of_evidence"
    # Abstention. The two parameterisations are NOT interchangeable and each
    # drives only its own rules -- see `decision_threshold`.
    #   abstain_band  is a PROBABILITY band, |p - 0.5| < band, applied as the
    #                 exactly equivalent |score| threshold
    #                 log((0.5+band)/(0.5-band)) = 0.6190. It drives
    #                 "winsorized_linear" and "weight_of_evidence", whose
    #                 scores are log-odds-shaped, and it keeps their behaviour
    #                 bit-identical.
    #   abstain_score is a threshold in SCORE units, applied directly. It
    #                 drives "noise_floored", whose |s| is a noise-normalised
    #                 margin and not a log-odds: pushing it through a sigmoid
    #                 to compare against a probability band would be
    #                 arithmetic without a meaning.
    abstain_band: float = 0.15     # legacy rules only (-> |score| >= 0.6190)
    # 0.40 is where the single-channel rule reaches the 82.5% POOLED decision
    # rate the calibration targets -- a target from the mixed development
    # domains, not a per-set delivery: on the clean held-out optical set the
    # same 0.40 decides 87.1%, and that is the shipped operating point. The
    # clean held-out set puts any t in 0.35..0.47 at the same accuracy, so the
    # value is a plateau, not a peak.
    abstain_score: float = 0.40    # "noise_floored" only, in score units
    # global ordering
    exact_max_nodes: int = 14      # exact DP up to this component size
    # evidence
    # With no image, or with this off, every crossing abstains: the geometry
    # still solves and nothing interpenetrates, but the vertical order is a
    # tie-break rather than a measurement, and the report says so.
    use_image_evidence: bool = True
    # radii
    # "measured" uses whatever the caller supplied (or the FWHM widths); "fixed"
    # ignores them and gives every instance default_radius_px, which is what a
    # mask-only run has to do.
    radius_mode: str = "measured"   # "measured" | "fixed"
    # What an instance with NO usable width measurement gets. "median" gives it
    # the median of whatever WAS measured in the same scene; "default" gives it
    # default_radius_px, a constant with no relation to the scene. On a real
    # 282-instance field that constant put 56 rods at 5.0 px while every rod
    # that could be measured came out between 6.3 and 16.8 px -- i.e. thinner
    # than anything actually present. Either way the count is reported as
    # solver_report.n_radius_imputed, and those instances carry "d_imputed".
    radius_fallback: str = "median"  # "median" | "default"
    # metric z (compact-stack assumption)
    default_radius_px: float = 5.0  # used when nothing at all could be measured
    z_gap_px: float = 0.0          # extra clearance added to r_i + r_j
    # Grazing pairs: two instances can overlap in projection without their
    # centrelines crossing (abutting ends, side-by-side runs, shallow grazes).
    # This stage constrains centreline CROSSINGS only, so those pairs are free
    # to interpenetrate. Off keeps that behaviour and reports the
    # interpenetration; on clears them too, as abstained pairs -- clearance, no
    # opinion about which is on top, exactly how an undecidable crossing is
    # already treated.
    clear_grazing_overlaps: bool = False
    # Grazing pairs, again: `clear_grazing_overlaps` gives them clearance with
    # no opinion. This instead ASKS the image about them -- the same occlusion
    # evidence a real crossing gets, sampled at the point of closest approach --
    # so each pair either earns a direction or abstains on its own merits.
    # Needs `use_image_evidence`; without an image there is nothing to ask.
    grazing_evidence: bool = False
    # how pairs the evidence could NOT decide are stacked; see compact_order
    undecided_order: str = "compact"    # "compact" | "id_based"


@dataclass
class PredCrossing:
    i: int                         # instance id
    j: int
    x: float
    y: float
    score: float = float("nan")    # signed evidence score; + favours i. This is
                                   # the quantity that decides and weights.
    p_over: float = float("nan")   # sigmoid(score), for reporting only
    features: Dict[str, float] = field(default_factory=dict)
    abstain: bool = False
    raw_over: int = -1             # local decision (or -1 when abstained)
    over: int = -1                 # after global correction
    flipped: bool = False          # global solver reversed the local relation
    grazing: bool = False          # strokes overlap but the centrelines never
                                   # meet: this is a closest-approach contact,
                                   # not a projected centreline crossing


# ── 1. crossings between instance centrelines ─────────────────────────────


def _seg_intersect(p, p2, q, q2):
    r = p2 - p
    s = q2 - q
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p + t * r
    return None


def identify_crossings(centrelines: Dict[int, np.ndarray],
                       params: Optional[DepthParams] = None
                       ) -> List[PredCrossing]:
    """Projected centreline intersections between every instance pair.

    `centrelines` maps instance id -> (K, 2) ordered [x, y] polyline. Same
    exact-segment-intersection construction the generator uses, so predicted
    and ground-truth crossings are commensurable.
    """
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]

    params = params or DepthParams()
    ids = sorted(centrelines)
    dense = {i: _densify(centrelines[i]) for i in ids}
    trees = {i: cKDTree(dense[i]) for i in ids if len(dense[i]) >= 2}
    out: List[PredCrossing] = []
    for ai in range(len(ids)):
        for bi in range(ai + 1, len(ids)):
            a, b = ids[ai], ids[bi]
            if a not in trees or b not in trees:
                continue
            pa, pb = dense[a], dense[b]
            near = trees[a].query_ball_tree(trees[b], r=4.0)
            hits: List[Tuple[float, float]] = []
            seen: Set[Tuple[int, int]] = set()
            for ka, lst in enumerate(near):
                for kb in lst:
                    for sa in (ka - 1, ka):
                        if not (0 <= sa < len(pa) - 1):
                            continue
                        for sb in (kb - 1, kb):
                            if not (0 <= sb < len(pb) - 1):
                                continue
                            if (sa, sb) in seen:
                                continue
                            seen.add((sa, sb))
                            hit = _seg_intersect(pa[sa], pa[sa + 1],
                                                 pb[sb], pb[sb + 1])
                            if hit is not None:
                                hits.append((float(hit[0]), float(hit[1])))
            merged: List[List[float]] = []
            for x, y in hits:
                for m in merged:
                    if math.hypot(x - m[0] / m[2], y - m[1] / m[2]) \
                            < params.merge_px:
                        m[0] += x; m[1] += y; m[2] += 1
                        break
                else:
                    merged.append([x, y, 1])
            for m in merged:
                out.append(PredCrossing(i=a, j=b, x=m[0] / m[2], y=m[1] / m[2]))
    return out


def identify_contacts(centrelines: Dict[int, np.ndarray],
                      radii: Dict[int, float],
                      params: Optional[DepthParams] = None,
                      exclude: Optional[Set[Tuple[int, int]]] = None
                      ) -> List[PredCrossing]:
    """Pairs whose STROKES overlap in projection without their centrelines
    crossing.

    `identify_crossings` answers the question this stage is about -- which
    instance is on top where two of them cross -- and constrains nothing else.
    That leaves a real gap for anyone wanting geometry out of it: two rods can
    overlap in projection, and so interpenetrate in 3-D, without their axes
    ever intersecting. On a 307-instance annotation, 414 pairs cross and a
    further 445 overlap without crossing.

    Each such pair is returned as an ABSTAINED crossing at the point of closest
    approach, so it carries clearance but no opinion about which rod is above.
    `exclude` is the set of (lo, hi) pairs already found by
    `identify_crossings`.
    """
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]

    params = params or DepthParams()
    dense = {i: _densify(np.asarray(centrelines[i])) for i in sorted(centrelines)}
    ids = [i for i in sorted(dense) if len(dense[i])]
    if len(ids) < 2:
        return []
    counts = [len(dense[i]) for i in ids]
    stacked = np.concatenate([dense[i] for i in ids])
    owner = np.repeat(np.arange(len(ids)), counts)
    radius = np.array([radii.get(i, params.default_radius_px) for i in ids])

    pairs = cKDTree(stacked).query_pairs(2.0 * float(radius.max()),
                                         output_type="ndarray")
    slot_a, slot_b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    keep = slot_a != slot_b
    pairs, slot_a, slot_b = pairs[keep], slot_a[keep], slot_b[keep]
    delta = stacked[pairs[:, 0]] - stacked[pairs[:, 1]]
    distance = np.hypot(delta[:, 0], delta[:, 1])
    keep = distance < radius[slot_a] + radius[slot_b] + params.z_gap_px
    pairs, slot_a, slot_b = pairs[keep], slot_a[keep], slot_b[keep]
    distance = distance[keep]

    exclude = exclude or set()
    midpoints = 0.5 * (stacked[pairs[:, 0]] + stacked[pairs[:, 1]])
    best: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for row in range(len(pairs)):
        a, b = ids[slot_a[row]], ids[slot_b[row]]
        key = (a, b) if a < b else (b, a)
        if key in exclude:
            continue
        current = best.get(key)
        if current is None or distance[row] < current[0]:
            best[key] = (float(distance[row]), float(midpoints[row, 0]),
                         float(midpoints[row, 1]))
    out = []
    for key, value in sorted(best.items()):
        c = PredCrossing(i=key[0], j=key[1], x=value[1], y=value[2])
        c.abstain = True
        c.grazing = True
        c.features["reason"] = 3.0          # grazing: nothing asked of the
                                            # image (see DepthParams.
                                            # grazing_evidence to ask)
        c.features["min_separation"] = value[0]
        out.append(c)
    return out


def clearance_violations(crossings: List[PredCrossing], z: Dict[int, float],
                         radii: Dict[int, float], params: DepthParams,
                         tolerance: float = 1e-6) -> List[dict]:
    """Pairs whose SOLVED heights still interpenetrate.

    Measured on the output rather than on intent, and reported rather than
    repaired: with `clear_grazing_overlaps` off this is how the geometry the
    stage does not constrain becomes visible instead of silent.
    """
    out = []
    for c in crossings:
        if c.i not in z or c.j not in z:
            continue
        need = (radii.get(c.i, params.default_radius_px)
                + radii.get(c.j, params.default_radius_px) + params.z_gap_px)
        gap = abs(z[c.i] - z[c.j])
        if gap < need - tolerance:
            out.append({"i": int(c.i), "j": int(c.j),
                        "separation": float(gap), "required": float(need)})
    return out


# ── 2. local over/under evidence ──────────────────────────────────────────


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """The edge image the sharpness channel is read from.

    A light Gaussian first, because a Sobel on raw SEM noise measures the noise;
    sigma = 1 px is small enough to leave a filament's own edge intact. Named
    rather than inlined so that anything driving this stage from outside reads
    the same edges the evidence does.
    """
    from scipy.ndimage import gaussian_filter, sobel

    smooth = gaussian_filter(np.asarray(image, dtype=np.float32), 1.0,
                             mode="reflect")
    return np.hypot(sobel(smooth, 0), sobel(smooth, 1)).astype(np.float32)


def _sample_image(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of `image` at [x, y] points."""
    H, W = image.shape
    x = np.clip(pts[:, 0], 0, W - 1.001)
    y = np.clip(pts[:, 1], 0, H - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    return (image[y0, x0] * (1 - fx) * (1 - fy)
            + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy
            + image[y0 + 1, x0 + 1] * fx * fy)


def _arc_window(dense: np.ndarray, x: float, y: float,
                window: float, core: float):
    """Split one centreline near (x, y) into core and flank samples.

    Returns (core_pts, flank_pts, flank_normals): unit normals accompany the
    flank points so a caller can sample the rod's *edges*, where a ridge's
    gradient actually lives -- on the centreline itself the gradient of a
    ridge is ~0 by symmetry, so sharpness must be measured off-axis.
    """
    d = np.hypot(dense[:, 0] - x, dense[:, 1] - y)
    k0 = int(np.argmin(d))
    seg = np.diff(dense, axis=0)
    step = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    s = step - step[k0]
    sel = np.abs(s) <= window
    in_core = sel & (np.abs(s) <= core)
    in_flank = sel & (np.abs(s) > core)
    normals = _unit_normals(dense)
    return dense[in_core], dense[in_flank], normals[in_flank]


def abstain_threshold(band: float) -> float:
    """A probability band, expressed in score units. LEGACY rules only.

    `abstain_band` is stated as a probability band, |p - 0.5| < band, which is
    how the two log-odds-shaped rules have always been documented. Since p is
    monotone in the score, that is *exactly* the same set as
    |score| < log((0.5+band)/(0.5-band)) -- band 0.15 becomes 0.6190.
    Converting once here means the decision never has to round-trip through a
    probability.

    This is the band -> score converter and nothing else. It is the right
    conversion only where the score really is a log-odds, which is
    "winsorized_linear" and "weight_of_evidence"; `decision_threshold` is what
    picks the right parameterisation per rule.
    """
    band = min(max(float(band), 0.0), 0.499999)
    return math.log((0.5 + band) / (0.5 - band))


def decision_threshold(params: DepthParams) -> float:
    """The |score| below which a crossing abstains, under THIS scoring rule.

    Two parameterisations, because the two families of rule do not mean the
    same thing by their score, and one number cannot honestly serve both.

    The legacy rules sum weighted, bounded channels the way log-odds add, so
    "abstain when the implied probability is within `abstain_band` of 0.5" is
    a statement about them: `abstain_threshold` converts it exactly, and their
    behaviour stays bit-identical to the record.

    "noise_floored" has no probability anywhere in it. |s| = |2 * F_inf| is a
    margin measured in units of the larger of the two rods' contrast and twice
    the flank noise; sigmoid(s) is a display transform (held-out ECE ~0.21),
    so thresholding a band around 0.5 of it would be arithmetic without a
    meaning. `abstain_score` is therefore stated and applied directly in score
    units -- 0.40, the value that puts the rule at the 82.5% POOLED decision
    rate the calibration targets. That pooled target comes from the mixed
    development domains and is recorded outside this repository; it is not
    what 0.40 delivers on any single set. On the clean held-out optical set
    the same 0.40 decides 87.1% of crossings (0.9347 accuracy, against the
    previous rule's 0.9217 at its 89.5%) -- that 87.1% is the shipped
    operating point, and the "+0.0188 at 82.5% coverage" quoted elsewhere in
    this module is a coverage-matched comparison reached by abstaining more
    than this default does.
    """
    if params.scoring == "noise_floored":
        return float(params.abstain_score)
    return abstain_threshold(params.abstain_band)


def flank_noise_sigma(flank_i, flank_j) -> float:
    """Robust NOISE scale of the flank greys: 1.4826 * MAD of the residuals
    of each rod's flank samples about THAT ROD's own median.

    Each rod's samples are centred on their own median *before* the two
    residual sets are pooled, so the rods' separation |m_i - m_j| cannot enter.

    This ordering is the whole point, and getting it wrong is silent. The MAD
    of the raw POOLED flank VALUES is taken over a bimodal set -- two rods, two
    brightnesses -- so it tracks |m_i - m_j| and becomes a CONTRAST floor
    instead of a noise floor. A denominator that grows with contrast exactly
    cancels the effect the floor exists to produce: the statistic would go on
    saturating whenever the two rods differ at all, which is the behaviour
    being replaced. Nothing about the code would look wrong.

    NaN when neither rod has a single flank sample; callers treat that as 0.
    """
    residuals = []
    for flank in (flank_i, flank_j):
        values = np.asarray(flank, dtype=np.float64).ravel()
        if values.size:
            residuals.append(values - np.median(values))
    if not residuals:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(np.concatenate(residuals))))


def noise_floored_feature(m_core: float, m_flank_i: float, m_flank_j: float,
                          sigma_hat: float) -> float:
    """F_inf -- the intensity evidence with a noise floor under it.

        F_inf = (|m_c - m_j| - |m_c - m_i|) / max(|m_i - m_j|, 2*sigma_hat)

    Same numerator as `feat_intensity`: how much closer the shared core sits
    to rod i's own flank brightness than to rod j's. Positive favours rod i.

    The denominator is the difference. `feat_intensity` divides by the two
    rods' contrast alone, so it reaches +/-1 whenever the core matches one rod
    outright -- including when the two rods differ by less than the flank
    noise, where the match means nothing. Flooring at twice the noise scale
    makes the feature report a margin in units of "how much bigger than the
    noise is this", and it stops claiming certainty the picture cannot supply.
    Above the floor the two agree exactly.

    Measured on the clean held-out optical set (20 scenes, 1355 crossings,
    disjoint seeds): same accuracy at the same coverage (+0.0025,
    [-0.0042, +0.0079]) and a much better-ordered confidence, AUC
    0.7772 -> 0.8698 (+0.0926, [+0.0563, +0.1321]).
    """
    sigma = 0.0 if not np.isfinite(sigma_hat) else float(sigma_hat)
    numerator = abs(m_core - m_flank_j) - abs(m_core - m_flank_i)
    return numerator / max(abs(m_flank_i - m_flank_j), 2.0 * sigma, 1e-12)


def combine_channels(feat_intensity: float, feat_sharpness: float,
                     params: DepthParams,
                     feat_noise_floored: Optional[float] = None) -> float:
    """The evidence combined into one signed score.

    Positive favours rod i. Every rule is odd in its features and carries no
    intercept, because swapping i and j must flip the score exactly.

    ``noise_floored`` (default, shipped)
        s = w_I F_inf.  ONE channel. F_inf shares its numerator with F_I and
        differs only in the denominator, which is floored at twice the
        within-rod flank noise (`noise_floored_feature`), so the sign is
        always the sign the other two rules give and only the magnitude --
        the confidence -- changes. Measured on the clean held-out optical set
        (20 scenes / 1355 crossings, seeds disjoint from every tuning set):
        the same accuracy at the same coverage (+0.0025, 95% CI
        [-0.0042, +0.0079]), AUC of correctness against |s| 0.7772 -> 0.8698
        (+0.0926, [+0.0563, +0.1321]), and +0.0188 / +0.0240 accuracy at 82.5%
        / 80% coverage -- coverage-matched comparison points BELOW where the
        default runs (at abstain_score = 0.40 the rule decides 87.1% on that
        set; see `decision_threshold`), not the shipped operating point. Same
        accuracy, better-ordered confidence, simpler evidence path -- not
        "more accurate".

        F_inf is passed in rather than derived from `feat_intensity`: the two
        differ by a ratio of denominators that this function cannot see, and
        reconstructing it from a rounded feature would be a quiet way to get a
        different number. `crossing_evidence` computes both from the same
        medians and hands both over.

    ``winsorized_linear``
        s = w_I F_I + w_S clip(F_S, -1, 1).  F_I is already in [-1, 1] by
        construction, so squashing it again is redundant; F_S is the unbounded
        one and clipping bounds it just as tanh did. Measured on 2649 held-out
        crossings (both sets) this changes the direction of **no** crossing;
        with w = (2, 1) it decides ~3% fewer of them and is 0.4-0.6 points more
        accurate on the ones it does decide.

    ``weight_of_evidence`` (what produced the recorded held-out aggregates)
        s = w_I tanh(F_I) + w_S tanh(F_S).  A log-odds / weight-of-evidence
        fusion: tanh bounds each channel before weighting, and the solver sums
        |s| because log-odds are additive.
    """
    if params.scoring == "noise_floored":
        if feat_noise_floored is None:
            raise ValueError(
                "scoring='noise_floored' needs feat_noise_floored; it is not "
                "recoverable from feat_intensity (different denominator).")
        return params.w_intensity * float(feat_noise_floored)
    if params.scoring == "winsorized_linear":
        return (params.w_intensity * feat_intensity
                + params.w_sharpness * min(1.0, max(-1.0, feat_sharpness)))
    return (params.w_intensity * math.tanh(feat_intensity)
            + params.w_sharpness * math.tanh(feat_sharpness))


def crossing_evidence(image: np.ndarray, grad_mag: Optional[np.ndarray],
                      centrelines_dense: Dict[int, np.ndarray],
                      c: PredCrossing,
                      params: DepthParams,
                      radii: Optional[Dict[int, float]] = None) -> None:
    """Occlusion-continuity evidence at one crossing, written into `c`.

    The physically grounded cue is occlusion: the crossing core shows the
    *upper* rod's appearance, uninterrupted, while the lower rod's ridge is
    interrupted there. The question asked of every channel is the same one --
    "how far is the core from this rod's own flanks" -- and the rod whose
    flank appearance the core matches better is called upper.

      intensity  |median(core) - median(flank_k)| per rod, on the grey image.
                 Normalised two ways, which is what `scoring` selects between:
                 `feat_intensity` divides by the two rods' contrast alone;
                 `feat_noise_floored` divides by that contrast OR twice the
                 within-rod flank noise, whichever is larger, so a crossing
                 whose rods differ by less than the noise can no longer
                 saturate. Both are always stored, so a record made under
                 either rule can be re-read under the other.
      sharpness  the same question on the gradient-magnitude image, at
                 `centreline +/- r*normal` because a ridge's gradient is ~0 on
                 its own axis. ONLY the two legacy rules read it. Under
                 `scoring == "noise_floored"` it is not computed, `grad_mag`
                 may be (and from `run_scene` is) None, and the two sharpness
                 features are ABSENT from `c.features` rather than reported as
                 zeros nobody measured.

    When the two rods' appearances barely differ (the generator's deliberately
    unidentifiable crossings) the numerator is ~0, |score| stays under the
    threshold, and the crossing abstains rather than being decided by noise.

    Radii. Under `core_mode == "flat"` -- the default -- nothing in this
    function consults a radius: the core is a constant, the window follows the
    core, and the only consumer of a per-rod radius was the sharpness channel,
    which the shipped rule does not have. That is deliberate. The radius the
    old geometry needed is an oracle quantity on synthetic data and an
    estimate everywhere else, and dropping it from the EVIDENCE costs nothing
    measurable: on the clean held-out optical set a flat core is never
    significantly worse than oracle per-rod radii anywhere in 6..14 px, and at
    7.5 px it matches the oracle's decision rate (88.0% vs 87.5%) and edges its
    accuracy (+0.0017 at matched coverage). Radii are still needed OUTSIDE the
    evidence path -- metric z, grazing detection, the exported tube -- and are
    untouched there.
    """
    radii = radii or {}
    r_i = radii.get(c.i, params.default_radius_px)
    r_j = radii.get(c.j, params.default_radius_px)
    if params.core_mode == "flat":
        # A constant. No radius, oracle or estimated, reaches the samples.
        core = float(params.core_px)
    else:
        # The historical geometry: the shared core spans the two silhouettes,
        # so its half-size follows the radii (falling back to the configured
        # default) -- flank samples must sit clear of the *other* rod's stroke
        # or they measure the union, not the rod. Retained so stored records
        # reproduce; note core_px is the FLOOR here, so reproducing a record
        # also needs the core_px that made it (6.0).
        core = max(params.core_px, 1.1 * (r_i + r_j) / 2.0 + 2.0)
    window = max(params.window_px, 3.0 * core)
    # Sharpness is a property of the RULE, not of what the caller happened to
    # pass: gating it here is what keeps the shipped path single-channel and
    # radius-free even if a caller hands over a gradient image anyway.
    use_sharpness = params.scoring != "noise_floored"
    if use_sharpness and grad_mag is None:
        raise ValueError(
            f"scoring={params.scoring!r} reads the sharpness channel and so "
            "needs grad_mag; only scoring='noise_floored' runs without one.")
    edge_image = grad_mag if grad_mag is not None else image
    ca, fa, na = _arc_window(centrelines_dense[c.i], c.x, c.y, window, core)
    cb, fb, nb = _arc_window(centrelines_dense[c.j], c.x, c.y, window, core)
    if len(fa) < params.min_flank_px or len(fb) < params.min_flank_px \
            or (len(ca) + len(cb)) == 0:
        c.abstain = True
        c.features["reason"] = 1.0   # insufficient support
        return
    core_pts = np.concatenate([ca, cb])

    # The grey samples themselves, kept rather than immediately reduced: the
    # noise floor is a statistic OF the flank samples, not of their medians.
    grey_fa = _sample_image(image, fa)
    grey_fb = _sample_image(image, fb)
    core_i = float(np.median(_sample_image(image, core_pts)))
    flank_ai = float(np.median(grey_fa))
    flank_bi = float(np.median(grey_fb))

    # positive -> core looks like rod i -> i on top
    feat_i = (abs(core_i - flank_bi) - abs(core_i - flank_ai)) \
        / max(abs(flank_ai - flank_bi), 0.02)
    sigma_hat = flank_noise_sigma(grey_fa, grey_fb)
    feat_n = noise_floored_feature(core_i, flank_ai, flank_bi, sigma_hat)

    feat_g = 0.0
    if use_sharpness:
        def edge_sharpness(pts, normals, r):
            """Median gradient magnitude on the rod's two edges (axis grad ~ 0)."""
            vals = []
            for sign in (1.0, -1.0):
                vals.append(_sample_image(edge_image, pts + sign * r * normals))
            return float(np.median(np.concatenate(vals)))

        # Sharpness continuity is tested per rod, on that rod's own geometry:
        # the core-adjacent samples of rod k sit on k's edges when k is on top
        # (edge stays crisp through the crossing) but inside the other rod's
        # stroke when k is underneath (its edge is overwritten there). The
        # normals of the nearest flank samples extend into the short core arc.
        def core_edge(pts_core, flank_normals, r):
            if not len(pts_core) or not len(flank_normals):
                return float("nan")
            normals = np.tile(flank_normals[:1], (len(pts_core), 1))
            return edge_sharpness(pts_core, normals, r)

        flank_ag = edge_sharpness(fa, na, r_i)
        flank_bg = edge_sharpness(fb, nb, r_j)
        core_g_a = core_edge(ca, na, r_i)
        core_g_b = core_edge(cb, nb, r_j)
        cont_a = abs(core_g_a - flank_ag) if np.isfinite(core_g_a) else None
        cont_b = abs(core_g_b - flank_bg) if np.isfinite(core_g_b) else None
        if cont_a is not None and cont_b is not None:
            feat_g = (cont_b - cont_a) / max(0.02, flank_ag + flank_bg)
        c.features.update(d_intensity=flank_ai - flank_bi,
                          d_sharpness=float(flank_ag) - float(flank_bg),
                          feat_intensity=float(feat_i),
                          feat_sharpness=float(feat_g))
    else:
        c.features.update(d_intensity=flank_ai - flank_bi,
                          feat_intensity=float(feat_i))
    # Stored under every rule, so the output JSON says what the noise floor
    # actually was at each crossing instead of leaving it to be re-derived
    # from an image nobody kept.
    c.features.update(sigma_hat=(0.0 if not np.isfinite(sigma_hat)
                                 else float(sigma_hat)),
                      feat_noise_floored=float(feat_n))
    c.score = combine_channels(float(feat_i), float(feat_g), params,
                               feat_noise_floored=float(feat_n))
    # A monotone display transform of the score, kept because the output schema
    # and the evaluator's calibration metric both expect it. It is NOT used to
    # decide anything: sign(score) decides and |score| weights (see
    # `resolve_global_order`), so this value is a ranking score wearing a
    # probability's clothes -- held-out ECE is ~0.21.
    c.p_over = 1.0 / (1.0 + math.exp(-c.score))
    # Both branches are stated. This used to only ever SET abstain, which is
    # invisible for a fresh PredCrossing -- `abstain` is already False -- but
    # silently wrong for a caller re-scoring something that came in abstained,
    # which is exactly what `grazing_evidence` does.
    if abs(c.score) < decision_threshold(params):
        c.abstain = True
        c.raw_over = -1
    else:
        c.abstain = False
        c.raw_over = c.i if c.score > 0.0 else c.j


# ── 3. global ordering ────────────────────────────────────────────────────


def _exact_order(nodes: List[int],
                 w: Dict[Tuple[int, int], float]) -> List[int]:
    """Max-weight consistent linear order by DP over subsets (n <= ~14).

    w[(u, v)] is the reward for placing u above v. Order is returned
    top-first. f(S) = best score placing some v of S at the *bottom* of S:
    edges from S\\{v} into v are then satisfied.
    """
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    gain = [[0.0] * n for _ in range(n)]
    for (u, v), weight in w.items():
        gain[idx[u]][idx[v]] += weight
    f = [0.0] * (1 << n)
    choice = [0] * (1 << n)
    for S in range(1, 1 << n):
        best, arg = -1.0, -1
        for v in range(n):
            if not S & (1 << v):
                continue
            rest = S & ~(1 << v)
            g = f[rest] + sum(gain[u][v] for u in range(n) if rest & (1 << u))
            if g > best:
                best, arg = g, v
        f[S] = best
        choice[S] = arg
    order_rev: List[int] = []
    S = (1 << n) - 1
    while S:
        v = choice[S]
        order_rev.append(nodes[v])
        S &= ~(1 << v)
    return order_rev                # bottom chosen first -> reversed = top-first


def _greedy_order(nodes: List[int],
                  w: Dict[Tuple[int, int], float]) -> List[int]:
    """Insertion order by net upward pull, improved by adjacent swaps."""
    score = {n: 0.0 for n in nodes}
    for (u, v), weight in w.items():
        score[u] += weight
        score[v] -= weight
    order = sorted(nodes, key=lambda n: -score[n])   # top-first

    def gain_of(order_):
        pos = {n: k for k, n in enumerate(order_)}
        return sum(weight for (u, v), weight in w.items()
                   if pos[u] < pos[v])

    best = gain_of(order)
    improved = True
    while improved:
        improved = False
        for k in range(len(order) - 1):
            trial = order[:k] + [order[k + 1], order[k]] + order[k + 2:]
            g = gain_of(trial)
            if g > best + 1e-12:
                order, best = trial, g
                improved = True
    return order


def _order_block(nodes: List[int],
                 w: Dict[Tuple[int, int], float],
                 params: DepthParams) -> Tuple[List[int], int]:
    """One set of nodes as one top-first order, by the FAS machinery.

    Returns (order, 1 if the exact DP could be afforded else 0), so the caller
    can keep reporting how much of the scene was solved exactly.
    """
    if len(nodes) <= params.exact_max_nodes:
        # the DP hands back bottom-first; the rest of this file is top-first
        return _exact_order(list(nodes), w)[::-1], 1
    return _greedy_order(list(nodes), w), 0


def resolve_global_order(instance_ids: Sequence[int],
                         crossings: List[PredCrossing],
                         params: DepthParams) -> dict:
    """Impose one consistent vertical order per crossing-graph component.

    Edge weight = |score| of every decided crossing, directed toward its
    locally more likely relation. The solver maximises the total weight of
    satisfied relations (equivalently minimum-weight feedback arc set);
    relations it cannot satisfy are *flipped* -- kept, but recorded, and the
    raw local score stays stored. Returns a report with cycle counts.

    The weight used to be recovered as |logit(p_over)|, which is |score| again:
    `crossing_evidence` had just built p as sigmoid(score), so the pair of
    transforms cancelled. Reading the score directly is the same number without
    the round trip, and it keeps working for a scoring rule that never forms a
    probability at all.
    """
    w: Dict[Tuple[int, int], float] = {}
    for c in crossings:
        if c.abstain or c.raw_over < 0:
            continue
        lo = c.j if c.raw_over == c.i else c.i
        if np.isfinite(c.score):
            weight = abs(float(c.score))
        else:   # a caller that supplied p_over without a score
            p = min(max(c.p_over, 1e-6), 1 - 1e-6)
            weight = abs(math.log(p / (1.0 - p)))
        key = (c.raw_over, lo)
        w[key] = w.get(key, 0.0) + weight

    # components over decided edges
    parent = {i: i for i in instance_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v) in w:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
    comps: Dict[int, List[int]] = {}
    for i in instance_ids:
        comps.setdefault(find(i), []).append(i)

    # count raw precedence cycles (strongly connected components > 1)
    n_cycles = _count_cyclic_components(instance_ids, w)

    position: Dict[int, int] = {}
    used_exact = 0
    offset = 0
    # MEASURED DEFECT, recorded here rather than repaired. This pass rebuilds a
    # whole connected component's total order at once (`_order_block`, i.e.
    # `_exact_order` or `_greedy_order`), so it can reverse local relations that
    # lie on no contradiction cycle at all. Fed every ground-truth crossing with
    # its true direction -- input that is acyclic by construction, and that a
    # minimum-feedback-arc-set solver should therefore leave untouched -- it
    # still reverses 24 relations across the 20 held-out scenes, capping layer
    # ARI at 0.886 instead of 1.000. On real evidence 37 of the 102 relations it
    # reverses lie on no cycle.
    #
    # A condense-then-order variant -- topologically order the condensation of
    # the strongly connected components, and run the reordering only inside
    # genuine cycles -- reverses none of those and reaches ARI 1.000 at that
    # ceiling, but measured worse on the actual noisy predictions: Kendall tau
    # -0.024, mean K 5.80 -> 7.45 against a true 5.95, depth RMSE 18.95 -> 23.80
    # px. The broader flipping incidentally suppresses some erroneous local
    # relations: it makes 60 more reversals than the condensing variant and 34
    # of those 60 land correct, 57%, which is a coin toss that happened to come
    # up heads on this set. That advantage is an accident rather than a
    # mechanism and is expected to invert as the local evidence improves. It was
    # implemented and removed rather than kept switchable; the implementation is
    # in git history at commit 7d369d8.
    for comp in comps.values():
        cw = {(u, v): weight for (u, v), weight in w.items()
              if u in comp and v in comp}
        order, exact = _order_block(comp, cw, params)
        used_exact += exact
        # positions are offset per component into one global total order, so
        # every downstream constraint direction (including for abstained
        # crossings between components) is drawn from a single acyclic order
        for rank, node in enumerate(order):
            position[node] = offset + rank     # smaller = nearer the top
        offset += len(order)

    n_flipped = 0
    for c in crossings:
        if c.abstain or c.raw_over < 0:
            c.over = -1
            continue
        lo = c.j if c.raw_over == c.i else c.i
        if position[c.raw_over] < position[lo]:
            c.over = c.raw_over
        else:
            c.over = lo
            c.flipped = True
            n_flipped += 1

    return {"position": position,
            "n_components": len(comps),
            "n_precedence_cycles_raw": n_cycles,
            "n_relations_flipped": n_flipped,
            "n_components_exact": used_exact,
            "optimisation_status": "ok"}


def _strong_components(nodes: Sequence[int],
                       adjacency: Mapping[int, Sequence[int]]) -> List[List[int]]:
    """Tarjan's strongly connected components of one digraph.

    One implementation for the two callers that need it: the per-scene cycle
    COUNT (`_count_cyclic_components`) and the cycle MEMBERSHIP a reader needs
    when a crossing comes back flipped (`raw_cycle_groups`). Two near-identical
    copies of this loop used to live in this file; a third was not going to make
    it more correct.

    Iterative rather than recursive, for the same reason `assign_layers` is: a
    few hundred instances in one chain is quite enough to exhaust the
    interpreter stack.

    Edges pointing outside `nodes` are dropped rather than followed, so a
    caller may hand over an induced subgraph without pruning it first. Members
    come back sorted and the outer scan runs over `nodes` in the caller's
    order, so the result is deterministic.
    """
    inside = set(nodes)
    adj = {n: [v for v in adjacency.get(n, ()) if v in inside] for n in nodes}
    index: Dict[int, int] = {}
    low: Dict[int, int] = {}
    on_stack: Set[int] = set()
    stack: List[int] = []
    counter = 0
    out: List[List[int]] = []
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(adj[child])))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    members.append(member)
                    if member == node:
                        break
                out.append(sorted(members))
    return out


def _count_cyclic_components(nodes: Sequence[int],
                             w: Dict[Tuple[int, int], float]) -> int:
    """Number of strongly connected components with more than one node.

    That is the count of groups of local calls that cannot all be true at
    once -- the contradictions the global order then has to pay for.
    """
    adjacency: Dict[int, List[int]] = {n: [] for n in nodes}
    for (u, v) in w:
        if u in adjacency:
            adjacency[u].append(v)
    return sum(1 for members in _strong_components(nodes, adjacency)
               if len(members) > 1)


def compact_order(instance_ids: Sequence[int], crossings: List[PredCrossing]
                  ) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Positions for the ABSTAINED pairs that keep the film thin.

    A crossing the evidence could not decide still has to be separated, but
    nothing says which way round. `resolve_global_order` breaks that tie with
    the order its components happen to fall in, which for singleton components
    is instance-id rank -- and id rank has no reason to put two filaments that
    do not overlap each other on the same level, so the stack comes out deeper
    than the geometry requires. On a six-filament raft (each overlapping only
    its neighbour) id rank builds six levels where two suffice.

    Stated properly this is graph colouring with precedence:

      * every crossing pair is a CONFLICT -- the two may not share a level;
      * every DECIDED relation is a PRECEDENCE constraint -- the upper one must
        sit strictly higher;
      * minimise the number of levels.

    which is the "minimal K" objective `assign_layers` already applies to the
    decided relations, extended to the ones this stage abstained on. Colouring
    is NP-hard, so this is DSATUR (take the vertex with the most distinct
    neighbouring levels next, ties by conflict degree, then by id) run in an
    order that respects precedence, taking the lowest feasible level each time.
    Deterministic. Returns (position, level): `position` is the top-first map
    `resolve_global_order` produces, and `level` is the colouring itself, which
    is what a viewer slices by.
    """
    ids = list(instance_ids)
    above: Dict[int, Set[int]] = {i: set() for i in ids}
    conflict: Dict[int, Set[int]] = {i: set() for i in ids}
    for c in crossings:
        if c.i not in conflict or c.j not in conflict:
            continue
        conflict[c.i].add(c.j)
        conflict[c.j].add(c.i)
        if c.over >= 0:
            lo = c.j if c.over == c.i else c.i
            above[c.over].add(lo)

    pending = {i: len(above[i]) for i in ids}
    dependents: Dict[int, List[int]] = {i: [] for i in ids}
    for upper, lowers in above.items():
        for lower in lowers:
            dependents[lower].append(upper)

    level: Dict[int, int] = {}
    ready = {i for i in ids if pending[i] == 0}
    while ready:
        node = max(sorted(ready),
                   key=lambda n: (len({level[m] for m in conflict[n] if m in level}),
                                  len(conflict[n])))
        ready.discard(node)
        floor = max((level[m] + 1 for m in above[node] if m in level), default=0)
        taken = {level[m] for m in conflict[node] if m in level}
        chosen = floor
        while chosen in taken:
            chosen += 1
        level[node] = chosen
        for upper in dependents[node]:
            pending[upper] -= 1
            if pending[upper] == 0:
                ready.add(upper)
    for node in ids:
        level.setdefault(node, 0)

    ranked = sorted(ids, key=lambda n: (-level[n], n))
    return {node: rank for rank, node in enumerate(ranked)}, level


def stacked_z(instance_ids: Sequence[int], crossings: List[PredCrossing],
              radii: Dict[int, float], params: DepthParams,
              order_position: Dict[int, int]) -> Dict[int, float]:
    """Heights for a given vertical order, for COSTING that order.

    Identical computation to `solve_metric_z`, but takes the order as an
    argument instead of reading it from a report, which is what `run_scene`
    needs when it costs the two candidate arrangements against each other.
    """
    ids = list(instance_ids)
    r = {n: max(0.5, radii.get(n, params.default_radius_px)) for n in ids}
    below: Dict[int, List[Tuple[int, float]]] = {}
    for c in crossings:
        if c.i not in r or c.j not in r:
            continue
        if c.over >= 0:
            hi, lo = c.over, (c.j if c.over == c.i else c.i)
        elif c.i in order_position and c.j in order_position:
            hi, lo = ((c.i, c.j) if order_position[c.i] < order_position[c.j]
                      else (c.j, c.i))
        else:
            continue
        below.setdefault(hi, []).append((lo, r[hi] + r[lo] + params.z_gap_px))
    z = {n: 0.0 for n in ids}
    for n in sorted(ids, key=lambda n: (-order_position.get(n, 0), n)):
        for lo, sep in below.get(n, ()):
            z[n] = max(z[n], z[lo] + sep)
    floor = min(z.values(), default=0.0)
    return {n: value - floor for n, value in z.items()}


def film_thickness(instance_ids: Sequence[int], z: Dict[int, float],
                   radii: Dict[int, float], params: DepthParams) -> float:
    """Top surface to bottom surface, the number a reader actually quotes."""
    if not z:
        return 0.0
    top = max(z[i] + radii.get(i, params.default_radius_px) for i in instance_ids)
    bottom = min(z[i] - radii.get(i, params.default_radius_px) for i in instance_ids)
    return float(top - bottom)


def apply_abstention(crossings: List[PredCrossing],
                     band: Optional[float] = None,
                     params: Optional[DepthParams] = None) -> None:
    """(Re-)apply the abstention threshold to already-computed scores.

    Separate from `crossing_evidence` because moving the threshold is a
    decision change, not a re-measurement: the image features are the
    expensive part and they do not depend on it. A caller that wants to sweep
    the operating point measures once and calls this per value.

    Exactly one of `band` and `params` is required, because the two scoring
    families are parameterised differently and guessing would silently apply
    the wrong one:

      params=  uses `decision_threshold(params)`, i.e. whatever THAT run's
               scoring rule abstains at -- `abstain_score` in score units for
               "noise_floored", the converted `abstain_band` for the two
               legacy rules. This is the form that works for all three.
      band=    the legacy probability band directly, for a caller sweeping
               |p - 0.5| over a record made under a log-odds-shaped rule.
    """
    if params is not None and band is None:
        threshold = decision_threshold(params)
    elif band is not None and params is None:
        threshold = abstain_threshold(band)
    else:
        raise TypeError("apply_abstention takes exactly one of band= or "
                        "params=; they parameterise different scoring rules.")
    for c in crossings:
        # the score is the decision quantity; fall back to p_over for a caller
        # that supplied probabilities without one
        score = c.score
        if not np.isfinite(score) and np.isfinite(c.p_over):
            p = min(max(c.p_over, 1e-6), 1 - 1e-6)
            score = math.log(p / (1.0 - p))
        if not np.isfinite(score) or abs(score) < threshold:
            c.abstain, c.raw_over = True, -1
        else:
            c.abstain = False
            c.raw_over = c.i if score > 0.0 else c.j
        c.over = -1
        c.flipped = False


def raw_cycle_groups(instance_ids: Sequence[int],
                     crossings: List[PredCrossing]) -> Dict[int, int]:
    """instance -> index of the contradictory group it belongs to, if any.

    `_count_cyclic_components` answers "how many groups of local calls cannot
    all be true at once". This answers "which ones", which is what a reader
    needs when a crossing comes back FLIPPED: the flip is not arbitrary, it is
    the cheapest way to break a specific contradiction, and that contradiction
    can then be pointed at.

    Computed on the RAW local decisions, before the global order corrects
    anything, so the groups describe the evidence rather than the fix. Numbered
    by lowest member, so the numbering is stable.
    """
    adjacency: Dict[int, List[int]] = {i: [] for i in instance_ids}
    for c in crossings:
        if c.abstain or c.raw_over < 0:
            continue
        lower = c.j if c.raw_over == c.i else c.i
        if c.raw_over in adjacency and lower in adjacency:
            adjacency[c.raw_over].append(lower)

    groups = [members for members in _strong_components(instance_ids, adjacency)
              if len(members) > 1]
    groups.sort(key=lambda members: members[0])
    return {member: number for number, members in enumerate(groups)
            for member in members}


def _longest_path_levels(instance_ids: Sequence[int],
                         above: Dict[int, Set[int]]) -> Dict[int, int]:
    """Longest-path level per node over `above`; -1 for a node never reached.

    Iterative post-order, because a deep DAG must not exhaust the interpreter
    stack -- which a few hundred instances in one chain is quite enough to do.
    The outer scan follows `instance_ids` and children are expanded in `sorted`
    order, so the result is deterministic.
    """
    level = {i: -1 for i in instance_ids}
    for start in instance_ids:
        if level[start] >= 0:
            continue
        stack = [(start, False)]
        active: Set[int] = set()
        while stack:
            node, expanded = stack.pop()
            if expanded:
                active.discard(node)
                level[node] = max((level[b] for b in above[node] if level[b] >= 0),
                                  default=-1) + 1
                continue
            if level[node] >= 0 or node in active:
                continue
            active.add(node)
            stack.append((node, True))
            for below_node in sorted(above[node]):
                if level[below_node] < 0 and below_node not in active:
                    stack.append((below_node, False))
    return level


def stack_levels(instance_ids: Sequence[int], crossings: List[PredCrossing],
                 order_position: Optional[Dict[int, int]] = None
                 ) -> Tuple[Dict[int, int], int]:
    """Levels over EVERY constraint, not only the evidence-decided ones.

    `assign_layers` answers this stage's own question -- the minimum number of
    layers the DECIDED over/under relations force -- and stays exactly that.
    But an abstained or grazing pair still takes a direction from the global
    order and still separates in metric z, so with no image evidence at all
    `assign_layers` returns K = 1 for a film that is demonstrably hundreds of
    pixels thick.

    This is the other number: the longest chain in the constraint set that
    metric z actually realises. It can only be larger than or equal to K, and
    it is the one to slice a viewer by.
    """
    above: Dict[int, Set[int]] = {i: set() for i in instance_ids}
    for c in crossings:
        if c.over >= 0:
            hi, lo = c.over, (c.j if c.over == c.i else c.i)
        elif (order_position is not None and c.i in order_position
              and c.j in order_position):
            hi, lo = ((c.i, c.j) if order_position[c.i] < order_position[c.j]
                      else (c.j, c.i))
        else:
            continue
        if lo in above:
            above[hi].add(lo)
    level = _longest_path_levels(instance_ids, above)
    count = max(level.values(), default=-1) + 1
    return level, max(count, 1 if len(instance_ids) else 0)


# ── 4. layers ─────────────────────────────────────────────────────────────


def assign_layers(instance_ids: Sequence[int],
                  crossings: List[PredCrossing]) -> Tuple[Dict[int, int], int]:
    """Minimal-K layering from the corrected (acyclic) relations.

    layer 0 is the bottom. Only actual above-relations constrain; rods that
    never conflict share a layer, so K is inferred from the DAG's longest
    path, never from the instance or junction count.
    """
    above: Dict[int, Set[int]] = {i: set() for i in instance_ids}
    for c in crossings:
        if c.over < 0:
            continue
        lo = c.j if c.over == c.i else c.i
        above[c.over].add(lo)
    layer = _longest_path_levels(instance_ids, above)
    k = max(layer.values(), default=-1) + 1
    return layer, max(k, 1 if instance_ids else 0)


# ── 5. metric z under the compact-stack assumption ────────────────────────


def solve_metric_z(instance_ids: Sequence[int],
                   crossings: List[PredCrossing],
                   radii: Dict[int, float],
                   layers: Dict[int, int],
                   params: DepthParams,
                   order_position: Optional[Dict[int, int]] = None
                   ) -> Tuple[Dict[int, float], str]:
    """The thinnest stack the ordering allows.

        minimise   sum of z
        subject to z_hi - z_lo >= r_hi + r_lo + gap   for every overlapping pair

    This is the *compact-stack assumption* made concrete: a projected crossing
    does not imply physical contact, so the heights are only meaningful under a
    stated prior, and the prior taken here is that nothing floats higher than it
    must. Every instance ends up as low as its own constraints allow.

    Every projected crossing constrains. An ABSTAINED crossing still means two
    rods overlap in projection and must clear each other, so it contributes an
    inequality (direction taken from the global order) even though it expresses
    no preference about which is on top.

    Because all constraint directions come from one total order the edge set is
    acyclic, so a single bottom-up sweep -- visit instances from the bottom of
    the order upward, lifting each just enough to clear everything beneath it --
    is the exact optimum, in O(edges), with no solver.

    Until 2026-08-19 this instead minimised a weighted least-squares pull toward
    touching (`sum w (z_hi - z_lo - (r_hi + r_lo))^2`) with SLSQP. That variant
    satisfied the same constraints and produced the same film thickness, but it
    also pulled an instance UP toward whatever sat above it, so instances with
    nothing beneath them floated mid-stack. It was removed: on a 307-instance
    scene it took 18 s against this sweep's 0.001 s, and it needs a dense solver
    that an unhealthy LAPACK can bring the whole process down with. `git log`
    has it if the comparison is ever wanted again.
    """
    ids = list(instance_ids)
    pos = {n: k for k, n in enumerate(ids)}
    r = np.array([max(0.5, radii.get(n, params.default_radius_px))
                  for n in ids])

    edges: List[Tuple[int, int]] = []
    for c in crossings:
        if c.over >= 0:
            lo = c.j if c.over == c.i else c.i
            edges.append((c.over, lo))
        elif order_position is not None \
                and c.i in order_position and c.j in order_position:
            hi, lo = ((c.i, c.j)
                      if order_position[c.i] < order_position[c.j]
                      else (c.j, c.i))
            edges.append((hi, lo))
    if not edges:
        return {n: 0.0 for n in ids}, "no_edges"

    below: Dict[int, List[Tuple[int, float]]] = {}
    for hi, lo in edges:
        below.setdefault(hi, []).append(
            (lo, r[pos[hi]] + r[pos[lo]] + params.z_gap_px))
    if order_position is not None:
        bottom_up = sorted(ids, key=lambda n: (-order_position.get(n, 0), n))
    else:
        bottom_up = sorted(ids, key=lambda n: (layers.get(n, 0), n))

    z = np.zeros(len(ids))
    for n in bottom_up:
        for lo, sep in below.get(n, ()):
            z[pos[n]] = max(z[pos[n]], z[pos[lo]] + sep)
    z = z - float(z.min())
    return {n: float(v) for n, v in zip(ids, z)}, "compact_stack"


# ── top level ─────────────────────────────────────────────────────────────


def run_scene(image: np.ndarray,
              centrelines: Dict[int, np.ndarray],
              radii: Optional[Dict[int, float]] = None,
              params: Optional[DepthParams] = None,
              instance_map: Optional[Dict[int, int]] = None,
              with_metric_z: bool = True,
              sem_image=None) -> dict:
    """Full post-hoc depth pass over one scene.

    `image` is the registered grayscale in [0, 1]; `centrelines` maps
    instance id -> ordered (K, 2) [x, y] polyline (PLECTA chains or oracle);
    `radii` per-instance physical radius estimates in px (falls back to
    `DepthParams.default_radius_px`). When `radii` is empty and `sem_image`
    (an `plecta.image.measurement.SemImage`) is given, diameters
    are measured from perpendicular FWHM profiles and converted through the
    domain broadening correction; the observed widths are stored either way.
    Returns the pred_depth dict.
    """
    params = params or DepthParams()
    radii = dict(radii or {})
    scored = params.use_image_evidence and image is not None
    ids = sorted(centrelines)

    crossings = identify_crossings(centrelines, params)
    widths: Dict[int, dict] = {}
    if not radii and sem_image is not None:
        widths = measure_diameters(sem_image, centrelines, crossings, params)
        for iid, rec in widths.items():
            d_est = width_to_diameter(rec["w_obs"])
            if d_est is not None:
                radii[iid] = d_est / 2.0
    if params.radius_mode == "fixed":
        # one radius for everything: what a run with no width measurement has
        radii = {i: params.default_radius_px for i in centrelines}

    # An instance whose width could not be measured has no radius, and every
    # downstream lookup would fall through to default_radius_px -- a constant
    # picked once, with no relation to this scene. It is used for the sampling
    # geometry, the grazing test, the clearance constraint and the exported
    # tube, so getting it badly wrong is not cosmetic. The scene's own median
    # is the better estimator and is applied only where there is nothing to
    # measure; if NOTHING was measured there is no median to take and the
    # constant still applies.
    imputed_ids: Set[int] = set()
    radius_fallback_px = float(params.default_radius_px)
    missing = [i for i in ids if i not in radii]
    if missing and radii and params.radius_fallback == "median":
        radius_fallback_px = float(np.median(list(radii.values())))
        for iid in missing:
            radii[iid] = radius_fallback_px
        imputed_ids = set(missing)

    dense = {i: _densify(np.asarray(p)) for i, p in centrelines.items()}
    grad = None
    if scored:
        image = np.asarray(image, dtype=np.float32)
        # The gradient image exists for exactly one consumer, the sharpness
        # channel, and the shipped rule has no sharpness channel. Building it
        # anyway would be a Gaussian plus two Sobels over the whole frame, per
        # scene, thrown away unread -- so on that path it is not built at all
        # and `crossing_evidence` is handed None.
        if params.scoring != "noise_floored":
            grad = gradient_magnitude(image)
        for c in crossings:
            crossing_evidence(image, grad, dense, c, params, radii)
    else:
        # No image, or evidence switched off: every crossing abstains. The
        # geometry still solves and nothing that IS constrained interpenetrates,
        # but the vertical order is then a deterministic tie-break, not a
        # measurement -- `evidence: false` in the report is what says so.
        for c in crossings:
            c.abstain = True
            c.raw_over = -1
            c.features["reason"] = 2.0

    # Grazing pairs -- strokes overlapping in projection without the
    # centrelines ever meeting. By default they carry no evidence and enter as
    # abstained: clearance, no opinion. Keeping them in the same list is what
    # makes them constrain metric z through the ordinary abstained path.
    contacts = identify_contacts(
        centrelines, radii, params,
        exclude={(min(c.i, c.j), max(c.i, c.j)) for c in crossings})
    graze_scored = bool(params.grazing_evidence and scored)
    if graze_scored:
        # Ask the image about them instead. `identify_contacts` puts each pair
        # at its point of CLOSEST APPROACH, and that point is where the two
        # strokes overlap, so the same occlusion question a real crossing is
        # asked has a subject here: does the overlap wear rod i's appearance
        # or rod j's? `crossing_evidence` reads its windows off each rod's own
        # centreline near that point, so it needs no shared intersection to
        # exist -- but it does assume the flanks are unoccluded, which for a
        # pair that runs alongside for longer than the window is not true.
        # Those pairs come back with a near-zero score and abstain, which is
        # the behaviour wanted: the ones that earn a direction earn it, the
        # rest keep the clearance-only treatment they already had.
        #
        # abstain is cleared first because these arrive abstained by
        # construction, and re-scoring has to be free to decide them.
        for c in contacts:
            c.abstain = False
            c.raw_over = -1
            crossing_evidence(image, grad, dense, c, params, radii)
    decided_grazing = [c for c in contacts
                       if not c.abstain and c.raw_over >= 0]
    # `clear_grazing_overlaps` still means "clear ALL of them, decided or not".
    # With only `grazing_evidence` on, a pair constrains exactly when the
    # evidence decided it. Separating a pair on a direction nobody measured is
    # measured to HURT -- Kendall tau 0.3541 -> 0.3420 over the 20 held-out
    # scenes with predicted instances -- so nothing is separated on a guess.
    constrained = crossings + (contacts if params.clear_grazing_overlaps
                               else decided_grazing)

    report = resolve_global_order(ids, constrained, params)
    layers, k = assign_layers(ids, constrained)
    z = None
    z_status = "not_requested"
    placement = "id_rank"
    if with_metric_z:
        if params.undecided_order == "compact":
            # Colouring minimises the LEVEL COUNT, but what a reader quotes is
            # the thickness, and thickness is the longest WEIGHTED chain: with
            # unequal radii an arrangement with fewer levels can still be
            # thicker (measured, on a real 307-instance scene: 113.5 px against
            # 111.8 px at the same eight levels). So both orders are costed and
            # the thinner one is kept -- compact is then never worse than
            # id_based.
            #
            # The costing uses `stacked_z` because it takes the order as an
            # argument; `solve_metric_z` reads one from a report, and there is
            # no report for a candidate arrangement. The sweep underneath is
            # the same, so this is not a cheaper approximation of it.
            coloured, _levels = compact_order(ids, constrained)
            if (film_thickness(ids, stacked_z(ids, constrained, radii, params,
                                              coloured), radii, params)
                    < film_thickness(ids, stacked_z(ids, constrained, radii, params,
                                                    report["position"]),
                                     radii, params) - 1e-9):
                report["position"] = coloured
                placement = "coloured"
        z, z_status = solve_metric_z(ids, constrained, radii, layers, params,
                                     order_position=report.get("position"))
    # The gate always measures BOTH sets, whatever was constrained, so turning
    # `clear_grazing_overlaps` off reports the interpenetration it leaves
    # instead of hiding it.
    violations = ([] if z is None
                  else clearance_violations(crossings + contacts, z, radii, params))
    report.update(n_layers=k, metric_z_status=z_status,
                  n_abstained=sum(c.abstain for c in crossings),
                  n_crossings=len(crossings),
                  image_evidence=bool(scored),
                  # What the evidence path actually was, so a stored run says
                  # which rule and which sampling geometry produced it rather
                  # than depending on the parameters.yaml of the day.
                  scoring=params.scoring,
                  abstain_threshold=round(decision_threshold(params), 6),
                  core_mode=params.core_mode,
                  core_px=float(params.core_px),
                  radius_mode=params.radius_mode,
                  n_grazing_overlaps=len(contacts),
                  grazing_constrained=bool(params.clear_grazing_overlaps),
                  grazing_evidence=graze_scored,
                  n_grazing_decided=len(decided_grazing),
                  radius_fallback=params.radius_fallback,
                  n_radius_imputed=len(imputed_ids),
                  radius_fallback_px=round(radius_fallback_px, 4),
                  n_interpenetrating=len(violations),
                  undecided_order=params.undecided_order,
                  undecided_placement=placement)

    out = {
        "assumptions": {
            "planar_instances": "dz/ds = 0",
            "cross_section": "circular",
            "metric_z_prior": "compact stack (equality pull toward contact)",
            "evidence": (
                "none: every crossing abstained" if not scored else
                ("occlusion continuity (intensity, noise-floored; "
                 "one channel, no radius)"
                 if params.scoring == "noise_floored"
                 else "occlusion continuity (intensity + edge sharpness)")),
        },
        "instances": [
            {"id": int(i),
             "z": (None if z is None else round(z[i], 4)),
             "layer": int(layers[i]),
             "d": (round(2.0 * radii[i], 4) if i in radii else None),
             # `d` used to be null for exactly the instances that had no width;
             # imputing one fills it, so the marker moves here rather than
             # disappearing.
             **({"d_imputed": True} if i in imputed_ids else {}),
             "centerline_xy": np.asarray(centrelines[i]).round(2).tolist(),
             **({"w_obs": widths[i]["w_obs"],
                 "w_mad": widths[i]["w_mad"],
                 "n_cuts": widths[i]["n_cuts"],
                 "width_quality": widths[i]["quality"]}
                if i in widths else {})}
            for i in ids
        ],
        "crossings": [
            {"i": int(c.i), "j": int(c.j),
             "x": round(c.x, 2), "y": round(c.y, 2),
             "score": (None if not np.isfinite(c.score)
                       else round(float(c.score), 4)),
             "p_over": (None if not np.isfinite(c.p_over)
                        else round(float(c.p_over), 4)),
             "abstain": bool(c.abstain),
             "raw_over": (None if c.raw_over < 0 else int(c.raw_over)),
             "over": (None if c.over < 0 else int(c.over)),
             "flipped": bool(c.flipped),
             # marks a closest-approach contact rather than a projected
             # centreline intersection; only ever present with
             # `grazing_evidence` on, so the default output is unchanged
             **({"grazing": True} if c.grazing else {}),
             "features": {k_: round(float(v), 4)
                          for k_, v in c.features.items()}}
            # A grazing pair the evidence DECIDED is a real over/under call
            # about a real projected overlap, and it is reported as one. The
            # ones that abstained are not: they would be indistinguishable
            # from an undecidable crossing while describing a pair whose
            # centrelines never meet at all.
            for c in crossings + (decided_grazing if graze_scored else [])
        ],
        "solver_report": report,
    }
    if instance_map is not None:
        out["instance_map"] = {str(k_): int(v) for k_, v in instance_map.items()}
    return out


def write_pred_depth(path, pred: dict) -> None:
    Path(path).write_text(json.dumps(pred, indent=1), encoding="utf-8")


# ── diameter measurement along instance centrelines ───────────────────────


# ── 3-D tube geometry ─────────────────────────────────────────────────────
#
# Kept behind a small interface so the circular section can later be swapped
# for ellipses/ribbons: `section_fn(theta) -> (u, v)` gives the section curve
# in the local (in-plane normal, e_z) frame, scaled by the local radius.


def _circular_section(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.cos(theta), np.sin(theta)


def tube_mesh(centreline_xy: np.ndarray, z: float, radius,
              n_theta: int = 12, step: float = 2.0,
              section_fn=_circular_section):
    """Swept-tube mesh for one planar instance.

    S(s, theta) = [x(s), y(s), z] + r(s) * (cos(theta) n(s) + sin(theta) e_z)
    with n(s) the in-plane unit normal of the planar centreline. Returns
    (vertices (N, 3) float32, quad faces (M, 4) int32). `radius` is a scalar
    or per-point array.
    """
    pts = _densify(np.asarray(centreline_xy, dtype=np.float64), step=step)
    if len(pts) < 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int32)
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(pts),))
    normal = _unit_normals(pts)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cu, sv = section_fn(theta)
    verts = np.empty((len(pts), n_theta, 3), dtype=np.float64)
    verts[:, :, 0] = pts[:, 0, None] + r[:, None] * cu[None, :] * normal[:, 0, None]
    verts[:, :, 1] = pts[:, 1, None] + r[:, None] * cu[None, :] * normal[:, 1, None]
    verts[:, :, 2] = z + r[:, None] * sv[None, :]
    faces = []
    for k in range(len(pts) - 1):
        base0 = k * n_theta
        base1 = (k + 1) * n_theta
        for t in range(n_theta):
            t2 = (t + 1) % n_theta
            faces.append((base0 + t, base0 + t2, base1 + t2, base1 + t))
    return (verts.reshape(-1, 3).astype(np.float32),
            np.asarray(faces, dtype=np.int32))


def scene_tubes(centrelines: Dict[int, np.ndarray],
                z: Dict[int, float], radii: Dict[int, float],
                default_radius: float = 5.0,
                n_theta: int = 12) -> Dict[int, tuple]:
    """One (vertices, faces) mesh per instance."""
    out = {}
    for iid, poly in centrelines.items():
        out[iid] = tube_mesh(poly, z.get(iid, 0.0),
                             radii.get(iid, default_radius), n_theta=n_theta)
    return out


def export_ply(path, meshes: Dict[int, tuple]) -> None:
    """ASCII PLY of all instance tubes, vertex-coloured by instance id."""
    palette = [(31, 111, 178), (224, 122, 31), (46, 158, 107),
               (176, 58, 106), (124, 92, 196), (138, 106, 58),
               (212, 80, 135), (76, 163, 199), (154, 159, 31),
               (192, 57, 43), (59, 122, 87), (142, 68, 173)]
    verts_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    offset = 0
    for k, iid in enumerate(sorted(meshes)):
        verts, faces = meshes[iid]
        if not len(verts):
            continue
        colour = np.tile(np.asarray(palette[k % len(palette)], np.uint8),
                         (len(verts), 1))
        verts_all.append(verts)
        cols_all.append(colour)
        faces_all.append(faces + offset)
        offset += len(verts)
    verts = np.concatenate(verts_all) if verts_all else np.zeros((0, 3))
    cols = np.concatenate(cols_all) if cols_all else np.zeros((0, 3), np.uint8)
    faces = np.concatenate(faces_all) if faces_all else np.zeros((0, 4), int)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(verts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {len(faces)}\n")
        fh.write("property list uchar int vertex_indices\nend_header\n")
        for v, c in zip(verts, cols):
            fh.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
        for f in faces:
            fh.write(f"4 {f[0]} {f[1]} {f[2]} {f[3]}\n")


def export_obj(path, meshes: Dict[int, tuple]) -> None:
    """Wavefront OBJ, one named object per instance."""
    with open(path, "w", encoding="ascii") as fh:
        offset = 1
        for iid in sorted(meshes):
            verts, faces = meshes[iid]
            if not len(verts):
                continue
            fh.write(f"o instance_{iid}\n")
            for v in verts:
                fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for f in faces:
                fh.write(f"f {f[0]+offset} {f[1]+offset} {f[2]+offset} "
                         f"{f[3]+offset}\n")
            offset += len(verts)


# ── centrelines from PLECTA's own chains (predicted mode) ─────────────────


def centrelines_from_chains(graph, matching: Dict[int, int],
                            chains: List[List[int]]) -> Dict[int, np.ndarray]:
    """Ordered [x, y] centreline per chain, walked along the matching.

    The walk itself is `plecta.image.bundles.order_chain` -- the one the width
    layer already uses -- so a chain's centreline is the same curve whichever
    stage asks for it, and there is one traversal to keep correct rather than
    two. Those helpers work in (row, col); the depth stage works in (x, y), so
    the two columns are swapped here. Instance ids follow
    `predict.instances_from_chains` numbering (chain index + 1).
    """
    from .image.bundles import chain_points, order_chain

    out: Dict[int, np.ndarray] = {}
    for key, chain in enumerate(chains, start=1):
        if not chain:
            continue
        pts, _aids = chain_points(graph, order_chain(graph, matching, chain))
        if len(pts) >= 2:
            out[key] = pts[:, ::-1].astype(np.float64)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────


def _read_gray(path) -> np.ndarray:
    """Grayscale in [0, 1] -- the image layer's reader, not a second one."""
    from .image.measurement import read_gray

    return read_gray(path)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Post-hoc depth stage (DEPTH_MODE=posthoc). The 2-D "
                    "grouping is not changed; oracle mode skips it entirely.")
    ap.add_argument("--scene", required=True,
                    help="scene folder containing the grayscale image")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--sem-name", default="sem.png")
    ap.add_argument("--mask-name", default="mask_w2.png")
    ap.add_argument("--oracle", action="store_true",
                    help="use ground-truth 2-D centrelines from "
                         "gt_depth.json instead of running PLECTA")
    ap.add_argument("--oracle-radii", action="store_true",
                    help="use ground-truth radii (with --oracle)")
    ap.add_argument("--no-metric-z", action="store_true")
    ap.add_argument("--undecided-order", choices=("compact", "id_based"),
                    default=None,
                    help="how to stack pairs the evidence could not decide "
                         "(default: whatever parameters.yaml says)")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    metavar="KEY=VALUE",
                    help="override any depth_3d parameter for this run")
    ap.add_argument("--no-image-evidence", action="store_true",
                    help="do not read the image: every crossing abstains and "
                         "the vertical order becomes a tie-break")
    ap.add_argument("--clear-grazing", action="store_true",
                    help="also clear pairs that overlap in projection without "
                         "their centrelines crossing")
    ap.add_argument("--grazing-evidence", action="store_true",
                    help="read the image at those pairs' closest approach too, "
                         "so each either earns an over/under or abstains")
    ap.add_argument("--fixed-radius", type=float, default=None,
                    help="give every instance this radius in px instead of a "
                         "measured one")
    ap.add_argument("--tubes", action="store_true",
                    help="also export tubes.ply / tubes.obj")
    args = ap.parse_args(argv)

    scene = Path(args.scene)
    image = _read_gray(scene / args.sem_name)
    # Every tuned value comes from parameters.yaml, like every other stage;
    # the flags below are overrides on top of it, not a second source of truth.
    from .parameters import build

    overrides = list(args.overrides)
    if args.undecided_order:
        overrides.append(f"undecided_order={args.undecided_order}")
    if args.no_image_evidence:
        overrides.append("use_image_evidence=false")
    if args.clear_grazing:
        overrides.append("clear_grazing_overlaps=true")
    if args.grazing_evidence:
        overrides.append("grazing_evidence=true")
    if args.fixed_radius:
        overrides.append("radius_mode=fixed")
        overrides.append(f"default_radius_px={args.fixed_radius}")
    params = build(DepthParams, overrides)
    radii: Dict[int, float] = {}

    if args.oracle:
        gt = json.loads((scene / "gt_depth.json").read_text(encoding="utf-8"))
        centrelines = {int(r["id"]): np.asarray(r["centerline_xy"], float)
                       for r in gt["instances"]}
        if args.oracle_radii:
            radii = {int(r["id"]): float(r["d_mean"]) / 2.0
                     for r in gt["instances"]}
    else:
        from .graph import read_mask
        from .predict import load_params, predict

        mask = read_mask(scene / args.mask_name)
        _masks, graph, matching, chains = predict(
            mask, load_params(), return_internals=True)
        centrelines = centrelines_from_chains(graph, matching, chains)

    sem_img = None
    if not radii:
        from plecta.image.measurement import load_sem

        sem_img = load_sem(scene / args.sem_name)
    pred = run_scene(image, centrelines, radii=radii, params=params,
                     with_metric_z=not args.no_metric_z, sem_image=sem_img)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pred_depth(out_dir / "pred_depth.json", pred)
    if args.tubes:
        zmap = {r["id"]: r["z"] for r in pred["instances"]
                if r["z"] is not None}
        rmap = {r["id"]: r["d"] / 2.0 for r in pred["instances"]
                if r.get("d")}
        meshes = scene_tubes(centrelines, zmap, rmap,
                             params.default_radius_px)
        export_ply(out_dir / "tubes.ply", meshes)
        export_obj(out_dir / "tubes.obj", meshes)
        print(f"  tubes -> {out_dir / 'tubes.ply'}")
    rep = pred["solver_report"]
    if rep["grazing_evidence"]:
        grazing_note = f" ({rep['n_grazing_decided']} decided)"
    elif rep["grazing_constrained"]:
        grazing_note = ""
    else:
        grazing_note = " (free)"
    print(f"{scene.name}: {len(centrelines)} instances, "
          f"{rep['n_crossings']} crossings "
          f"({rep['n_abstained']} abstained, "
          f"{rep['n_relations_flipped']} flipped, "
          f"{rep['n_precedence_cycles_raw']} raw cycles), "
          f"K={rep['n_layers']}, "
          f"{rep['n_grazing_overlaps']} grazing{grazing_note}, "
          f"{rep['n_interpenetrating']} interpenetrating")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
