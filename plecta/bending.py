"""Heights that let a filament bend, instead of holding it flat.

THE OPTION. `height_model = "bending"` in `DepthParams` selects this module;
the default `"compact_stack"` never imports anything from it at run time and
is untouched by it. Everything bending-specific lives here; the hooks
elsewhere (`depth.run_scene`, `depth.tube_mesh`, `parameters.yaml`) are
fenced with `# -- bending option --` comments. See BENDING.md for the method
and for how to remove the option.

`solve_metric_z` gives every instance ONE height, which is the assumption
`assumptions.planar_instances: dz/ds = 0` states. A filament that passes over
one neighbour and under the next cannot be drawn that way, so each transitive
over/under step is paid as global stack height, and the film grows with the
crossing graph rather than with the specimen.

This solves a height every few pixels along each filament instead, under the
same compact prior -- nothing floats higher than it must -- plus one physical
constraint: a filament may not be bent more sharply than its own stiffness
and adhesion allow. Bending enters as a CONSTRAINT rather than an energy, so
the whole thing stays a linear programme.

What is solved
--------------
For filament n, heights h_n,k at knots spaced d_n apart along its in-plane
arclength (about `bend_sample_px`, with a knot at each end). Between knots
the profile is the cubic Hermite curve with central-difference slopes
(Catmull-Rom); that curve is linear in the knot heights, so its curvature can
be bounded INSIDE the LP and the filament that is drawn is the filament that
was constrained, with no safety factor.

    minimise    sum_n d_n * sum_k h_n,k                      (lifted area)
    then, among the minimisers,   sum |h_n,k+1 - h_n,k|      (flatness)

    subject to  h_n,k >= r_n                                 (on the substrate)
                h_hi(s*) - h_lo(t*) >= r_hi + r_lo + gap      at every constrained
                                                             crossing, imposed on
                                                             the knots bracketing
                                                             it with the chord
                                                             margin d^2/(8 R)
                h_hi,a - h_lo,b >= sqrt(sep^2 - delta_ab^2)  for every knot pair
                                                             of the two within
                                                             sep in projection
                |z_n''(s)| <= 1 / R_min(d_n)                 on the drawn curve,
                                                             everywhere
                z_n'(0) = z_n'(L_n) = 0                      flat at both ends

The two objectives are lexicographic: the film is first as compact as the
constraints allow (the flat model's prior, integrated along each filament),
and only ties are broken toward flatness. There is no exchange rate between
height and shape to choose.

The order
---------
Directions come from `PredCrossing.over` where the evidence decided one and
from the global `order_position` where it abstained -- exactly as
`solve_metric_z` reads them. Nothing here writes to a crossing, to the order,
or to a centreline: bending changes heights and only heights.

The one number
--------------
rho is the arclength, in filament diameters, over which a filament rises one
diameter. It follows from balancing the energy of bending against the
adhesion given up by lifting off a neighbour, for a clamped cubic ramp of
rise h over length L: E = 6 EI h^2 / L^3 + w_a L, minimal at L^4 = 18 EI h^2
/ w_a, so with h = d and L = rho d

    rho^4 = 18 EI / (w_a d^2).

The end curvature of that ramp, 6 h / L^2, is independent of the rise, and
is the constraint applied:

    R_min = rho^2 d / 6 = sqrt(EI / (2 w_a)),

which is the classical peel radius -- the curvature at which the bending
energy per unit length EI kappa^2 / 2 equals the adhesion per unit length
w_a. It does not depend on the ramp shape assumed; only rho's literal
reading does (the solved shape, two arcs of radius R_min, rises one diameter
in 2 sqrt(R_min d) = 0.82 rho d).

For a rope of (10,10) tubes free to shear past each other, EI = N EI_1 and N
grows as d^2, so the d^2 cancels: rho does not depend on the bundle diameter,
and is about 8.1 for the constants below. R_min itself is proportional to d,
so a thicker bundle is held to a gentler arc. The default is 8.09; 0 asks
`rho_shear_free` for it at run time. rho -> infinity is the flat model: zero
curvature and flat ends leave every filament horizontal, and the compact
prior then places it exactly where `solve_metric_z` does.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: (10,10) armchair geometry and rope constants; see `rho_shear_free`.
#: Alternatives from the same literature, and the rho each gives, are
#: tabulated in BENDING.md; rho goes as the fourth root of every one of
#: them, which is why the choice among them moves it by a fifth at most.
_A_CC_NM = 0.142            # C-C bond length
_TUBE_D_NM = _A_CC_NM * np.sqrt(3.0) * np.sqrt(300.0) / np.pi   # 1.356 nm
_ROPE_A0_NM = 1.67          # hexagonal lattice constant of the rope, Thess 1996 / Lu 1997
_GAP_NM = 0.317             # wall-to-wall spacing, a0 - d_tube
_C_INPLANE = 345.0          # N/m, graphene in-plane stiffness (Kudin,
                            # Scuseria & Yakobson 2001); Yakobson 1996 gave 360
_W_ADHESION = 0.44e-9       # N, three tube-pair lines of contact at the
                            # Girifalco 2000 well depth; the adhesion a bundle
                            # gives up per unit length lifted off a neighbour

#: Weight on total height variation, in px^2 of lifted area per px of
#: variation, or None for the two-stage lexicographic solve (area first, then
#: variation among its minimisers). The objective is lexicographic in intent
#: -- there is no exchange rate between height and shape to choose -- and
#: 1e-3 is a tie-break below the solver's tolerance rather than a trade:
#: on B58-B3-S2_100 it leaves the lifted area within 1e-9 of the two-stage
#: minimum and the variation within 4e-5 of it, at one ninth of the cost
#: (0.7 s against 6.2 s). None selects the two-stage solve for verification.
_FLATNESS_WEIGHT: Optional[float] = 1e-3

#: Sample points per knot interval at which the drawn curve is held clear of
#: an overlapping neighbour. The curve is linear in the knot heights at any
#: point, so these are extra linear rows, not extra variables. 1 clears the
#: knots only; the requirement sqrt(sep^2 - delta^2) then changes between
#: knots and a residual interpenetration of up to 0.29 px was measured on
#: the reference field; BENDING.md tabulates the residual and cost per value.
_ZONE_SUBSAMPLES = 2

#: Second-difference cap, in px, below which a chain is pinned exactly
#: horizontal rather than bounded. A chain of m knots with |d2h| <= cap
#: strays at most cap m^2 / 8 from straight, 0.03 px for 500 knots at this
#: value, while HiGHS reports caps of 1e-9 and below as infeasible.
_STRAIGHT_CAP_PX = 1e-6


def tube_bending_stiffness() -> float:
    """EI of one (10,10) tube as a thin shell, pi C R^3, in N m^2."""
    return float(np.pi * _C_INPLANE * (_TUBE_D_NM * 1e-9 / 2.0) ** 3)


def rope_tube_count(diameter_nm: float) -> float:
    """Tubes in a hexagonally packed rope of this outer diameter."""
    d = max(float(diameter_nm), _TUBE_D_NM)
    return max(1.0, (np.pi / 4.0) * (d - _GAP_NM) ** 2
               / ((np.sqrt(3.0) / 2.0) * _ROPE_A0_NM ** 2))


def bundle_stiffness(diameter_nm: float) -> float:
    """EI of the rope with the tubes free to shear: N EI_1, in N m^2."""
    return rope_tube_count(diameter_nm) * tube_bending_stiffness()


def peel_radius_nm(diameter_nm: float) -> float:
    """sqrt(EI / (2 w_a)): the bend radius at which bending energy per unit
    length equals the adhesion per unit length. This is `min_radius_px` in
    physical units, and does not depend on any assumed ramp shape."""
    return float(np.sqrt(bundle_stiffness(diameter_nm) / (2.0 * _W_ADHESION))
                 * 1e9)


def rho_shear_free(diameter_nm: float = 0.0) -> float:
    """Diameters of arclength a filament needs to rise one diameter.

    From rho^4 = 18 EI / (w_a d^2) with the tubes free to shear, where the
    diameter cancels. The argument is accepted only so a caller can show it;
    the answer moves by under 1 % across 10-25 nm bundles.
    """
    d_nm = max(float(diameter_nm) or 13.4, _TUBE_D_NM)
    ei = bundle_stiffness(d_nm)
    return float((18.0 * ei / (_W_ADHESION * (d_nm * 1e-9) ** 2)) ** 0.25)


def min_radius_px(rho: float, diameter_px: float) -> float:
    """rho^2 d / 6: the tightest bend allowed, in the units of `diameter_px`.

    Equal to sqrt(EI / (2 w_a)) when rho is `rho_shear_free`; the radius the
    energy balance gives for a ramp of ANY rise.
    """
    return float(rho) * float(rho) * float(diameter_px) / 6.0


def ramp_px(rho: float, diameter_px: float) -> float:
    """Arclength over which the solved shape (two arcs of R_min) rises one
    diameter: 2 sqrt(R_min d) = rho d sqrt(2/3)."""
    return 2.0 * float(np.sqrt(min_radius_px(rho, diameter_px) * diameter_px))


def _arclength(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros(len(p))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def project_arc(points: np.ndarray, x: float, y: float) -> float:
    """Arclength along `points` of the point on the polyline nearest (x, y).

    A crossing lands between two centreline vertices, not on one; this is
    the exact foot of the perpendicular on the nearest segment.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        return 0.0
    a, ab = p[:-1], p[1:] - p[:-1]
    ln2 = (ab ** 2).sum(axis=1)
    q = np.array([x, y], dtype=float)
    t = np.clip(((q - a) * ab).sum(axis=1) / np.where(ln2 > 0, ln2, 1.0),
                0.0, 1.0)
    foot = a + t[:, None] * ab
    k = int(np.argmin(((foot - q) ** 2).sum(axis=1)))
    return float(_arclength(p)[k] + t[k] * np.sqrt(ln2[k]))


def height_profile(points: np.ndarray,
                   sample_at: Sequence[float],
                   sample_h: Sequence[float],
                   at: np.ndarray = None) -> np.ndarray:
    """The solved profile at every centreline point (or at `at`).

    The curve between knots is the cubic Hermite interpolant with
    central-difference slopes -- the SAME curve whose curvature the LP
    bounded -- so what is drawn is what was constrained. A PCHIP or a spline
    threaded through the knots afterwards is not: measured on a real field,
    PCHIP reached three times the curvature the chain was allowed.
    """
    from scipy.interpolate import CubicHermiteSpline

    cum = _arclength(points) if at is None else np.asarray(at, dtype=float)
    x = np.asarray(sample_at, dtype=float)
    y = np.asarray(sample_h, dtype=float)
    if len(cum) == 0:
        return np.zeros(0)
    if len(x) < 2:
        return np.full(len(cum), float(y[0]) if len(y) else 0.0)
    curve = CubicHermiteSpline(x, y, np.gradient(y, x))
    return np.asarray(curve(np.clip(cum, x[0], x[-1])), dtype=float)


def _hermite_curvature_rows(m: int) -> List[Dict[int, float]]:
    """Second derivative of the Catmull-Rom Hermite curve at both ends of
    every knot interval, as stencils over local knot indices, for a chain
    of m knots at unit spacing. z'' is linear within an interval, so
    bounding it at the two ends bounds it everywhere on the curve.

    z(u) = h_k H00 + m_k H10 + h_k+1 H01 + m_k+1 H11 on interval k, with
    m_k = (h_k+1 - h_k-1)/2 inside and one-sided at the ends;
    z''(0) = -6 h_k - 4 m_k + 6 h_k+1 - 2 m_k+1,
    z''(1) =  6 h_k + 2 m_k - 6 h_k+1 + 4 m_k+1.
    """
    if m < 2:
        return []
    slope: List[Dict[int, float]] = []
    for k in range(m):
        if k == 0:
            slope.append({0: -1.0, 1: 1.0})
        elif k == m - 1:
            slope.append({m - 2: -1.0, m - 1: 1.0})
        else:
            slope.append({k - 1: -0.5, k + 1: 0.5})

    def combine(*terms):
        out: Dict[int, float] = {}
        for coef, stencil in terms:
            for j, v in stencil.items():
                out[j] = out.get(j, 0.0) + coef * v
        return {j: v for j, v in out.items() if abs(v) > 1e-12}

    rows = []
    for k in range(m - 1):
        p0, p1 = {k: 1.0}, {k + 1: 1.0}
        rows.append(combine((-6.0, p0), (-4.0, slope[k]), (6.0, p1),
                            (-2.0, slope[k + 1])))
        rows.append(combine((6.0, p0), (2.0, slope[k]), (-6.0, p1),
                            (4.0, slope[k + 1])))
    return rows


def _directed(crossings, inside, order_position):
    """(k, hi, lo) per constrained crossing, read as `solve_metric_z` reads
    them: `over` where decided, the global order where abstained."""
    out = []
    for k, c in enumerate(crossings):
        if c.i not in inside or c.j not in inside:
            continue
        if c.over >= 0:
            hi, lo = c.over, (c.j if c.over == c.i else c.i)
        elif c.i in order_position and c.j in order_position:
            hi, lo = ((c.i, c.j) if order_position[c.i] < order_position[c.j]
                      else (c.j, c.i))
        else:
            continue
        out.append((k, hi, lo))
    return out


def _combine(*terms) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for coef, stencil in terms:
        for j, v in stencil.items():
            out[j] = out.get(j, 0.0) + coef * v
    return {j: v for j, v in out.items() if abs(v) > 1e-12}


def _slope_stencils(m: int) -> List[Dict[int, float]]:
    """Catmull-Rom slopes in per-interval units, m_k = (h_k+1 - h_k-1) / 2
    inside and one-sided at the ends, as stencils over local knot indices."""
    out = []
    for k in range(m):
        if k == 0:
            out.append({0: -1.0, 1: 1.0})
        elif k == m - 1:
            out.append({m - 2: -1.0, m - 1: 1.0})
        else:
            out.append({k - 1: -0.5, k + 1: 0.5})
    return out


def _sample_stencils(m: int, q: int) -> List[Tuple[float, Dict[int, float]]]:
    """(position along the chain in intervals, stencil) for q samples per
    knot interval plus the last knot: the drawn Hermite curve at each, as a
    linear combination of the knot heights.

    z(u) = h_k H00(u) + m_k H10(u) + h_k+1 H01(u) + m_k+1 H11(u) on interval k.
    """
    if m < 2:
        return [(0.0, {0: 1.0})]
    slope = _slope_stencils(m)
    out = []
    for k in range(m - 1):
        for i in range(q):
            u = i / q
            if i == 0:
                out.append((float(k), {k: 1.0}))
                continue
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            out.append((k + u, _combine((h00, {k: 1.0}), (h10, slope[k]),
                                        (h01, {k + 1: 1.0}),
                                        (h11, slope[k + 1]))))
    out.append((float(m - 1), {m - 1: 1.0}))
    return out


def solve_bending_z(instance_ids: Sequence[int],
                    crossings: list,
                    centrelines: Dict[int, np.ndarray],
                    radii: Dict[int, float],
                    params,
                    order_position: Dict[int, int],
                    rho: float = 0.0,
                    ):
    """(mean height, (low, high), status, info) per instance.

    `info` carries the rho used, the bend radius it implies, the per-instance
    height profiles at the centreline points (`profiles`) and the solved
    chains (`chains`: {id: (arclengths, heights)}), which `clearance_violations`
    and `height_profile` read. A solver failure returns the flat sweep for the
    same order with status `bending_failed:...`, never zeros.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    from scipy.spatial import cKDTree

    ids = list(instance_ids)
    order_position = order_position or {}
    r = {n: max(0.5, float(radii.get(n, params.default_radius_px))) for n in ids}
    info: Dict[str, object] = {"rho": 0.0, "min_radius_px": 0.0,
                               "ramp_px": 0.0, "profiles": {}, "chains": {}}

    def flat_fallback(status: str):
        # the flat sweep for this same order, so a failure here degrades to
        # the shipped model rather than to nonsense
        from .depth import stacked_z
        z = stacked_z(ids, crossings, radii, params, order_position)
        return z, {n: (z[n], z[n]) for n in ids}, status, info

    # -- 1. one chain per instance, a knot at each end and one every `step`
    step = max(1.0, float(getattr(params, "bend_sample_px", 0)) or 6.0)
    q = max(1, int(_ZONE_SUBSAMPLES))
    knots_of: Dict[int, np.ndarray] = {}
    cum_of: Dict[int, np.ndarray] = {}
    offset: Dict[int, int] = {}
    samples: Dict[int, list] = {}          # [(arclength, global stencil)]
    sample_xy: Dict[int, np.ndarray] = {}
    n_h = 0
    for n in ids:
        pts = np.asarray(centrelines.get(n, np.zeros((0, 2))),
                         dtype=float).reshape(-1, 2)
        cum = _arclength(pts)
        total = float(cum[-1]) if len(cum) else 0.0
        if total > 0.0:
            knots = np.linspace(0.0, total, max(3, int(total / step) + 1))
        else:
            # a one-point or zero-length instance is one flat height, so it
            # still takes part in every clearance, as it does in the flat model
            knots = np.zeros(1)
        knots_of[n], cum_of[n], offset[n] = knots, cum, n_h
        d = float(knots[1] - knots[0]) if len(knots) > 1 else 0.0
        local = _sample_stencils(len(knots), q)
        arcs = np.array([f * d for f, _st in local])
        if total > 0.0:
            xy = np.column_stack([np.interp(arcs, cum, pts[:, 0]),
                                  np.interp(arcs, cum, pts[:, 1])])
        else:
            xy = np.repeat(pts[:1] if len(pts) else np.full((1, 2), np.nan),
                           len(local), axis=0)
        samples[n] = [(float(a), {n_h + j: v for j, v in st.items()})
                      for a, (_f, st) in zip(arcs, local)]
        sample_xy[n] = xy
        n_h += len(knots)

    pairs = _directed(crossings, set(ids), order_position)
    if not pairs:
        return ({n: 0.0 for n in ids}, {n: (0.0, 0.0) for n in ids},
                "no_edges", info)

    rho = float(rho) if rho and rho > 0 else rho_shear_free()
    # rho is dimensionless, so the bend it allows scales with the filament it
    # is applied to: a thicker bundle is held to a gentler radius
    r_min_of = {n: min_radius_px(rho, 2.0 * r[n]) for n in ids}
    spacing = {n: float(knots_of[n][1] - knots_of[n][0])
               if len(knots_of[n]) > 1 else 0.0 for n in ids}
    # a curve with |z''| <= 1/R stays within d^2 / (8 R) of the chord of any
    # interval of length d: the margin that makes a bound on the samples a
    # bound on the drawn curve between them
    bulge = {n: (spacing[n] / q) ** 2 / (8.0 * r_min_of[n]) for n in ids}
    diameter = 2.0 * float(np.median([r[n] for n in ids]))
    info.update(rho=round(rho, 4),
                min_radius_px=round(min_radius_px(rho, diameter), 3),
                min_radius_px_range=[round(min(r_min_of.values()), 3),
                                     round(max(r_min_of.values()), 3)],
                ramp_px=round(ramp_px(rho, diameter), 3),
                sample_px=round(step, 2))

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    rhs: List[float] = []

    def row(coef: Dict[int, float], bound: float) -> None:      # coef . h <= bound
        rw = len(rhs)
        for j, v in coef.items():
            rows.append(rw)
            cols.append(j)
            vals.append(v)
        rhs.append(bound)

    def bracket(n: int, s: float) -> Tuple[int, ...]:
        """Indices of the samples of `n` on either side of arclength s."""
        arcs = np.array([a for a, _st in samples[n]])
        if len(arcs) == 1:
            return (0,)
        j = int(np.searchsorted(arcs, s))
        return tuple(sorted({min(max(j - 1, 0), len(arcs) - 1),
                             min(max(j, 0), len(arcs) - 1)}))

    # -- 2. clearance: at the contact point of every constrained pair, on the
    # samples bracketing it, and at every sample pair of the two that
    # overlaps in projection (the flat model clears the whole overlap at
    # once; this is the same guarantee for a filament that varies along its
    # length, at the sample resolution)
    n_contact_rows = 0
    n_zone_rows = 0
    trees: Dict[int, object] = {}
    for k, hi, lo in pairs:
        c = crossings[k]
        sep = r[hi] + r[lo] + params.z_gap_px
        margin = bulge[hi] + bulge[lo]
        s_hi = project_arc(centrelines.get(hi, ()), c.x, c.y)
        s_lo = project_arc(centrelines.get(lo, ()), c.x, c.y)
        for a in bracket(hi, s_hi):
            for b in bracket(lo, s_lo):
                row(_combine((1.0, samples[lo][b][1]), (-1.0, samples[hi][a][1])),
                    -(sep + margin))
                n_contact_rows += 1
        if np.isnan(sample_xy[hi]).any() or np.isnan(sample_xy[lo]).any():
            continue
        if lo not in trees:
            trees[lo] = cKDTree(sample_xy[lo])
        near = trees[lo].query_ball_point(sample_xy[hi], r=sep)
        for a, lst in enumerate(near):
            for b in lst:
                delta = float(np.hypot(*(sample_xy[hi][a] - sample_xy[lo][b])))
                need = float(np.sqrt(max(sep * sep - delta * delta, 0.0)))
                row(_combine((1.0, samples[lo][b][1]), (-1.0, samples[hi][a][1])),
                    -(need + margin))
                n_zone_rows += 1

    # -- 3. curvature of the drawn curve, everywhere, and flat ends
    n_curvature_rows = 0
    for n in ids:
        m = len(knots_of[n])
        d = spacing[n]
        if m < 2 or d <= 0.0:
            continue
        base = offset[n]
        cap = d * d / r_min_of[n]
        if cap < _STRAIGHT_CAP_PX:
            # A cap below the solver's feasibility tolerance is not a small
            # curvature but a numerically inconsistent one (HiGHS reports the
            # rows infeasible from 1e-9 down). Such a filament is straight,
            # and with flat ends horizontal: say exactly that instead.
            for k in range(1, m):
                row({base: 1.0, base + k: -1.0}, 0.0)
                row({base: -1.0, base + k: 1.0}, 0.0)
                n_curvature_rows += 2
            continue
        for stencil in _hermite_curvature_rows(m):
            shifted = {base + j: v for j, v in stencil.items()}
            row(shifted, cap)
            row({j: -v for j, v in shifted.items()}, cap)
            n_curvature_rows += 2
        # nothing lifts an end: a filament is horizontal where it stops or
        # leaves the frame, and rho -> infinity is then exactly the flat model
        row({base: 1.0, base + 1: -1.0}, 0.0)
        row({base: -1.0, base + 1: 1.0}, 0.0)
        row({base + m - 2: 1.0, base + m - 1: -1.0}, 0.0)
        row({base + m - 2: -1.0, base + m - 1: 1.0}, 0.0)

    # -- 4. objective: lifted area (each knot weighs its spacing; the ends
    # half of it), then flatness among the minimisers
    weight = np.zeros(n_h)
    lower = np.zeros(n_h)
    for n in ids:
        m = len(knots_of[n])
        w = np.full(m, spacing[n] if m > 1 else 1.0)
        if m > 1:
            w[0] *= 0.5
            w[-1] *= 0.5
        weight[offset[n]:offset[n] + m] = w
        lower[offset[n]:offset[n] + m] = r[n]
    info.update(n_height_variables=int(n_h),
                n_rows={"contact": n_contact_rows, "overlap": n_zone_rows,
                        "curvature": n_curvature_rows})

    steps: List[Tuple[int, int]] = []
    for n in ids:
        for k in range(len(knots_of[n]) - 1):
            steps.append((offset[n] + k, offset[n] + k + 1))
    n_total = n_h + len(steps)
    for t, (u, v) in enumerate(steps):
        row({v: 1.0, u: -1.0, n_h + t: -1.0}, 0.0)
        row({u: 1.0, v: -1.0, n_h + t: -1.0}, 0.0)
    a_ub = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_total)).tocsr()
    b_ub = np.array(rhs)
    bounds = [(float(v), None) for v in lower] + [(0.0, None)] * len(steps)

    if _FLATNESS_WEIGHT is not None:
        cost = np.concatenate([weight, np.full(len(steps), _FLATNESS_WEIGHT)])
        res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            return flat_fallback("bending_failed:" + res.message)
        h = res.x[:n_h]
        info["objective"] = {"lifted_area_px2": float(weight @ h),
                             "variation_px": float(res.x[n_h:].sum())}
    else:
        cost = np.concatenate([weight, np.zeros(len(steps))])
        first = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                        method="highs")
        if not first.success:
            return flat_fallback("bending_failed:" + first.message)
        area = float(first.fun)
        # the second stage may not lift the film: the area is pinned at its
        # minimum (to solver tolerance) and only the variation is minimised
        area_row = coo_matrix((weight, (np.zeros(n_h, dtype=int),
                                        np.arange(n_h))),
                              shape=(1, n_total)).tocsr()
        cost2 = np.concatenate([np.zeros(n_h), np.ones(len(steps))])
        second = linprog(cost2, A_ub=vstack([a_ub, area_row]).tocsr(),
                         b_ub=np.concatenate([b_ub, [area * (1.0 + 1e-9) + 1e-7]]),
                         bounds=bounds, method="highs")
        res = second if second.success else first
        h = res.x[:n_h]
        info["objective"] = {"lifted_area_px2": float(weight @ h),
                             "variation_px": float(sum(abs(h[v] - h[u])
                                                       for u, v in steps)),
                             "second_stage": bool(second.success)}

    z: Dict[int, float] = {}
    extent: Dict[int, Tuple[float, float]] = {}
    chains: Dict[int, np.ndarray] = {}
    for n in ids:
        seg = h[offset[n]:offset[n] + len(knots_of[n])]
        chains[n] = seg
        z[n] = float(np.mean(seg))
        extent[n] = (float(seg.min()), float(seg.max()))
    floor = min(low for low, _high in extent.values())
    z = {n: v - floor for n, v in z.items()}
    extent = {n: (low - floor, high - floor) for n, (low, high) in extent.items()}

    profiles: Dict[int, np.ndarray] = {}
    for n in ids:
        if n not in centrelines:
            continue
        pts = np.asarray(centrelines[n], dtype=float).reshape(-1, 2)
        profiles[n] = height_profile(pts, knots_of[n], chains[n] - floor,
                                     at=cum_of[n])
    info["profiles"] = profiles
    info["chains"] = {n: (knots_of[n], chains[n] - floor) for n in ids}
    return z, extent, "bending", info


def profile_height_at(centreline: np.ndarray, chain: Tuple[np.ndarray, np.ndarray],
                      x: float, y: float) -> float:
    """Height of the solved curve where the centreline passes (x, y)."""
    pts = np.asarray(centreline, dtype=float).reshape(-1, 2)
    knots, heights = chain
    s = project_arc(pts, x, y)
    return float(height_profile(pts, knots, heights, at=np.array([s]))[0])


def clearance_violations(crossings: list,
                         chains: Dict[int, Tuple[np.ndarray, np.ndarray]],
                         centrelines: Dict[int, np.ndarray],
                         radii: Dict[int, float],
                         params,
                         tolerance: float = 1e-6) -> List[dict]:
    """Pairs whose solved PROFILES still interpenetrate where they meet.

    The flat gate `depth.clearance_violations` reads one height per instance;
    under this model that would be the mean of a profile and say nothing
    about the crossing. This evaluates each filament's curve at the contact
    point instead, and reports the same record.
    """
    out = []
    for c in crossings:
        if c.i not in chains or c.j not in chains:
            continue
        need = (radii.get(c.i, params.default_radius_px)
                + radii.get(c.j, params.default_radius_px) + params.z_gap_px)
        zi = profile_height_at(centrelines[c.i], chains[c.i], c.x, c.y)
        zj = profile_height_at(centrelines[c.j], chains[c.j], c.x, c.y)
        gap = abs(zi - zj)
        if gap < need - tolerance:
            out.append({"i": int(c.i), "j": int(c.j),
                        "separation": float(gap), "required": float(need)})
    return out


def film_thickness(instance_ids: Sequence[int],
                   extent: Dict[int, Tuple[float, float]],
                   radii: Dict[int, float],
                   params) -> float:
    """Top surface to bottom surface, over the solved PROFILES.

    `depth.film_thickness` takes one height per instance, which under this
    model would read the mean of a profile and understate the envelope.
    """
    present = [i for i in instance_ids if i in extent]
    if not present:
        return 0.0
    top = max(extent[i][1] + radii.get(i, params.default_radius_px)
              for i in present)
    bottom = min(extent[i][0] - radii.get(i, params.default_radius_px)
                 for i in present)
    return float(top - bottom)
