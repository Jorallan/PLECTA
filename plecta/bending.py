"""Heights that let a filament bend, instead of holding it flat.

`solve_metric_z` gives every instance ONE height, which is the assumption
`assumptions.planar_instances: dz/ds = 0` states. A filament that passes over
one neighbour and under the next cannot be drawn that way, so each transitive
over/under step is paid as global stack height, and the film grows with the
crossing graph rather than with the specimen.

This solves a height per (instance, crossing) instead, under the same compact
prior -- nothing floats higher than it must -- plus one constraint: a filament
may not be bent more sharply than its own stiffness allows. Bending enters as
a CONSTRAINT rather than an energy, which keeps the whole thing a single LP
with no exchange rate between evidence and shape to choose.

Everything is controlled by `rho`, the one dimensionless number the mechanics
reduces to: **the arclength, in filament diameters, that a filament needs to
rise by one diameter.** Small rho is a floppy filament that can weave tightly;
large rho is a stiff one that must stay flat over long runs. rho -> infinity
recovers the flat-filament model exactly.

rho follows from the balance between the energy of bending and the adhesion
given up by lifting off a neighbour:

    rho^4 = 18 EI / (w_a d^2)

For a bundle of (10,10) tubes free to shear past each other, EI = N EI_1 and
the tube count N grows as d^2, so the d^2 cancels: **rho does not depend on
the bundle diameter at all**, and comes out near 8.1 for any such rope. That
is the default. Independently, the sharpest in-plane bend a human traced on a
real field implies rho ~ 8.6 on the same bundles, 12 % away, which is the only
place in this stage where mechanics and image agree without being fitted.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

#: (10,10) armchair geometry and rope constants; see `rho_shear_free`.
_A_CC_NM = 0.142            # C-C bond length
_TUBE_D_NM = _A_CC_NM * np.sqrt(3.0) * np.sqrt(300.0) / np.pi   # 1.356 nm
_ROPE_A0_NM = 1.67          # hexagonal lattice constant of the rope, Lu 1997
_GAP_NM = 0.317             # wall-to-wall equilibrium spacing
_C_INPLANE = 345.0          # J/m^2, in-plane stiffness (Yakobson 1996)
_W_ADHESION = 0.44e-9       # N, three-pair line contact (Girifalco 2000)

#: Weight on total height variation, against 1.0 per unit of height.
#:
#: Minimising height alone bends a filament wherever bending is free, which
#: is most places, so the film came out wavy everywhere rather than where the
#: geometry demands it. This charges for bending as well, and 5.0 is the
#: largest weight that buys flatness for nothing on B58-B3-S2_100: it leaves
#: the film at 87.1 nm while taking the filaments that never rise a nanometre
#: from 148 of 307 to 169, and the summed rise down 9 %. Above it thickness
#: starts to pay (89.1 nm at 8, 102 nm at 15) and by 50 the planar model is
#: back. Chosen on ONE field, so it is a tie-break rather than a measurement.
_FLATNESS = 5.0

#: The curvature bound is applied to the second differences of a chain
#: sampled every few pixels, which is a weaker statement than bounding the
#: curvature of the smooth filament that chain stands for: a curve drawn
#: through samples sitting exactly at the limit exceeds it between them, by
#: up to a factor of two at 6 px sampling (measured, and only slowly improved
#: by refining -- 1.5x at 1.5 px and 25x the variables). The constraint is
#: therefore applied this much tighter, so that the filament the model draws,
#: and not merely the chain it solves, respects the radius the mechanics
#: gives. Reported bounds are the physical ones, not these.
_CURVATURE_SAFETY = 2.0


def rho_shear_free(diameter_nm: float = 0.0) -> float:
    """Diameters of arclength a filament needs to rise one diameter.

    From rho^4 = 18 EI / (w_a d^2) with the tubes free to shear, where the
    diameter cancels. The argument is accepted only so a caller can show it;
    the answer moves by under 1 % across 10-25 nm bundles.
    """
    d_m = max(float(diameter_nm) or 13.4, _TUBE_D_NM) * 1e-9
    ei_1 = np.pi * _C_INPLANE * (_TUBE_D_NM * 1e-9 / 2.0) ** 3
    n_tubes = (np.pi / 4.0) * (d_m - _GAP_NM * 1e-9) ** 2 / (
        (np.sqrt(3.0) / 2.0) * (_ROPE_A0_NM * 1e-9) ** 2)
    return float((18.0 * max(n_tubes, 1.0) * ei_1
                  / (_W_ADHESION * d_m ** 2)) ** 0.25)


def min_radius_px(rho: float, diameter_px: float) -> float:
    """rho^2 d / 6: the radius a ramp of ANY rise adopts at this rho."""
    return float(rho) * float(rho) * float(diameter_px) / 6.0


def _arclength(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return np.zeros(len(p))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def height_profile(points: np.ndarray,
                   sample_at: Sequence[float],
                   sample_h: Sequence[float],
                   at: np.ndarray = None) -> np.ndarray:
    """Height at every centreline point, from the heights that were SOLVED.

    There is no assumed shape here any more. `solve_bending_z` carries a
    height every few pixels along the filament and bounds the curvature of
    that chain directly, so the profile is the answer rather than a curve
    threaded through a handful of knots afterwards. This only resamples it
    onto the centreline's own points.
    """
    from scipy.interpolate import PchipInterpolator

    cum = _arclength(np.asarray(points, dtype=float)) if at is None         else np.asarray(at, dtype=float)
    x = np.asarray(sample_at, dtype=float)
    y = np.asarray(sample_h, dtype=float)
    if len(cum) == 0:
        return np.zeros(0)
    if len(x) < 3:
        return np.interp(cum, x, y)
    # PCHIP rather than a natural spline: an interpolating spline overshoots
    # between samples and so reports curvature the solver never permitted
    # (measured: radii 12 % tighter than the bound). PCHIP does not overshoot,
    # and np.interp would instead put a corner at every sample.
    return PchipInterpolator(x, y)(np.clip(cum, x[0], x[-1]))


def solve_bending_z(instance_ids: Sequence[int],
                    crossings: list,
                    centrelines: Dict[int, np.ndarray],
                    radii: Dict[int, float],
                    params,
                    order_position: Dict[int, int],
                    rho: float = 0.0,
                    ):
    """(mean height, (low, high), status, info) per instance.

    Directions come from `over` where the evidence decided one and from
    `order_position` where it abstained, exactly as `solve_metric_z` reads
    them, so the ordering stage is untouched by this. `info` carries the rho
    used, the derived bend radius and the per-instance height profiles.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    ids = list(instance_ids)
    r = {n: max(0.5, float(radii.get(n, params.default_radius_px))) for n in ids}
    inside = set(ids)
    flat = ({n: 0.0 for n in ids}, {n: (0.0, 0.0) for n in ids})
    info: Dict[str, object] = {"rho": 0.0, "min_radius_px": 0.0,
                               "ramp_px": 0.0, "profiles": {}}

    # One height every `step` pixels along each filament, with a sample
    # landing on each crossing. Bounding the curvature of THIS chain bounds
    # the shape the model actually claims, everywhere along it -- which
    # bounding the rise across a span, or a curve fitted through the crossings
    # afterwards, does not.
    step = max(1.0, float(getattr(params, "bend_sample_px", 0)) or 6.0)
    samples: Dict[int, np.ndarray] = {}
    offset: Dict[int, int] = {}
    n_var = 0
    cum = {n: _arclength(centrelines[n]) for n in ids if n in centrelines}
    for n in ids:
        if n not in cum or len(cum[n]) < 2:
            continue
        total = float(cum[n][-1])
        if total <= 0:
            continue
        knots = np.linspace(0.0, total, max(3, int(total / step) + 1))
        samples[n] = knots
        offset[n] = n_var
        n_var += len(knots)

    def sample_of(n: int, arc: float) -> int:
        return offset[n] + int(np.argmin(np.abs(samples[n] - arc)))

    index: Dict[Tuple[int, int], int] = {}
    pairs: List[Tuple[int, int, int]] = []          # (k, hi, lo)
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
        if hi not in samples or lo not in samples:
            continue
        pairs.append((k, hi, lo))
        for t in (c.i, c.j):
            p = np.asarray(centrelines[t], dtype=float)
            nearest = int(np.argmin(np.hypot(p[:, 0] - c.x, p[:, 1] - c.y)))
            index[(t, k)] = sample_of(t, float(cum[t][nearest]))

    if not index:
        return flat[0], flat[1], "no_edges", info

    rho = float(rho) if rho and rho > 0 else rho_shear_free()
    # rho is dimensionless, so the bend it allows scales with the filament it
    # is applied to: a thicker bundle is held to a gentler radius and takes a
    # longer arc to rise, which is what "harder to bend" means in absolute
    # terms even though rho itself does not move with diameter.
    r_min_of = {n: min_radius_px(rho, 2.0 * r[n]) for n in ids}
    ramp_of = {n: rho * 2.0 * r[n] for n in ids}
    diameter = 2.0 * float(np.median([r[n] for n in ids]))
    info.update(rho=round(rho, 4),
                min_radius_px=round(min_radius_px(rho, diameter), 3),
                min_radius_px_range=[round(min(r_min_of.values()), 3),
                                     round(max(r_min_of.values()), 3)],
                ramp_px=round(rho * diameter, 3))

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

    # clearance, at the sample each crossing lands on
    for k, hi, lo in pairs:
        a_i, b_i = index.get((hi, k)), index.get((lo, k))
        if a_i is None or b_i is None:
            continue
        row({b_i: 1.0, a_i: -1.0}, -(r[hi] + r[lo] + params.z_gap_px))

    # Curvature, on every consecutive triple of the chain. With uniform
    # spacing d the second difference IS d^2 * d2h/ds2 for a shallow profile,
    # so |h[m-1] - 2 h[m] + h[m+1]| <= d^2 / r_min holds the filament to its
    # own bend radius at EVERY point, not just across a span or at a knot.
    # The bending is then spread the way an elastic filament spreads it,
    # because no part of the chain may take more than its share.
    n_h = n_var
    for n, knots in samples.items():
        d = float(knots[1] - knots[0]) if len(knots) > 1 else 0.0
        if d <= 0:
            continue
        cap = d * d / (r_min_of[n] * _CURVATURE_SAFETY)
        base = offset[n]
        for m in range(1, len(knots) - 1):
            i0, i1, i2 = base + m - 1, base + m, base + m + 1
            row({i0: 1.0, i1: -2.0, i2: 1.0}, cap)
            row({i0: -1.0, i1: 2.0, i2: -1.0}, cap)

    # Height alone leaves the solver indifferent wherever bending is free, so
    # charge for bending too: one slack per step carrying |dh|.
    slack: List[Tuple[int, int, int]] = []
    for n, knots in samples.items():
        base = offset[n]
        for m in range(len(knots) - 1):
            slack.append((n_h + len(slack), base + m, base + m + 1))
    for t, u, v in slack:
        row({v: 1.0, u: -1.0, t: -1.0}, 0.0)
        row({u: 1.0, v: -1.0, t: -1.0}, 0.0)

    n_total = n_h + len(slack)
    a_ub = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_total)).tocsr()
    lower = np.zeros(n_total)
    for n, knots in samples.items():
        lower[offset[n]:offset[n] + len(knots)] = r[n]
    cost = np.concatenate([np.ones(n_h), np.full(len(slack), _FLATNESS)])
    info["n_height_variables"] = int(n_h)
    info["sample_px"] = round(step, 2)
    res = linprog(cost, A_ub=a_ub, b_ub=np.array(rhs),
                  bounds=list(zip(lower, [None] * n_total)), method="highs")
    if not res.success:
        return flat[0], flat[1], "infeasible", info

    h = res.x
    z: Dict[int, float] = {}
    extent: Dict[int, Tuple[float, float]] = {}
    chains: Dict[int, np.ndarray] = {}
    for n in ids:
        if n in samples:
            seg = h[offset[n]:offset[n] + len(samples[n])]
            chains[n] = seg
            z[n] = float(np.mean(seg))
            extent[n] = (float(seg.min()), float(seg.max()))
        else:
            z[n] = r[n]
            extent[n] = (r[n], r[n])
    floor = min(low for low, _high in extent.values())
    z = {n: v - floor for n, v in z.items()}
    extent = {n: (low - floor, high - floor) for n, (low, high) in extent.items()}

    profiles: Dict[int, np.ndarray] = {}
    for n in ids:
        if n not in centrelines:
            continue
        if n in chains:
            profiles[n] = height_profile(centrelines[n], samples[n],
                                         chains[n] - floor, at=cum.get(n))
        else:
            profiles[n] = np.full(len(np.asarray(centrelines[n])), z[n])
    info["profiles"] = profiles
    info["chains"] = {n: (samples[n], chains[n] - floor) for n in chains}
    return z, extent, "bending", info


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
