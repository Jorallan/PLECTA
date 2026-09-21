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


def _smoothstep(t):
    """Clamped cubic: flat at both ends, which is the shape a bend takes."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def height_profile(points: np.ndarray,
                   arcs: Sequence[float],
                   heights: Sequence[float],
                   rest: float,
                   ramp_px: float,
                   at: np.ndarray = None) -> np.ndarray:
    """Height at every point of a centreline, given the solved crossings.

    Between two crossings the filament takes a clamped S-bend, flat at each
    crossing. Beyond the outermost crossing it ramps down to `rest` over
    `ramp_px * sqrt(rise / diameter)` -- a ramp SCALES with its rise, which is
    what keeps a filament coming down from four diameters from doing it as
    sharply as one coming down from a single diameter. Where there is not
    enough arclength for the ramp, the filament simply stays up: it has run
    out of room to come down, and the model never claimed otherwise.
    """
    pts = np.asarray(points, dtype=float)
    cum = _arclength(pts) if at is None else np.asarray(at, dtype=float)
    out = np.full(len(cum), float(rest))
    arcs = np.asarray(arcs, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if len(arcs) == 0:
        return out
    order = np.argsort(arcs)
    arcs, heights = arcs[order], heights[order]
    total = float(cum[-1]) if len(cum) else 0.0

    def ramp_for(rise):
        return float(ramp_px) * np.sqrt(max(float(rise), 0.0) / max(ramp_px, 1e-9))

    head = cum <= arcs[0]
    if head.any():
        length = ramp_for(heights[0] - rest)
        if length > 1e-9 and arcs[0] >= length:
            out[head] = heights[0] + (rest - heights[0]) * _smoothstep(
                (arcs[0] - cum[head]) / length)
        else:
            out[head] = heights[0]
    tail = cum >= arcs[-1]
    if tail.any():
        length = ramp_for(heights[-1] - rest)
        if length > 1e-9 and (total - arcs[-1]) >= length:
            out[tail] = heights[-1] + (rest - heights[-1]) * _smoothstep(
                (cum[tail] - arcs[-1]) / length)
        else:
            out[tail] = heights[-1]
    for a1, a2, h1, h2 in zip(arcs, arcs[1:], heights, heights[1:]):
        span = (cum >= a1) & (cum <= a2)
        if not span.any():
            continue
        out[span] = h1 + (h2 - h1) * _smoothstep((cum[span] - a1) / max(a2 - a1, 1e-9))
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

    index: Dict[Tuple[int, int], int] = {}
    per: Dict[int, List[Tuple[float, int]]] = {}
    pairs: List[Tuple[int, int, int]] = []          # (k, hi, lo)
    cum = {n: _arclength(centrelines[n]) for n in ids if n in centrelines}
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
        pairs.append((k, hi, lo))
        for s in (c.i, c.j):
            if (s, k) in index or s not in cum or len(cum[s]) < 2:
                continue
            index[(s, k)] = len(index)
            p = np.asarray(centrelines[s], dtype=float)
            nearest = int(np.argmin(np.hypot(p[:, 0] - c.x, p[:, 1] - c.y)))
            per.setdefault(s, []).append((float(cum[s][nearest]), k))

    if not index:
        return flat[0], flat[1], "no_edges", info

    diameter = 2.0 * float(np.median([r[n] for n in ids]))
    rho = float(rho) if rho and rho > 0 else rho_shear_free()
    r_min = min_radius_px(rho, diameter)
    ramp = rho * diameter
    info.update(rho=round(rho, 4), min_radius_px=round(r_min, 3),
                ramp_px=round(ramp, 3))

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

    for k, hi, lo in pairs:
        a, b = index.get((hi, k)), index.get((lo, k))
        if a is None or b is None:
            continue
        row({b: 1.0, a: -1.0}, -(r[hi] + r[lo] + params.z_gap_px))

    # A clamped S-bend of rise dh over arclength L has minimum radius
    # L^2 / (6 dh), so "no tighter than r_min" is |dh| <= L^2 / (6 r_min).
    # The TRUE arclength is used: two crossings within a diameter of each
    # other are one junction, and the cap then holds them level rather than
    # licensing a kink between them.
    for s, hits in per.items():
        ordered = sorted(set(hits))
        for (a1, k1), (a2, k2) in zip(ordered, ordered[1:]):
            cap = (a2 - a1) ** 2 / (6.0 * r_min)
            u, v = index[(s, k1)], index[(s, k2)]
            row({v: 1.0, u: -1.0}, cap)
            row({u: 1.0, v: -1.0}, cap)

    n_var = len(index)
    a_ub = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_var)).tocsr()
    order = sorted(index, key=index.get)
    lower = np.array([r[s] for (s, _k) in order])
    res = linprog(np.ones(n_var), A_ub=a_ub, b_ub=np.array(rhs),
                  bounds=list(zip(lower, [None] * n_var)), method="highs")
    if not res.success:
        return flat[0], flat[1], "infeasible", info

    h = res.x
    z: Dict[int, float] = {}
    extent: Dict[int, Tuple[float, float]] = {}
    solved: Dict[int, Tuple[List[float], List[float]]] = {}
    for n in ids:
        hits = sorted(set(per.get(n, ())))
        got = [h[index[(n, k)]] for (_a, k) in hits if (n, k) in index]
        if got:
            z[n] = float(np.mean(got))
            extent[n] = (float(min(got)), float(max(got)))
            solved[n] = ([a for a, _k in hits], got)
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
        if n in solved:
            arcs, hs = solved[n]
            profiles[n] = height_profile(centrelines[n], arcs,
                                         [v - floor for v in hs],
                                         r[n], ramp, at=cum.get(n))
        else:
            profiles[n] = np.full(len(np.asarray(centrelines[n])), z[n])
    info["profiles"] = profiles
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
