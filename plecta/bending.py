"""Heights that let a filament bend, instead of holding it flat.

`solve_metric_z` gives every instance ONE height, which is the assumption
`assumptions.planar_instances: dz/ds = 0` states. A filament that passes over
one neighbour and under the next cannot be drawn that way, so each transitive
over/under step is paid as global stack height, and the film grows with the
crossing graph rather than with the specimen.

This solves a height per (instance, crossing) instead, under the same compact
prior -- nothing floats higher than it must -- plus one constraint: a filament
may not be bent to a radius tighter than `r_min`. Bending enters as a
CONSTRAINT rather than an energy, which is what keeps the whole thing a single
LP with no free exchange rate.

The default `r_min` comes from the bundle the filament is. For a rope of
(10,10) tubes free to shear past each other the adhesion-bending balance gives
a ramp radius of rho^2 d / 6, with rho itself independent of d. On a 13.4 nm
bundle that is 146 nm, against the 165 nm sharpest in-plane bend a human
traced on the same field: two independent routes, 12 % apart.
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


def rho_shear_free(diameter_nm: float) -> float:
    """Ramp length in bundle diameters, from the tube packing alone.

    rho^4 = 18 EI / (w_a d^2). With the tubes free to shear past each other
    EI = N EI_1 and N ~ d^2, so the d^2 cancels: rho is a constant of the
    material and not of the bundle. Returns about 8.1 for any (10,10) rope.
    """
    d_m = max(float(diameter_nm), _TUBE_D_NM) * 1e-9
    ei_1 = np.pi * _C_INPLANE * (_TUBE_D_NM * 1e-9 / 2.0) ** 3
    n_tubes = (np.pi / 4.0) * (d_m - _GAP_NM * 1e-9) ** 2 / (
        (np.sqrt(3.0) / 2.0) * (_ROPE_A0_NM * 1e-9) ** 2)
    return float((18.0 * max(n_tubes, 1.0) * ei_1
                  / (_W_ADHESION * d_m ** 2)) ** 0.25)


def default_min_radius_px(diameter_px: float, nm_per_px: float = 1.0) -> float:
    """rho^2 d / 6, the radius a ramp of any rise adopts under that balance."""
    rho = rho_shear_free(float(diameter_px) * float(nm_per_px))
    return rho * rho * float(diameter_px) / 6.0


def _arclength(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return np.zeros(len(p))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def solve_bending_z(instance_ids: Sequence[int],
                    crossings: list,
                    centrelines: Dict[int, np.ndarray],
                    radii: Dict[int, float],
                    params,
                    order_position: Dict[int, int],
                    r_min_px: float = 0.0,
                    ):
    """(mean height per instance, (low, high) per instance, status, r_min).

    Directions come from `over` where the evidence decided one and from
    `order_position` where it abstained, exactly as `solve_metric_z` reads
    them, so the ordering stage is untouched by this.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    ids = list(instance_ids)
    r = {n: max(0.5, float(radii.get(n, params.default_radius_px))) for n in ids}
    inside = set(ids)
    flat = ({n: 0.0 for n in ids}, {n: (0.0, 0.0) for n in ids})

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
        return flat[0], flat[1], "no_edges", 0.0

    if r_min_px <= 0:
        r_min_px = default_min_radius_px(
            2.0 * float(np.median([r[n] for n in ids])))

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
            cap = (a2 - a1) ** 2 / (6.0 * r_min_px)
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
        return flat[0], flat[1], "infeasible", float(r_min_px)

    h = res.x
    z: Dict[int, float] = {}
    extent: Dict[int, Tuple[float, float]] = {}
    for n in ids:
        got = [h[index[(n, k)]] for (_a, k) in per.get(n, ()) if (n, k) in index]
        if got:
            z[n] = float(np.mean(got))
            extent[n] = (float(min(got)), float(max(got)))
        else:
            z[n] = r[n]
            extent[n] = (r[n], r[n])
    floor = min(low for low, _high in extent.values())
    z = {n: v - floor for n, v in z.items()}
    extent = {n: (low - floor, high - floor) for n, (low, high) in extent.items()}
    return z, extent, "bending", float(r_min_px)


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
