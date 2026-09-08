"""Local geometry of a stub, and the cost of joining two stubs.

A *stub* is one end of an arm.  Everything the linker needs to know about it is
summarised by a frame: where the tip is, which way the arm leaves the tip, and
how fast it is turning there.

The frame is fitted in arclength, with the tip at ``s = 0`` and ``s`` growing
*into* the arm, then re-expressed outward, because every question the linker
asks ("does this arm continue into that one?") is about the outward direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

Pixel = Tuple[int, int]


@dataclass
class Frame:
    tip: np.ndarray          # (row, col) float
    tangent: np.ndarray      # unit vector pointing OUT of the arm
    curvature: float         # signed, 1/px, in the outward parameterisation
    support: int             # how many pixels the fit actually used
    reliable: bool           # enough support for the tangent to mean anything


def arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean length along a polyline, starting at 0."""
    if len(points) < 2:
        return np.zeros(len(points))
    step = np.hypot(*(np.diff(points, axis=0).T))
    return np.concatenate([[0.0], np.cumsum(step)])


def _densify(pts: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Resample a polyline at a fixed arclength step.

    Also available through `plecta.depth`.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    total = float(s[-1])
    if total < step:
        return pts
    t = np.linspace(0.0, total, max(2, int(np.ceil(total / step)) + 1))
    return np.stack([np.interp(t, s, pts[:, 0]),
                     np.interp(t, s, pts[:, 1])], axis=1)


def _unit_normals(pts: np.ndarray) -> np.ndarray:
    """In-plane unit normals of a planar polyline: the tangent, quarter-turned.

    A point whose neighbours coincide has no direction; its tangent is left at
    length 1 so the division is safe, and the normal there means nothing.
    """
    tang = np.gradient(pts, axis=0)
    norm = np.hypot(tang[:, 0], tang[:, 1])
    norm[norm < 1e-9] = 1.0
    tang = tang / norm[:, None]
    return np.stack([-tang[:, 1], tang[:, 0]], axis=1)


def fit_frame(points: Sequence[Pixel], window: float = 14.0,
              min_quadratic: int = 9) -> Frame:
    """Fit an outward frame at ``points[0]``.

    ``points`` runs from the tip inward.  ``window`` is in *arclength*, not in
    pixel count, so a staircased diagonal arm gets the same physical support as
    a straight horizontal one.

    Degree is chosen by how much support there is: a quadratic needs enough of
    the arm to be worth its extra variance, otherwise the fit is a straight line
    and the curvature is reported as 0.  ``min_quadratic`` is compared against
    the **arclength** spanned, not the pixel count -- a diagonal run covers
    sqrt(2) px of arclength per pixel, so counting pixels would hand an
    axis-aligned arm a curved fit and its 45-degree twin a straight one on
    identical physical support.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return Frame(np.zeros(2), np.zeros(2), 0.0, 0, False)
    if len(pts) == 1:
        return Frame(pts[0].copy(), np.zeros(2), 0.0, 1, False)

    s = arclength(pts)
    keep = s <= window
    if keep.sum() < 2:
        keep = np.zeros(len(pts), bool)
        keep[:2] = True
    pts = pts[keep]
    s = s[keep]
    n = len(pts)

    degree = 2 if float(s[-1]) >= float(min_quadratic) else 1
    degree = min(degree, n - 1)

    coef_r = np.polyfit(s, pts[:, 0], degree)
    coef_c = np.polyfit(s, pts[:, 1], degree)
    # polyfit returns highest power first; d/ds at s=0 is the linear coefficient
    dr, dc = coef_r[-2], coef_c[-2]
    if degree >= 2:
        ddr, ddc = 2.0 * coef_r[-3], 2.0 * coef_c[-3]
    else:
        ddr = ddc = 0.0

    # Re-express in u = -s so that u grows outward from the tip.
    out_r, out_c = -dr, -dc
    norm = math.hypot(out_r, out_c)
    if norm < 1e-9:
        return Frame(pts[0].copy(), np.zeros(2), 0.0, n, False)
    tangent = np.array([out_r / norm, out_c / norm])
    # second derivative is even in u, so it carries over unchanged
    curv = (out_r * ddc - out_c * ddr) / (norm ** 3)

    reliable = float(s[-1]) >= 3.0 and n >= 3
    return Frame(pts[0].copy(), tangent, float(curv), n, reliable)


# ── stub-to-stub linking cost ──────────────────────────────────────────────


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0)))


@dataclass
class LinkCost:
    cost: float
    theta: float      # tangent-vs-tangent turn (0 = straight through)
    phi_a: float      # turn from a's tangent onto the chord
    phi_b: float      # turn from b's tangent onto the chord
    length: float     # tip-to-tip distance


def link_cost(
    fa: Frame, fb: Frame,
    w_turn: float = 1.0,
    w_direct: float = 1.0,
    w_len: float = 0.0,
    len_scale: float = 12.0,
    chord_floor: float = 3.0,
    w_kappa: float = 0.0,
    kappa_scale: float = 30.0,
) -> LinkCost:
    """How badly a stub-to-stub join violates "filaments are smooth".

    Two independent pieces of evidence, because each fails on its own:

    * ``theta`` -- the angle between a's outward tangent and b's *reversed*
      tangent. Robust when the tips are close (a crossing), because it never
      touches the tip-to-tip chord, which is badly quantised at short range.
      Blind to lateral offset: two parallel arms that never meet score 0.
    * ``phi_a + phi_b`` -- how far each tangent has to turn onto the chord that
      actually connects the tips. Sees lateral offset, and is what matters
      across a gap, but is meaningless when the tips are a couple of pixels
      apart. ``chord_floor`` fades it out at short range instead of letting
      quantisation noise dominate.

    ``w_kappa`` adds curvature continuity.  Signed curvature flips sign when a
    curve is traversed backwards, and the two frames look outward from opposite
    ends of the join, so a smooth continuation wants ``kappa_a + kappa_b`` near
    zero.

    This signature defaults it to 0 only so the function is usable standalone.
    **The shipped configuration turns it on**: ``j_w_kappa = 1.0`` at crossings
    and ``g_w_kappa = 0.5`` across gaps (``parameters.yaml``, frozen in
    ``params.json``), worth about +0.005 F1 on the development set.  Do not read
    the defaults here, or on ``linking.Params``, as the configuration -- see
    ``plecta.parameters``.

    The worry that motivated the low weight -- that a second derivative fitted
    on a short arm is mostly noise -- is handled upstream rather than by leaving
    the term off: ``fit_frame`` drops to a straight-line fit below
    ``min_quadratic`` pixels of arclength and reports curvature exactly 0, so a
    stub too short to support the fit contributes nothing here.
    """
    if fa.tangent is None or fb.tangent is None:
        return LinkCost(math.inf, math.pi, math.pi, math.pi, math.inf)
    if not np.any(fa.tangent) or not np.any(fb.tangent):
        return LinkCost(math.inf, math.pi, math.pi, math.pi, math.inf)

    d = fb.tip - fa.tip
    length = float(math.hypot(d[0], d[1]))
    theta = _angle(fa.tangent, -fb.tangent)

    if length >= 1e-6:
        u = d / length
        phi_a = _angle(fa.tangent, u)
        phi_b = _angle(fb.tangent, -u)
    else:
        phi_a = phi_b = 0.0

    # Trust the chord only once it is long enough to have a direction.
    chord_trust = min(1.0, length / chord_floor) if chord_floor > 0 else 1.0
    cost = (w_direct * theta
            + w_turn * chord_trust * (phi_a + phi_b)
            + w_len * (length / len_scale))
    if w_kappa:
        cost += w_kappa * abs(fa.curvature + fb.curvature) * kappa_scale
    return LinkCost(float(cost), theta, phi_a, phi_b, length)
