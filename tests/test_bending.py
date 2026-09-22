"""The bending height model (plecta/bending.py): known answers and properties.

Known answers are geometries whose solution can be written down; properties
are what the option promises about every scene: it reads the order and never
changes it, it touches nothing two-dimensional, it clears every constrained
pair where the pair meets, the curve it draws respects the radius it
reports, and with rho -> infinity it is the flat model again.

    python -m pytest -q tests/test_bending.py
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from plecta import bending as B
from plecta import depth as D

pytest.importorskip("scipy")


# ── scenes ────────────────────────────────────────────────────────────────

def line(x0, y0, x1, y1, n=None):
    """A straight centreline with about one vertex per 2 px."""
    length = math.hypot(x1 - x0, y1 - y0)
    n = n or max(2, int(length / 2.0) + 1)
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def crossing(i, j, x, y, over=None):
    c = D.PredCrossing(i=i, j=j, x=x, y=y)
    if over is None:
        c.abstain = True
    else:
        c.raw_over = over
        c.over = over
    return c


def params(**kw):
    p = D.DepthParams(height_model="bending", **kw)
    return p


def solve(centrelines, crossings, radii, position=None, rho=8.09, **kw):
    ids = sorted(centrelines)
    p = params(**kw)
    position = position if position is not None else {i: k for k, i in enumerate(ids)}
    return B.solve_bending_z(ids, crossings, centrelines, radii, p, position, rho)


def hermite_min_radius(knots, heights):
    """Tightest radius of the drawn curve, from its analytic second derivative."""
    from scipy.interpolate import CubicHermiteSpline

    if len(knots) < 3:
        return math.inf
    curve = CubicHermiteSpline(knots, heights, np.gradient(heights, knots))
    s = np.linspace(knots[0], knots[-1], int(knots[-1] * 8) + 2)
    k = float(np.abs(curve.derivative(2)(s)).max())
    return math.inf if k == 0.0 else 1.0 / k


# ── the constants ──────────────────────────────────────────────────────────

class TestConstants:
    def test_rho_is_the_documented_value_and_diameter_free(self):
        assert abs(B.rho_shear_free() - 8.09) < 0.005
        for d in (8.0, 10.0, 13.4, 20.0, 30.0):
            assert abs(B.rho_shear_free(d) - 8.09) < 0.07

    def test_min_radius_is_the_peel_radius(self):
        """rho^2 d / 6 = sqrt(EI / 2 w_a): the two derivations agree exactly,
        so the radius does not depend on the ramp shape rho was derived
        with."""
        for d in (8.0, 13.4, 25.0):
            rho = B.rho_shear_free(d)
            assert B.min_radius_px(rho, d) == pytest.approx(B.peel_radius_nm(d), rel=1e-12)
        assert B.peel_radius_nm(13.4) == pytest.approx(146.2, abs=0.3)

    def test_single_tube_stiffness_sits_between_the_two_transcribed_values(self):
        """Volkov & Zhigilei K_bnd = 3.04e-25 N m^2; Ostanin Y I = 3.58e-25."""
        assert 3.04e-25 < B.tube_bending_stiffness() < 3.58e-25

    def test_the_radius_scales_with_the_diameter(self):
        assert B.min_radius_px(8.09, 20.0) == pytest.approx(2 * B.min_radius_px(8.09, 10.0))


# ── known answers ──────────────────────────────────────────────────────────

class TestKnownAnswers:
    def test_over_one_and_under_the_next_is_two_diameters_not_three(self):
        """F runs over A and under B, far apart. Flat: A at 0, F at 2r, B at
        4r -- three diameters. Bent: A and B rest on the substrate, F humps
        over A, B humps over F -- two diameters. The hump's half-length is
        2 sqrt(R h) for two arcs of radius R rising h, and F is back on the
        substrate beyond it."""
        r = 5.0
        cl = {1: line(0, 50, 600, 50), 2: line(100, 0, 100, 100), 3: line(500, 0, 500, 100)}
        cs = [crossing(1, 2, 100, 50, over=1), crossing(1, 3, 500, 50, over=3)]
        radii = {1: r, 2: r, 3: r}
        z, ext, status, info = solve(cl, cs, radii, position={1: 1, 2: 2, 3: 0},
                                     bend_sample_px=3.0)
        assert status == "bending"
        p = params()
        thick = B.film_thickness([1, 2, 3], ext, radii, p)
        assert thick == pytest.approx(4 * r, abs=0.15)
        flat, _ = D.solve_metric_z([1, 2, 3], cs, radii, {}, p, {1: 1, 2: 2, 3: 0})
        assert D.film_thickness([1, 2, 3], flat, radii, p) == pytest.approx(6 * r, abs=1e-9)
        knots, h = info["chains"][1]
        rise = 2 * r
        R = B.min_radius_px(8.09, 2 * r)
        half = 2 * math.sqrt(R * rise)
        # at the crossing F sits a full separation over A
        assert B.profile_height_at(cl[1], info["chains"][1], 100, 50) - \
            B.profile_height_at(cl[2], info["chains"][2], 100, 50) >= 2 * r - 1e-6
        # and is back on the substrate one ramp away, but not half a ramp away
        floor = h.min()
        assert np.interp(100 + 1.15 * half, knots, h) == pytest.approx(floor, abs=0.05)
        assert np.interp(100 + 0.5 * half, knots, h) > floor + 0.2 * rise
        # the drawn curve never bends tighter than the radius it reports
        assert hermite_min_radius(knots, h) >= R * (1 - 1e-6)

    def test_a_symmetric_weave_stays_two_diameters_deep(self):
        """One filament over, under, over, under, over five equally spaced
        neighbours. Under the flat model the stack is three deep; bent it is
        two deep and the profile is mirror-symmetric about the middle
        neighbour."""
        r = 4.0
        cl = {0: line(0, 50, 1200, 50)}
        cs = []
        radii = {0: r}
        for k in range(5):
            x = 200 + 200 * k
            cl[k + 1] = line(x, 0, x, 100)
            radii[k + 1] = r
            cs.append(crossing(0, k + 1, x, 50, over=(0 if k % 2 == 0 else k + 1)))
        z, ext, status, info = solve(cl, cs, radii, position={i: i for i in cl},
                                     bend_sample_px=4.0)
        assert B.film_thickness(sorted(cl), ext, radii, params()) == pytest.approx(4 * r, abs=0.2)
        knots, h = info["chains"][0]
        # where the constraints act the answer is pinned: over a neighbour
        # by one separation, on the substrate under one
        at = [float(np.interp(x, knots, h)) for x in (200, 400, 600, 800, 1000)]
        assert at[0] == pytest.approx(at[4], abs=1e-6) and at[1] == pytest.approx(at[3], abs=1e-6)
        assert at[1] == pytest.approx(h.min(), abs=1e-6)
        assert at[0] >= 2 * r - 1e-6
        # between constraints the hump has the freedom of an LP vertex within
        # the discretisation: the mirror image is another minimiser, so the
        # profile is symmetric only to that freedom (0.5 px at 4 px knots)
        mirrored = np.interp(knots[-1] - knots, knots, h)
        assert np.abs(h - mirrored).max() < 0.1 * (2 * r)

    def test_two_crossings_within_a_ramp_hold_the_filament_level(self):
        """Neighbours closer together than the ramp cannot be woven between:
        F over both at the same height, no dip."""
        r = 5.0
        cl = {1: line(0, 50, 400, 50), 2: line(190, 0, 190, 100), 3: line(210, 0, 210, 100)}
        cs = [crossing(1, 2, 190, 50, over=1), crossing(1, 3, 210, 50, over=1)]
        radii = {1: r, 2: r, 3: r}
        z, ext, status, info = solve(cl, cs, radii, position={1: 0, 2: 1, 3: 2},
                                     bend_sample_px=3.0)
        knots, h = info["chains"][1]
        between = np.interp(200.0, knots, h)
        at = np.interp(190.0, knots, h)
        assert between >= at - 0.05

    @pytest.mark.parametrize("rho, tol", [(1e3, 1e-2), (1e5, 1e-5), (1e7, 1e-6)])
    def test_the_stiff_limit_is_the_flat_model(self, rho, tol):
        """rho -> infinity: no curvature and flat ends make every filament
        horizontal, and the compact prior then gives the same heights the
        flat sweep gives, to solver tolerance, when every radius is equal.
        (With unequal radii the two differ only by the substrate convention:
        bottoms coplanar here, centrelines coplanar there.)"""
        r = 4.0
        cl = {1: line(0, 50, 300, 50), 2: line(150, 0, 150, 100),
              3: line(0, 80, 300, 80), 4: line(60, 0, 60, 100)}
        cs = [crossing(1, 2, 150, 50, over=1), crossing(2, 3, 150, 80, over=2),
              crossing(1, 4, 60, 50), crossing(3, 4, 60, 80, over=4)]
        radii = {i: r for i in cl}
        position = {2: 0, 1: 1, 4: 2, 3: 3}
        z, ext, status, info = solve(cl, cs, radii, position=position, rho=rho)
        assert status == "bending"
        flat, _ = D.solve_metric_z(sorted(cl), cs, radii, {}, params(), position)
        # at rho = 1e3 the cap is still 2.7e-5 px per knot, which over 50
        # knots lets a chain stray by cap m^2 / 8 = 0.008 px: the tolerance
        # follows the model, and vanishes with the cap
        for i in cl:
            assert z[i] == pytest.approx(flat[i], abs=tol)
            low, high = ext[i]
            assert high - low < tol

    def test_unequal_radii_in_the_stiff_limit_differ_only_by_the_substrate(self):
        r = {1: 3.0, 2: 6.0, 3: 4.0}
        cl = {1: line(0, 50, 300, 50), 2: line(150, 0, 150, 100), 3: line(0, 90, 300, 90)}
        cs = [crossing(1, 2, 150, 50, over=2), crossing(2, 3, 150, 90, over=2)]
        position = {2: 0, 1: 1, 3: 2}
        z, ext, status, info = solve(cl, cs, r, position=position, rho=1e6)
        # least element of {h >= r_n, h_hi - h_lo >= r_hi + r_lo}, floored
        h = {1: 3.0, 3: 4.0}
        h[2] = max(6.0, h[1] + 9.0, h[3] + 10.0)
        floor = min(h.values())
        for i in cl:
            assert z[i] == pytest.approx(h[i] - floor, abs=1e-5)


# ── properties ─────────────────────────────────────────────────────────────

def weave_scene(seed=0, n=14, size=600.0):
    """Random straight filaments with random radii and random decisions."""
    rng = np.random.default_rng(seed)
    cl, radii = {}, {}
    for i in range(n):
        if i % 2 == 0:
            y0, y1 = rng.uniform(0, size, 2)
            cl[i] = line(0, y0, size, y1)
        else:
            x0, x1 = rng.uniform(0, size, 2)
            cl[i] = line(x0, 0, x1, size)
        radii[i] = float(rng.uniform(2.5, 7.0))
    p = params()
    cs = D.identify_crossings(cl, p)
    for c in cs:
        u = rng.uniform()
        if u < 0.3:
            c.abstain = True
        else:
            c.raw_over = c.i if u < 0.65 else c.j
            c.score = 1.0 if c.raw_over == c.i else -1.0
    ids = sorted(cl)
    report = D.resolve_global_order(ids, cs, p)
    return cl, cs, radii, report["position"]


class TestProperties:
    def test_it_reads_the_order_and_changes_nothing(self):
        cl, cs, radii, position = weave_scene(1)
        before = copy.deepcopy([(c.i, c.j, c.over, c.raw_over, c.abstain, c.flipped) for c in cs])
        pos_before = dict(position)
        cl_before = {i: p.copy() for i, p in cl.items()}
        z, ext, status, info = solve(cl, cs, radii, position=position)
        assert status == "bending"
        assert [(c.i, c.j, c.over, c.raw_over, c.abstain, c.flipped) for c in cs] == before
        assert position == pos_before
        for i in cl:
            assert np.array_equal(cl[i], cl_before[i])
        # and every constrained pair is the right way up where it meets
        for c in cs:
            hi = c.over if c.over >= 0 else (c.i if position[c.i] < position[c.j] else c.j)
            lo = c.j if hi == c.i else c.i
            assert (B.profile_height_at(cl[hi], info["chains"][hi], c.x, c.y)
                    > B.profile_height_at(cl[lo], info["chains"][lo], c.x, c.y))

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_every_constrained_pair_clears_where_it_meets(self, seed):
        cl, cs, radii, position = weave_scene(seed)
        z, ext, status, info = solve(cl, cs, radii, position=position)
        p = params()
        assert B.clearance_violations(cs, info["chains"], cl, radii, p) == []
        # the flat model clears the same pairs; both gates agree on "none"
        flat, _ = D.solve_metric_z(sorted(cl), cs, radii, {}, p, position)
        assert D.clearance_violations(cs, flat, radii, p) == []

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_interpenetration_across_the_overlap_zone(self, seed):
        """Everywhere two constrained filaments overlap in projection, the
        drawn curves keep sqrt(sep^2 - delta^2) apart, to a residual that
        is the sub-sample spacing's, not the flat model's zero: stated as
        a bound rather than hidden."""
        from scipy.spatial import cKDTree

        cl, cs, radii, position = weave_scene(seed)
        z, ext, status, info = solve(cl, cs, radii, position=position)
        dense = {i: D._densify(cl[i]) for i in cl}
        heights = {i: B.height_profile(cl[i], *info["chains"][i], at=B._arclength(dense[i]))
                   for i in cl}
        worst = 0.0
        for c in cs:
            hi = c.over if c.over >= 0 else (c.i if position[c.i] < position[c.j] else c.j)
            lo = c.j if hi == c.i else c.i
            sep = radii[hi] + radii[lo]
            near = cKDTree(dense[lo]).query_ball_point(dense[hi], r=sep)
            for a, lst in enumerate(near):
                if not lst:
                    continue
                delta = np.hypot(*(dense[lo][lst] - dense[hi][a]).T)
                need = np.sqrt(np.maximum(sep * sep - delta * delta, 0.0))
                worst = max(worst, float((need - (heights[hi][a] - heights[lo][lst])).max()))
        assert worst < 0.15

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_drawn_curve_respects_the_reported_radius(self, seed):
        cl, cs, radii, position = weave_scene(seed)
        z, ext, status, info = solve(cl, cs, radii, position=position)
        for i in cl:
            knots, h = info["chains"][i]
            R = B.min_radius_px(info["rho"], 2 * radii[i])
            assert hermite_min_radius(knots, h) >= R * (1 - 1e-6)
        assert info["min_radius_px_range"][0] <= info["min_radius_px"] <= info["min_radius_px_range"][1]

    def test_the_film_is_never_thicker_than_the_flat_model(self):
        for seed in range(4):
            cl, cs, radii, position = weave_scene(seed)
            p = params()
            z, ext, status, info = solve(cl, cs, radii, position=position)
            flat, _ = D.solve_metric_z(sorted(cl), cs, radii, {}, p, position)
            assert (B.film_thickness(sorted(cl), ext, radii, p)
                    <= D.film_thickness(sorted(cl), flat, radii, p) + 1e-6)

    def test_sample_spacing_moves_the_film_by_a_stated_amount(self):
        """A discretisation, converging from above: finer is thinner, and
        halving the spacing from 6 px changes the film by under a tenth."""
        cl, cs, radii, position = weave_scene(5, n=16)
        p = params()
        films = {}
        for step in (3.0, 6.0, 12.0):
            z, ext, status, info = solve(cl, cs, radii, position=position, bend_sample_px=step)
            films[step] = B.film_thickness(sorted(cl), ext, radii, p)
        assert films[3.0] <= films[6.0] + 1e-6 <= films[12.0] + 2e-6
        assert abs(films[3.0] - films[6.0]) <= 0.1 * films[6.0]

    def test_deterministic(self):
        cl, cs, radii, position = weave_scene(2)
        a = solve(cl, cs, radii, position=position)
        b = solve(cl, cs, radii, position=position)
        for i in cl:
            assert np.array_equal(a[3]["chains"][i][1], b[3]["chains"][i][1])

    def test_two_stage_lexicographic_solve_agrees_with_the_weighted_one(self):
        """The shipped weight is a tie-break, not a trade: the two-stage
        solve (area first, then variation) gives the same area and the same
        film."""
        cl, cs, radii, position = weave_scene(3)
        p = params()
        weighted = solve(cl, cs, radii, position=position)
        saved = B._FLATNESS_WEIGHT
        try:
            B._FLATNESS_WEIGHT = None
            lexicographic = solve(cl, cs, radii, position=position)
        finally:
            B._FLATNESS_WEIGHT = saved
        assert lexicographic[3]["objective"]["second_stage"]
        a, b = weighted[3]["objective"], lexicographic[3]["objective"]
        assert a["lifted_area_px2"] == pytest.approx(b["lifted_area_px2"], rel=1e-6)
        assert (B.film_thickness(sorted(cl), weighted[1], radii, p)
                == pytest.approx(B.film_thickness(sorted(cl), lexicographic[1], radii, p), abs=1e-3))


# ── degenerate geometry ────────────────────────────────────────────────────

class TestDegenerate:
    def test_no_crossings(self):
        cl = {1: line(0, 0, 100, 0), 2: line(0, 50, 100, 50)}
        z, ext, status, info = solve(cl, [], {1: 3.0, 2: 3.0})
        assert status == "no_edges"
        assert z == {1: 0.0, 2: 0.0}

    def test_one_crossing_touches_exactly(self):
        cl = {1: line(0, 50, 100, 50), 2: line(50, 0, 50, 100)}
        cs = [crossing(1, 2, 50, 50, over=2)]
        z, ext, status, info = solve(cl, cs, {1: 3.0, 2: 4.0}, position={2: 0, 1: 1})
        gap = (B.profile_height_at(cl[2], info["chains"][2], 50, 50)
               - B.profile_height_at(cl[1], info["chains"][1], 50, 50))
        # touching, plus the chord margin (d/q)^2 / (8 R) of each filament,
        # 0.033 px here: the guarantee is >= sep, the slack is the margin
        assert 7.0 - 1e-9 <= gap <= 7.05

    def test_closed_loop_short_filament_zero_length_and_no_radius(self):
        t = np.linspace(0, 2 * np.pi, 80)
        loop = np.column_stack([50 + 30 * np.cos(t), 50 + 30 * np.sin(t)])
        cl = {1: loop,
              2: line(0, 50, 100, 50),
              3: np.array([[20.0, 20.0], [20.5, 20.0]]),          # very short
              4: np.array([[80.0, 50.0], [80.0, 50.0]]),          # zero length
              5: np.array([[20.0, 50.0]])}                        # one point
        cs = [crossing(1, 2, 20, 50, over=2), crossing(1, 2, 80, 50, over=2),
              crossing(2, 4, 80, 50, over=4), crossing(2, 5, 20, 50, over=2),
              crossing(2, 3, 20, 50)]
        radii = {1: 3.0, 2: 3.0, 3: 2.0, 4: 2.5}                   # 5 has none
        z, ext, status, info = solve(cl, cs, radii, position={4: 0, 2: 1, 1: 2, 3: 3, 5: 4})
        assert status == "bending"
        assert set(info["chains"]) == {1, 2, 3, 4, 5}
        assert len(info["profiles"][5]) == 1
        assert B.clearance_violations(cs, info["chains"], cl, radii, params()) == []

    def test_coincident_crossings_between_different_pairs(self):
        cl = {1: line(0, 50, 100, 50), 2: line(50, 0, 50, 100), 3: line(0, 0, 100, 100)}
        cs = [crossing(1, 2, 50, 50, over=1), crossing(1, 3, 50, 50, over=3),
              crossing(2, 3, 50, 50, over=3)]
        z, ext, status, info = solve(cl, cs, {1: 3.0, 2: 3.0, 3: 3.0}, position={3: 0, 1: 1, 2: 2})
        assert B.clearance_violations(cs, info["chains"], cl, {1: 3.0, 2: 3.0, 3: 3.0}, params()) == []

    def test_a_missing_order_position_leaves_an_abstained_pair_free(self):
        cl = {1: line(0, 50, 100, 50), 2: line(50, 0, 50, 100)}
        cs = [crossing(1, 2, 50, 50)]
        z, ext, status, info = solve(cl, cs, {1: 3.0, 2: 3.0}, position={})
        assert status == "no_edges"


# ── the hook in run_scene ──────────────────────────────────────────────────

class TestRunScene:
    def scene(self):
        canvas = np.full((120, 120), 0.1, dtype=np.float32)
        cl = {1: line(10, 60, 110, 60), 2: line(60, 10, 60, 110)}
        return canvas, cl, {1: 4.0, 2: 4.0}

    def test_default_output_carries_no_bending_field(self):
        canvas, cl, radii = self.scene()
        out = D.run_scene(canvas, cl, radii=radii, params=D.DepthParams())
        assert out["assumptions"]["planar_instances"] == "dz/ds = 0"
        assert "height_model" not in out["solver_report"]
        assert "bend_rho" not in out["solver_report"]

    def test_bending_output_says_so(self):
        canvas, cl, radii = self.scene()
        out = D.run_scene(canvas, cl, radii=radii, params=D.DepthParams(height_model="bending"))
        rep = out["solver_report"]
        assert rep["metric_z_status"] == "bending"
        assert rep["height_model"] == "bending"
        assert rep["bend_rho"] == pytest.approx(8.09)
        assert rep["bend_min_radius_px"] == pytest.approx(B.min_radius_px(8.09, 8.0), abs=1e-3)
        assert rep["film_thickness_px"] == pytest.approx(16.0, abs=0.05)
        assert out["assumptions"]["planar_instances"].startswith("|d2z/ds2|")
        assert rep["n_interpenetrating"] == 0

    def test_tube_mesh_follows_a_profile(self):
        pts = line(0, 0, 100, 0)
        profile = np.linspace(0.0, 10.0, len(pts))
        verts, faces = D.tube_mesh(pts, profile, 2.0, n_theta=8)
        zc = verts[:, 2].reshape(-1, 8).mean(axis=1)
        assert zc[0] == pytest.approx(0.0, abs=1e-6)
        assert zc[-1] == pytest.approx(10.0, abs=1e-6)
        flat_verts, _ = D.tube_mesh(pts, 3.0, 2.0, n_theta=8)
        assert np.allclose(flat_verts[:, 2].reshape(-1, 8).mean(axis=1), 3.0)
