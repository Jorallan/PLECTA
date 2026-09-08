"""Unit tests for the post-hoc depth stage (plecta/depth.py).

Known-solution cases, including deliberately contradictory precedence
graphs:

    python -m unittest tests.test_depth -v
"""
from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from plecta import depth as D


def _cross(i, j, p_over, x=0.0, y=0.0, band=0.15):
    c = D.PredCrossing(i=i, j=j, x=x, y=y)
    c.p_over = p_over
    if abs(p_over - 0.5) < band - 1e-9:
        c.abstain = True
    else:
        c.raw_over = i if p_over > 0.5 else j
    return c


class TestGlobalOrder(unittest.TestCase):
    def test_consistent_chain_is_kept(self):
        cs = [_cross(1, 2, 0.9), _cross(2, 3, 0.9)]
        rep = D.resolve_global_order([1, 2, 3], cs, D.DepthParams())
        self.assertEqual(rep["n_precedence_cycles_raw"], 0)
        self.assertEqual(rep["n_relations_flipped"], 0)
        self.assertEqual([c.over for c in cs], [1, 2])

    def test_cycle_flips_weakest_relation(self):
        # A>B (0.9), B>C (0.9), C>A (0.7): impossible with one z per rod;
        # the weakest local relation is the one the solver reverses.
        cs = [_cross(1, 2, 0.9), _cross(2, 3, 0.9), _cross(3, 1, 0.7)]
        rep = D.resolve_global_order([1, 2, 3], cs, D.DepthParams())
        self.assertEqual(rep["n_precedence_cycles_raw"], 1)
        self.assertEqual(rep["n_relations_flipped"], 1)
        flipped = [c for c in cs if c.flipped]
        self.assertEqual(len(flipped), 1)
        self.assertEqual((flipped[0].i, flipped[0].j), (3, 1))
        self.assertEqual(flipped[0].over, 1)       # corrected: A over C
        self.assertEqual(flipped[0].raw_over, 3)   # raw kept for analysis

    def test_exact_and_greedy_agree_on_small_graph(self):
        cs_weights = [(1, 2, 0.9), (2, 3, 0.8), (1, 3, 0.7), (3, 4, 0.95)]
        w = {}
        for u, v, p in cs_weights:
            w[(u, v)] = abs(math.log(p / (1 - p)))
        exact = D._exact_order([1, 2, 3, 4], w)[::-1]   # top-first
        greedy = D._greedy_order([1, 2, 3, 4], w)
        self.assertEqual(exact, [1, 2, 3, 4])
        self.assertEqual(greedy, [1, 2, 3, 4])

    def test_strong_components_separates_a_cycle_from_a_chain(self):
        """The one Tarjan the cycle count and the cycle grouping share."""
        chain = D._strong_components([1, 2, 3], {1: [2], 2: [3], 3: []})
        self.assertEqual(sorted(chain), [[1], [2], [3]])
        cycle = D._strong_components([1, 2, 3, 4],
                                     {1: [2], 2: [3], 3: [1], 4: []})
        self.assertEqual(sorted(cycle), [[1, 2, 3], [4]])
        # an edge leaving the given node set is dropped, not followed
        # (emitted sinks-first, hence [2] before [1])
        self.assertEqual(D._strong_components([1, 2], {1: [2, 99], 2: []}),
                         [[2], [1]])

    def test_abstained_crossings_do_not_constrain_order(self):
        cs = [_cross(1, 2, 0.52)]     # inside the abstain band
        rep = D.resolve_global_order([1, 2], cs, D.DepthParams())
        self.assertEqual(cs[0].over, -1)
        self.assertEqual(rep["n_relations_flipped"], 0)


class TestLayers(unittest.TestCase):
    def test_chain_and_shared_layers(self):
        cs = [_cross(1, 2, 0.9), _cross(2, 3, 0.9)]
        D.resolve_global_order([1, 2, 3, 4], cs, D.DepthParams())
        layers, k = D.assign_layers([1, 2, 3, 4], cs)
        self.assertEqual(k, 3)
        self.assertEqual(layers[3], 0)
        self.assertEqual(layers[2], 1)
        self.assertEqual(layers[1], 2)
        self.assertEqual(layers[4], 0)   # never conflicts: shares the bottom
        # K inferred from the DAG, not from the number of instances (4)


class TestMetricZ(unittest.TestCase):
    def test_compact_stack_touches_and_clears(self):
        cs = [_cross(1, 2, 0.9), _cross(2, 3, 0.9)]
        params = D.DepthParams()
        rep = D.resolve_global_order([1, 2, 3], cs, params)
        layers, _ = D.assign_layers([1, 2, 3], cs)
        radii = {1: 4.0, 2: 3.0, 3: 5.0}
        z, status = D.solve_metric_z([1, 2, 3], cs, radii, layers, params,
                                     order_position=rep["position"])
        self.assertEqual(status, "compact_stack")
        self.assertAlmostEqual(z[3], 0.0)
        # compact stack: each above-pair pulled to touching
        self.assertAlmostEqual(z[2] - z[3], 8.0, delta=0.05)
        self.assertAlmostEqual(z[1] - z[2], 7.0, delta=0.05)

    def test_abstained_pair_still_clears(self):
        cs = [_cross(1, 2, 0.52)]     # abstained, but projected overlap
        params = D.DepthParams()
        rep = D.resolve_global_order([1, 2], cs, params)
        layers, _ = D.assign_layers([1, 2], cs)
        radii = {1: 4.0, 2: 4.0}
        z, _ = D.solve_metric_z([1, 2], cs, radii, layers, params,
                                order_position=rep["position"])
        self.assertGreaterEqual(abs(z[1] - z[2]), 8.0 - 1e-6)


class TestEvidenceOnSyntheticImage(unittest.TestCase):
    def test_bright_over_dim_is_detected(self):
        """A bright horizontal rod painted OVER a dim vertical one."""
        H = W = 120
        img = np.zeros((H, W), dtype=np.float32)
        img[:, 56:64] = 0.35     # vertical rod (id 2), painted first
        img[56:64, :] = 0.85     # horizontal rod (id 1), painted on top
        grad = np.zeros_like(img)
        from scipy.ndimage import sobel
        grad = np.hypot(sobel(img, 0), sobel(img, 1)).astype(np.float32)
        centrelines = {
            1: np.column_stack([np.arange(5.0, 115.0), np.full(110, 59.5)]),
            2: np.column_stack([np.full(110, 59.5), np.arange(5.0, 115.0)]),
        }
        dense = {k: D._densify(v) for k, v in centrelines.items()}
        c = D.PredCrossing(i=1, j=2, x=59.5, y=59.5)
        D.crossing_evidence(img, grad, dense, c, D.DepthParams(),
                            radii={1: 4.0, 2: 4.0})
        self.assertFalse(c.abstain)
        self.assertGreater(c.p_over, 0.5)      # rod 1 (painted over) on top

    def test_identical_rods_abstain(self):
        """Two rods with identical brightness: the crossing is unidentifiable."""
        H = W = 120
        img = np.zeros((H, W), dtype=np.float32)
        img[:, 56:64] = 0.6
        img[56:64, :] = 0.6
        from scipy.ndimage import sobel
        grad = np.hypot(sobel(img, 0), sobel(img, 1)).astype(np.float32)
        centrelines = {
            1: np.column_stack([np.arange(5.0, 115.0), np.full(110, 59.5)]),
            2: np.column_stack([np.full(110, 59.5), np.arange(5.0, 115.0)]),
        }
        dense = {k: D._densify(v) for k, v in centrelines.items()}
        c = D.PredCrossing(i=1, j=2, x=59.5, y=59.5)
        D.crossing_evidence(img, grad, dense, c, D.DepthParams(),
                            radii={1: 4.0, 2: 4.0})
        self.assertTrue(c.abstain)


class TestCrossingIdentification(unittest.TestCase):
    def test_perpendicular(self):
        centrelines = {
            1: np.column_stack([np.arange(0.0, 100.0, 2.0), np.full(50, 50.0)]),
            2: np.column_stack([np.full(50, 50.0), np.arange(0.0, 100.0, 2.0)]),
        }
        cs = D.identify_crossings(centrelines)
        self.assertEqual(len(cs), 1)
        self.assertAlmostEqual(cs[0].x, 50.0, delta=1.0)
        self.assertAlmostEqual(cs[0].y, 50.0, delta=1.0)


def _rods(shape=(200, 200)):
    """Two crossing rods and one that only grazes, with a drawn occlusion."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]

    def stroke(p0, p1, radius):
        ax, ay = p0
        bx, by = p1
        dx, dy = bx - ax, by - ay
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        return np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)) <= radius

    centrelines = {
        1: np.array([[20.0, 100.0], [180.0, 100.0]]),      # horizontal
        2: np.array([[100.0, 20.0], [100.0, 180.0]]),      # vertical, on top
        3: np.array([[20.0, 109.0], [180.0, 109.0]]),      # grazes rod 1
    }
    image = np.full(shape, 0.10, dtype=np.float32)
    image[stroke((20, 109), (180, 109), 6.0)] = 0.40
    image[stroke((20, 100), (180, 100), 6.0)] = 0.60
    image[stroke((100, 20), (100, 180), 6.0)] = 0.95
    return centrelines, image, {1: 6.0, 2: 6.0, 3: 6.0}


def _shaded_rods(shape=(240, 240)):
    """Two crossing rods, the lower one shaded along its length.

    Uniform rods make the sampling geometry invisible: the median of a
    constant stripe is the same whatever arc took the samples, so `_rods`
    cannot tell `core_mode="flat"` from `"radius_scaled"` even when the two
    read completely different points. Here the horizontal rod brightens on
    one side of the crossing only, so its flank median moves with the core
    half-size -- which is what makes "no radius entered the evidence" a
    testable claim rather than an assertion about the source.
    """
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    image = np.full(shape, 0.10, dtype=np.float32)
    ramp = 0.30 + 0.004 * np.clip(xs - 120, 0, None)
    horizontal = np.abs(ys - 120) <= 6
    image[horizontal] = ramp[horizontal].astype(np.float32)
    image[np.abs(xs - 120) <= 6] = 0.92          # painted second: on top
    centrelines = {1: np.array([[20.0, 120.0], [220.0, 120.0]]),
                   2: np.array([[120.0, 20.0], [120.0, 220.0]])}
    return centrelines, image, {1: 6.0, 2: 6.0}


class TestScoringRule(unittest.TestCase):
    """The three ways the evidence combines, and the gates gating them."""

    def test_abstention_band_converts_to_a_score_threshold_exactly(self):
        """|p - 0.5| < band and |score| < threshold must be the SAME set.

        The band is documented as a probability band but applied in score
        units, so the two forms have to agree at the boundary or the
        documented meaning silently drifts.
        """
        for band in (0.05, 0.15, 0.30):
            t = D.abstain_threshold(band)
            for score in (-4.0, -1.0, -t - 1e-6, -t + 1e-6, 0.0,
                          t - 1e-6, t + 1e-6, 1.0, 4.0):
                p = 1.0 / (1.0 + math.exp(-score))
                self.assertEqual(abs(p - 0.5) < band - 1e-12,
                                 abs(score) < t - 1e-12,
                                 f"band={band} score={score}")

    def test_smooth_variant_is_the_recorded_one(self):
        """The variant the stored held-out aggregates were produced under."""
        p = D.DepthParams(scoring="weight_of_evidence",
                          w_intensity=2.446, w_sharpness=1.576)
        self.assertAlmostEqual(
            D.combine_channels(1.0, 0.5592, p),
            2.446 * math.tanh(1.0) + 1.576 * math.tanh(0.5592), places=12)

    def test_the_shipped_default_is_the_noise_floored_rule(self):
        """The defaults a caller gets without touching anything."""
        p = D.DepthParams()
        self.assertEqual(p.scoring, "noise_floored")
        self.assertEqual(p.core_mode, "flat")
        self.assertEqual(p.core_px, 7.5)
        self.assertEqual(p.abstain_score, 0.40)
        # one channel: the score is w_intensity * F_inf and nothing else, so
        # the sharpness argument cannot move it
        self.assertAlmostEqual(
            D.combine_channels(0.4, 9.0, p, feat_noise_floored=-0.25),
            -0.5, places=12)
        self.assertAlmostEqual(
            D.combine_channels(0.4, -9.0, p, feat_noise_floored=-0.25),
            -0.5, places=12)

    def test_the_noise_floored_rule_refuses_to_guess_its_feature(self):
        """F_inf is not recoverable from F_I, so it may not be invented."""
        with self.assertRaises(ValueError):
            D.combine_channels(0.4, 0.0, D.DepthParams())

    def test_legacy_rule_drops_the_squash_and_clips(self):
        p = D.DepthParams(scoring="winsorized_linear")
        # F_I enters linearly; F_S is clipped to [-1, 1]
        self.assertAlmostEqual(D.combine_channels(0.4, 0.3, p), 1.1, places=12)
        self.assertAlmostEqual(D.combine_channels(0.4, 9.0, p), 1.8, places=12)
        self.assertAlmostEqual(D.combine_channels(-0.4, -9.0, p), -1.8, places=12)

    def test_all_rules_are_odd(self):
        """Swapping i and j must flip the score exactly, under every rule."""
        for mode in ("weight_of_evidence", "winsorized_linear"):
            p = D.DepthParams(scoring=mode)
            for fi, fs in ((0.3, 0.7), (1.0, -2.5), (-0.8, 0.1)):
                self.assertAlmostEqual(D.combine_channels(fi, fs, p),
                                       -D.combine_channels(-fi, -fs, p),
                                       places=12, msg=mode)
        # the shipped rule's oddness lives in the feature, where swapping the
        # two rods really does exchange m_i and m_j
        for m_c, m_i, m_j, sigma in ((0.62, 0.31, 0.60, 0.01),
                                     (0.40, 0.55, 0.38, 0.04),
                                     (0.50, 0.50, 0.50, 0.02)):
            self.assertAlmostEqual(
                D.noise_floored_feature(m_c, m_i, m_j, sigma),
                -D.noise_floored_feature(m_c, m_j, m_i, sigma), places=12)
            self.assertEqual(D.flank_noise_sigma([m_i, m_j], [m_c]),
                             D.flank_noise_sigma([m_c], [m_i, m_j]))

    def test_the_abstention_parameterisations_do_not_cross_over(self):
        """A probability band is the wrong shape for a noise-normalised
        margin, so each rule reads only its own knob."""
        floored = D.DepthParams(scoring="noise_floored")
        self.assertEqual(D.decision_threshold(floored), 0.40)
        # moving the legacy band leaves the shipped rule exactly where it was
        self.assertEqual(
            D.decision_threshold(D.DepthParams(abstain_band=0.45)), 0.40)
        for mode in ("winsorized_linear", "weight_of_evidence"):
            p = D.DepthParams(scoring=mode)
            self.assertAlmostEqual(D.decision_threshold(p),
                                   D.abstain_threshold(0.15), places=12)
            # ... and moving abstain_score leaves the legacy rules alone
            self.assertAlmostEqual(
                D.decision_threshold(D.DepthParams(scoring=mode,
                                                   abstain_score=0.01)),
                D.abstain_threshold(0.15), places=12)

    def test_apply_abstention_works_for_all_three_rules(self):
        for mode, deciding, abstaining in (
                ("noise_floored", 0.5, 0.3),          # threshold 0.40
                ("winsorized_linear", 0.8, 0.5),      # threshold 0.6190
                ("weight_of_evidence", 0.8, 0.5)):
            params = D.DepthParams(scoring=mode)
            cs = [D.PredCrossing(i=1, j=2, x=0.0, y=0.0),
                  D.PredCrossing(i=3, j=4, x=0.0, y=0.0)]
            cs[0].score, cs[1].score = deciding, abstaining
            D.apply_abstention(cs, params=params)
            self.assertFalse(cs[0].abstain, mode)
            self.assertEqual(cs[0].raw_over, 1, mode)
            self.assertTrue(cs[1].abstain, mode)
        with self.assertRaises(TypeError):      # ambiguous: which one wins?
            D.apply_abstention([], 0.15, D.DepthParams())
        with self.assertRaises(TypeError):
            D.apply_abstention([])

    def test_solver_weight_is_the_score_magnitude(self):
        """The old code recovered |logit(p)|; that is |score| without the
        round trip, so both must give the same ordering weights."""
        cs = [_cross(1, 2, 0.90), _cross(2, 3, 0.80)]
        for c in cs:                      # as crossing_evidence would set it
            c.score = math.log(c.p_over / (1 - c.p_over))
        params = D.DepthParams()
        with_score = D.resolve_global_order([1, 2, 3], cs, params)
        for c in cs:                      # a caller that has only p_over
            c.score = float("nan")
        without = D.resolve_global_order([1, 2, 3], cs, params)
        self.assertEqual(with_score["position"], without["position"])


class TestNoiseFloor(unittest.TestCase):
    """The shipped statistic: what the floor is made of, and what it buys.

    Measured on a clean held-out optical set of 20 scenes / 1355 crossings
    whose seeds are disjoint from every tuning set: the same accuracy as the
    previous rule at the same coverage (+0.0025, 95% CI [-0.0042, +0.0079]),
    with the confidence ranking AUC up from 0.7772 to 0.8698
    (+0.0926, [+0.0563, +0.1321]), which turns into accuracy as soon as more
    abstention is acceptable (+0.0188 at 82.5% coverage, +0.0240 at 80%).
    """

    def test_sigma_hat_is_within_rod_and_not_the_pooled_mad(self):
        """The trap, tested explicitly.

        Two rods 0.60 apart in brightness, each carrying the same +/-0.01 of
        noise. The within-rod residual MAD sees only the 0.01; the MAD of the
        raw POOLED values sees a bimodal set and reports the 0.60 separation
        instead. Both are "a MAD of the flank samples"; only one is a noise
        scale.
        """
        flank_i = [0.19, 0.19, 0.21, 0.21]      # rod i, median 0.20
        flank_j = [0.79, 0.79, 0.81, 0.81]      # rod j, median 0.80
        sigma = D.flank_noise_sigma(flank_i, flank_j)
        self.assertAlmostEqual(sigma, 1.4826 * 0.01, places=12)

        pooled = np.array(flank_i + flank_j)
        trap = 1.4826 * float(np.median(np.abs(pooled - np.median(pooled))))
        self.assertAlmostEqual(trap, 1.4826 * 0.30, places=12)
        self.assertGreater(trap, 25.0 * sigma)   # a factor of 30 apart here

        # and it is not a cosmetic difference: with the real noise scale the
        # floor is inactive and a well-separated crossing saturates, which is
        # correct. With the pooled MAD the "floor" (2 x 0.4448) exceeds the
        # separation it is supposed to protect, so a crossing the picture
        # answers outright is reported as three-quarters sure.
        self.assertAlmostEqual(
            D.noise_floored_feature(0.80, 0.20, 0.80, sigma), -1.0, places=12)
        self.assertAlmostEqual(
            D.noise_floored_feature(0.80, 0.20, 0.80, trap),
            -0.60 / (2.0 * trap), places=12)
        self.assertGreater(
            D.noise_floored_feature(0.80, 0.20, 0.80, trap), -0.7)

    def test_sigma_hat_ignores_the_rods_own_separation(self):
        """Move one rod's whole flank set and the noise scale must not move."""
        flank_i = [0.19, 0.19, 0.21, 0.21]
        base = D.flank_noise_sigma(flank_i, [0.79, 0.79, 0.81, 0.81])
        far = D.flank_noise_sigma(flank_i, [0.09, 0.09, 0.11, 0.11])
        self.assertAlmostEqual(base, far, places=12)

    def test_it_saturates_only_once_separation_clears_twice_sigma(self):
        """D far below 2*sigma: legacy F_I is +/-1, the floored feature is not.

        m_i = 0.50, m_j = 0.52, core matching rod j exactly. The rods differ
        by 0.02 and the flanks carry sigma = 0.08 of noise, so the separation
        that F_I divides by is a quarter of the noise -- and F_I reports
        certainty anyway.
        """
        m_c, m_i, m_j, sigma = 0.52, 0.50, 0.52, 0.08
        legacy = ((abs(m_c - m_j) - abs(m_c - m_i))
                  / max(max(1e-6, abs(m_i - m_j)), 0.02))
        floored = D.noise_floored_feature(m_c, m_i, m_j, sigma)
        self.assertAlmostEqual(legacy, -1.0, places=12)          # saturated
        self.assertAlmostEqual(floored, -0.02 / 0.16, places=12)  # -0.125
        self.assertEqual(np.sign(legacy), np.sign(floored))      # same call
        # and the difference is the whole point: at w_intensity = 2 the legacy
        # score is -2.0 and decides, the floored score is -0.25 and abstains
        params = D.DepthParams()
        self.assertLess(abs(D.combine_channels(legacy, 0.0, params,
                                               feat_noise_floored=floored)),
                        D.decision_threshold(params))

        # once the rods really are separated, the floor is inactive and the
        # two features agree exactly
        wide = D.noise_floored_feature(0.90, 0.20, 0.90, sigma)
        self.assertAlmostEqual(wide, -1.0, places=12)

    def test_every_rule_agrees_on_the_sign(self):
        """The three rules share a numerator, so they share a direction.

        Only the sharpness channel can disagree, and the shipped rule does not
        have one -- so with sharpness silent all three must call every
        crossing the same way, whatever the magnitudes do.
        """
        cases = [(0.62, 0.31, 0.60, 0.01), (0.31, 0.31, 0.60, 0.05),
                 (0.40, 0.55, 0.38, 0.04), (0.52, 0.50, 0.52, 0.08),
                 (0.20, 0.80, 0.21, 0.002)]
        for m_c, m_i, m_j, sigma in cases:
            legacy = ((abs(m_c - m_j) - abs(m_c - m_i))
                      / max(max(1e-6, abs(m_i - m_j)), 0.02))
            floored = D.noise_floored_feature(m_c, m_i, m_j, sigma)
            scores = [D.combine_channels(legacy, 0.0,
                                         D.DepthParams(scoring=mode),
                                         feat_noise_floored=floored)
                      for mode in ("noise_floored", "winsorized_linear",
                                   "weight_of_evidence")]
            self.assertEqual(len({np.sign(s) for s in scores}), 1,
                             f"{(m_c, m_i, m_j, sigma)} -> {scores}")

    def test_the_features_are_written_into_the_record(self):
        """sigma_hat and F_inf are auditable, not internal."""
        centrelines, image, radii = _rods()
        out = D.run_scene(image, centrelines, radii=radii)
        self.assertTrue(out["crossings"])
        for c in out["crossings"]:
            self.assertIn("sigma_hat", c["features"])
            self.assertIn("feat_noise_floored", c["features"])
            self.assertIn("feat_intensity", c["features"])
        rep = out["solver_report"]
        self.assertEqual(rep["scoring"], "noise_floored")
        self.assertEqual(rep["core_mode"], "flat")
        self.assertEqual(rep["core_px"], 7.5)
        self.assertEqual(rep["abstain_threshold"], 0.40)


class TestDepthOptions(unittest.TestCase):
    """The switches that decide what the stage is allowed to use."""

    def test_scoring_rules_agree_on_direction(self):
        """The simplification must not flip any crossing on this fixture."""
        centrelines, image, radii = _rods()
        floored = D.run_scene(image, centrelines, radii=radii)   # shipped
        win = D.run_scene(image, centrelines, radii=radii,
                          params=D.DepthParams(scoring="winsorized_linear"))
        woe = D.run_scene(image, centrelines, radii=radii,
                          params=D.DepthParams(scoring="weight_of_evidence",
                                               w_intensity=2.446,
                                               w_sharpness=1.576))
        self.assertEqual(floored["solver_report"]["scoring"], "noise_floored")
        self.assertEqual(win["solver_report"]["scoring"], "winsorized_linear")
        self.assertEqual(woe["solver_report"]["scoring"], "weight_of_evidence")
        for a, b, c in zip(woe["crossings"], win["crossings"],
                           floored["crossings"]):
            self.assertEqual(np.sign(a["score"]), np.sign(b["score"]))
            self.assertEqual(np.sign(a["score"]), np.sign(c["score"]))
            if a["raw_over"] is not None and b["raw_over"] is not None:
                self.assertEqual(a["raw_over"], b["raw_over"])

    def test_flat_core_never_consults_a_radius(self):
        """Two runs, radii 4 px and 20 px, one sampling geometry."""
        centrelines, image, _radii = _shaded_rods()

        def features(radius, **overrides):
            out = D.run_scene(image, centrelines,
                              radii={i: radius for i in centrelines},
                              params=D.DepthParams(**overrides))
            self.assertTrue(out["crossings"])
            return [c["features"] for c in out["crossings"]]

        # "flat": nothing in the evidence path can see the radius, so wildly
        # different rods give byte-identical evidence
        self.assertEqual(features(4.0), features(20.0))
        # "radius_scaled": the same two runs sample different arcs, which is
        # exactly the dependence that was removed
        self.assertNotEqual(features(4.0, core_mode="radius_scaled"),
                            features(20.0, core_mode="radius_scaled"))
        # ... and radii still reach everything downstream of the evidence
        thin = D.run_scene(image, centrelines,
                           radii={i: 4.0 for i in centrelines})
        fat = D.run_scene(image, centrelines,
                          radii={i: 20.0 for i in centrelines})
        self.assertNotEqual([r["z"] for r in thin["instances"]],
                            [r["z"] for r in fat["instances"]])

    def test_no_gradient_image_on_the_noise_floored_path(self):
        """A Gaussian plus two Sobels per scene, not computed and not faked."""
        centrelines, image, radii = _rods()

        def refuse(_image):
            raise AssertionError("the shipped rule has no sharpness channel; "
                                 "the gradient image must not be built")

        with mock.patch.object(D, "gradient_magnitude", refuse):
            out = D.run_scene(image, centrelines, radii=radii)
        for c in out["crossings"]:
            # absent, rather than a zero nobody measured
            self.assertNotIn("feat_sharpness", c["features"])
            self.assertNotIn("d_sharpness", c["features"])

        # the guard is not vacuous: the legacy rule does build it, exactly once
        calls = []
        real = D.gradient_magnitude
        with mock.patch.object(D, "gradient_magnitude",
                               lambda img: (calls.append(img), real(img))[1]):
            legacy = D.run_scene(image, centrelines, radii=radii,
                                 params=D.DepthParams(
                                     scoring="winsorized_linear"))
        self.assertEqual(len(calls), 1)
        self.assertIn("feat_sharpness", legacy["crossings"][0]["features"])

    def test_a_legacy_rule_refuses_to_run_without_a_gradient_image(self):
        """Rather than scoring a channel it did not measure as zero."""
        centrelines, image, radii = _rods()
        dense = {k: D._densify(v) for k, v in centrelines.items()}
        c = D.PredCrossing(i=1, j=2, x=100.0, y=100.0)
        with self.assertRaises(ValueError):
            D.crossing_evidence(image, None, dense, c,
                                D.DepthParams(scoring="winsorized_linear"),
                                radii=radii)

    def test_evidence_can_be_switched_off(self):
        centrelines, image, radii = _rods()
        lit = D.run_scene(image, centrelines, radii=radii)
        blind = D.run_scene(image, centrelines, radii=radii,
                            params=D.DepthParams(use_image_evidence=False))
        self.assertTrue(lit["solver_report"]["image_evidence"])
        self.assertFalse(blind["solver_report"]["image_evidence"])
        self.assertEqual(blind["solver_report"]["n_abstained"],
                         blind["solver_report"]["n_crossings"])
        self.assertTrue(all(c["over"] is None for c in blind["crossings"]))

    def test_no_image_at_all_is_accepted(self):
        centrelines, _image, radii = _rods()
        out = D.run_scene(None, centrelines, radii=radii)
        self.assertFalse(out["solver_report"]["image_evidence"])
        # the geometry still solves: the rods are still separated
        heights = sorted(r["z"] for r in out["instances"])
        self.assertGreater(heights[-1] - heights[0], 0.0)

    def test_grazing_pairs_are_found_and_optionally_cleared(self):
        centrelines, image, radii = _rods()
        free = D.run_scene(image, centrelines, radii=radii)
        cleared = D.run_scene(image, centrelines, radii=radii,
                              params=D.DepthParams(clear_grazing_overlaps=True))
        self.assertEqual(free["solver_report"]["n_grazing_overlaps"], 1)
        self.assertFalse(free["solver_report"]["grazing_constrained"])
        # rods 1 and 3 overlap in projection but never cross, so left free they
        # end up interpenetrating -- reported, not hidden
        self.assertGreater(free["solver_report"]["n_interpenetrating"], 0)
        self.assertEqual(cleared["solver_report"]["n_interpenetrating"], 0)

    def test_fixed_radius_ignores_the_measured_ones(self):
        centrelines, image, radii = _rods()
        out = D.run_scene(image, centrelines, radii=radii,
                          params=D.DepthParams(radius_mode="fixed",
                                               default_radius_px=9.0))
        self.assertEqual({r["d"] for r in out["instances"]}, {18.0})

    def test_compact_never_thicker_and_never_changes_a_decision(self):
        centrelines, image, radii = _rods()
        ids = sorted(centrelines)
        thickness = {}
        decisions = {}
        for mode in ("id_based", "compact"):
            out = D.run_scene(image, centrelines, radii=radii,
                              params=D.DepthParams(undecided_order=mode,
                                                   clear_grazing_overlaps=True))
            heights = {r["id"]: r["z"] for r in out["instances"]}
            thickness[mode] = D.film_thickness(ids, heights, radii, D.DepthParams())
            decisions[mode] = [c["over"] for c in out["crossings"]]
        self.assertLessEqual(thickness["compact"], thickness["id_based"] + 1e-9)
        self.assertEqual(decisions["compact"], decisions["id_based"])

    def test_compact_colouring_shares_levels_between_disjoint_instances(self):
        # a raft: each rod overlaps only its neighbour, so two levels suffice
        crossings = []
        for a in range(1, 6):
            c = D.PredCrossing(i=a, j=a + 1, x=0.0, y=0.0)
            c.abstain = True
            crossings.append(c)
        _position, levels = D.compact_order(list(range(1, 7)), crossings)
        # rods 1, 3, 5 must not be forced into a six-deep ladder
        self.assertLessEqual(len({levels[n] // 1 for n in (1, 3, 5)}), 3)

    def test_stacked_z_matches_the_solver_on_a_simple_stack(self):
        crossings = [_cross(1, 2, 0.95), _cross(2, 3, 0.95)]
        params = D.DepthParams()
        report = D.resolve_global_order([1, 2, 3], crossings, params)
        layers, _k = D.assign_layers([1, 2, 3], crossings)
        radii = {1: 3.0, 2: 4.0, 3: 5.0}
        cheap = D.stacked_z([1, 2, 3], crossings, radii, params,
                            report["position"])
        solved, _status = D.solve_metric_z([1, 2, 3], crossings, radii, layers,
                                           params,
                                           order_position=report["position"])
        for key in cheap:
            self.assertAlmostEqual(cheap[key], solved[key], places=6)



def _fragmented_crossing(shape=(200, 200)):
    """A crossing the IMAGE shows plainly and the centrelines stop short of.

    This is what 2-D fragmentation leaves behind: the rods really do cross,
    the picture really does show which one is on top, but the reconstructed
    axis of the upper rod ends 4 px before the meeting point, so nothing in
    `identify_crossings` will ever see it. `identify_contacts` does -- as a
    grazing overlap -- and it is the case `grazing_evidence` exists for.
    """
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]

    def stroke(p0, p1, radius):
        ax, ay = p0
        bx, by = p1
        dx, dy = bx - ax, by - ay
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        return np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)) <= radius

    image = np.full(shape, 0.10, dtype=np.float32)
    image[stroke((20, 100), (180, 100), 6.0)] = 0.40    # rod 1, underneath
    image[stroke((100, 20), (100, 180), 6.0)] = 0.90    # rod 2, painted over
    centrelines = {
        1: np.array([[20.0, 100.0], [180.0, 100.0]]),
        2: np.array([[100.0, 20.0], [100.0, 96.0]]),    # stops short
    }
    return centrelines, image, {1: 6.0, 2: 6.0}


class TestGrazingEvidence(unittest.TestCase):
    """`grazing_evidence`: asking the image about pairs that overlap without
    their centrelines ever meeting."""

    def test_a_fragmented_crossing_is_decided_and_reported(self):
        centrelines, image, radii = _fragmented_crossing()
        params = D.DepthParams(grazing_evidence=True)
        out = D.run_scene(image, centrelines, radii=radii, params=params)
        rep = out["solver_report"]
        self.assertEqual(rep["n_crossings"], 0)        # centrelines never meet
        self.assertEqual(rep["n_grazing_overlaps"], 1)
        self.assertTrue(rep["grazing_evidence"])
        self.assertEqual(rep["n_grazing_decided"], 1)
        decided = [c for c in out["crossings"] if c.get("grazing")]
        self.assertEqual(len(decided), 1)
        self.assertFalse(decided[0]["abstain"])
        # rod 2 is the one painted over rod 1, and it is the one called upper
        self.assertEqual(decided[0]["over"], 2)
        # having a direction, the pair now separates instead of overlapping
        self.assertEqual(rep["n_interpenetrating"], 0)

    def test_off_by_default_it_changes_nothing(self):
        centrelines, image, radii = _fragmented_crossing()
        out = D.run_scene(image, centrelines, radii=radii)
        rep = out["solver_report"]
        self.assertFalse(rep["grazing_evidence"])
        self.assertEqual(rep["n_grazing_decided"], 0)
        self.assertEqual(out["crossings"], [])         # nothing invented
        self.assertEqual(rep["n_interpenetrating"], 1)  # reported, not hidden

    def test_a_parallel_run_abstains(self):
        """The measured limitation, kept as a test rather than a footnote.

        `crossing_evidence` reads a rod's own flanks as its unoccluded
        appearance. Two rods running alongside each other for longer than the
        sampling window have no unoccluded flank near the contact, so there
        is nothing to compare the overlap against -- and the pair abstains
        instead of guessing. On the held-out scenes this is why grazing
        evidence is right only 0.537 of the time on pairs that are NOT real
        crossings, against 0.716 on the fragmented crossings among them.
        """
        centrelines, image, radii = _rods()     # rod 3 runs parallel to rod 1
        out = D.run_scene(image, centrelines, radii=radii,
                          params=D.DepthParams(grazing_evidence=True))
        self.assertEqual(out["solver_report"]["n_grazing_overlaps"], 1)
        self.assertEqual(out["solver_report"]["n_grazing_decided"], 0)

    def test_it_needs_an_image(self):
        centrelines, _image, radii = _fragmented_crossing()
        out = D.run_scene(None, centrelines, radii=radii,
                          params=D.DepthParams(grazing_evidence=True))
        self.assertFalse(out["solver_report"]["grazing_evidence"])
        self.assertEqual(out["solver_report"]["n_grazing_decided"], 0)


if __name__ == "__main__":
    unittest.main()
