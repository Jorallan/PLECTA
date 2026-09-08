"""Joint mode: a MEASURED NEGATIVE RESULT, retained as a record.

Joint mode lets a registered greyscale image influence which arm continues
into which at a 2-D junction. It was built, measured on 2026-08-20, and
**rejected**:

* mean F1 **-0.0129** as shipped, over 12 synthetic development scenes and 3
  real manually-annotated scenes -- never positive on any set;
* **+0.0005** -- neutral -- after repairing three implementation defects, so
  the loss was not merely a bug;
* on the ambiguous stub pairs it acts on, appearance separates right from
  wrong at AUC **0.63** pooled and **~0.53** on real SEM, against geometry's
  own **0.78-0.82** on the same pairs;
* its pre-registered stop criterion -- AUC below ~0.75 -- was met.

**No improvement to 2-D reconstruction is claimed here or anywhere.** The
module is kept, not deleted, so that the finding does not have to be
re-derived by the next person who reads the design document and has the same
idea. It is not part of the shipped pipeline: `plecta/predict.py` never
imports it, there is no console-script entry point for it, and the only way
to run it is `python -m plecta.joint`. Its parameters are recorded
under `joint_greyscale` in
`plecta/parameters.yaml`.

What ships instead is `plecta/depth.py` in posthoc mode: the 2-D core runs
completely unchanged, and only afterward is it asked, of the FINISHED
instances, who is on top at each crossing. That keeps the 2-D grouping
byte-identical to the mask-only core.

Nothing here modifies `plecta/graph.py`, `plecta/linking.py`, or
`plecta/predict.py`. Every function reused from them is imported, not
copied. `predict_joint(mask, sem=None, jparams=None)` -- or any call with
`jparams.lambda_appearance == 0` -- takes a fast path that calls the exact
same `_max_weight_pairs` on the exact same costs `linking.resolve_junctions`
would, so it is not merely intended to match the geometry-only core, it is
structurally unable to differ from it. `tests/test_joint.py` proves this on
real scenes, not just by inspection.

Mechanism, per junction (node) with two or more stubs:

  1. solve the geometry-only matching, exactly as the core does;
  2. for every pair the core chose, ask how much cost the best alternative
     matching would need to pay if that specific pair were forbidden -- the
     margin. A pair is "ambiguous" when that margin is smaller than
     `JointParams.margin_threshold`, in the same cost units `params.j_unmatched`
     already uses;
  3. only for stubs touched by an ambiguous pair, sample a short window of
     the arm's own image appearance (background-subtracted intensity AND
     local edge-gradient magnitude, a few px in from the tip so the sample
     never touches the junction itself), combine the two channels through a
     calibrated logistic model (`W_INTENSITY`, `W_SHARPNESS`, `B_INTERCEPT`
     -- fit on the depth development set, see their docstring), and add the
     resulting cost to every candidate edge that touches one of those stubs;
  4. re-solve that one node's matching with the blended costs. Edges neither
     stub of which was ambiguous are untouched, so a confident geometric
     decision elsewhere in the same node cannot be moved by this step.

This is the appearance channel only (`E_geometry + lambda_a * E_appearance`
from the design document, with `E_appearance` itself a calibrated blend of
an intensity and a sharpness sub-channel); width/focus/occlusion channels
were never implemented, and the result above is why they were not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple, overload

import numpy as np

from .geometry import Frame
from .graph import SkelGraph
from .linking import (Matching, Params, best_gap_option, break_cycles,
                      bridge_gaps, build_frames, chains_from_matching,
                      junction_costs, matching_objective, _max_weight_pairs)


@dataclass
class JointParams:
    #: weight on the appearance term, in the same cost units as the
    #: geometry link cost (radians-equivalent); 0 disables joint mode exactly.
    #:
    #: INTENT: this stays 1.5 and is deliberately NOT set to 0.0. 1.5 is the
    #: weight the rejected result (-0.0129 mean F1, 2026-08-20) was obtained
    #: at, and both this default and `joint_greyscale.lambda_appearance` in
    #: parameters.yaml exist to record the configuration that was measured,
    #: not to ship a safe one. Disabling is already total and structural --
    #: nothing in the shipped pipeline imports this module -- so the value
    #: cannot reach any result. Re-run the module at 1.5 to reproduce the
    #: negative; get the geometry-only core by not running the module.
    lambda_appearance: float = 1.5
    #: a chosen pairing is "ambiguous" when the best forbidding-it
    #: alternative costs less than this much more, in link-cost units
    margin_threshold: float = 0.15
    #: px from the stub tip to start sampling -- skips the junction itself
    skip_px: float = 1.5
    #: px of arm sampled beyond skip_px
    window_px: float = 6.0
    n_samples: int = 6


# ── per-node ambiguity + appearance ────────────────────────────────────────


def _node_prices(stubs: Sequence[int], relief: dict, params: Params):
    if relief:
        return {sid: min(params.j_unmatched,
                         params.j_gap_relief * relief[sid])
                if sid in relief else params.j_unmatched
                for sid in stubs}
    return params.j_unmatched


def _matching_cost(stubs: Sequence[int], pairs: List[Tuple[int, int]],
                   costs: Dict[Tuple[int, int], float], prices) -> float:
    matched = set()
    total = 0.0
    for a, b in pairs:
        total += costs.get((a, b), costs.get((b, a), math.inf))
        matched.add(a)
        matched.add(b)
    for sid in stubs:
        if sid not in matched:
            total += prices[sid] if isinstance(prices, dict) else float(prices)
    return total


def _pair_margins(stubs: Sequence[int], chosen: List[Tuple[int, int]],
                  costs: Dict[Tuple[int, int], float], prices
                  ) -> Dict[Tuple[int, int], float]:
    """For each pair the geometry-only solve chose, the cost of the best
    matching that does not use that pair, minus the chosen matching's own
    cost -- how much the decision to pair these two specifically is worth,
    against the best matching that pairs them with someone else instead.
    """
    chosen_cost = _matching_cost(stubs, chosen, costs, prices)
    margins = {}
    for (a, b) in chosen:
        alt_costs = {k: v for k, v in costs.items() if k != (a, b)}
        alt = _max_weight_pairs(stubs, alt_costs, prices)
        margins[(a, b)] = _matching_cost(stubs, alt, alt_costs, prices) - chosen_cost
    return margins


def _bilinear(image: np.ndarray, rc_points: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    return map_coordinates(image, [rc_points[:, 0], rc_points[:, 1]],
                           order=1, mode="nearest")


#: Calibrated two-channel appearance-difference -> same-filament classifier,
#: logistic-regression fit on 10908 candidate stub pairs from the 12-scene
#: depth development set (synthetic_depth_dev, 2026-08-18): every candidate
#: pair at every multi-stub junction, labelled by whether the two arms share
#: a common-fragment-assigned ground-truth owner. Both weights fit negative,
#: as expected (larger appearance difference -> less likely the same
#: filament); train accuracy 0.81 at the 0.5 threshold, and the two-channel
#: fit beats an intensity-only fit (NLL 0.434 vs 0.440) enough to justify the
#: second channel. Development data only -- frozen before touching held-out.
W_INTENSITY = -2.5814
W_SHARPNESS = -1.0127
B_INTERCEPT = 1.3926


def stub_appearance(sem, grad, frame: Frame, jparams: JointParams
                    ) -> Optional[dict]:
    """Background-subtracted intensity, and local edge-gradient magnitude
    (`grad`, precomputed once per image), a short way into the arm from its
    tip -- the arm's own local appearance, not the junction's. `None` when
    the frame has no reliable direction (nothing to sample along).
    """
    if not frame.reliable or not np.any(frame.tangent):
        return None
    s = np.linspace(jparams.skip_px, jparams.skip_px + jparams.window_px,
                    jparams.n_samples)
    # tangent points OUT of the arm (geometry.Frame's convention); moving
    # against it from the tip walks back INTO the arm's own pixels.
    pts = frame.tip[None, :] - s[:, None] * frame.tangent[None, :]
    H, W = sem.image.shape
    pts = np.stack([np.clip(pts[:, 0], 0, H - 1.001),
                    np.clip(pts[:, 1], 0, W - 1.001)], axis=1)
    ridge = _bilinear(sem.image, pts) - _bilinear(sem.background, pts)
    sharp = _bilinear(grad, pts)
    return {"intensity": float(np.median(ridge)),
           "sharpness": float(np.median(sharp))}


def appearance_features(image: np.ndarray):
    """The gradient-magnitude field `stub_appearance` samples from, computed
    once per image: `depth.gradient_magnitude` itself, so this is the depth
    stage's own occlusion-evidence channel rather than a copy of it."""
    from .depth import gradient_magnitude

    return gradient_magnitude(image)


def appearance_cost(fa: Optional[dict], fb: Optional[dict],
                    noise: float) -> Optional[float]:
    """Calibrated cost, in nats: 0 when the two arms' appearance is
    identical, growing as the fitted same-filament probability drops below
    its zero-difference baseline. `None` when either side has no feature --
    the channel simply disappears rather than forcing a value, and a pairing
    ambiguous only for OTHER reasons is left exactly as geometry scored it.
    """
    if fa is None or fb is None:
        return None
    scale = noise if (noise and np.isfinite(noise) and noise > 1e-6) else 0.02
    feat_i = np.tanh(abs(fa["intensity"] - fb["intensity"]) / scale)
    feat_s = np.tanh(abs(fa["sharpness"] - fb["sharpness"]) / scale)
    z = W_INTENSITY * feat_i + W_SHARPNESS * feat_s + B_INTERCEPT
    # softplus(-z) - softplus(-b): -log p_same(z), relative to -log p_same(0)
    softplus = lambda x: np.log1p(np.exp(-abs(x))) + max(x, 0.0)
    return float(softplus(-z) - softplus(-B_INTERCEPT))


def resolve_junctions_joint(graph: SkelGraph, frames: Dict[int, Frame],
                            sem, params: Params,
                            jparams: Optional[JointParams]) -> Tuple[Matching, dict]:
    """Geometry-only matching, with appearance blended in only where the
    geometry-only decision was ambiguous. Disabled (`jparams` absent,
    `lambda_appearance <= 0`, or no image) takes a fast path that calls
    `_max_weight_pairs` on the untouched geometry costs -- identical to
    `linking.resolve_junctions`, not merely intended to match it.
    """
    relief = (best_gap_option(graph, frames, params)
              if params.j_gap_relief > 0 else {})
    matching: Matching = {}
    report = dict(n_nodes=0, n_ambiguous_pairs=0, n_edges_blended=0,
                  n_flipped=0)
    if jparams is not None and sem is not None and jparams.lambda_appearance > 0:
        active, grad = True, appearance_features(sem.image)
    else:
        active, grad = False, None
    feats: Dict[int, Optional[dict]] = {}
    for node in graph.nodes:
        stubs = sorted(node.stubs)
        if len(stubs) < 2:
            continue
        report["n_nodes"] += 1
        costs = junction_costs(graph, frames, stubs, params)
        prices = _node_prices(stubs, relief, params)
        chosen = _max_weight_pairs(stubs, costs, prices)
        if not active or jparams is None or sem is None:
            for a, b in chosen:
                matching[a] = b
                matching[b] = a
            continue

        margins = _pair_margins(stubs, chosen, costs, prices)
        ambiguous_pairs = [p for p, m in margins.items()
                           if m < jparams.margin_threshold]
        if not ambiguous_pairs:
            for a, b in chosen:
                matching[a] = b
                matching[b] = a
            continue
        report["n_ambiguous_pairs"] += len(ambiguous_pairs)
        ambiguous_stubs = {sid for pair in ambiguous_pairs for sid in pair}

        for sid in stubs:
            if sid not in feats:
                feats[sid] = stub_appearance(sem, grad, frames[sid], jparams)

        blended = dict(costs)
        for (u, v) in costs:
            if u not in ambiguous_stubs and v not in ambiguous_stubs:
                continue
            ac = appearance_cost(feats.get(u), feats.get(v), sem.noise)
            if ac is None:
                continue
            report["n_edges_blended"] += 1
            blended[(u, v)] = costs[(u, v)] + jparams.lambda_appearance * ac

        final = _max_weight_pairs(stubs, blended, prices)
        if {frozenset(p) for p in final} != {frozenset(p) for p in chosen}:
            report["n_flipped"] += 1
        for a, b in final:
            matching[a] = b
            matching[b] = a
    return matching, report


# ── top-level round loop (mirrors linking.solve) ───────────────────────────


def solve_joint(graph: SkelGraph, params: Params, sem,
                jparams: Optional[JointParams]
                ) -> Tuple[Matching, List[List[int]], dict]:
    """Same round structure as `linking.solve`: frames are re-estimated as
    chains are built, and both crossings and gaps are re-solved each round.
    Only the junction step differs (`resolve_junctions_joint` in place of
    `linking.resolve_junctions`); gap bridging is geometry-only in both
    modes, since the design scope is the continuation decision at a
    junction, not gap matching.
    """
    rounds = max(1, params.n_rounds)
    matching: Matching = {}
    best: Tuple[float, Matching] = (math.inf, {})
    best_report: dict = {}
    for round_index in range(rounds):
        window = params.window_local if round_index == 0 else params.window_chain
        frac = round_index / max(1, rounds - 1)
        strict = params.anneal_start + (1.0 - params.anneal_start) * frac
        stage = (params if strict >= 1.0 else replace(
            params,
            j_unmatched=params.j_unmatched * strict,
            g_unmatched=params.g_unmatched * strict,
            gap_max_phi=params.gap_max_phi * strict,
            gap_max_theta=params.gap_max_theta * strict,
        ))
        frames = build_frames(graph, matching, window, params)
        junctions, rep = resolve_junctions_joint(graph, frames, sem, stage,
                                                 jparams)
        junctions = break_cycles(graph, junctions)
        full = break_cycles(graph, bridge_gaps(graph, frames, junctions, stage))
        if strict >= 1.0:
            value = matching_objective(graph, full, frames, params)
            if value < best[0]:
                best = (value, full)
                best_report = rep
        if full == matching and strict >= 1.0:
            break
        matching = full
    chosen = best[1] if best[0] < math.inf else matching
    return chosen, chains_from_matching(graph, chosen), best_report


@overload
def predict_joint(mask: np.ndarray, sem=..., params: Optional[Params] = ...,
                  jparams: Optional[JointParams] = ...,
                  include_nodes: bool = ..., min_isolated_px: int = ...,
                  return_internals: Literal[False] = ...
                  ) -> Dict[int, np.ndarray]: ...


@overload
def predict_joint(mask: np.ndarray, sem=..., params: Optional[Params] = ...,
                  jparams: Optional[JointParams] = ...,
                  include_nodes: bool = ..., min_isolated_px: int = ...,
                  *, return_internals: Literal[True]
                  ) -> Tuple[Dict[int, np.ndarray], SkelGraph, Matching,
                             List[List[int]], dict]: ...


def predict_joint(mask: np.ndarray, sem=None, params: Optional[Params] = None,
                  jparams: Optional[JointParams] = None,
                  include_nodes: bool = True, min_isolated_px: int = 6,
                  return_internals: bool = False):
    """Same contract as `plecta.predict.predict`, with an optional image.

    `sem=None` or `jparams=None` (or `jparams.lambda_appearance <= 0`)
    reproduces `predict.predict`'s output exactly -- see
    `tests/test_joint.py::test_disabled_matches_core_exactly`.
    """
    from .graph import build_graph
    from .predict import _emitted_chains, instances_from_chains, load_params

    params = params or load_params()
    graph = build_graph(mask, spur_px=params.spur_px,
                        bridge_px=params.bridge_px,
                        absorb_free_px=params.absorb_free_px,
                        join_px=params.join_px)
    matching, chains, report = solve_joint(graph, params, sem, jparams)
    chains = _emitted_chains(graph, chains, min_isolated_px)
    masks = instances_from_chains(graph, chains, include_nodes=include_nodes)
    if return_internals:
        return masks, graph, matching, chains, report
    return masks


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Joint mode: image evidence at ambiguous 2-D junctions "
                    "only. --lambda-appearance 0 reproduces the geometry-"
                    "only core exactly.")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--sem", help="registered grayscale image; omit for "
                                 "geometry-only (identical to the core)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-appearance", type=float, default=None,
                    help="override parameters.yaml for this run")
    ap.add_argument("--margin-threshold", type=float, default=None,
                    help="override parameters.yaml for this run")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    metavar="KEY=VALUE",
                    help="override any joint_greyscale parameter for this run")
    args = ap.parse_args(argv)

    from pathlib import Path

    from .graph import read_mask
    from .parameters import build
    from .predict import load_params, save_multilabel_npz

    mask = read_mask(args.mask)
    sem = None
    if args.sem:
        from .image.measurement import load_sem
        sem = load_sem(args.sem)
    # Every tuned value comes from parameters.yaml, like every other stage;
    # the flags above are overrides on top of it, not a second source of truth.
    overrides = list(args.overrides)
    if args.lambda_appearance is not None:
        overrides.append(f"lambda_appearance={args.lambda_appearance}")
    if args.margin_threshold is not None:
        overrides.append(f"margin_threshold={args.margin_threshold}")
    jparams = build(JointParams, overrides)
    masks, graph, matching, chains, report = predict_joint(
        mask, sem, load_params(), jparams, return_internals=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_multilabel_npz(out, masks, mask.shape)
    print(f"{len(masks)} instances -> {out}")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
