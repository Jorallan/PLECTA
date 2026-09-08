"""Decide which arm continues into which, and stitch the results into chains.

Two decisions, both expressed as *matchings on stubs*:

* **at a crossing** -- the arms meeting at one node have to be paired up.  This
  is solved exactly and jointly per node (maximum-weight matching), not greedily
  arm-by-arm, because the arms constrain each other: pairing a with b is only
  attractive if what is left over for c and d is also cheap.
* **across a gap** -- ends left free after the crossings are resolved may be two
  halves of one filament that the mask broke.  Same machinery, one global
  matching over all free ends.

Leaving a stub unmatched is always an option and costs a fixed
``unmatched_penalty``; that is what lets a filament genuinely end, and what
stops a T-junction from inventing a continuation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from .geometry import Frame, LinkCost, fit_frame, link_cost
from .graph import SkelGraph

Matching = Dict[int, int]


@dataclass
class Params:
    """Tunables for graph construction, stub frames and both matchings.

    The defaults below exist only to keep the class constructible on its own.
    They are **not** the shipped configuration and several differ from it
    sharply -- ``j_w_kappa`` is 0 here and 1.0 as shipped, ``n_rounds`` 3
    against 8, ``join_px`` 0 against 14. Every real run builds this through
    ``plecta.parameters.build``, which fills all of it from ``parameters.yaml``
    and refuses to leave a field unset. Read that file, not these lines, to know
    what a run actually did.
    """

    # ── graph construction ────────────────────────────────────────────
    spur_px: int = 3                # dead-end hairs this short are skeleton noise
    bridge_px: int = 5              # arms this short between two crossings are
                                    # part of one bigger crossing (must be <= 5)
    absorb_free_px: int = 5         # free-ended arms this short are crossing debris
    join_px: int = 0                # arms this short merge their two crossings for
                                    # matching but keep their own identity

    # ── frames ────────────────────────────────────────────────────────
    window_local: float = 14.0      # arclength used for a first-pass tangent
    window_chain: float = 42.0      # arclength once arms have been chained
    min_quadratic: int = 9          # pixels needed before a curved fit is used
    n_rounds: int = 3               # frame/matching refinement rounds
    anneal_start: float = 1.0       # first round's acceptance limits, as a
                                    # fraction of the final ones (1.0 = off)
    break_rings: bool = True        # enforce "an instance is an open curve" by
                                    # opening any matched ring. Off admits
                                    # closed and looping strands as instances.

    # ── crossings ─────────────────────────────────────────────────────
    j_w_direct: float = 1.0         # weight on tangent-vs-tangent turn
    j_w_turn: float = 0.5           # weight on turn-onto-chord
    j_chord_floor: float = 4.0
    j_w_kappa: float = 0.0          # weight on curvature continuity
    j_unmatched: float = 0.62       # rad; pairing costs above 2x this are refused
    j_gap_relief: float = 0.0       # >0: a stub with a good gap partner waiting
                                    # pays less to decline pairing at a crossing

    # ── shared by both cost functions ─────────────────────────────────
    # Multiplies the curvature term in `geometry.link_cost` for crossings AND
    # for gaps.  It is redundant with ``j_w_kappa``/``g_w_kappa`` -- only the
    # product reaches the cost -- but it was previously a bare default inside
    # `geometry.link_cost`, i.e. an effective tuned value that no parameter file
    # recorded.  It is carried here so that reading the configuration tells you
    # the whole configuration.  Changing it is equivalent to rescaling both
    # kappa weights.
    kappa_scale: float = 30.0

    # ── gaps ──────────────────────────────────────────────────────────
    # Measured on the development set: true gap links have median length 25 px
    # (p90 = 46), and 95% of *wrong* candidates turn more than 38 deg onto the
    # chord while 75% of right ones turn less than 20.  So the cut-off that
    # does the work is the turn, not the length -- the length only breaks ties.
    gap_max_len: float = 58.0       # px, longest bridge considered
    g_w_direct: float = 0.35        # weight on tangent-vs-tangent turn
    g_w_turn: float = 1.0           # weight on turn-onto-chord
    g_w_len: float = 0.30
    g_len_scale: float = 60.0
    g_chord_floor: float = 3.0
    g_w_kappa: float = 0.0
    g_unmatched: float = 0.55
    gap_max_theta: float = 0.70     # rad, hard reject on tangent disagreement
    gap_max_phi: float = 0.50       # rad, hard reject on turn-onto-chord


# ── stub frames ────────────────────────────────────────────────────────────


def walk_from_stub(graph: SkelGraph, sid: int, matching: Matching,
                   max_arclen: float) -> List[Tuple[int, int]]:
    """Pixels from a stub tip inward, following the current matching.

    Once arms are chained, the trajectory available for estimating "where is
    this filament heading" is the whole chain, not the one arm the stub sits
    on.  That matters because a third of the arms in a dense scene are under
    10 px long, which is far too short to measure a direction on.
    """
    points: List[Tuple[int, int]] = []
    total = 0.0
    cur = sid
    seen: Set[int] = set()
    while True:
        stub = graph.stubs[cur]
        if stub.aid in seen:
            break
        seen.add(stub.aid)
        arm = graph.arms[stub.aid]
        segment = arm.path if stub.end == 0 else arm.path[::-1]
        for p in segment:
            if points:
                total += math.hypot(p[0] - points[-1][0], p[1] - points[-1][1])
            points.append(p)
            if total >= max_arclen:
                return points
        far = graph.other_stub(cur)
        nxt = matching.get(far)
        if nxt is None:
            break
        cur = nxt
    return points


def build_frames(graph: SkelGraph, matching: Matching, window: float,
                 params: Params) -> Dict[int, Frame]:
    frames: Dict[int, Frame] = {}
    for stub in graph.stubs:
        pts = walk_from_stub(graph, stub.sid, matching, window)
        frames[stub.sid] = fit_frame(pts, window=window,
                                     min_quadratic=params.min_quadratic)
    return frames


# ── exact matching on a small stub set ─────────────────────────────────────


def _max_weight_pairs(stubs: Sequence[int],
                      costs: Dict[Tuple[int, int], float],
                      unmatched_penalty,
                      ) -> List[Tuple[int, int]]:
    """Minimum-total-cost partial matching, with a price for opting out.

    Maximising ``sum(P_i + P_j - c_ij)`` over the chosen pairs is identical to
    minimising ``sum(c_ij) + sum of P over the stubs left unmatched``: every pair
    chosen removes two opt-out payments and adds its own cost, and the total of
    all P is a constant.  Blossom (networkx) solves that exactly for any node
    degree, so nothing has to be greedy and nothing has to be capped.

    ``unmatched_penalty`` is either one number for every stub, or a mapping from
    stub id to its own price.  Per-stub prices are what let a stub that has a
    good continuation waiting on the far side of a mask gap decline to pair at
    the crossing.
    """
    import networkx as nx

    def price(sid: int) -> float:
        if isinstance(unmatched_penalty, dict):
            return float(unmatched_penalty[sid])
        return float(unmatched_penalty)

    graph = nx.Graph()
    graph.add_nodes_from(stubs)
    any_edge = False
    for (i, j), c in costs.items():
        limit = price(i) + price(j)
        if c < limit:
            graph.add_edge(i, j, weight=limit - c)
            any_edge = True
    if not any_edge:
        return []
    matched = nx.max_weight_matching(graph, maxcardinality=False)
    return [(int(min(a, b)), int(max(a, b))) for a, b in matched]


# ── crossings ──────────────────────────────────────────────────────────────


def _junction_link(fa: Frame, fb: Frame, params: Params) -> LinkCost:
    return link_cost(fa, fb, w_turn=params.j_w_turn,
                     w_direct=params.j_w_direct, w_len=0.0,
                     chord_floor=params.j_chord_floor,
                     w_kappa=params.j_w_kappa,
                     kappa_scale=params.kappa_scale)


def _gap_link(fa: Frame, fb: Frame, params: Params) -> LinkCost:
    return link_cost(fa, fb, w_turn=params.g_w_turn,
                     w_direct=params.g_w_direct, w_len=params.g_w_len,
                     len_scale=params.g_len_scale,
                     chord_floor=params.g_chord_floor,
                     w_kappa=params.g_w_kappa,
                     kappa_scale=params.kappa_scale)


def _gap_candidates(graph: SkelGraph, frames: Dict[int, Frame],
                    ids: Sequence[int], params: Params):
    """Every eligible gap link among ``ids``, in kd-tree pair order.

    Two ends of one arm, or two ends sitting on the *same* crossing, are never
    bridged -- that decision was already made at the crossing, and re-making it
    here would just undo it.
    """
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]

    tips = np.array([frames[sid].tip for sid in ids])
    tree = cKDTree(tips)
    for ii, jj in tree.query_pairs(params.gap_max_len, output_type="ndarray"):
        a, b = ids[int(ii)], ids[int(jj)]
        sa, sb = graph.stubs[a], graph.stubs[b]
        if sa.aid == sb.aid:
            continue
        if sa.node is not None and sa.node == sb.node:
            continue
        lc = _gap_link(frames[a], frames[b], params)
        if not math.isfinite(lc.cost):
            continue
        if lc.theta > params.gap_max_theta:
            continue
        if max(lc.phi_a, lc.phi_b) > params.gap_max_phi:
            continue
        yield a, b, lc


def junction_costs(graph: SkelGraph, frames: Dict[int, Frame],
                   stubs: Sequence[int], params: Params
                   ) -> Dict[Tuple[int, int], float]:
    costs: Dict[Tuple[int, int], float] = {}
    for ii in range(len(stubs)):
        for jj in range(ii + 1, len(stubs)):
            a, b = stubs[ii], stubs[jj]
            if graph.stubs[a].aid == graph.stubs[b].aid:
                continue  # an arm may not close on itself at one crossing
            lc = _junction_link(frames[a], frames[b], params)
            if math.isfinite(lc.cost):
                costs[(a, b)] = lc.cost
    return costs


def best_gap_option(graph: SkelGraph, frames: Dict[int, Frame],
                    params: Params) -> Dict[int, float]:
    """For every stub, the cost of the best gap link it could take instead.

    Crossings are resolved before gaps, so a stub the crossing stage pairs up is
    gone before gap bridging ever sees it -- a filament whose continuation lies
    across a mask gap can be captured by the wrong arm at the crossing and never
    released.  Knowing what each stub is giving up lets the crossing stage price
    its refusal properly instead of charging everyone the same.
    """
    ids = [s.sid for s in graph.stubs if frames[s.sid].reliable]
    if len(ids) < 2:
        return {}
    best: Dict[int, float] = {}
    for a, b, lc in _gap_candidates(graph, frames, ids, params):
        for sid in (a, b):
            if lc.cost < best.get(sid, math.inf):
                best[sid] = lc.cost
    return best


def resolve_junctions(graph: SkelGraph, frames: Dict[int, Frame],
                      params: Params) -> Matching:
    relief = (best_gap_option(graph, frames, params)
              if params.j_gap_relief > 0 else {})
    matching: Matching = {}
    for node in graph.nodes:
        stubs = sorted(node.stubs)
        if len(stubs) < 2:
            continue
        costs = junction_costs(graph, frames, stubs, params)
        if relief:
            prices = {sid: min(params.j_unmatched,
                               params.j_gap_relief * relief[sid])
                      if sid in relief else params.j_unmatched
                      for sid in stubs}
        else:
            prices = params.j_unmatched
        for a, b in _max_weight_pairs(stubs, costs, prices):
            matching[a] = b
            matching[b] = a
    return matching


# ── gaps ───────────────────────────────────────────────────────────────────


def bridge_gaps(graph: SkelGraph, frames: Dict[int, Frame],
                matching: Matching, params: Params) -> Matching:
    """Pair up ends the crossings left free, across empty mask.

    A free end is either a real skeleton endpoint or an arm the crossing logic
    refused to continue.  Both can be one side of a mask gap, so both take part.
    Two ends sitting on the *same* crossing are never bridged -- that decision
    was already made, and re-making it here would just undo it.
    """
    free = [s.sid for s in graph.stubs if s.sid not in matching]
    free = [sid for sid in free if frames[sid].reliable]
    if len(free) < 2:
        return dict(matching)

    costs: Dict[Tuple[int, int], float] = {}
    for a, b, lc in _gap_candidates(graph, frames, free, params):
        costs[(min(a, b), max(a, b))] = lc.cost

    out = dict(matching)
    for a, b in _max_weight_pairs(free, costs, params.g_unmatched):
        out[a] = b
        out[b] = a
    return out


# ── chains ─────────────────────────────────────────────────────────────────


def matching_objective(graph: SkelGraph, matching: Matching,
                       frames: Dict[int, Frame], params: Params) -> float:
    """Total cost of a whole matching: chosen links, plus a price per free stub.

    The same quantity each per-node matching minimises locally, summed. The
    refinement loop needs it because it is not a contraction: re-estimating the
    frames can lead the next round to a *worse* configuration, and without an
    objective the loop would return whichever round happened to be last. Only
    6 of 20 development scenes reach a fixed point within the round budget, so
    this is the common case, not a corner.
    """
    total = 0.0
    seen: Set[Tuple[int, int]] = set()
    for a, b in matching.items():
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        sa, sb = graph.stubs[a], graph.stubs[b]
        at_node = sa.node is not None and sa.node == sb.node
        if at_node:
            lc = _junction_link(frames[a], frames[b], params)
        else:
            lc = _gap_link(frames[a], frames[b], params)
        penalty = params.j_unmatched if at_node else params.g_unmatched
        total += (lc.cost if math.isfinite(lc.cost) else 4.0 * penalty) - 2.0 * penalty
    total += sum(params.j_unmatched if s.node is not None else params.g_unmatched
                 for s in graph.stubs)
    return total


def _arm_union(graph: SkelGraph, links) -> "Callable[[int], int]":
    """Union arms joined by ``links``, in the given order, and return ``find``.

    The order matters and belongs to the caller: with path compression but no
    union-by-rank the surviving root depends on it, and `chains_from_matching`
    keys its output on that root.
    """
    parent = list(range(len(graph.arms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(graph.stubs[a].aid), find(graph.stubs[b].aid)
        if ra != rb:
            parent[rb] = ra
    return find


def break_cycles(graph: SkelGraph, matching: Matching) -> Matching:
    """Filaments are open curves; a matching that closes a ring is wrong.

    A ring shows up as a connected group of arms in which *every* stub is
    matched. The longest jump in the ring is dropped to open it.

    On the development set this never fires -- 0 links dropped across all 84
    scenes -- so it is a guard against a configuration the data does not
    produce, not a working part of the method.
    """
    def jump(link: Tuple[int, int]) -> float:
        a, b = link
        pa = np.asarray(graph.stubs[a].tip, float)
        pb = np.asarray(graph.stubs[b].tip, float)
        return float(np.hypot(*(pa - pb)))

    links = set()
    for a, b in matching.items():
        if a < b:
            links.add((a, b))
    find = _arm_union(graph, sorted(links))

    groups: Dict[int, List[Tuple[int, int]]] = {}
    for a, b in links:
        groups.setdefault(find(graph.stubs[a].aid), []).append((a, b))
    members: Dict[int, Set[int]] = {}
    for aid in range(len(graph.arms)):
        members.setdefault(find(aid), set()).add(aid)

    out = dict(matching)
    for root, group_links in groups.items():
        n_arms = len(members[root])
        # a path over n arms has n-1 links; anything more closes a ring
        while len(group_links) >= n_arms:
            worst = max(group_links, key=jump)
            group_links.remove(worst)
            out.pop(worst[0], None)
            out.pop(worst[1], None)
    return out


def chains_from_matching(graph: SkelGraph, matching: Matching) -> List[List[int]]:
    """Group arm ids into chains induced by the stub matching."""
    find = _arm_union(graph, ((a, b) for a, b in matching.items() if a < b))

    groups: Dict[int, List[int]] = {}
    for aid in range(len(graph.arms)):
        groups.setdefault(find(aid), []).append(aid)
    return [sorted(v) for _, v in sorted(groups.items())]


# ── top level ──────────────────────────────────────────────────────────────


def solve(graph: SkelGraph, params: Params) -> Tuple[Matching, List[List[int]]]:
    """Resolve crossings and bridge gaps, refining the frames as chains grow.

    The two decisions are re-made together on every round rather than once each
    in sequence.  They feed each other: bridging a gap lengthens a chain, which
    sharpens the direction estimate at every crossing that chain still has to
    get through, and resolving a crossing does the same for the gaps.  Round 0
    can only see one arm at a time -- a third to a half of them under 10 px --
    so its estimates are the worst ones the method ever uses.
    """
    rounds = max(1, params.n_rounds)
    matching: Matching = {}
    best: Tuple[float, Matching] = (math.inf, {})
    for round_index in range(rounds):
        window = params.window_local if round_index == 0 else params.window_chain
        # Confident first: early rounds only accept joins that are clearly
        # right, so the chains they build are trustworthy enough to measure the
        # harder crossings against.  A tangle resolved from round-0 estimates on
        # 5-px arms is a coin flip; the same tangle resolved from a chain that
        # has already been assembled either side of it is not.
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
        #  Each matching is opened before the next stage sees it, so the two
        #  calls cannot be collapsed into one at the end.  With break_rings
        #  off, a matched ring stands and a closed or looping strand survives
        #  as one instance.
        junctions = resolve_junctions(graph, frames, stage)
        if params.break_rings:
            junctions = break_cycles(graph, junctions)
        full = bridge_gaps(graph, frames, junctions, stage)
        if params.break_rings:
            full = break_cycles(graph, full)
        # Score the candidate against the frames it was chosen under, and keep
        # the best round rather than the last one.  Only rounds at full
        # strictness compete: an annealed round is deliberately under-linked and
        # would win on an objective that charges for every link it declined.
        if strict >= 1.0:
            value = matching_objective(graph, full, frames, params)
            if value < best[0]:
                best = (value, full)
        if full == matching and strict >= 1.0:
            break
        matching = full
    chosen = best[1] if best[0] < math.inf else matching
    return chosen, chains_from_matching(graph, chosen)
