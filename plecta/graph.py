"""Turn a thin binary filament mask into a graph of arms and crossings.

The mask is a 1-px axis mask, so its skeleton is essentially the mask itself.
Removing the pixels where three or more skeleton pixels meet cuts the mask into
*arms*: maximal pieces that contain no ambiguity at all.  Everything the method
has to decide is then expressed on this graph:

* at a **node** (a crossing) -- which arm continues into which,
* between two **free stub ends** -- whether a mask gap should be bridged.

Nothing here looks at ground truth.

Two deliberate choices that make the output line up with the project scorer's
atomic units (``eval/core/common_metric.common_fragments``):

* the same 3-px dead-end spur rule is applied before splitting, so the arms and
  the scorer's fragments are the same pieces of skeleton;
* junction *clusters* are merged when they are joined by a stub shorter than
  ``bridge_px`` (<= 5).  The scorer discards fragments below 6 px, so no scored
  fragment is ever swallowed into a node by this merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Pixel = Tuple[int, int]

NEI8: Tuple[Pixel, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


# ── input ──────────────────────────────────────────────────────────────────


def read_mask(path) -> np.ndarray:
    """Binarise a mask image using a strictly positive intensity threshold."""
    from skimage import io as skio

    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return np.asarray(img) > 0


# ── skeleton helpers ───────────────────────────────────────────────────────


def degree_map(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours of every skeleton pixel (0 off the skeleton)."""
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0) * skel


def prune_spurs(skel: np.ndarray, max_len: int = 3, max_iters: int = 20) -> np.ndarray:
    """Drop dead-end skeleton hairs of at most ``max_len`` px.

    A hair is a non-junction piece with a free end that touches a junction.
    Removing one can demote its junction to an ordinary path pixel and expose
    another hair, hence the iteration.  Same rule (and same 3-px default) as the
    scorer's fragment construction, on purpose -- see the module docstring.
    """
    from scipy.ndimage import binary_dilation
    from skimage.measure import label as cc_label

    skel = np.asarray(skel, dtype=bool).copy()
    if max_len <= 0:
        return skel
    box = np.ones((3, 3), dtype=bool)
    for _ in range(max_iters):
        deg = degree_map(skel)
        junctions = skel & (deg >= 3)
        if not junctions.any():
            break
        free_ends = skel & (deg == 1)
        pieces = np.asarray(cc_label(skel & ~junctions, connectivity=2))
        n_pieces = int(pieces.max())
        if n_pieces == 0:
            break
        near_junction = binary_dilation(junctions, structure=box)
        # Per-piece tests by scattering into per-label arrays.  The obvious
        # `pieces == k` loop builds a full H x W boolean per piece and costs 97%
        # of graph construction on a dense mask (2.4 s for 6300 pieces).
        flat = pieces.ravel()
        nz = flat > 0
        labels = flat[nz]
        sizes = np.bincount(labels, minlength=n_pieces + 1)
        touches = np.zeros(n_pieces + 1, dtype=bool)
        np.logical_or.at(touches, labels, near_junction.ravel()[nz])
        has_end = np.zeros(n_pieces + 1, dtype=bool)
        np.logical_or.at(has_end, labels, free_ends.ravel()[nz])
        doomed = (sizes <= max_len) & touches & has_end
        doomed[0] = False
        if not doomed.any():
            break
        skel[doomed[pieces]] = False
    return skel


def _order_pixels(pixels: Sequence[Pixel]) -> List[Pixel]:
    """Order one connected piece of skeleton along itself, tip to tip.

    Junction pixels are removed before the pieces are labelled, so every pixel
    here has at most two neighbours inside its own piece: a piece is either a
    simple path or a closed ring.  A path is returned whole.  A ring has no
    ends, and a geodesic between two of its pixels would cover only half of it,
    so the ring is cut at one pixel first and the resulting path is returned --
    the cut pixel stays in the piece's pixel set, it just carries no arclength.
    """
    pixels = sorted(pixels)
    if len(pixels) <= 2:
        return list(pixels)
    members: Set[Pixel] = set(pixels)

    def degree_in(p: Pixel) -> int:
        r, c = p
        return sum(1 for dr, dc in NEI8 if (r + dr, c + dc) in members)

    if all(degree_in(p) >= 2 for p in pixels):
        members = members - {pixels[0]}   # open the ring
        pixels = sorted(members)
        if len(pixels) <= 2:
            return list(pixels)

    def neighbours(p: Pixel) -> List[Pixel]:
        r, c = p
        return [(r + dr, c + dc) for dr, dc in NEI8 if (r + dr, c + dc) in members]

    def farthest(source: Pixel) -> Tuple[Pixel, Dict[Pixel, Optional[Pixel]]]:
        parent: Dict[Pixel, Optional[Pixel]] = {source: None}
        frontier = [source]
        last = source
        while frontier:
            nxt: List[Pixel] = []
            for p in frontier:
                for q in neighbours(p):
                    if q not in parent:
                        parent[q] = p
                        nxt.append(q)
            if nxt:
                last = min(nxt)  # deterministic pick among the equidistant
            frontier = nxt
        return last, parent

    ends = [p for p in pixels if len(neighbours(p)) <= 1]
    start = ends[0] if ends else pixels[0]
    a, _ = farthest(start)
    b, parent = farthest(a)
    path: List[Pixel] = []
    cur: Optional[Pixel] = b
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ── graph objects ──────────────────────────────────────────────────────────


@dataclass
class Arm:
    """A maximal ambiguity-free piece of skeleton."""

    aid: int
    path: List[Pixel]            # ordered, path[0] is stub end 0
    pixels: List[Pixel]          # every pixel of the piece
    stubs: Tuple[int, int] = (-1, -1)


@dataclass
class Stub:
    """One end of an arm: the unit the method has to pair up."""

    sid: int
    aid: int
    end: int                      # 0 or 1
    tip: Pixel
    node: Optional[int] = None    # crossing it sits on, None = free end


@dataclass
class Node:
    """A crossing: junction pixels plus any absorbed sub-bridge debris."""

    pixels: List[Pixel]
    stubs: List[int] = field(default_factory=list)


@dataclass
class SkelGraph:
    shape: Tuple[int, int]
    arms: List[Arm]
    stubs: List[Stub]
    nodes: List[Node]

    # convenience -------------------------------------------------------
    def other_stub(self, sid: int) -> int:
        st = self.stubs[sid]
        a, b = self.arms[st.aid].stubs
        return b if sid == a else a


# ── construction ───────────────────────────────────────────────────────────


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_graph(
    mask: np.ndarray,
    spur_px: int = 3,
    bridge_px: int = 5,
    absorb_free_px: int = 0,
    join_px: int = 0,
) -> SkelGraph:
    """Decompose ``mask`` into arms and crossings.

    ``bridge_px``  arms this short that join two crossings are treated as part
                   of one bigger crossing rather than as a real arm.  Must stay
                   <= 5 so the scorer's 6-px fragment floor is never crossed.
    ``absorb_free_px``  arms this short with a free end are absorbed as crossing
                   debris (0 disables).
    ``join_px``    arms up to this long joining two crossings put both crossings
                   into ONE node for matching purposes, but stay real arms with
                   their own two stubs.  This is the same idea as ``bridge_px``
                   without the erasure, so it is not limited by the scorer's
                   6-px floor: the arm keeps its own identity and one owner.
                   0 disables.
    """
    from skimage.measure import label as cc_label
    from skimage.morphology import skeletonize
    from scipy.ndimage import binary_dilation

    if spur_px > 3:
        raise ValueError(
            "spur_px above 3 prunes skeleton the scorer's own fragment "
            "construction keeps (it prunes at 3), so arms would stop tiling the "
            "scored fragments and some fragments would have no owner at all"
        )
    if bridge_px > 5 or absorb_free_px > 5:
        raise ValueError(
            "bridge_px / absorb_free_px above 5 would swallow pieces the scorer "
            "still scores as fragments (its floor is 6 px), which would make one "
            "piece of skeleton the shared property of every chain through a node"
        )

    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    skel = prune_spurs(skeletonize(mask), spur_px)

    deg = degree_map(skel)
    junction = skel & (deg >= 3)

    # Cluster junction pixels.  Dilating first also merges junction pixels that
    # are not touching -- anything within Chebyshev distance 3, which is what a
    # shallow crossing produces when its overlap region breaks into two blobs.
    if junction.any():
        grown = np.asarray(cc_label(
            binary_dilation(junction, structure=np.ones((3, 3), bool)),
            connectivity=2))
        cluster_of = np.where(junction, grown, 0)
    else:
        cluster_of = np.zeros(shape, dtype=np.int32)
    n_clusters = int(cluster_of.max())

    piece_labels = np.asarray(cc_label(skel & ~junction, connectivity=2))
    n_pieces = int(piece_labels.max())

    piece_pixels: List[List[Pixel]] = [[] for _ in range(n_pieces + 1)]
    rr, cc = np.nonzero(piece_labels)
    for r, c in zip(rr.tolist(), cc.tolist()):
        piece_pixels[piece_labels[r, c]].append((r, c))

    def touching_clusters(pixels: Sequence[Pixel]) -> Set[int]:
        found: Set[int] = set()
        for r, c in pixels:
            for dr, dc in NEI8:
                rr2, cc2 = r + dr, c + dc
                if 0 <= rr2 < shape[0] and 0 <= cc2 < shape[1]:
                    k = int(cluster_of[rr2, cc2])
                    if k:
                        found.add(k)
        return found

    # ── merge crossings joined by a sub-bridge, absorb crossing debris ──
    uf = _UnionFind(n_clusters + 1)
    absorbed_pieces: List[int] = []
    piece_paths: List[List[Pixel]] = [[] for _ in range(n_pieces + 1)]
    for k in range(1, n_pieces + 1):
        pix = piece_pixels[k]
        piece_paths[k] = _order_pixels(pix)
        near = touching_clusters(pix)
        if len(pix) <= bridge_px and len(near) >= 2:
            near_list = sorted(near)
            for other in near_list[1:]:
                uf.union(near_list[0], other)
            absorbed_pieces.append(k)
        elif join_px and len(pix) <= join_px and len(near) >= 2:
            # merge the crossings so their arms are paired jointly, but leave
            # this arm in place: it keeps its pixels and gets a single owner.
            near_list = sorted(near)
            for other in near_list[1:]:
                uf.union(near_list[0], other)
        elif absorb_free_px and len(pix) <= absorb_free_px and len(near) == 1:
            absorbed_pieces.append(k)

    absorbed_set = set(absorbed_pieces)

    # ── build the merged nodes ─────────────────────────────────────────
    node_id_of_cluster: Dict[int, int] = {}
    node_pixel_lists: List[List[Pixel]] = []
    for k in range(1, n_clusters + 1):
        root = uf.find(k)
        if root not in node_id_of_cluster:
            node_id_of_cluster[root] = len(node_pixel_lists)
            node_pixel_lists.append([])
    jr, jc = np.nonzero(junction)
    for r, c in zip(jr.tolist(), jc.tolist()):
        nid = node_id_of_cluster[uf.find(int(cluster_of[r, c]))]
        node_pixel_lists[nid].append((r, c))

    for k in absorbed_pieces:
        near = sorted(touching_clusters(piece_pixels[k]))
        nid = node_id_of_cluster[uf.find(near[0])]
        node_pixel_lists[nid].extend(piece_pixels[k])

    node_pixels = np.full(shape, -1, dtype=np.int32)
    for nid, pix in enumerate(node_pixel_lists):
        for r, c in pix:
            node_pixels[r, c] = nid

    # ── arms and their stubs ───────────────────────────────────────────
    arms: List[Arm] = []
    stubs: List[Stub] = []
    nodes = [Node(pix) for pix in node_pixel_lists]

    def node_at(tip: Pixel) -> Optional[int]:
        r, c = tip
        votes: Dict[int, int] = {}
        for dr, dc in NEI8:
            rr2, cc2 = r + dr, c + dc
            if 0 <= rr2 < shape[0] and 0 <= cc2 < shape[1]:
                nid = int(node_pixels[rr2, cc2])
                if nid >= 0:
                    votes[nid] = votes.get(nid, 0) + 1
        if not votes:
            return None
        return max(sorted(votes), key=lambda n: votes[n])

    for k in range(1, n_pieces + 1):
        if k in absorbed_set:
            continue
        path = piece_paths[k]
        aid = len(arms)
        arm = Arm(aid=aid, path=path, pixels=piece_pixels[k])
        s0 = Stub(sid=len(stubs), aid=aid, end=0, tip=path[0])
        stubs.append(s0)
        s1 = Stub(sid=len(stubs), aid=aid, end=1, tip=path[-1])
        stubs.append(s1)
        arm.stubs = (s0.sid, s1.sid)
        arms.append(arm)

    for st in stubs:
        st.node = node_at(st.tip)
        if st.node is not None:
            nodes[st.node].stubs.append(st.sid)

    return SkelGraph(shape=shape, arms=arms, stubs=stubs, nodes=nodes)
