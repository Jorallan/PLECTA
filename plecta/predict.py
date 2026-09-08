"""PLECTA filament instance reconstruction: binary axis mask in, instances out.

    # a directory of scene folders, each containing mask_w1.png
    plecta --scenes <root> --out <outdir>

    # a single mask
    plecta --mask <scene>/mask_w1.png --out pred.npz

The only input ever read is the binary mask.  Ground truth, `mask_clean.png` and
`sem.png` are never opened -- there is no code path here that could.

Output per scene is ``pred_multilabel.npz`` with ``shape``, ``ids``, ``indptr``,
and flattened ``indices`` arrays. Instances may overlap: a crossing pixel is
written into every filament that runs through it.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, overload

import numpy as np

from .graph import SkelGraph, build_graph, read_mask
from .linking import Params, solve

PARAMS_PATH = Path(__file__).resolve().parent / "params.json"


def load_params(overrides: List[str] | None = None) -> Params:
    """The tuned 2-D grouping configuration, from ``parameters.yaml``.

    Values live in ``plecta/parameters.yaml`` under ``grouping_2d``; the
    frozen ``params.json`` beside it is retained as the immutable record of
    the published configuration and is asserted against the YAML by
    ``plecta.parameters.verify_against_frozen``.
    """
    from .parameters import build

    return build(Params, overrides)


def _emitted_chains(graph: SkelGraph, chains: List[List[int]],
                    min_isolated_px: int) -> List[List[int]]:
    """Drop isolated debris once and preserve the emitted chain ordering."""
    if not min_isolated_px:
        return list(chains)
    kept = []
    for chain in chains:
        if len(chain) == 1:
            arm = graph.arms[chain[0]]
            isolated = all(graph.stubs[s].node is None for s in arm.stubs)
            if len(arm.pixels) < min_isolated_px and isolated:
                continue
        kept.append(chain)
    return kept


def instances_from_chains(graph: SkelGraph, chains: List[List[int]],
                          include_nodes: bool = True) -> Dict[int, np.ndarray]:
    """Paint chains as layers, sharing crossing pixels between instances."""
    masks: Dict[int, np.ndarray] = {}
    for key, chain in enumerate(chains, start=1):
        canvas = np.zeros(graph.shape, dtype=bool)
        nodes_touched = set()
        for arm_id in chain:
            arm = graph.arms[arm_id]
            rows, cols = zip(*arm.pixels)
            canvas[list(rows), list(cols)] = True
            for stub_id in arm.stubs:
                node = graph.stubs[stub_id].node
                if node is not None:
                    nodes_touched.add(node)
        if include_nodes:
            for node_id in nodes_touched:
                pixels = graph.nodes[node_id].pixels
                if pixels:
                    rows, cols = zip(*pixels)
                    canvas[list(rows), list(cols)] = True
        if canvas.any():
            masks[key] = canvas
    return masks


@overload
def predict(mask: np.ndarray, params: Optional[Params] = ...,
            include_nodes: bool = ..., min_isolated_px: int = ...,
            return_internals: Literal[False] = ...
            ) -> Dict[int, np.ndarray]: ...


@overload
def predict(mask: np.ndarray, params: Optional[Params] = ...,
            include_nodes: bool = ..., min_isolated_px: int = ...,
            *, return_internals: Literal[True]
            ) -> Tuple[Dict[int, np.ndarray], SkelGraph, Dict[int, int],
                       List[List[int]]]: ...


def predict(mask: np.ndarray, params: Optional[Params] = None,
            include_nodes: bool = True, min_isolated_px: int = 6,
            return_internals: bool = False):
    """Reconstruct overlapping filament instances from one binary mask."""
    params = params or load_params()
    graph = build_graph(mask, spur_px=params.spur_px,
                        bridge_px=params.bridge_px,
                        absorb_free_px=params.absorb_free_px,
                        join_px=params.join_px)
    matching, chains = solve(graph, params)
    chains = _emitted_chains(graph, chains, min_isolated_px)
    masks = instances_from_chains(graph, chains,
                                  include_nodes=include_nodes)
    if return_internals:
        return masks, graph, matching, chains
    return masks


def save_multilabel_npz(path, masks: Dict[int, np.ndarray],
                        shape: Tuple[int, int]) -> None:
    """Write sparse overlapping layers for interoperable, compact storage."""
    ids = sorted(masks)
    indices = [np.flatnonzero(np.asarray(masks[key], dtype=bool).ravel())
               .astype(np.int64) for key in ids]
    indptr = np.cumsum([0] + [item.size for item in indices], dtype=np.int64)
    stacked = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
    np.savez_compressed(
        str(path),
        shape=np.asarray(shape, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        indptr=indptr,
        indices=stacked,
    )


def find_scenes(root: Path, mask_name: str) -> List[Path]:
    root = Path(root)
    if (root / mask_name).is_file():
        return [root]
    return sorted(root.rglob(mask_name))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes", help="directory of scene folders")
    src.add_argument("--mask", help="a single binary mask image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-name", default="mask_w1.png")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--params", default=None, metavar="TUNED.yaml",
                    help="a tuned configuration from `plecta-tune`, applied on "
                         "top of the frozen parameters. The frozen values and "
                         "params.json are not modified.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    params = load_params(args.overrides)
    if args.params:
        # A tuned configuration is applied on top of the frozen values, and
        # after --set so an explicit override still wins.
        from .tune import apply_config

        params = apply_config(params, args.params)
    out_root = Path(args.out)

    if args.mask:
        mask = read_mask(args.mask)
        masks = predict(mask, params)
        out_root.parent.mkdir(parents=True, exist_ok=True)
        save_multilabel_npz(out_root, masks, mask.shape)
        print(f"{len(masks)} instances -> {out_root}")
        return 0

    hits = find_scenes(Path(args.scenes), args.mask_name)
    if not hits:
        raise SystemExit(f"no {args.mask_name} found under {args.scenes}")
    out_root.mkdir(parents=True, exist_ok=True)
    for hit in hits:
        scene = hit if hit.is_dir() else hit.parent
        mask_path = scene / args.mask_name
        t0 = time.time()
        mask = read_mask(mask_path)
        masks = predict(mask, params)
        dest = out_root / f"{scene.parent.name}__{scene.name}"
        dest.mkdir(parents=True, exist_ok=True)
        save_multilabel_npz(dest / "pred_multilabel.npz", masks, mask.shape)
        if not args.quiet:
            print(f"{scene.parent.name}/{scene.name}: {len(masks)} instances  "
                  f"{time.time() - t0:.2f}s", flush=True)
    (out_root / "params_used.json").write_text(json.dumps(asdict(params), indent=1))
    print(f"{len(hits)} scenes -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
