"""Run PLECTA grouping and downstream image characterization for one scene.

    binary mask  --plecta/predict.py, unmodified-->  overlapping instance masks
                 --sem.png-->                 the same masks + a width and a
                                              brightness, each with a spread

The grouping call uses ``plecta.predict.predict`` with
``plecta/params.json``. Image measurements cannot revise that grouping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from plecta.graph import read_mask
from plecta.linking import Params
from plecta.predict import load_params, predict

from .bundles import Bundle, SampleParams, aggregate, collect_cuts, node_distance
from .measurement import CutParams, SemImage, load_sem


@dataclass
class SceneResult:
    scene: Path
    mask_name: str
    shape: tuple
    n_instances: int
    bundles: List[Bundle]
    masks: Dict[int, np.ndarray]
    graph: object
    matching: Dict[int, int]
    chains: List[List[int]]
    sem: SemImage
    seconds_stage1: float
    seconds_stage2: float
    cut_width_cap: Optional[float] = None

    def rows(self) -> List[dict]:
        return [b.row() for b in self.bundles]


def measure_scene(scene: Path, mask_name: str = "mask_w1.png",
                  sem_name: str = "sem.png",
                  params: Optional[Params] = None,
                  sp: Optional[SampleParams] = None,
                  cp: Optional[CutParams] = None,
                  bg_sigma: float = 24.0) -> SceneResult:
    scene = Path(scene)
    params = params or load_params()
    sp = sp or SampleParams()
    cp = cp or CutParams()

    mask = read_mask(scene / mask_name)

    t0 = time.time()
    masks, graph, matching, chains = predict(mask, params, return_internals=True)
    t1 = time.time()

    sem = load_sem(scene / sem_name, sigma=bg_sigma)
    if sem.shape != mask.shape:
        raise RuntimeError(f"{sem_name} is {sem.shape}, mask is {mask.shape}")
    ndist = node_distance(graph)
    cutsets = [collect_cuts(sem, ndist, graph, matching, chains[k - 1], k,
                            int(masks[k].sum()), sp, cp)
               for k in sorted(masks)]
    # The scene-level width guard needs every cut in the image before it can
    # say what "much wider than usual here" means, so aggregation is a second
    # pass over cuts that are already measured -- it costs nothing extra.
    pooled = np.concatenate([c.width for c in cutsets if c.width.size]) \
        if any(c.width.size for c in cutsets) else np.zeros(0)
    cap = (float(sp.scene_width_cap * np.median(pooled))
           if sp.scene_width_cap > 0 and pooled.size >= 20 else None)
    bundles = [aggregate(c, sp, cap) for c in cutsets]
    t2 = time.time()

    return SceneResult(scene=scene, mask_name=mask_name, shape=mask.shape,
                       n_instances=len(masks), bundles=bundles, masks=masks,
                       graph=graph, matching=matching, chains=chains, sem=sem,
                       seconds_stage1=t1 - t0,
                       seconds_stage2=t2 - t1, cut_width_cap=cap)
