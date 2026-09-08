"""Adapt PLECTA's grouping parameters to a new domain from a few annotated images.

PLECTA's parameters are physical -- pixel lengths and radians describing how long
a filament is, how far apart fragments may be, how sharply objects bend. A new
domain changes those facts even though the algorithm is unchanged, so the numbers
have to move. This finds them.

Nothing here writes to `params.json` or to the frozen `grouping_2d` section.
A tuned configuration is a SEPARATE file, applied with `plecta --params`, so
`verify_against_frozen()` keeps passing and every published number stays
reproducible.

Design, each part chosen because it was measured rather than assumed
-------------------------------------------------------------------
* **Random search is the engine.** At equal budget it matched a directed
  pattern search on held-out data (0.6274 vs 0.6189) -- so the optimiser was
  never the bottleneck, and Bayesian optimisation is not worth a dependency.
* **Because a stronger search would be WORSE.** Pattern search reached train
  0.749 / test 0.619 (gap +0.13) where random reached 0.626 / 0.627 (gap
  -0.00). Best-of-N random rarely lands in a sharp optimum because a spike
  occupies almost no volume, so it is implicitly biased toward broad basins,
  which generalise. A harder-climbing optimiser overfits harder.
* **Random parallelises; directed search does not.** 66 s on 19 workers against
  209-277 s serial for the same budget.
* **A short refinement pass follows, for explanation not accuracy.** It is what
  produced statements like "gap_max_len 85 -> 13.6", which is how a tuned
  configuration becomes something a human can check. Every refinement move must
  improve the validation score or it is discarded, so in the worst case this
  degrades to "random search plus a log".
* **Selection is gated on PAIRS, not images.** Holding out images for validation
  only helps when those images carry enough scoreable pairs: it failed on worm
  images (~20 pairs each) and works on SEM scenes (2257-5812 each). When the
  validation fold is too thin the tuner says so instead of trusting it.
* **One annotated image is supported** by tiling it into folds. Tiling cuts
  objects at tile borders, which biases the objective -- but it biases every
  candidate equally, so the ranking selection depends on still holds. Tile
  scores are NOT reported as expected performance: measured against reality they
  were off by -0.069 to +0.034, in both directions.

Windows note: this uses a process pool, and Windows spawns rather than forks, so
every worker re-imports the calling module. A caller without an
``if __name__ == "__main__":`` guard will re-enter the tuner inside each worker
and die with `BrokenProcessPool`. `tune()` catches that and says so.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .linking import Params

CONFIG_VERSION = 1

#: name -> (kind, low, high). Angles are radians.
#:
#: The lengths here do NOT all move together -- worms wanted `window_local`
#: down 24 -> 9.6 while `join_px` went up 14 -> 35 -- so a single global scale
#: multiplier was measured and contributed exactly nothing.
#:
#: `j_unmatched` and `g_unmatched` are deliberately ABSENT. The matching is
#: invariant to scaling all weights and the unmatched price by a common factor
#: (cost = sum w_k f_k, eligible iff cost < price_i + price_j, objective
#: maximises sum(price_i + price_j - cost)), so tuning them adds a flat
#: direction that wastes budget and makes the result non-reproducible. Holding
#: them fixed reaches the same configurations in canonical form.
#:
#: `g_len_scale` and `kappa_scale` are absent for the same reason: they enter
#: only as `w_len / g_len_scale` and `w_kappa * kappa_scale`, pure products with
#: weights that ARE tuned. `j_chord_floor` / `g_chord_floor` ARE tuned, because
#: they sit inside `min(1, length/chord_floor)`, a nonlinear ramp that changes
#: the shape of the cost rather than only its magnitude.
#:
#: `n_rounds` is absent: it trades compute for quality and is not a property of
#: the domain.
TUNABLE: Dict[str, Tuple[str, float, float]] = {
    "window_local":  ("float", 6.0, 80.0),
    "window_chain":  ("float", 10.0, 160.0),
    "min_quadratic": ("int", 4, 40),
    "join_px":       ("int", 0, 40),
    "gap_max_len":   ("float", 5.0, 200.0),
    "j_chord_floor": ("float", 0.5, 30.0),
    "g_chord_floor": ("float", 0.5, 30.0),
    "j_w_direct":    ("float", 0.05, 5.0),
    "j_w_turn":      ("float", 0.05, 5.0),
    "j_w_kappa":     ("float", 0.0, 5.0),
    "g_w_direct":    ("float", 0.05, 5.0),
    "g_w_turn":      ("float", 0.05, 5.0),
    "g_w_len":       ("float", 0.0, 3.0),
    "g_w_kappa":     ("float", 0.0, 5.0),
    "gap_max_theta": ("float", 0.05, 1.6),
    "gap_max_phi":   ("float", 0.05, 1.6),
    "anneal_start":  ("float", 0.5, 1.0),
}


@dataclass
class TuneConfig:
    budget: int = 300              # total candidate evaluations
    random_frac: float = 0.6       # share spent on the parallel random phase
    top_k: int = 8                 # finalists re-scored on validation
    val_frac: float = 0.34         # share of scenes (or tiles) held out
    tile_grid: Tuple[int, int] = (2, 2)   # only used for a single scene
    min_val_pairs: int = 200       # below this, validation is not trusted
    workers: Optional[int] = None
    seed: int = 0
    step0: float = 0.4


# ── one scene, with the parameter-independent work done once ──────────────


class Scene:
    """A mask + its ground truth, with fragments and the GT assignment cached.

    Fragments and the GT assignment do not depend on the parameters, so they are
    computed once per scene rather than once per candidate. That is what makes a
    few hundred evaluations affordable.
    """

    def __init__(self, mask_path, gt_path, min_len_px=6, prune_spur_px=3):
        from .graph import read_mask
        from .score import assign_fragments, common_fragments

        self.mask = read_mask(mask_path)
        self.frag = common_fragments(self.mask, min_len_px, prune_spur_px)
        self.gt_assign = assign_fragments(self.frag, _load_gt(gt_path,
                                                              self.mask.shape))
        self.n_gt_pairs = _positive_pairs(self.gt_assign)

    def counts(self, params: Params) -> Tuple[float, float, float]:
        from .predict import predict
        from .score import assign_fragments, pairwise_scores

        pa = assign_fragments(self.frag, predict(self.mask, params))
        s = pairwise_scores(self.gt_assign, pa)
        return float(s["tp"]), float(s["fp"]), float(s["fn"])


def _load_gt(gt_path, shape):
    """Ground truth as {id: bool mask}, from a multilabel npz or a label image."""
    gt_path = Path(gt_path)
    if gt_path.suffix.lower() == ".npz":
        d = np.load(str(gt_path), allow_pickle=False)
        ids = [int(v) for v in d["ids"]]
        indptr = np.asarray(d["indptr"]).astype(np.int64)
        indices = np.asarray(d["indices"]).astype(np.int64)
        out = {}
        for k, cid in enumerate(ids):
            m = np.zeros(int(np.prod(shape)), dtype=bool)
            m[indices[indptr[k]:indptr[k + 1]]] = True
            out[cid] = m.reshape(shape)
        return out
    from skimage import io as skio
    lab = np.asarray(skio.imread(str(gt_path)))
    if lab.ndim == 3:
        lab = lab[..., 0]
    return {int(v): (lab == int(v)) for v in np.unique(lab) if int(v) != 0}


def _positive_pairs(assign: Dict[int, int]) -> int:
    """How many same-instance fragment pairs the ground truth contains.

    This -- not the image count -- is what says whether a fold carries enough
    evidence to rank candidate configurations.
    """
    sizes: Dict[int, int] = {}
    for v in assign.values():
        if v:
            sizes[v] = sizes.get(v, 0) + 1
    return int(sum(n * (n - 1) // 2 for n in sizes.values()))


# ── worker plumbing ───────────────────────────────────────────────────────

_SCENES: Dict[str, List[Scene]] = {}


def _init_worker(train_specs, val_specs):
    global _SCENES
    _SCENES = {"train": [Scene(m, g) for m, g in train_specs],
               "val": [Scene(m, g) for m, g in val_specs]}


def _score_one(args):
    params, which = args
    tp = fp = fn = 0.0
    for sc in _SCENES[which]:
        try:
            a, b, c = sc.counts(params)
        except Exception:
            return 0.0
        tp += a
        fp += b
        fn += c
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


# ── candidate generation ──────────────────────────────────────────────────


def _clip(name: str, value):
    kind, lo, hi = TUNABLE[name]
    value = min(max(value, lo), hi)
    return int(round(value)) if kind == "int" else float(value)


def _sample(rng, base: Params) -> Params:
    """One random configuration, log-uniform over each positive range."""
    vals = {}
    for name, (kind, lo, hi) in TUNABLE.items():
        v = (math.exp(rng.uniform(math.log(lo), math.log(hi))) if lo > 0
             else rng.uniform(lo, hi))
        vals[name] = int(round(v)) if kind == "int" else float(v)
    return replace(base, **vals)


#: Filename patterns tried, in order, when a scene folder is given without
#: explicit names. Annotation tools name their outputs after the scene rather
#: than to a fixed convention, so insisting on `mask.png`/`gt_multilabel.npz`
#: would mean copying files around just to rename them.
MASK_PATTERNS = ("mask.png", "*_mask.png", "mask_w1.png", "*_mask255.png",
                 "mask_w2.png")
GT_PATTERNS = ("gt_multilabel.npz", "*_multilabel.npz",   # overlap-aware first
               "gt_labels.tif", "*_labels.tif")           # flattened fallback


def discover_scene(folder) -> Tuple[str, str]:
    """Find the mask and the ground truth in a scene folder.

    Overlap-aware ground truth is preferred over a flattened label image,
    because a label image cannot say that a crossing pixel belongs to two
    instances -- which is the thing PLECTA exists to represent.
    """
    folder = Path(folder)

    def first(patterns):
        for pat in patterns:
            hits = sorted(folder.glob(pat))
            if hits:
                return hits[0]
        return None

    mask, gt = first(MASK_PATTERNS), first(GT_PATTERNS)
    if mask is None:
        raise FileNotFoundError(
            f"{folder}: no mask found (looked for {', '.join(MASK_PATTERNS)})")
    if gt is None:
        raise FileNotFoundError(
            f"{folder}: no ground truth found (looked for "
            f"{', '.join(GT_PATTERNS)})")
    return str(mask), str(gt)


def tile_scene(mask_path, gt_path, grid, outdir: Path) -> List[Tuple[str, str]]:
    """Cut one annotated scene into tiles so a single image still has folds."""
    from skimage import io as skio

    from .graph import read_mask
    from .predict import save_multilabel_npz

    outdir.mkdir(parents=True, exist_ok=True)
    mask = read_mask(mask_path)
    gt = _load_gt(gt_path, mask.shape)
    H, W = mask.shape
    gh, gw = grid
    specs = []
    for i in range(gh):
        for j in range(gw):
            r0, r1 = i * H // gh, (i + 1) * H // gh
            c0, c1 = j * W // gw, (j + 1) * W // gw
            sub = mask[r0:r1, c0:c1]
            if sub.sum() < 200:
                continue
            subs = {int(iid): m[r0:r1, c0:c1] for iid, m in gt.items()
                    if m[r0:r1, c0:c1].sum() >= 20}
            if len(subs) < 2:
                continue
            mp, gp = outdir / f"tile_{i}{j}.png", outdir / f"tile_{i}{j}_gt.npz"
            skio.imsave(str(mp), (sub.astype(np.uint8) * 255),
                        check_contrast=False)
            save_multilabel_npz(gp, subs, sub.shape)
            specs.append((str(mp), str(gp)))
    return specs


# ── the tuner ─────────────────────────────────────────────────────────────


def tune(scene_specs: Sequence[Tuple[str, str]], base: Optional[Params] = None,
         cfg: TuneConfig = TuneConfig(), workdir: Optional[Path] = None,
         verbose: bool = True) -> dict:
    """Search for a configuration that suits these annotated scenes.

    `scene_specs` is a sequence of (mask_path, gt_path). A single pair is
    accepted and tiled. Returns a dict describing what was found, including
    whether it should be accepted at all.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    from .predict import load_params

    base = base or load_params()
    specs = [(str(m), str(g)) for m, g in scene_specs]
    if not specs:
        raise ValueError("no scenes given")
    tiled = False
    if len(specs) == 1:
        outdir = Path(workdir or Path.cwd()) / "_plecta_tune_tiles"
        specs = tile_scene(specs[0][0], specs[0][1], cfg.tile_grid, outdir)
        tiled = True
        if len(specs) < 2:
            raise RuntimeError(
                "a single scene tiled into fewer than 2 usable folds; use a "
                "finer tile_grid or supply a second annotated image")
        if verbose:
            print(f"  one scene -> {len(specs)} tiles used as folds")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(specs))
    n_val = max(1, int(round(len(specs) * cfg.val_frac)))
    val_specs = [specs[i] for i in order[:n_val]]
    train_specs = [specs[i] for i in order[n_val:]] or [specs[int(order[-1])]]

    val_pairs = sum(Scene(m, g).n_gt_pairs for m, g in val_specs)
    trusted = val_pairs >= cfg.min_val_pairs
    if verbose:
        print(f"  {len(train_specs)} train / {len(val_specs)} val "
              f"{'tiles' if tiled else 'scenes'}; {val_pairs} validation pairs"
              f" -> selection {'trusted' if trusted else 'NOT trusted'}")

    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
    n_random = max(4, int(cfg.budget * cfg.random_frac))
    trace: List[dict] = []

    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(train_specs, val_specs)) as pool:
            base_train = list(pool.map(_score_one, [(base, "train")]))[0]
            base_val = list(pool.map(_score_one, [(base, "val")]))[0]
            if verbose:
                print(f"  frozen: train {base_train:.4f}  val {base_val:.4f}")

            # phase 1 -- broad, parallel, implicitly biased toward flat optima
            cands = [_sample(rng, base) for _ in range(n_random)]
            tr = list(pool.map(_score_one, [(c, "train") for c in cands],
                               chunksize=1))
            ranked = sorted(range(len(cands)), key=lambda i: -tr[i])
            finalists = [cands[i] for i in ranked[:cfg.top_k]]
            fv = list(pool.map(_score_one, [(c, "val") for c in finalists]))
            k = int(np.argmax(fv))
            best, best_val, best_train = finalists[k], float(fv[k]), tr[ranked[k]]
            if best_val <= base_val:
                best, best_val, best_train = base, base_val, base_train
            evals = n_random + cfg.top_k + 2
            if verbose:
                print(f"  phase 1: {n_random} random evals -> val {best_val:.4f}")

            # phase 2 -- short, sequential, for the interpretable trace
            cur, cur_train = best, best_train
            step = cfg.step0
            while step > 0.05 and evals < cfg.budget:
                improved = False
                for name in TUNABLE:
                    if evals >= cfg.budget:
                        break
                    batch = []
                    for d in (1.0 + step, 1.0 - step):
                        v = getattr(cur, name)
                        nv = _clip(name, (v * d) if v > 1e-9
                                   else TUNABLE[name][2] * 0.1 * step)
                        if nv != v:
                            batch.append((replace(cur, **{name: nv}), nv))
                    if not batch:
                        continue
                    got = list(pool.map(_score_one,
                                        [(c, "train") for c, _ in batch]))
                    evals += len(batch)
                    b = int(np.argmax(got))
                    if got[b] > cur_train + 1e-6:
                        cand, nv = batch[b]
                        vs = list(pool.map(_score_one, [(cand, "val")]))[0]
                        evals += 1
                        cur, cur_train = cand, got[b]
                        improved = True
                        kept = vs > best_val + 1e-6
                        if kept:
                            best, best_val, best_train = cand, vs, got[b]
                        trace.append({"param": name, "from": getattr(base, name),
                                      "to": nv, "train": round(got[b], 4),
                                      "val": round(vs, 4), "kept": kept})
                        if verbose:
                            print(f"    {'*' if kept else ' '} {name} -> {nv}"
                                  f"   train {got[b]:.4f}  val {vs:.4f}")
                if not improved:
                    step *= 0.5
    except BrokenProcessPool as exc:      # the Windows spawn trap
        raise RuntimeError(
            "the tuner's process pool died. On Windows every worker re-imports "
            "the calling module, so the call to tune() must sit under "
            "`if __name__ == \"__main__\":` (or inside a function that is not "
            "executed at import time). See the module docstring.") from exc

    changed = {n: [getattr(base, n), getattr(best, n)] for n in TUNABLE
               if getattr(base, n) != getattr(best, n)}
    return {
        "tiled": tiled, "n_train": len(train_specs), "n_val": len(val_specs),
        "validation_pairs": val_pairs, "selection_trusted": trusted,
        "workers": workers, "evals": evals,
        "frozen": {"train": round(base_train, 4), "val": round(base_val, 4)},
        "tuned": {"train": round(best_train, 4), "val": round(best_val, 4)},
        "val_gain": round(best_val - base_val, 4),
        "accept": bool(best_val > base_val),
        "params": best, "changed": changed, "trace": trace,
    }


# ── the tuned configuration file ──────────────────────────────────────────


HEADER = """\
# PLECTA tuned configuration -- NOT the frozen release configuration.
#
# Produced by `plecta-tune` from annotated images of one domain. Apply it with
#     plecta --mask <mask> --out <out> --params <this file>
#
# `plecta/params.json` and the frozen `grouping_2d` section of parameters.yaml
# are untouched by tuning, so the published numbers remain reproducible; this
# file only overrides values at run time.
#
# `validation` below is a RANKING score used to choose between candidates. When
# `tiled` is true it comes from tiles of a single image and is NOT an estimate
# of performance -- measured against real held-out images it has been wrong in
# both directions by up to 0.07.
"""


def write_config(path, result: dict, domain: str = "", notes: str = "") -> Path:
    """Write a tuned configuration as a standalone file."""
    import yaml

    path = Path(path)
    payload = {
        "plecta_tuned_config": CONFIG_VERSION,
        "domain": domain,
        "notes": notes,
        "provenance": {
            "n_train_folds": result["n_train"], "n_val_folds": result["n_val"],
            "tiled": result["tiled"], "validation_pairs": result["validation_pairs"],
            "selection_trusted": result["selection_trusted"],
            "evaluations": result["evals"],
            "frozen_validation": result["frozen"]["val"],
            "tuned_validation": result["tuned"]["val"],
            "validation_gain": result["val_gain"],
        },
        # only the values that actually moved, so the file reads as a diff
        # against the release configuration rather than a wall of numbers
        "grouping_2d": {n: (float(b) if isinstance(b, float) else int(b))
                        for n, (_a, b) in sorted(result["changed"].items())},
    }
    path.write_text(HEADER + yaml.safe_dump(payload, sort_keys=False),
                    encoding="utf-8")
    return path


def load_config(path) -> Dict[str, float]:
    """Read a tuned configuration file into {parameter: value}."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "plecta_tuned_config" not in data:
        raise RuntimeError(f"{path} is not a PLECTA tuned configuration file")
    over = data.get("grouping_2d") or {}
    unknown = set(over) - {f.name for f in fields(Params)}
    if unknown:
        raise RuntimeError(f"{path} sets unknown parameters: {sorted(unknown)}")
    return over


def apply_config(base: Params, path) -> Params:
    """`base` with a tuned configuration file applied on top."""
    over = load_config(path)
    coerced = {}
    for k, v in over.items():
        proto = getattr(base, k)
        coerced[k] = int(round(v)) if isinstance(proto, int) else float(v)
    return replace(base, **coerced)


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    import argparse

    from .predict import load_params

    ap = argparse.ArgumentParser(
        description="Adapt PLECTA's grouping parameters to a domain from a few "
                    "annotated images. Writes a separate config file; never "
                    "modifies params.json or the frozen parameters.")
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="scene directories, each containing the mask and the "
                         "ground truth (see --mask-name/--gt-name)")
    ap.add_argument("--mask-name", default=None,
                    help="exact mask filename; omitted, it is discovered "
                         f"({', '.join(MASK_PATTERNS)})")
    ap.add_argument("--gt-name", default=None,
                    help="exact ground-truth filename; omitted, it is "
                         f"discovered ({', '.join(GT_PATTERNS)})")
    ap.add_argument("--out", required=True, help="tuned config file to write")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tile-grid", type=int, nargs=2, default=(2, 2),
                    metavar=("ROWS", "COLS"),
                    help="how to tile when only one scene is given")
    ap.add_argument("--domain", default="", help="free-text label for the file")
    ap.add_argument("--force", action="store_true",
                    help="write the config even if it did not beat the frozen "
                         "parameters on validation")
    args = ap.parse_args(argv)

    specs = []
    for s in args.scenes:
        d = Path(s)
        try:
            if args.mask_name or args.gt_name:
                m, g = d / (args.mask_name or ""), d / (args.gt_name or "")
                if not m.is_file() or not g.is_file():
                    raise FileNotFoundError(
                        f"{d}: need both {args.mask_name} and {args.gt_name}")
                m, g = str(m), str(g)
            else:
                m, g = discover_scene(d)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))
        print(f"  {d.name}: {Path(m).name} + {Path(g).name}")
        specs.append((m, g))

    cfg = TuneConfig(budget=args.budget, seed=args.seed, workers=args.workers,
                     tile_grid=tuple(args.tile_grid))
    result = tune(specs, load_params(), cfg, workdir=Path(args.out).parent)

    print(f"\n  validation {result['frozen']['val']:.4f} -> "
          f"{result['tuned']['val']:.4f}  ({result['val_gain']:+.4f})")
    if not result["selection_trusted"]:
        print(f"  WARNING: only {result['validation_pairs']} validation pairs; "
              f"the selection is weakly evidenced. More annotated images, or "
              f"denser scenes, would make this trustworthy.")
    if result["tiled"]:
        print("  NOTE: validation came from tiles of a single image. Use it to "
              "rank configurations, not as an estimate of performance.")
    if not result["accept"] and not args.force:
        print("\n  Tuning did not beat the frozen parameters on validation, so "
              "no file was written. Re-run with --force to write it anyway.")
        return 1
    for name, (a, b) in sorted(result["changed"].items()):
        print(f"    {name:16s} {a}  ->  {b}")
    dest = write_config(args.out, result, domain=args.domain)
    print(f"\n  wrote {dest}\n  apply with:  plecta --mask <mask> --out <out> "
          f"--params {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
