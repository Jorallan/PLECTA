# Optional SEM characterization

Runs the fixed PLECTA grouping, then measures each instance against the SEM
image: how wide it is, how bright, and how much to trust either number.

**SEM intensity never affects grouping.** Everything here runs after the
grouping is fixed, and the numbers it produces cannot revise it. None of it
contributed to the held-out result.

```powershell
python -m pip install ".[image]"
plecta-image --scene path\to\scene --out output
```

The scene folder needs a mask (`mask_w1.png` by default) and `sem.png`.

## What each file does

| File | What it does |
|:--|:--|
| [`pipeline.py`](pipeline.py) | The single library entry point: group the mask, load the SEM, cut every instance, aggregate. `measure_scene` is the one implementation — the CLI calls it rather than repeating it. |
| [`measurement.py`](measurement.py) | Reads the image and measures one cut across one ridge. Estimates the background as a *field* rather than a constant, from the pixels that do not look like filament, so a faint bundle is not measured against its bright neighbour. Fits full width at half maximum above that local background, and refuses a cut it cannot justify — too little contrast, crest off axis, a neighbouring ridge in the way — with a named reason. Also converts a measured width to a physical diameter for the depth stage. |
| [`bundles.py`](bundles.py) | Decides *where* to cut, and turns many cuts into one number per instance. Places cuts along the axis, skipping crossings (a cut through a crossing measures two bundles at once). Aggregates with a moving-block bootstrap, and separates the spread that is measurement noise from the spread that is the bundle genuinely varying along its length. |
| [`refine.py`](refine.py) | Optional re-rendering. Orders each chain, interpolates the bridged gaps the core output leaves unpainted, smooths with the endpoints pinned, and draws a ribbon at the fitted width. |
| [`overlay.py`](overlay.py) | Optional three-panel figure: the SEM alone, the instances coloured by fitted width with the cuts that produced them marked, and the width profile along each instance — where a wrong merge shows up as a step rather than a drift. |
| [`predict.py`](predict.py) | The `plecta-image` command line. Writes the layers, a per-instance CSV, a metadata JSON, and the figures. |

## `--refine`, and the one flag that changes grouping

`--refine` renders each chain as a smooth, gap-connected ribbon at its fitted
width. At the default `--absorb-thr 0` this **does not change grouping** — it
redraws the same instances.

A **positive** absorption threshold unions instances whose rendered silhouettes
largely coincide. That is a separate, experimental grouping change, and it is
absent from the 128-scene held-out result.

Two rendering options ship **off**, and their own measurements say why:

- `clip_to_mask` — cannot change the score at all (every common fragment lies
  inside the mask, so clipping removes only pixels no fragment occupies), and
  puts the ragged mask boundary back into the delivered shape.
- `taper` — more faithful to a genuinely tapering bundle, but measured at
  −0.003 to −0.012 F1: where the local width dips below the mask's own width,
  the ribbon stops covering its own fragment.

Both are kept selectable, because "this cannot help" is worth being able to
re-run.
