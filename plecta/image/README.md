# Optional greyscale characterization

Runs the fixed PLECTA grouping, then measures each instance against a
registered greyscale image: how wide it is, how bright, and how much to trust
either number.

**Image intensity never affects grouping.** Everything here runs after the
grouping is fixed, and the numbers it produces cannot revise it. None of it
contributed to the held-out result.

Nothing here is specific to a modality — it needs a greyscale image registered
to the mask, and reads it as intensities. It was developed and validated on
SEM, which is why the default filename and the figure labels say so.

```powershell
python -m pip install ".[image]"
plecta-image --scene path\to\scene --out output
```

The scene folder needs a mask (`mask_w1.png` by default) and a greyscale
image (`sem.png` by default; `--sem-name` overrides it).

## What each file does

| File | What it does |
|:--|:--|
| [`pipeline.py`](pipeline.py) | The single library entry point: group the mask, load the image, cut every instance, aggregate. `measure_scene` is the one implementation — the CLI calls it rather than repeating it. |
| [`measurement.py`](measurement.py) | Reads the image and measures one cut across one ridge. Estimates the background as a *field* rather than a constant, from the pixels that do not look like filament, so a faint bundle is not measured against its bright neighbour. Fits full width at half maximum above that local background, and refuses a cut it cannot justify — too little contrast, crest off axis, a neighbouring ridge in the way — with a named reason. Also converts a measured width to a physical diameter for the depth stage. |
| [`bundles.py`](bundles.py) | Decides *where* to cut, and turns many cuts into one number per instance. Places cuts along the axis, skipping crossings (a cut through a crossing measures two bundles at once). Aggregates with a moving-block bootstrap, and separates the spread that is measurement noise from the spread that is the bundle genuinely varying along its length. |
| [`refine.py`](refine.py) | Optional re-rendering. Orders each chain, interpolates the bridged gaps the core output leaves unpainted, smooths with the endpoints pinned, and draws a ribbon at the fitted width. |
| [`overlay.py`](overlay.py) | Optional three-panel figure: the image alone, the instances coloured by fitted width with the cuts that produced them marked, and the width profile along each instance — where a wrong merge shows up as a step rather than a drift. |
| [`predict.py`](predict.py) | The `plecta-image` command line. Writes the layers, a per-instance CSV, a metadata JSON, and the figures. |

## Assumptions, and where they hold

Numbers are on the reference SEM field B58-B3-S2_100 (307 filaments traced by
hand, 1536 × 1024 px), with its traced centrelines as the mask.

- **The FWHM is the diameter.** The width the depth stage receives is the
  median full width at half maximum, with no broadening correction. That holds
  for a grey profile flat across the bundle, with edges sharp next to its
  width; a profile following a round bundle's projected thickness would read
  (√3/2) d = 0.87 d, and blur widens either. On the reference field the median
  FWHM is 13.4 px against 12.43 px traced by hand, +0.80 px per filament
  (median).
- **The background is whatever is not filament.** A pixel counts as filament
  above a fifth of the way from the image's 20th to its 99th grey percentile,
  grown by 3 px; a bundle fainter than that enters the background field. On the
  reference field 7.4 % of the pixels the field averages lie inside traced
  filaments.
- **Skipping 6 px round a crossing clears it.** Cuts within `node_clear_px` of
  a crossing pixel are skipped, but the other bundle reaches
  r_i / tan θ + r_j / sin θ along this one, further at shallow crossings: 9.1 %
  of the accepted cuts lie within that reach and read a median 2.1 px wider
  than the rest of their instance.
- **Neighbouring cuts differ by noise alone.** The noise split takes the
  difference of two cuts 2 px apart as measurement noise, which assumes their
  noise independent and the width unchanged between them. On the reference
  field the image's background noise still correlates at 0.72 over 2 px,
  consecutive differences correlate at +0.05 where independent noise gives
  −0.5, and the estimate grows with the spacing (0.98, 1.30, 1.64 and 1.80 px
  at 2, 4, 6 and 8 px): the noise is read low and the real variation high.

## `--refine`, and the one flag that changes grouping

`--refine` renders each chain as a smooth, gap-connected ribbon at its fitted
width. At the default `--absorb-thr 0` this **does not change grouping** — it
redraws the same instances.

A **positive** absorption threshold unions instances whose rendered silhouettes
largely coincide. That is a separate, experimental grouping change, and it is
absent from the held-out result.

Two rendering options ship **off**, and their own measurements say why:

- `clip_to_mask` — cannot change the score at all (every common fragment lies
  inside the mask, so clipping removes only pixels no fragment occupies), and
  puts the ragged mask boundary back into the delivered shape.
- `taper` — more faithful to a genuinely tapering bundle, but measured worse:
  where the local width dips below the mask's own width, the ribbon stops
  covering its own fragment.

Both are kept selectable, because "this cannot help" is worth being able to
re-run.
