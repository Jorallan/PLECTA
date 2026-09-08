# PLECTA core

The evaluated mask-to-instance implementation. It reads a binary mask and
writes overlap-aware instance layers, and nothing else: not the SEM image, not
ground truth, not a clean reference mask.

```powershell
plecta --mask input.png --out pred_multilabel.npz
```

## What each file does

### The grouping — mask in, instances out

These four are the method. They run in the order listed, and read nothing but
the mask and their parameters.

| File | What it does |
|:--|:--|
| [`graph.py`](graph.py) | Skeletonizes the mask and cuts it into a graph. An **arm** is a maximal run of skeleton with no ambiguity in it; a **node** is a crossing; a **stub** is one end of an arm, and stubs are the units the method has to pair up. Also merges junction clusters joined by a sub-`bridge_px` connector, so one crossing is one node. |
| [`geometry.py`](geometry.py) | Local shape, per stub and per polyline. Fits an outward **frame** at a stub tip — position, tangent, curvature — in arclength rather than pixel count, so a diagonal arm gets the same support as a horizontal one. `link_cost` prices a candidate join from how far it bends. Also holds the shared polyline utilities: resample at a fixed step, and unit normals. |
| [`linking.py`](linking.py) | Decides which arm continues into which. Maximum-weight matching at each crossing, then gated bridging across mask gaps, repeated over several rounds — because every join lengthens a chain, which sharpens the direction estimate for the crossings that chain still has to get through. Ends by walking the matching into chains. |
| [`predict.py`](predict.py) | Ties the three together, paints each chain as a layer, and writes the sparse NPZ. A crossing pixel is written into *every* instance that runs through it. This is also the `plecta` command line. |

### Configuration — one file holds every tunable

| File | What it does |
|:--|:--|
| [`parameters.yaml`](parameters.yaml) | Every tunable of every stage, grouped for reading. The run-time source of truth. |
| [`parameters.py`](parameters.py) | Builds each stage's parameter object from that file. Refuses to run if the file and the dataclass disagree in *either* direction, which is what makes "the YAML is the configuration" true rather than aspirational — the dataclass defaults are deliberately *not* the tuned values (`linking.Params.join_px` defaults to 0 and is tuned to 14). Also asserts the frozen check below. |
| [`params.json`](params.json) | Immutable record of the `grouping_2d` section alone, 27 values. Its SHA-256 is the anchor for [`../evidence/`](../evidence/). |

### Tools that sit beside the core, not inside it

| File | What it does |
|:--|:--|
| [`score.py`](score.py) | The common-fragment metric behind every number PLECTA reports. Cuts both the prediction and the truth into the same atomic fragments, then scores agreement about which pairs share an instance — so the score does not depend on how either side numbered its instances. |
| [`tune.py`](tune.py) | Domain adaptation from a few annotated images: a random phase then coordinate descent, scored on held-out folds of the images you give it. Writes a *separate* config file and never touches `params.json`. |

### Optional stages — they run after grouping and cannot change it

| File | What it does |
|:--|:--|
| [`depth.py`](depth.py) | The 2.5-D stage. Finds projected crossings, reads one grayscale channel to decide which instance is on top at each, resolves those local calls into one globally consistent order, infers the fewest layers that order needs, solves a height per instance under a compact-stack prior, and sweeps circular tubes to PLY/OBJ. Also the `python -m plecta.depth` command line. |
| [`joint.py`](joint.py) | A **measured negative result**, kept as a record. Blends stub appearance into the crossing decision where the geometry alone was ambiguous; measured at −0.0129 mean F1 and retained so the configuration that produced that number still runs. Nothing in the shipped pipeline imports it. |
| [`image/`](image/) | Optional width and brightness measurement, and rendering. See [`image/README.md`](image/README.md). |

## Why the grouping is split into four files

`graph.py` and `geometry.py` are independent — neither imports the other. One
knows about *connectivity* (what touches what), the other about *shape* (which
way a tip points). `linking.py` is the only module that needs both, because
pairing stubs is exactly the question that requires connectivity and shape at
once. `predict.py` sits on top and knows only about output.

That layering is load-bearing rather than cosmetic: `geometry.py` is also used
by `depth.py` and by `image/measurement.py`, and `graph.py` by six other
modules. Folding either into `linking.py` would drag the matching solver into
every module that only wanted a tangent or a skeleton.

## What is guaranteed, and by what

- **Every effective value comes from `parameters.yaml`.** The code does not
  fall back to dataclass defaults. Guarded by
  [`../tests/test_parameters.py`](../tests/test_parameters.py).
- **The published configuration has not moved.**
  `plecta.parameters.verify_against_frozen()` asserts `grouping_2d` still
  equals `params.json`.
- **The public surface is stable.**
  [`../tests/test_core_contract.py`](../tests/test_core_contract.py).
- **Grouping reads only the mask.** `depth.py`, `joint.py` and `image/` all run
  after the grouping is fixed and cannot change which pixels were grouped
  together. `joint.py` is the one exception in intent, and it is disabled
  precisely because it was measured and rejected.

## Evidence

Held-out F1 **0.8545** over 128 disjoint-seed scenes; development F1 **0.8671**
over 84. Aggregates and provenance are in
[`../evidence/benchmark_summary.json`](../evidence/benchmark_summary.json),
which is the canonical machine-readable copy and the one the SHA-256 provenance
is built around.
