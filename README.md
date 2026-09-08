# PLECTA

Licensed under the [MIT License](LICENSE).

PLECTA, after the Latin *plecta*, a braid,
reconstructs overlapping strand instances from a thin binary mask. It
skeletonizes the mask, decomposes it into arms and junctions, and alternates
exact crossing matching with gated gap matching as chain geometry is refined.
Crossing pixels may belong to several output instances.

Grouping is deterministic and mask-only: it does not read the SEM image,
ground truth, or a clean reference mask.

## Install and run

```powershell
python -m pip install .
plecta --mask path\to\mask.png --out pred_multilabel.npz
```

For a directory tree containing `mask_w1.png`:

```powershell
plecta --scenes path\to\scenes --out predictions
```

The output is a sparse multilabel NPZ.

## Tests

```powershell
python -m pip install ".[test]"
python -m pytest -q
```

Tests that require external evaluation scenes skip when those scenes are
unavailable. Joint-mode scene tests accept `PLECTA_REAL_SCENE` (a directory
containing `mask.png`) and `PLECTA_SYNTH_ROOT` (a tree containing `mask_w2.png`).
Their defaults are under `.local/evaluation/data/`.

## Where to look

| If you want | Read |
|:--|:--|
| The algorithm, and what was measured | [`METHOD.md`](METHOD.md) |
| Source modules and architecture | [`plecta/README.md`](plecta/README.md) |
| Every tunable of every stage | [`plecta/parameters.yaml`](plecta/parameters.yaml) |
| The frozen published configuration | [`plecta/params.json`](plecta/params.json) |
| Benchmark aggregates and provenance | [`evidence/`](evidence/) |
| How the evaluation split was drawn | [`evidence/README.md`](evidence/README.md#evaluation-split) |

`parameters.yaml` is the whole tunable surface and what the executable actually
reads. `params.json` is the frozen record of the `grouping_2d` section alone;
its SHA-256 is the hash anchor for `evidence/`.

## Evidence

The fixed method scored **0.8545** mean common-fragment F1 on 128 independently
seeded held-out scenes from the same generator (precision 0.8676, recall 0.8430,
recovery 0.7994, ARI 0.8508).

This tests **sampling robustness, not transfer to a new distribution** — the
held-out scenes are independent draws from the same generator. Compact
provenance is in [`evidence/`](evidence/); evaluation data, detailed rows,
diagnostics, and experimental scripts remain local.

## Repository map

| Path | Purpose |
|:--|:--|
| [`plecta/`](plecta/) | The core mask-to-instance method, its configuration, and the secondary stages |
| [`plecta/parameters.yaml`](plecta/parameters.yaml) | Every tunable of every stage — the run-time source of truth |
| [`evidence/`](evidence/) | Benchmark aggregates, seed manifest, and release parity records |
| [`tests/`](tests/) | Software contract checks: the public surface, the parameter file's claim to be the configuration, and the seed manifest |
| [`tools/`](tools/) | Prints a configuration scaffold for comparison with the maintained `parameters.yaml` |


---

## SEM characterization

```powershell
python -m pip install ".[image]"
plecta-image --scene <dir> --out <dir>
```

Measures width and brightness **after** grouping. `--refine` renders smooth
ribbons; silhouette absorption is off by default. None of these options
contributed to the held-out grouping result. See
[`plecta/image/README.md`](plecta/image/README.md).

## Depth stage (2.5-D)

```powershell
python -m plecta.depth --scene path\to\scene --out path\to\out --tubes
```

Given a registered grayscale image as well as the mask, this infers which
instance is on top at each projected crossing, a globally consistent vertical
order, a minimal number of layers, one depth per instance, and sweeps circular
3-D tubes (PLY/OBJ).

Every instance is modelled as **planar** — one scalar depth, constant along its
length — and no two reconstructed instances interpenetrate. It runs after
grouping and cannot change it; not running it reproduces the grouping result
above exactly. **Validated on synthetic scenes only.**

| Flag | Effect |
|:--|:--|
| `--no-image-evidence` | Runs the same geometry with every crossing abstained |
| `--clear-grazing` | Also separates pairs that overlap without their centrelines crossing |
| `--grazing-evidence` | Reads the image at those pairs instead of merely separating them |
| `--fixed-radius` | Replaces measured widths with one constant |
| `--undecided-order` | Chooses how undecidable pairs are stacked |

### The over/under evidence

A **single** grayscale channel: how far the shared crossing core sits from each
rod's own flanks, normalised by the larger of the two rods' contrast and twice
the within-rod flank noise — so a crossing whose rods differ by less than the
noise cannot report certainty. It reads a constant-size core and therefore
**no radius**, and builds no gradient image.

Measured on a clean held-out optical set of 20 scenes / 1355 crossings whose
seeds are disjoint from every tuning set, against the rule shipped before
2026-08-22:

| Against the previous rule | Result |
|:--|:--|
| Accuracy at the same coverage | +0.0025, 95% CI [−0.0042, +0.0079] |
| Ranking AUC of correctness against \|s\| | 0.7772 → 0.8698 (+0.0926, [+0.0563, +0.1321]) |
| Accuracy at 82.5% coverage | +0.0188 |

So: **same accuracy, better-ordered confidence, simpler evidence path — not
more accurate.** The better ordering turns into accuracy only wherever more
abstention is acceptable. The 82.5% row is a coverage-matched comparison point,
**not the shipped operating point**: at its own `abstain_score` = 0.40 the rule
decides 87.1% of that set, at 0.9347 accuracy against 0.9217 at 89.5%.

Those seeds are clean for the *rule*; the one constant selected on them is
`core_px` = 7.5, which [`METHOD.md`](METHOD.md) says so of.

The previous rule is still selectable and reproduces its records bit for bit.
The depth aggregates recorded elsewhere — and the in-repo grazing-evidence
figures — were produced under it and need regenerating.
