# PLECTA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22687992.svg)](https://doi.org/10.5281/zenodo.22687992)

Licensed under the [MIT License](LICENSE): free to use, modify and
redistribute, provided the copyright notice and license text travel with any
copy or substantial portion of the code. If PLECTA or its method contributes to
work you publish, please also [cite it](#citing-plecta).

PLECTA, after the Latin *plecta*, a braid,
reconstructs overlapping strand instances from a thin binary mask. It
skeletonizes the mask, decomposes it into arms and junctions, and alternates
exact crossing matching with gated gap matching as chain geometry is refined.
Crossing pixels may belong to several output instances.

Grouping is deterministic and mask-only: it does not read the greyscale
image, ground truth, or a clean reference mask.

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
| The algorithm, described | [`METHOD.md`](METHOD.md) |
| Source modules and architecture | [`plecta/README.md`](plecta/README.md) |
| Every tunable of every stage | [`plecta/parameters.yaml`](plecta/parameters.yaml) |
| The frozen published configuration | [`plecta/params.json`](plecta/params.json) |
| Benchmark aggregates and provenance | [`evidence/`](evidence/) |
| How to cite this work | [Citing PLECTA](#citing-plecta) |
| How the evaluation split was drawn | [`evidence/README.md`](evidence/README.md#evaluation-split) |

`parameters.yaml` is the whole tunable surface and what the executable actually
reads. `params.json` is the frozen record of the `grouping_2d` section alone;
its SHA-256 is the hash anchor for `evidence/`.

## Evidence

The fixed method is evaluated with common-fragment pairwise F1 on held-out
scenes drawn from seeds disjoint from every development scene. That tests
**sampling robustness, not transfer to a new distribution** — the held-out
scenes are independent draws from the same generator.

The aggregates, ablations, seed manifest and release-parity records are in
[`evidence/`](evidence/), with the SHA-256 that ties them to the configuration
they were produced under. Evaluation data, detailed diagnostics and
experimental scripts remain local.

## Repository map

| Path | Purpose |
|:--|:--|
| [`plecta/`](plecta/) | The core mask-to-instance method, its configuration, and the secondary stages |
| [`plecta/parameters.yaml`](plecta/parameters.yaml) | Every tunable of every stage — the run-time source of truth |
| [`evidence/`](evidence/) | Benchmark aggregates, seed manifest, and release parity records |
| [`tests/`](tests/) | Software contract checks: the public surface, the parameter file's claim to be the configuration, and the seed manifest |
| [`tools/`](tools/) | Prints a configuration scaffold for comparison with the maintained `parameters.yaml` |


---

## Greyscale characterization

```powershell
python -m pip install ".[image]"
plecta-image --scene <dir> --out <dir>
```

Measures each instance against a registered greyscale image — width and
brightness — **after** grouping. `--refine` renders smooth ribbons; silhouette
absorption is off by default. None of these options contributed to the held-out
grouping result. Developed and validated on SEM, and the scene loader still
expects `sem.png` by default, but nothing in the stage is specific to that
modality. See
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
grouping and cannot change it; not running it leaves the grouping result unchanged. **Validated on synthetic scenes only.**

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

Against the rule shipped before it, this is **same accuracy, better-ordered
confidence, simpler evidence path — not more accurate**. The better ordering
turns into accuracy only wherever more abstention is acceptable. The previous
rule stays selectable and reproduces its records bit for bit.

It was checked on a clean optical set whose seeds are disjoint from every
tuning set. See the limitations noted in [`METHOD.md`](METHOD.md#scope).

## Citing PLECTA

The MIT license requires that the copyright notice be preserved in copies of
the code. Citation is asked of you separately, as a scholarly courtesy: if
PLECTA or the method described in [`METHOD.md`](METHOD.md) contributes to work
you publish, cite the software.

GitHub reads [`CITATION.cff`](CITATION.cff) and offers a **Cite this
repository** button with the current entry. In BibTeX:

```bibtex
@software{allan_plecta_2026,
  author    = {Allan, Oday},
  title     = {{PLECTA}: overlap-aware reconstruction of filament instances
               from binary axis masks},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22687992},
  url       = {https://doi.org/10.5281/zenodo.22687992}
}
```

That is the **concept** DOI: it always resolves to the newest version, so it
stays correct across releases. Each release also gets its own version DOI —
[`10.5281/zenodo.22687993`](https://doi.org/10.5281/zenodo.22687993) for
v0.1.0 — which is what to cite when the exact version matters for
reproducibility.

Once the accompanying paper is published it becomes the preferred citation,
via `preferred-citation` in `CITATION.cff`, and this DOI goes on identifying
the software itself.
