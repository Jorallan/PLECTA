# Evidence

Machine-readable provenance for the recorded grouping benchmarks.
Raw scenes and the scene generator are not included in this repository.

## What each file records

| File | What it records |
|:--|:--|
| [`benchmark_summary.json`](benchmark_summary.json) | The paper-facing aggregate results, and the SHA-256 of `plecta/params.json` that ties them to a configuration |
| [`heldout_per_scene.json`](heldout_per_scene.json) | The per-scene rows behind that aggregate — the aggregate is their mean |
| [`seed_manifest.json`](seed_manifest.json) | Every declared development, held-out, and untouched seed. [`tests/test_seed_manifest.py`](../tests/test_seed_manifest.py) is its executable verifier |
| [`release_parity.json`](release_parity.json) | That this release and the manuscript's frozen copy produced byte-identical outputs on all 128 held-out masks, with per-scene SHA-256 hashes |
| [`environment.json`](environment.json) | The interpreter and dependency versions that run was made under |
| [`configuration_lineage.json`](configuration_lineage.json) | Append-only record of every `plecta/params.json` this repository has shipped, and for each one after the anchor, whether the published outputs still hold and how that was checked |

## Why the recorded hashes are never rewritten

`benchmark_summary.json` and `release_parity.json` cite the SHA-256 of the
`params.json` their runs were made under. Those are **dated records of runs
that happened**, so the hash in them is never updated — doing so would assert a
run that never took place.

When `params.json` does change, the change is appended to
[`configuration_lineage.json`](configuration_lineage.json) instead, with what
changed and whether the published outputs still hold.
[`../tests/test_configuration_lineage.py`](../tests/test_configuration_lineage.py)
then enforces both halves: the live file must be the newest recorded entry, and
the two evidence files must still cite the anchor.

## The development ablations

Each disables one mechanism at a time, leaving everything else fixed:

| Key | What is disabled |
|:--|:--|
| `no_gap_bridging_f1` | Gap-bridging candidates |
| `no_connector_cluster_consolidation_f1` | Connector-based consolidation of junction clusters (`bridge_px = join_px = 0`); junction matching still runs |
| `no_junction_chord_turn_f1` | The junction chord-turn term (`j_w_turn = 0`) |

## Evaluation split

The seed manifest records the scene seeds and generator arguments.

### Scene sets

| Role | Set | Scenes | Coverages |
|:--|:--|--:|:--|
| Development | `synthetic_thick` | 20 | 20, 30, 40, 50, 60% |
| Development | `dev_ext` | 48 | 20, 30, 60% |
| Development | `gen_dev40` | 16 | 40% |
| **Development total** | | **84** | |
| Held out | `val` | 96 | 20, 30, 60% |
| Held out | `gen_test40` | 32 | 40% |
| **Held-out total** | | **128** | |
| Locked, unevaluated | `synthetic_locked_v1` | 125 | 20, 30, 40, 50, 60% |

### Interpretation

All listed seeds are mutually disjoint, so no development scene appears in the
held-out set.

The held-out scenes are **independent draws from the same generator**, not an
independent distribution. A held-out score therefore measures sampling
robustness, not transfer to new imaging conditions.

`synthetic_locked_v1` is the stronger set and has **no reported PLECTA result**;
it is locked and untouched.
