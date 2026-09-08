"""The executable verifier for the evaluation split.

evidence/README.md declares the split; evidence/seed_manifest.json records it. This test
checks: counts per set, mutual disjointness of
every seed across every set, and the presence of the metadata a re-generation
needs. It reads only the manifest, so it runs on any checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "evidence" / "seed_manifest.json"

#: scene counts per set as declared in evidence/README.md
DECLARED_COUNTS = {
    "synthetic_thick": 20,
    "dev_ext": 48,
    "gen_dev40": 16,
    "val": 96,
    "gen_test40": 32,
    "synthetic_locked_v1": 125,
}
DECLARED_ROLES = {
    "synthetic_thick": "development",
    "dev_ext": "development",
    "gen_dev40": "development",
    "val": "heldout",
    "gen_test40": "heldout",
    "synthetic_locked_v1": "locked",
}


def load():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def expand(entry) -> list[int]:
    seeds = []
    for block in entry["seed_blocks"]:
        seeds.extend(range(block["start"], block["start"] + block["count"]))
    return seeds


def test_sets_and_counts_match_declared_split():
    manifest = load()
    assert set(manifest["sets"]) == set(DECLARED_COUNTS)
    for name, entry in manifest["sets"].items():
        seeds = expand(entry)
        assert len(seeds) == DECLARED_COUNTS[name], name
        assert len(set(seeds)) == len(seeds), f"{name}: duplicate seed inside set"
        assert entry["role"] == DECLARED_ROLES[name], name


def test_all_seeds_mutually_disjoint():
    manifest = load()
    seen: dict[int, str] = {}
    for name, entry in manifest["sets"].items():
        for seed in expand(entry):
            assert seed not in seen, (
                f"seed {seed} appears in both {seen[seed]} and {name}")
            seen[seed] = name
    assert len(seen) == sum(DECLARED_COUNTS.values())


def test_scene_metadata_present():
    manifest = load()
    assert manifest["schema"] == 1
    assert manifest["generator"].endswith("synth_thick.py")
    assert "seed_rule" in manifest
    for name, entry in manifest["sets"].items():
        args = entry["generator_args"]
        for flag in ("--coverages", "--n", "--size", "--seed0"):
            assert flag in args, f"{name}: {flag} missing from generator_args"


def test_development_and_heldout_totals():
    manifest = load()
    totals = {"development": 0, "heldout": 0, "locked": 0}
    for entry in manifest["sets"].values():
        totals[entry["role"]] += len(expand(entry))
    assert totals["development"] == 84
    assert totals["heldout"] == 128
    assert totals["locked"] == 125
