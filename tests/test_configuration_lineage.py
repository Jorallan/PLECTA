"""The frozen configuration and the evidence that cites it must stay in step.

`plecta.parameters.verify_against_frozen()` already checks `parameters.yaml`
against `params.json`. Nothing checked `params.json` against `evidence/`, so a
change to the frozen file could -- and did -- leave the published SHA-256
anchor pointing at a file that no longer exists, with a green suite.

The rule enforced here is deliberately not "params.json may never change".
It is:

* the anchor hash is the one the published numbers were produced under, and
  `benchmark_summary.json` and `release_parity.json` must keep recording it,
  because those are dated records of runs that actually happened;
* the live `params.json` must be the newest entry in the lineage, so a change
  is either recorded with its justification or fails here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "plecta" / "params.json"
EVIDENCE = ROOT / "evidence"
LINEAGE = EVIDENCE / "configuration_lineage.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lineage() -> dict:
    return _load(LINEAGE)


def test_the_live_params_file_is_the_newest_recorded_configuration(lineage):
    """A change to params.json has to be recorded, or this fails."""
    entries = lineage["lineage"]
    assert entries, "the lineage may not be empty"
    newest = entries[-1]["sha256"]
    actual = _sha256(PARAMS)
    assert actual == newest, (
        "plecta/params.json has changed without being recorded.\n"
        f"  file now:            {actual}\n"
        f"  newest lineage entry: {newest}\n"
        "Append an entry to evidence/configuration_lineage.json saying what "
        "changed and whether the published outputs still hold, or revert the "
        "change to params.json."
    )


def test_every_lineage_entry_is_distinct_and_well_formed(lineage):
    seen = set()
    for entry in lineage["lineage"]:
        digest = entry["sha256"]
        assert len(digest) == 64 and not set(digest) - set("0123456789abcdef"), \
            f"{digest!r} is not a sha256"
        assert digest not in seen, f"{digest} appears twice in the lineage"
        seen.add(digest)
        assert entry.get("date"), "every entry needs a date"
        assert entry.get("note"), "every entry needs a note saying what changed"


def test_exactly_one_entry_produced_the_published_results(lineage):
    anchors = [e for e in lineage["lineage"]
               if e.get("produced_the_published_results")]
    assert len(anchors) == 1, \
        "exactly one configuration produced the published numbers"
    assert anchors[0]["sha256"] == lineage["anchor_sha256"]


@pytest.mark.parametrize("name,field", [
    ("benchmark_summary.json", "parameter_sha256"),
    ("release_parity.json", "params_sha256_both"),
])
def test_the_evidence_still_cites_the_anchor(lineage, name, field):
    """These are dated records of runs that happened under the anchor.

    They are not updated when params.json moves -- doing so would assert a run
    that never took place. They are checked, so that a later configuration
    cannot quietly be mistaken for the one that was measured.
    """
    recorded = _load(EVIDENCE / name)[field]
    assert recorded == lineage["anchor_sha256"], (
        f"{name}:{field} no longer cites the anchor configuration. If the "
        "evidence was genuinely regenerated, move the anchor in the lineage "
        "too; if it was not, restore the recorded hash."
    )


def test_a_superseded_configuration_says_whether_outputs_still_hold(lineage):
    """Any entry after the anchor has to answer the question that matters."""
    for entry in lineage["lineage"]:
        if entry.get("produced_the_published_results"):
            continue
        assert "outputs_unchanged_vs_anchor" in entry, (
            f"{entry['sha256'][:12]}: say whether this configuration still "
            "reproduces the published outputs"
        )
        if entry["outputs_unchanged_vs_anchor"]:
            assert entry.get("why") and entry.get("verified"), (
                f"{entry['sha256'][:12]}: claiming the outputs are unchanged "
                "requires the reason and what was actually checked"
            )
