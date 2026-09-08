"""Print a configuration scaffold from dataclass fields and frozen grouping values.

The maintained `plecta/parameters.yaml` includes additional documentation and
stage-specific values. Compare the scaffold with that file before applying
changes; the scaffold is not a replacement for the runtime configuration.

Output goes to stdout unless `--out` is supplied. The layout completeness
check rejects dataclass fields that have not been assigned to a group.
`grouping_2d` values come from `plecta/params.json`; other sections use their
dataclass defaults.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plecta.depth import DepthParams          # noqa: E402
from plecta.image.bundles import SampleParams  # noqa: E402
from plecta.image.measurement import CutParams  # noqa: E402
from plecta.image.refine import RefineParams   # noqa: E402
from plecta.joint import JointParams           # noqa: E402
from plecta.linking import Params              # noqa: E402

OUT = REPO / "plecta" / "parameters.yaml"
FROZEN = REPO / "plecta" / "params.json"

# section -> (dataclass, {group: [field, ...]}, group comments)
LAYOUT = {
    "grouping_2d": (Params, {
        "topology": ["spur_px", "bridge_px", "absorb_free_px", "join_px"],
        "tangent_windows": ["window_local", "window_chain", "min_quadratic"],
        "solver": ["n_rounds", "anneal_start"],
        "junction_cost": ["j_w_direct", "j_w_turn", "j_chord_floor",
                          "j_w_kappa", "j_unmatched", "j_gap_relief"],
        "gap_cost": ["gap_max_len", "g_w_direct", "g_w_turn", "g_w_len",
                     "g_len_scale", "g_chord_floor", "g_w_kappa",
                     "g_unmatched", "gap_max_theta", "gap_max_phi"],
        "shared": ["kappa_scale"],
    }),
    "depth_3d": (DepthParams, {
        "crossing_detection": ["merge_px", "window_px", "core_px",
                               "core_mode", "min_flank_px"],
        "evidence_weights": ["scoring", "w_intensity", "w_sharpness",
                             "abstain_band", "abstain_score",
                             "use_image_evidence"],
        "solver": ["exact_max_nodes", "undecided_order"],
        "geometry": ["radius_mode", "radius_fallback", "default_radius_px",
                     "z_gap_px", "clear_grazing_overlaps",
                     "grazing_evidence"],
    }),
    "joint_greyscale": (JointParams, {
        "appearance": ["lambda_appearance", "margin_threshold"],
        "sampling": ["skip_px", "window_px", "n_samples"],
    }),
    "width_sampling": (SampleParams, {
        "cuts": ["cut_step", "node_clear_px", "tangent_sigma", "join_px",
                 "min_cuts"],
        "bootstrap": ["block", "n_boot", "boot_min_cuts"],
        "quality": ["scene_width_cap", "weak_min_cuts", "weak_max_mad"],
    }),
    "width_profile": (CutParams, {
        "extent": ["half_len", "min_reach", "step", "recentre_px"],
        "acceptance": ["min_contrast_sigma", "min_contrast_abs", "min_width",
                       "max_width", "rise_tol", "require_bg_confident"],
    }),
    "rendering": (RefineParams, {
        "geometry": ["smooth_window", "resample_px", "min_width_px",
                     "max_width_px"],
        "absorption": ["absorb_thr", "absorb_min_px"],
        "drawing": ["draw_bridges", "keep_node_pixels", "clip_to_mask",
                    "bridge_dilate_px", "taper", "taper_smooth_px"],
    }),
}

HEADER = """\
# PLECTA tunable parameters -- the single source of truth at run time.
#
# Every adjustable number in the package is here. Nothing in the code carries a
# tuned value of its own: the dataclass defaults still exist so the classes are
# constructible in isolation, but `plecta.parameters` overwrites them from this
# file on every load.
#
# `grouping_2d` is FROZEN. Those values produced every published number, and
# `plecta/params.json` is retained beside this file as the immutable record of
# them. `plecta.parameters.verify_against_frozen()` asserts the two agree and
# runs in the test suite -- if you change a value here, that check fails until
# you consciously update the frozen record too. That is deliberate.
#
# Units are pixels unless a name says otherwise; angles are radians.
"""

SECTION_DOC = {
    "grouping_2d": "The 2-D grouping core: which arms continue into which.\n"
                   "  FROZEN -- see the note above before editing.",
    "depth_3d": "The post-hoc depth stage: which of two crossing instances\n"
                "  is in front, and how the global stack is solved.",
    "joint_greyscale": "Greyscale evidence inside the 2-D junction matching\n"
                       "  -- a MEASURED NEGATIVE RESULT, kept only as a record.\n"
                       "  Measured 2026-08-20: mean F1 -0.0129 as shipped, and\n"
                       "  +0.0005 (neutral) after repairing three implementation\n"
                       "  defects; appearance AUC 0.63 pooled and ~0.53 on real\n"
                       "  SEM, against geometry's own 0.78-0.82 on the same\n"
                       "  pairs. Its pre-registered stop criterion (AUC below\n"
                       "  ~0.75) was met, and it is disabled: nothing in the\n"
                       "  shipped pipeline imports plecta/joint.py, so this\n"
                       "  section is read only by `python -m plecta.joint`.\n"
                       "  Kept as the record of a rejected mechanism. No\n"
                       "  2-D improvement is claimed. See plecta/joint.py.",
    "width_sampling": "Where perpendicular cuts are taken along an axis.",
    "width_profile": "How one perpendicular intensity profile becomes a width.",
    "rendering": "Stage 4 only: redrawing an instance at its measured width.\n"
                 "  Never affects the grouping scores.",
}

FIELD_DOC = {
    "spur_px": "dead-end hairs this short are skeleton noise",
    "bridge_px": "arms this short between two crossings are crossing debris",
    "absorb_free_px": "free-ended arms this short are crossing debris",
    "join_px": "arms this short merge their two crossings into one",
    "window_local": "arclength for the first-pass tangent fit",
    "window_chain": "arclength once arms have been chained",
    "min_quadratic": "pixels needed before a curved fit is used",
    "n_rounds": "frame/matching refinement rounds",
    "anneal_start": "first round's acceptance limit, as a fraction of the "
                    "final one (1.0 = no annealing)",
    "j_unmatched": "rad; pairings costing above 2x this are refused",
    "j_gap_relief": ">0: a stub with a good gap partner waiting is relieved",
    "gap_max_len": "longest bridge considered",
    "gap_max_theta": "rad; hard reject on tangent disagreement",
    "gap_max_phi": "rad; hard reject on turn-onto-chord",
    "abstain_band": "|p - 0.5| below this and the crossing abstains "
                    "(legacy scoring rules only)",
    "abstain_score": "|score| below this and the crossing abstains "
                     "(noise_floored only)",
    "exact_max_nodes": "exact DP up to this component size, then greedy",
    "use_image_evidence": "false: every crossing abstains, order is a tie-break",
    "clear_grazing_overlaps": "also clear pairs overlapping without crossing",
    "grazing_evidence": "instead read the image at those pairs' closest approach",
    "undecided_order": "compact | id_based",
    "scoring": "noise_floored | winsorized_linear | weight_of_evidence",
    "core_mode": "flat | radius_scaled -- constant core, or radius-derived",
    "radius_mode": "measured | fixed",
    "radius_fallback": "median | default",
    "kappa_scale": "multiplies the curvature term in geometry.link_cost",
    "absorb_thr": "0 disables; >0 merges two instances at this overlap",
}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


BANNER = """\
# ---------------------------------------------------------------------------
# SCAFFOLD ONLY -- generated by tools/make_parameters_yaml.py.
# The checked-in plecta/parameters.yaml is hand-maintained and carries per-field
# measurement prose that this generator cannot emit. DIFF, do not overwrite.
# ---------------------------------------------------------------------------
"""


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Print a field-complete parameters.yaml skeleton. The "
                    "checked-in plecta/parameters.yaml is hand-maintained; "
                    "diff against this, do not overwrite it.")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="write to PATH instead of stdout. Pointing this at "
                         "plecta/parameters.yaml DELETES its hand-written "
                         "per-field prose -- diff first.")
    args = ap.parse_args(argv)

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    lines = [BANNER + HEADER]
    for section, (cls, groups) in LAYOUT.items():
        defaults = cls()
        known = {f.name for f in fields(cls)}
        placed = {n for g in groups.values() for n in g}
        missing = known - placed
        if missing:
            raise SystemExit(
                f"{section}: {sorted(missing)} are not placed in any group. "
                "Add them to LAYOUT so no parameter is silently omitted.")
        lines.append(f"\n{section}:")
        for doc_line in SECTION_DOC[section].split("\n"):
            lines.append(f"  # {doc_line.lstrip()}" if doc_line.startswith("  ")
                         else f"  # {doc_line}")
        for group, names in groups.items():
            lines.append(f"\n  {group}:")
            for name in names:
                value = (frozen[name] if section == "grouping_2d"
                         else getattr(defaults, name))
                comment = FIELD_DOC.get(name)
                suffix = f"   # {comment}" if comment else ""
                lines.append(f"    {name}: {_fmt(value)}{suffix}")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    n = sum(len(v) for _, groups in LAYOUT.values() for v in groups.values())
    if args.out is None:
        sys.stdout.write(text)
        return 0
    dest = Path(args.out).resolve()
    if dest == OUT.resolve():
        print(f"refusing to clobber {OUT}: it is hand-maintained and carries "
              "per-field prose this generator cannot reproduce. Diff against "
              "stdout instead, then hand-edit.", file=sys.stderr)
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(f"wrote {dest}  ({n} parameters in {len(LAYOUT)} sections)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
