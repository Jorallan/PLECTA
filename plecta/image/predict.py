"""Measure image-derived width and brightness for PLECTA instances.

    plecta-image --scene <dir> --out <dir>

The fixed mask-only method determines grouping. Image intensities are used only
for downstream measurements. Ribbon rendering is opt-in with ``--refine``.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plecta.predict import load_params, save_multilabel_npz
from .overlay import overlay_figure, width_table_figure
from .pipeline import measure_scene


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="a scene folder")
    ap.add_argument("--mask-name", default="mask_w1.png")
    ap.add_argument("--sem-name", default="sem.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refine", action="store_true",
                    help="stage 4: emit one smooth ribbon per instance at its "
                         "fitted width instead of raw skeleton pixels plus "
                         "crossing clusters")
    ap.add_argument("--absorb-thr", type=float, default=0.0,
                    help="with --refine: merge instances whose rendered masks "
                         "overlap by this fraction of the smaller (0 = off; "
                         "values above 0 change grouping)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--verify-core", action="store_true",
                    help="report that instance-aligned chains came from the "
                         "same core prediction")
    args = ap.parse_args(argv)

    scene = Path(args.scene)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = load_params()
    res = measure_scene(scene, mask_name=args.mask_name, sem_name=args.sem_name,
                        params=params)
    masks = res.masks
    bundles = res.bundles

    if args.verify_core:
        print("--verify-core: instance-aligned chains supplied by core prediction.")

    refine_log = None
    if args.refine:
        from .refine import RefineParams, refine_scene

        masks, _poly, refine_log = refine_scene(
            res, RefineParams(absorb_thr=args.absorb_thr))
        print(f"--refine: {len(masks)} instances re-rendered as ribbons"
              + (f", {len(refine_log)} absorbed" if refine_log else ""))

    save_multilabel_npz(out / "pred_multilabel.npz", masks, res.shape)
    cols = ["iid", "quality", "width", "width_unc_lo", "width_unc_hi",
            "width_mad", "sigma_noise", "sigma_real", "width_p16", "width_p84",
            "width_se", "width_se_noise", "n_eff", "brightness",
            "brightness_p90", "brightness_mad", "n_cuts_valid",
            "n_cuts_attempted", "axis_length", "n_arms", "n_pixels",
            "step_abs", "step_rel", "step_at", "excursion_z",
            "width_ci_lo", "width_ci_hi"]
    with (out / "bundle_widths.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for b in bundles:
            w.writerow(b.row())

    ok = [b for b in bundles if b.ok]
    widths = np.array([b.width for b in ok]) if ok else np.zeros(0)
    meta = {
        "scene": str(scene), "mask_name": args.mask_name,
        "refine": bool(args.refine),
        "absorb_thr": float(args.absorb_thr) if args.refine else None,
        "n_absorbed": len(refine_log) if refine_log else 0,
        "sem_name": args.sem_name,
        "n_instances": len(masks), "n_measured": len(ok),
        "n_good": sum(1 for b in ok if b.quality == "good"),
        "cut_width_cap_px": res.cut_width_cap,
        "width_median_px": float(np.median(widths)) if widths.size else None,
        "seconds_stage1": round(res.seconds_stage1, 2),
        "seconds_stage2": round(res.seconds_stage2, 2),
        "params": asdict(params),
    }
    (out / "characterization_meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")

    if not args.no_figures:
        r = SimpleNamespace(sem=res.sem, masks=masks, bundles=bundles,
                            shape=res.shape, n_instances=len(masks))
        overlay_figure(r, out / "overlay.png", title=f"{scene.name} ({args.mask_name})")
        width_table_figure(r, out / "widths.png",
                           title=f"{scene.name}: width per instance")

    if widths.size:
        print(f"{len(masks)} instances, {len(ok)} measured "
              f"({meta['n_good']} good), median width "
              f"{meta['width_median_px']:.2f} px")
    else:
        print(f"{len(masks)} instances, none measured")
    print(f"wrote {out}/pred_multilabel.npz, bundle_widths.csv, "
          "characterization_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
