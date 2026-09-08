"""Draw the answer on top of the SEM so a human can disagree with it.

Three panels, same geometry, side by side:

1. the SEM as it is, so the eye has an unannotated reference;
2. instances over the SEM, coloured by fitted width on a shared scale, with the
   cuts that were actually used marked -- if the colour of a filament does not
   match how wide it looks, that is visible immediately;
3. the width profile along every instance, which is where a wrong merge shows
   up as a step rather than a drift.

Matplotlib only; no seaborn, no style sheet, nothing that has to be installed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _colour_for(value: float, lo: float, hi: float):
    import matplotlib.cm as cm

    if not math.isfinite(value):
        return (0.55, 0.55, 0.55, 1.0)
    t = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    return cm.viridis(float(np.clip(t, 0.0, 1.0)))


def overlay_figure(res, out_path: Path, title: str = "",
                   label_top: int = 12, dpi: int = 130) -> Path:
    """Write the three-panel overlay for one measured scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    widths = [b.width for b in res.bundles if b.ok]
    if widths:
        lo = float(np.percentile(widths, 5))
        hi = float(np.percentile(widths, 95))
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
    else:
        lo, hi = 0.0, 1.0

    img = res.sem.image
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.0))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("SEM (unannotated)", fontsize=11)

    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    rgba = np.zeros((*res.shape, 4), dtype=float)
    for b in res.bundles:
        m = res.masks[b.iid]
        rgba[m] = _colour_for(b.width, lo, hi)
    alpha = np.zeros(res.shape, dtype=float)
    for b in res.bundles:
        alpha[res.masks[b.iid]] = 0.95
    rgba[..., 3] = alpha
    axes[1].imshow(rgba, interpolation="nearest")

    # the cuts that produced the numbers, and a label on the longest instances
    for b in res.bundles:
        if not b.ok or not b.cut_rc:
            continue
        pts = np.asarray(b.cut_rc)
        axes[1].plot(pts[:, 1], pts[:, 0], ".", ms=0.7, color="white", alpha=0.35)
    ranked = sorted((b for b in res.bundles if b.ok),
                    key=lambda b: -b.axis_length)[:label_top]
    for b in ranked:
        if not b.cut_rc:
            continue
        r, c = b.cut_rc[len(b.cut_rc) // 2]
        half = 0.5 * (b.width_unc_hi - b.width_unc_lo) if math.isfinite(b.width_unc_hi) else float("nan")
        txt = (f"{b.width:.1f}±{half:.1f}" if math.isfinite(half)
               else f"{b.width:.1f}")
        axes[1].annotate(txt, (c, r), color="white", fontsize=7.0,
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15", fc="black",
                                   ec="none", alpha=0.55))
    axes[1].set_title(f"{res.n_instances} instances, coloured by fitted width "
                      f"(white dots = cuts used)", fontsize=11)

    for b in res.bundles:
        if not b.ok or len(b.cut_s) < 3:
            continue
        axes[2].plot(b.cut_s, b.cut_width, "-", lw=0.8,
                     color=_colour_for(b.width, lo, hi), alpha=0.85)
    axes[2].set_xlabel("arclength along the instance (px)")
    axes[2].set_ylabel("fitted width (px)")
    axes[2].set_title("width profile along each instance", fontsize=11)
    axes[2].grid(alpha=0.25, lw=0.5)
    if widths:
        axes[2].set_ylim(max(0.0, lo - 4.0), hi + 6.0)

    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap="viridis")
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("fitted width (px)", fontsize=9)

    n_ok = sum(1 for b in res.bundles if b.ok)
    n_good = sum(1 for b in res.bundles if b.quality == "good")
    fig.suptitle(
        f"{title}   |   {n_ok}/{res.n_instances} instances measured "
        f"({n_good} good), "
        f"{sum(b.n_cuts_valid for b in res.bundles)} valid cuts of "
        f"{sum(b.n_cuts_attempted for b in res.bundles)}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def width_table_figure(res, out_path: Path, title: str = "",
                       truth: Optional[Dict[int, float]] = None,
                       dpi: int = 130) -> Path:
    """Per-instance widths with their uncertainty intervals, sorted by width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = sorted((b for b in res.bundles if b.ok), key=lambda b: b.width)
    if not ok:
        raise RuntimeError("no measured instance to plot")
    y = np.arange(len(ok))
    w = np.array([b.width for b in ok])
    lo = np.array([b.width - b.width_unc_lo if math.isfinite(b.width_unc_lo)
                   else 0.0 for b in ok])
    hi = np.array([b.width_unc_hi - b.width if math.isfinite(b.width_unc_hi)
                   else 0.0 for b in ok])
    colours = ["#1a7f37" if b.quality == "good" else "#b26a00" for b in ok]

    fig, ax = plt.subplots(figsize=(8.4, max(4.0, 0.20 * len(ok) + 1.6)))
    ax.errorbar(w, y, xerr=np.vstack([lo, hi]), fmt="none", ecolor="#888",
                elinewidth=1.0, capsize=2.0, zorder=1)
    ax.scatter(w, y, c=colours, s=18, zorder=2)
    if truth:
        tv = [truth.get(b.iid, float("nan")) for b in ok]
        ax.scatter(tv, y, marker="|", s=70, c="crimson", zorder=3,
                   label="true width")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{b.iid} ({b.n_cuts_valid} cuts)" for b in ok],
                       fontsize=6.5)
    ax.set_xlabel("fitted width (px), with the 90% interval")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.text(0.99, 0.005, "green = good, orange = weak", ha="right",
             fontsize=8, color="#555")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
