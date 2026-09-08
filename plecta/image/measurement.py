"""Image normalization, local background estimation, and ridge width fitting.

The per-instance diameter measurement at the end (`measure_diameters`) is the
depth stage's width input; it lives here, with the cut machinery it is built
on, rather than beside the physics that consumes it.

A full width at half maximum needs a *maximum* and a *half*, and the half is
measured above the local background -- not above zero.  Both the synthetic SEM
(``bg_level + bg_amp * low_frequency_field``) and the real crop have a
slowly-varying background that is a large fraction of a faint bundle's
amplitude, so using a single global number costs real accuracy on the faint
ones.

The background is therefore estimated as a *field*: a normalised convolution
over the pixels that do not look like filament.  That is
``blur(image * background_weight) / blur(background_weight)``, which is the
local mean of the background pixels only -- neighbouring filaments never enter
it, which is the failure mode a plain local minimum or a plain blur has.

Where a scene is so crowded that a neighbourhood contains almost no background
pixel at all (this happens at cov60), the field falls back to a global robust
background rather than reporting whatever little it found; ``bg_confident``
records where that happened so a caller can refuse those cuts. Accepted
perpendicular profiles are measured by full width at half maximum; ambiguous,
low-contrast, and border-truncated profiles carry explicit refusal reasons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..geometry import _densify, _unit_normals

if TYPE_CHECKING:                       # annotations only; no runtime coupling
    from ..depth import DepthParams, PredCrossing


@dataclass
class SemImage:
    """A grey-scale image in [0, 1] plus its estimated background field."""

    image: np.ndarray          # float32 (H, W), 0..1
    background: np.ndarray     # float32 (H, W), the local background level
    bg_confident: np.ndarray   # bool (H, W), False where the field was guessed
    noise: float               # robust sigma of the grain, in image units
    source: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape


def read_gray(path) -> np.ndarray:
    """Read an image as float32 in [0, 1].

    ``skimage.io`` is used rather than PIL so that this matches how the rest of
    the project reads its images (``plecta.graph.read_mask``).
    """
    from skimage import io as skio

    img = skio.imread(str(path))
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float32)
    if img.max() > 1.5:            # 8- or 16-bit integer image
        img = img / (255.0 if img.max() <= 255 else float(img.max()))
    return np.clip(img, 0.0, 1.0)


def robust_sigma(values: np.ndarray) -> float:
    """MAD-based sigma; immune to the filaments sitting in the upper tail."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2:
        return float("nan")
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def background_field(image: np.ndarray, sigma: float = 24.0,
                     fg_frac_of_range: float = 0.20,
                     dilate_px: int = 3,
                     min_weight: float = 0.02):
    """Local background level, estimated from background pixels only.

    ``fg_frac_of_range`` sets how aggressively a pixel is called "filament" and
    therefore excluded: a pixel is foreground if it lies more than that fraction
    of the way from the 20th to the 99th percentile of the image.  It is
    deliberately low -- excluding a bundle's dim shoulder from the background
    estimate matters much more than excluding a few extra background pixels.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    image = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(image, 20))
    hi = float(np.percentile(image, 99))
    thr = lo + fg_frac_of_range * max(1e-6, hi - lo)
    fg = np.asarray(image > thr, dtype=bool)
    if dilate_px > 0:
        fg = np.asarray(binary_dilation(fg, iterations=int(dilate_px)),
                        dtype=bool)

    weight = (~fg).astype(np.float32)
    num = gaussian_filter(image * weight, sigma=sigma, mode="reflect")
    den = gaussian_filter(weight, sigma=sigma, mode="reflect")
    confident = den >= min_weight
    fallback = float(np.median(image[~fg])) if (~fg).any() else lo
    field = np.where(confident, num / np.maximum(den, 1e-6), fallback)

    bg_pixels = image[~fg]
    noise = robust_sigma(bg_pixels) if bg_pixels.size >= 8 else float("nan")
    return field.astype(np.float32), confident, float(noise)


def load_sem(path, sigma: float = 24.0) -> SemImage:
    image = read_gray(path)
    field, confident, noise = background_field(image, sigma=sigma)
    return SemImage(image=image, background=field, bg_confident=confident,
                    noise=noise, source=str(Path(path)))


@dataclass
class CutParams:
    """Rules for accepting a perpendicular image-profile measurement."""

    half_len: float = 22.0
    min_reach: float = 10.0
    step: float = 0.25
    recentre_px: float = 3.0
    min_contrast_sigma: float = 4.0
    min_contrast_abs: float = 0.05
    min_width: float = 1.5
    max_width: float = 36.0
    rise_tol: float = 0.25
    require_bg_confident: bool = True


@dataclass
class CutResult:
    ok: bool
    width: float = float("nan")
    amplitude: float = float("nan")
    peak: float = float("nan")
    background: float = float("nan")
    offset: float = float("nan")
    reason: str = ""


def reach(centre: np.ndarray, normal: np.ndarray, shape: Tuple[int, int],
          want: float, margin: float = 1.0) -> Tuple[float, float]:
    """Return the available distance along each side of an image cut."""
    limits = []
    for axis in (0, 1):
        p = float(centre[axis])
        d = float(normal[axis])
        lo, hi = margin, shape[axis] - 1.0 - margin
        if abs(d) < 1e-9:
            limits.append((want, want) if lo <= p <= hi else (0.0, 0.0))
            continue
        t_lo = (lo - p) / d
        t_hi = (hi - p) / d
        forward = max(t_lo, t_hi)
        backward = -min(t_lo, t_hi)
        limits.append((max(0.0, min(want, backward)),
                       max(0.0, min(want, forward))))
    back = min(limits[0][0], limits[1][0])
    fwd = min(limits[0][1], limits[1][1])
    return back, fwd


def sample_line(image: np.ndarray, centre: np.ndarray, normal: np.ndarray,
                t_lo: float, t_hi: float, step: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample ``image`` along ``centre + t * normal``."""
    from scipy.ndimage import map_coordinates

    t = np.arange(-t_lo, t_hi + 0.5 * step, step, dtype=np.float64)
    rows = centre[0] + normal[0] * t
    cols = centre[1] + normal[1] * t
    vals = map_coordinates(image, np.vstack([rows, cols]), order=1,
                           mode="nearest", prefilter=False)
    return t, vals.astype(np.float64)


def _peak_near(values: np.ndarray, index: int, window: int) -> int:
    """Index of the largest sample within +/- ``window`` of ``index``."""
    lo = max(0, index - window)
    hi = min(len(values), index + window + 1)
    return lo + int(np.argmax(values[lo:hi]))


def _cross_outward(t: np.ndarray, values: np.ndarray, start: int,
                   level: float, direction: int, amplitude: float,
                   rise_tol: float) -> Tuple[Optional[float], str]:
    """Walk outward from a ridge crest to its interpolated half-height edge."""
    if values[start] <= level:
        return None, "start_below_half"
    running_min = values[start]
    i = start
    while True:
        j = i + direction
        if j < 0 or j >= len(values):
            reason = "no_return_left" if direction < 0 else "no_return_right"
            return None, reason
        if values[j] < running_min:
            running_min = values[j]
        elif (values[j] > running_min + rise_tol * amplitude
              and values[j] > level):
            reason = "neighbour_left" if direction < 0 else "neighbour_right"
            return None, reason
        if values[j] <= level:
            hi, lo = values[i], values[j]
            fraction = 0.0 if hi == lo else (hi - level) / (hi - lo)
            return float(t[i] + fraction * (t[j] - t[i])), ""
        i = j


def measure_cut(image: np.ndarray, background: float, noise: float,
                centre: np.ndarray, normal: np.ndarray,
                params: Optional[CutParams] = None,
                bg_confident: bool = True) -> CutResult:
    """Measure a ridge's full width at half maximum on one image cut."""
    params = params or CutParams()
    back, fwd = reach(centre, normal, image.shape, params.half_len)
    if min(back, fwd) < params.min_reach:
        return CutResult(False, reason="outside_image")
    if params.require_bg_confident and not bg_confident:
        return CutResult(False, reason="background_not_confident")

    t, values = sample_line(image, centre, normal, back, fwd, params.step)
    window = max(1, int(round(params.recentre_px / params.step)))
    i_peak = _peak_near(values, int(np.argmin(np.abs(t))), window)

    peak = float(values[i_peak])
    amplitude = peak - background
    floor = max(
        params.min_contrast_abs,
        params.min_contrast_sigma * (noise if math.isfinite(noise) else 0.0),
    )
    if amplitude < floor:
        return CutResult(False, amplitude=amplitude, peak=peak,
                         background=background, reason="low_contrast")

    level = background + 0.5 * amplitude
    left, reason = _cross_outward(t, values, i_peak, level, -1, amplitude,
                                  params.rise_tol)
    if left is None:
        return CutResult(False, amplitude=amplitude, peak=peak,
                         background=background, reason=reason)
    right, reason = _cross_outward(t, values, i_peak, level, 1, amplitude,
                                   params.rise_tol)
    if right is None:
        return CutResult(False, amplitude=amplitude, peak=peak,
                         background=background, reason=reason)

    # Re-centre on the two crossings. This avoids choosing a noisy point on a
    # flat-topped ridge as its physical centre.
    midpoint = 0.5 * (left + right)
    i_peak2 = _peak_near(values, int(np.argmin(np.abs(t - midpoint))), window)
    peak2 = float(values[i_peak2])
    amplitude2 = peak2 - background
    if amplitude2 >= floor:
        level2 = background + 0.5 * amplitude2
        left2, _ = _cross_outward(t, values, i_peak2, level2, -1,
                                   amplitude2, params.rise_tol)
        right2, _ = _cross_outward(t, values, i_peak2, level2, 1,
                                    amplitude2, params.rise_tol)
        if left2 is not None and right2 is not None:
            left, right, peak, amplitude = left2, right2, peak2, amplitude2

    offset = 0.5 * (left + right)
    if abs(offset) > params.recentre_px:
        return CutResult(False, amplitude=amplitude, peak=peak,
                         background=background, offset=offset,
                         reason="crest_off_axis")

    width = right - left
    if not params.min_width <= width <= params.max_width:
        return CutResult(False, width=width, amplitude=amplitude, peak=peak,
                         background=background, offset=offset,
                         reason="width_out_of_range")

    return CutResult(True, width=float(width), amplitude=float(amplitude),
                     peak=float(peak), background=float(background),
                     offset=float(offset), reason="ok")


# ── per-instance diameters, for the depth stage ───────────────────────────


def measure_diameters(sem, centrelines: Dict[int, np.ndarray],
                      crossings: List["PredCrossing"],
                      params: "DepthParams",
                      cut_step: float = 3.0,
                      clear_margin_px: float = 4.0,
                      min_cuts: int = 5) -> Dict[int, dict]:
    """Observed FWHM width per instance, cuts kept clear of crossings.

    Reuses the existing profile machinery
    (`plecta.image.measurement.measure_cut`): perpendicular cuts every `cut_step` px of
    arclength, refused near any crossing of that instance (a cut through a
    crossing measures two rods stacked and reports the union). Returns per
    instance: `w_obs` (median FWHM, px), `w_mad`, `n_cuts`, `quality`.
    `w_obs` is an OBSERVED width -- physical diameter only after the
    domain's broadening correction (`d` stays None here).
    """
    from plecta.image.measurement import CutParams, measure_cut

    cp = CutParams()
    out: Dict[int, dict] = {}
    H, W = sem.image.shape
    cross_pts: Dict[int, List[Tuple[float, float, float]]] = {}
    excl = (1.1 * (params.default_radius_px + params.default_radius_px)
            + clear_margin_px)
    for c in crossings:
        cross_pts.setdefault(c.i, []).append((c.x, c.y, excl))
        cross_pts.setdefault(c.j, []).append((c.x, c.y, excl))
    for iid, poly in centrelines.items():
        dense = _densify(np.asarray(poly), step=cut_step)
        if len(dense) < 3:
            out[iid] = {"w_obs": None, "w_mad": None, "n_cuts": 0,
                        "quality": "none"}
            continue
        normals_xy = _unit_normals(dense)
        widths = []
        for (x, y), (nx, ny) in zip(dense, normals_xy):
            near = False
            for cx, cy, excl in cross_pts.get(iid, []):
                if (x - cx) ** 2 + (y - cy) ** 2 < excl * excl:
                    near = True
                    break
            if near:
                continue
            rr, cc = int(round(y)), int(round(x))
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            res = measure_cut(sem.image,
                              float(sem.background[rr, cc]),
                              float(sem.noise),
                              centre=np.array([y, x]),        # (row, col)
                              normal=np.array([ny, nx]),
                              params=cp,
                              bg_confident=bool(sem.bg_confident[rr, cc]))
            if res.ok:
                widths.append(res.width)
        if len(widths) >= min_cuts:
            w = np.asarray(widths)
            med = float(np.median(w))
            mad = float(1.4826 * np.median(np.abs(w - med)))
            out[iid] = {"w_obs": med, "w_mad": mad, "n_cuts": len(widths),
                        "quality": "good" if len(widths) >= 12 and
                        mad < 0.25 * med else "weak"}
        else:
            out[iid] = {"w_obs": None, "w_mad": None, "n_cuts": len(widths),
                        "quality": "none"}
    return out


#: Observed-width -> physical-diameter correction, fitted on the depth
#: development nucleus (12 microscopy-domain scenes, 340 instances,
#: 2026-08-18) as d = W2D_SLOPE * w_obs + W2D_OFFSET on oracle centrelines.
#: Residual RMS 1.44 px on 7-16 px diameters; residual bias vs blur is
#: -0.93 px for sigma > 2 px (FWHM broadening not separately inverted --
#: profile shape, not the PSF, dominates the correction on this domain).
#: The raw w_obs is always stored alongside so the correction stays
#: auditable, and the correction is DOMAIN calibration: under a shifted
#: rendering law it is expected to degrade, and that is reported, not hidden.
W2D_SLOPE = 1.176
W2D_OFFSET = 2.559


def width_to_diameter(w_obs: Optional[float]) -> Optional[float]:
    if w_obs is None or not np.isfinite(w_obs):
        return None
    return max(0.5, W2D_SLOPE * float(w_obs) + W2D_OFFSET)
