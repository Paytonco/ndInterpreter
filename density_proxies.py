"""Cell-density proxies for the phase contrast inside a trap.

``nd2_workflow.py`` measures one of these per trap and timepoint, chosen with
``--density`` (default ``glcm_asm_flipped``):

  area_fraction     fraction of trap pixels above one global Otsu threshold
  std               standard deviation
  gradient          mean gradient magnitude (Sobel)
  local_variance    mean local variance (circular window of radius ~d/2)
  entropy           Shannon entropy of the intensity histogram (fixed range, bins)
  glcm_contrast     GLCM contrast (displacement ~d)
  glcm_entropy      GLCM entropy
  glcm_asm          GLCM angular second moment (falls as the trap fills)
  glcm_asm_flipped  1 - GLCM angular second moment (rises as the trap fills)

``d`` is the cell diameter (``CELL_DIAMETER_UM``). Every method works on
``B``, the (bandpassed) phase frame, inside the trap mask ``M``. Anything that
must be fixed across frames (the threshold, the intensity range) is computed
once from pixels pooled over many frames and traps (``intensity_settings``),
so frames and traps stay comparable.

``extract_trap_crops`` and ``all_proxies`` compute every proxy for one
position at once, for comparing them side by side. Reading the nd2 is the slow
part, so the crops are pickled as ``objects/<stem>_pos<#>_trap_crops.pkl`` and
reused on later runs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

# Typical cell diameter (um): E. coli is ~1 um wide. Sets the local variance
# window and the GLCM displacement.
CELL_DIAMETER_UM = 1.0
# Pixels kept around each trap's bounding box, so gradients and local windows
# at the mask border are computed from real image rather than a cut edge.
CROP_MARGIN_PX = 8
# Percentiles of the pooled masked pixels that fix the intensity range for
# the entropy histogram and the GLCM quantization.
RANGE_PERCENTILES = (0.1, 99.9)
ENTROPY_BINS = 256
GLCM_LEVELS = 32
GLCM_ANGLES_DEG = (0, 45, 90, 135)

PROXY_LABELS = {
    "area_fraction": "Area fraction (Otsu)",
    "std": "Standard deviation",
    "gradient": "Mean gradient magnitude",
    "local_variance": "Mean local variance",
    "entropy": "Shannon entropy (bits)",
    "glcm_contrast": "GLCM contrast",
    "glcm_entropy": "GLCM entropy (bits)",
    "glcm_asm": "GLCM angular second moment",
    "glcm_asm_flipped": "1 - GLCM angular second moment",
}
DEFAULT_PROXY = "glcm_asm_flipped"
# Proxies that need the series-wide threshold / intensity range.
NEEDS_INTENSITY = {"area_fraction", "entropy",
                   "glcm_contrast", "glcm_entropy", "glcm_asm", "glcm_asm_flipped"}


# --------------------------------------------------------------------------- #
# Reading one position's traps
# --------------------------------------------------------------------------- #

def crops_path(objects_dir: Path, stem: str, scope_number: int) -> Path:
    return Path(objects_dir) / f"{stem}_pos{scope_number}_trap_crops.pkl"


def trap_box(mask, margin=CROP_MARGIN_PX):
    """Bounding box of ``mask`` grown by ``margin`` pixels (clipped to the image)."""
    rows, cols = np.nonzero(mask)
    return (slice(max(rows.min() - margin, 0), rows.max() + 1 + margin),
            slice(max(cols.min() - margin, 0), cols.max() + 1 + margin))


def extract_trap_crops(nd2_path, traps_path, scope_number=1, objects_dir=None,
                       bandpass=True, refresh=False) -> dict:
    """Bandpassed phase crops around every trap of one position, over time.

    Returns ``{"Time": times, "pixel_um": um, "position": name, "traps":
    {trap_key: {"stack": (T, h, w) float32, "mask": (h, w) bool}}}``. Each
    crop is the trap's bounding box plus ``CROP_MARGIN_PX`` on every side;
    ``mask`` marks the trap's own pixels (same rasterization as the workflow).
    Cached in ``objects_dir`` (default ``objects/``); pass ``refresh=True`` to
    read the nd2 again.
    """
    import nd2_workflow as w

    nd2_path = Path(nd2_path)
    cache = crops_path(objects_dir or w.OBJECTS_DIR, nd2_path.stem, scope_number)
    if cache.exists() and not refresh:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    with open(traps_path, "rb") as fh:
        saved = pickle.load(fh)
    position = next((p for p in saved["positions"] if p["scope_number"] == scope_number), None)
    if position is None:
        raise ValueError(f"{traps_path} has no traps for position #{scope_number}.")

    with w.open_nd2(nd2_path) as f:
        channels = w.classify_channels(f)
        phase = [c for c in channels if c["role"] == "phase"]
        info = w.describe_file(f)
        shape = (info["sizes"]["Y"], info["sizes"]["X"])
        pixel_um = (info["voxel_um"]["x"], info["voxel_um"]["y"])
        times, _ = w.time_axis(f)
        seqs = w.frame_sequence_map(f)
        filt = w.SpatialBandpass(shape, pixel_um) if bandpass else None

        boxes = {}
        for trap in position["traps"]:
            mask = w.trap_mask(shape, [trap])
            box = trap_box(mask)
            boxes[w.trap_key(position, trap)] = (box, mask[box])
        stacks = {key: np.empty((len(times),) + m.shape, np.float32)
                  for key, (_, m) in boxes.items()}

        with w.ProgressBar(len(times), f"reading position #{scope_number} phase") as bar:
            for t in range(len(times)):
                frame = w.read_channels(f, seqs, t, position["index"], phase)[w.PHASE_KEY]
                frame = filt(frame) if filt is not None else frame.astype(np.float32)
                for key, (box, _) in boxes.items():
                    stacks[key][t] = frame[box]
                bar.update()

    crops = {"Time": np.asarray(times), "pixel_um": float(max(pixel_um)),
             "position": position["name"], "bandpass": filt.describe() if filt else None,
             "traps": {key: {"stack": stacks[key], "mask": boxes[key][1]} for key in boxes}}
    w.save_pickle(crops, cache)
    return crops


# --------------------------------------------------------------------------- #
# Global (series-wide) settings
# --------------------------------------------------------------------------- #

def pooled_pixels(traps) -> np.ndarray:
    """Every masked pixel of every frame of every trap in ``traps``, flattened."""
    return np.concatenate([t["stack"][:, t["mask"]].ravel() for t in traps.values()])


def global_threshold(pooled, method="otsu") -> float:
    """One threshold for the whole series, from the pooled histogram."""
    from skimage.filters import threshold_otsu, threshold_triangle

    funcs = {"otsu": threshold_otsu, "triangle": threshold_triangle}
    return float(funcs[method](pooled, nbins=ENTROPY_BINS))


def global_range(pooled, percentiles=RANGE_PERCENTILES) -> tuple[float, float]:
    """Intensity range shared by the entropy histogram and GLCM quantization."""
    lo, hi = np.percentile(pooled, percentiles)
    return float(lo), float(hi)


def diameter_px(pixel_um, diameter_um=CELL_DIAMETER_UM) -> float:
    return diameter_um / pixel_um


def scale_settings(pixel_um, diameter_um=CELL_DIAMETER_UM, levels=GLCM_LEVELS) -> dict:
    """Window radius and GLCM displacement for cells ``diameter_um`` across."""
    d_px = diameter_px(pixel_um, diameter_um)
    return {
        "diameter_um": diameter_um,
        "diameter_px": d_px,
        "local_variance_radius_px": max(round(0.5 * d_px), 1),
        "glcm_distance_px": max(d_px, 1.0),
        "glcm_levels": levels,
        "glcm_offsets": glcm_offsets(max(d_px, 1.0)),
    }


def intensity_settings(pooled, threshold_method="otsu") -> dict:
    """Threshold and intensity range fixed from ``pooled`` masked pixels."""
    return {
        "threshold_method": threshold_method,
        "threshold": global_threshold(pooled, threshold_method),
        "value_range": global_range(pooled),
    }


# --------------------------------------------------------------------------- #
# The proxies. Each takes an (h, w) image and an (h, w) mask and returns one
# value. Image-wide operations (gradients, local windows) run on the whole
# image before masking, so pixels at the mask border see the real image next
# to them, not a cut edge.
# --------------------------------------------------------------------------- #

def frame_area_fraction(image, mask, threshold) -> float:
    """Fraction of masked pixels above the global ``threshold``."""
    return float((image[mask] > threshold).mean())


def frame_std(image, mask) -> float:
    """Standard deviation of B over M."""
    return float(np.std(image[mask], dtype=np.float64))


def frame_gradient(image, mask) -> float:
    """Mean Sobel gradient magnitude over M."""
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    return float(np.hypot(gx, gy)[mask].mean())


def disk(radius_px) -> np.ndarray:
    r = max(round(radius_px), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2 <= r ** 2).astype(np.float64)


def frame_local_variance(image, mask, radius_px) -> float:
    """Mean over M of the variance in a circular window of ``radius_px``.

    The radius is rounded to whole pixels, minimum 1.
    """
    kernel = disk(radius_px)
    kernel /= kernel.sum()
    image = image.astype(np.float64)
    mean = ndimage.convolve(image, kernel, mode="reflect")
    mean_sq = ndimage.convolve(image ** 2, kernel, mode="reflect")
    return float(np.maximum(mean_sq - mean ** 2, 0)[mask].mean())


def quantize(values, value_range, levels) -> np.ndarray:
    """Map ``value_range`` onto integer levels 0..levels-1, clipping outside it."""
    lo, hi = value_range
    q = np.floor((values - lo) / (hi - lo) * levels)
    return np.clip(q, 0, levels - 1).astype(np.intp)


def frame_entropy(image, mask, value_range, bins=ENTROPY_BINS) -> float:
    """Entropy (bits) of the masked-pixel histogram on a fixed range/bins."""
    counts = np.bincount(quantize(image[mask], value_range, bins), minlength=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def glcm_offsets(distance_px, angles_deg=GLCM_ANGLES_DEG):
    """(row, col) displacement per angle, rounded to whole pixels as skimage does."""
    offsets = []
    for angle in np.deg2rad(angles_deg):
        dr = round(-np.sin(angle) * distance_px)
        dc = round(np.cos(angle) * distance_px)
        offsets.append((dr, dc))
    return offsets


def _shifted_pairs(a, dr, dc):
    """Views ``(a[r, c], a[r + dr, c + dc])`` over every position where both exist."""
    h, w_ = a.shape
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w_, w_ - dc)
    return a[r0:r1, c0:c1], a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]


def glcm(image, mask, value_range, levels, offsets) -> np.ndarray:
    """Symmetric, normalized co-occurrence matrix averaged over ``offsets``.

    Only pairs with both pixels inside ``mask`` are counted.
    """
    q = quantize(image, value_range, levels)
    total = np.zeros((levels, levels))
    for dr, dc in offsets:
        qa, qb = _shifted_pairs(q, dr, dc)
        ma, mb = _shifted_pairs(mask, dr, dc)
        both = ma & mb
        counts = np.bincount(qa[both] * levels + qb[both], minlength=levels ** 2)
        counts = counts.reshape(levels, levels).astype(np.float64)
        counts += counts.T
        total += counts / counts.sum()
    return total / len(offsets)


def frame_glcm_features(image, mask, value_range, offsets, levels=GLCM_LEVELS) -> dict:
    """GLCM contrast, entropy (bits) and angular second moment of one frame."""
    p = glcm(image, mask, value_range, levels, offsets)
    i, j = np.indices((levels, levels))
    nz = p[p > 0]
    return {"contrast": float(np.sum((i - j) ** 2 * p)),
            "entropy": float(-np.sum(nz * np.log2(nz))),
            "asm": float(np.sum(p ** 2))}


def measure(name, image, mask, settings) -> float:
    """Proxy ``name`` (a key of ``PROXY_LABELS``) for one frame.

    ``settings`` merges ``scale_settings`` and, for proxies in
    ``NEEDS_INTENSITY``, ``intensity_settings``.
    """
    image = np.asarray(image, dtype=np.float32)  # raw phase is unsigned ints
    if name == "area_fraction":
        return frame_area_fraction(image, mask, settings["threshold"])
    if name == "std":
        return frame_std(image, mask)
    if name == "gradient":
        return frame_gradient(image, mask)
    if name == "local_variance":
        return frame_local_variance(image, mask, settings["local_variance_radius_px"])
    if name == "entropy":
        return frame_entropy(image, mask, settings["value_range"])
    if name.startswith("glcm_"):
        g = frame_glcm_features(image, mask, settings["value_range"],
                                settings["glcm_offsets"], settings["glcm_levels"])
        if name == "glcm_asm_flipped":
            return 1.0 - g["asm"]
        return g[name.removeprefix("glcm_")]
    raise ValueError(f"Unknown density proxy {name!r}; choose from {', '.join(PROXY_LABELS)}.")


# --------------------------------------------------------------------------- #
# Whole stacks. Each takes a (T, h, w) stack and an (h, w) mask and returns
# one value per frame.
# --------------------------------------------------------------------------- #

def _per_frame(func, stack, *args) -> np.ndarray:
    return np.array([func(image, *args) for image in stack])


def area_fraction(stack, mask, threshold) -> np.ndarray:
    return (stack[:, mask] > threshold).mean(axis=1)


def std_dev(stack, mask) -> np.ndarray:
    return stack[:, mask].std(axis=1, dtype=np.float64)


def mean_gradient(stack, mask) -> np.ndarray:
    return _per_frame(frame_gradient, stack, mask)


def mean_local_variance(stack, mask, radius_px) -> np.ndarray:
    return _per_frame(frame_local_variance, stack, mask, radius_px)


def shannon_entropy(stack, mask, value_range, bins=ENTROPY_BINS) -> np.ndarray:
    return _per_frame(frame_entropy, stack, mask, value_range, bins)


def glcm_features(stack, mask, value_range, distance_px, levels=GLCM_LEVELS) -> dict:
    """GLCM contrast, entropy (bits) and angular second moment per frame."""
    offsets = glcm_offsets(distance_px)
    frames = [frame_glcm_features(image, mask, value_range, offsets, levels) for image in stack]
    return {k: np.array([fr[k] for fr in frames]) for k in ("contrast", "entropy", "asm")}


def all_proxies(crops, threshold_method="otsu",
                diameter_um=CELL_DIAMETER_UM, levels=GLCM_LEVELS) -> tuple[dict, dict]:
    """Every proxy for every trap in ``crops`` (from ``extract_trap_crops``).

    Returns ``(proxies, settings)``: ``proxies[name][trap_key]`` is an array
    over time (names as in ``PROXY_LABELS``), and ``settings`` records the
    global threshold, range, window radius and GLCM displacement used. The
    threshold and range are pooled over every frame of every trap.
    """
    traps = crops["traps"]
    settings = {**intensity_settings(pooled_pixels(traps), threshold_method),
                **scale_settings(crops["pixel_um"], diameter_um, levels)}
    proxies = {name: {} for name in PROXY_LABELS}
    for key, trap in traps.items():
        stack, mask = trap["stack"], trap["mask"]
        proxies["area_fraction"][key] = area_fraction(stack, mask, settings["threshold"])
        proxies["std"][key] = std_dev(stack, mask)
        proxies["gradient"][key] = mean_gradient(stack, mask)
        proxies["local_variance"][key] = mean_local_variance(
            stack, mask, settings["local_variance_radius_px"])
        proxies["entropy"][key] = shannon_entropy(stack, mask, settings["value_range"])
        g = glcm_features(stack, mask, settings["value_range"],
                          settings["glcm_distance_px"], levels)
        proxies["glcm_contrast"][key] = g["contrast"]
        proxies["glcm_entropy"][key] = g["entropy"]
        proxies["glcm_asm"][key] = g["asm"]
        proxies["glcm_asm_flipped"][key] = 1.0 - g["asm"]
    return proxies, settings
