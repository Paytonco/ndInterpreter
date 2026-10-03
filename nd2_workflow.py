#!/usr/bin/env python3
"""Standardized nd2 microscopy workflow.

Takes a Nikon NIS-Elements ``.nd2`` file (the format in ``data/``) from a
microfluidics experiment through the standard pipeline:

  1. open the nd2 (lazily -- pixel data is only read on demand) and report its
     axes, dtype, and pixel size
  2. identify channels: exactly one phase contrast channel (keyed ``"phase"``)
     plus up to three fluorescence channels keyed by fluorophore (``"GFP"``,
     ``"RFP"``, ...). Identification uses the microscope modality recorded in
     the file, not the channel name, since NIS lets channels be named anything
  3. list the stage positions and build the time axis in seconds, one entry
     per timepoint. Per-frame timestamps are used when they're sane, otherwise
     the nominal time-loop period
  4. read the first frame of every position/channel as a quick sanity check
  5. per position, name it and draw its traps (one or more simple polygons)
     on the first phase frame, then pickle them as ``objects/<stem>_traps.pkl``
  6. per position and fluorescence channel, take the median of the first
     frame outside all traps as the background; it is subtracted from every
     fluorescence frame of that position
  7. pickle the experiment layout as ``objects/<stem>_metadata.pkl``
  8. set up a spatial bandpass filter (FFT, order-2 Butterworth transfer
     function, passing structure between 0.35 um and 4 um by default) that
     every phase frame goes through before it reaches a video. Fluorescence
     is not filtered
  9. write three mp4s per position into ``videos/``: phase, phase + all
     fluorescence channels, and phase with the traps overlaid (for spotting
     drift). The file is read once, in on-disk order, feeding every video

The pickled dictionaries carry a top-level ``"Time"`` key: the time axis in
seconds since the run started (e.g. ``d["Time"]``).

Positions are referred to by the scope's own 1-based numbering (#1, #2, ...,
as NIS shows them); the 0-based ``"index"`` is only used inside the pickles.

Run it with no arguments to use the newest nd2 in ``data/``:

    python nd2_workflow.py
    python nd2_workflow.py /mnt/e/SOC/Scope/SOC-EXP-14/soc-exp-14.nd2

Reuse previously drawn traps (skips the UI), restrict positions, or override
phase-channel detection:

    python nd2_workflow.py --traps objects/soc-exp-14_traps.pkl
    python nd2_workflow.py --positions "1-4,7"
    python nd2_workflow.py --phase-channel "New"

Change the bandpass cutoffs, or turn it off:

    python nd2_workflow.py --bandpass-min-um 0.5 --bandpass-max-um 10
    python nd2_workflow.py --no-bandpass

Requires ``nd2``, ``dask``, ``numpy``, ``scipy``, ``scikit-image``, ``matplotlib`` (Qt backend for the
trap UI), and ``imageio-ffmpeg`` (see ``environment.yml``).
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from itertools import pairwise
from pathlib import Path

import nd2
import numpy as np
import scipy.fft

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
OBJECTS_DIR = REPO_DIR / "objects"
VIDEOS_DIR = REPO_DIR / "videos"

PHASE_KEY = "phase"
MAX_FLUOR_CHANNELS = 3

# Modality flags (as recorded by NIS) that mean transmitted light, i.e. the
# phase contrast channel. Brightfield is included because NIS often records a
# phase ring acquisition as plain "brightfield".
TRANSMITTED_MODALITIES = {"phaseContrast", "brightfield", "diContrast"}

# Fluorophore name patterns -> output key, checked against the channel name.
FLUOR_NAME_PATTERNS = [
    (r"gfp|fitc|488", "GFP"),
    (r"yfp|venus|citrine", "YFP"),
    (r"rfp|mcherry|tdtomato|mscarlet|texas|txred|tritc|cy3|561", "RFP"),
    (r"cfp|cerulean|turquoise", "CFP"),
    (r"bfp|dapi|hoechst|405", "BFP"),
    (r"cy5|far.?red|640|647", "Cy5"),
]
# Fallback when the name is unhelpful: emission wavelength (nm) bands -> key.
FLUOR_EMISSION_BANDS = [
    (400, 470, "BFP"),
    (470, 500, "CFP"),
    (500, 530, "GFP"),
    (530, 560, "YFP"),
    (560, 650, "RFP"),
    (650, 800, "Cy5"),
]

# Per-frame timestamps beyond this (seconds) are treated as corrupt.
MAX_SANE_TIME_S = 30 * 86400

# Spatial bandpass defaults: keep structure between these length scales (um).
# 0.35 um is the diffraction limit of a 20x/0.75 NA objective at GFP emission
# (520 nm / (2 * 0.75)); 4 um is a few bacterial cell lengths, so anything
# broader (uneven illumination, halos, background gradients) is removed.
DEFAULT_BANDPASS_MIN_UM = 0.35
DEFAULT_BANDPASS_MAX_UM = 4.0
# Butterworth order: 2 gives a gentle rolloff (-12 dB/octave asymptotically)
# with little ringing; higher orders get steeper and ring more.
BANDPASS_ORDER = 2
# Mirror padding added on each side before the FFT, in multiples of the
# longest passed wavelength, so the image edges don't wrap around and ring.
BANDPASS_PAD_WAVELENGTHS = 4

# Video defaults.
DEFAULT_FPS = 10
# Display color (RGB, 0-1) for each fluorescence key in the composite video.
FLUOR_COLORS = {
    "GFP": (0.0, 1.0, 0.0),
    "YFP": (1.0, 1.0, 0.0),
    "RFP": (1.0, 0.0, 1.0),
    "CFP": (0.0, 1.0, 1.0),
    "BFP": (0.3, 0.5, 1.0),
    "Cy5": (1.0, 0.5, 0.0),
}
# Phase brightness in the composite, so fluorescence stands out on top of it.
COMPOSITE_PHASE_WEIGHT = 0.5
# Contrast limits (percentiles) used to scale each channel to 8 bits.
PHASE_PERCENTILES = (0.5, 99.5)
FLUOR_PERCENTILES = (0.5, 99.9)
# Timepoints sampled per position (spread over the run) to set contrast limits.
CONTRAST_SAMPLES = 3
# Fill opacity of the trap polygons in the drift video.
TRAP_ALPHA = 0.3
# Trap colors, cycled per trap (matplotlib's tab10).
TRAP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# --------------------------------------------------------------------------- #
# 1. Open the nd2
# --------------------------------------------------------------------------- #

def open_nd2(path: Path) -> nd2.ND2File:
    """Open an nd2 for lazy reading. Close it (or use ``with``) when done."""
    return nd2.ND2File(path)


def describe_file(f: nd2.ND2File) -> dict:
    voxel = f.voxel_size()
    return {
        "sizes": dict(f.sizes),
        "dtype": str(f.dtype),
        "voxel_um": {"x": voxel.x, "y": voxel.y, "z": voxel.z},
    }


# --------------------------------------------------------------------------- #
# 2. Channel identification
# --------------------------------------------------------------------------- #

def _fluor_key(name: str, emission_nm) -> str:
    lowered = name.lower()
    for pattern, key in FLUOR_NAME_PATTERNS:
        if re.search(pattern, lowered):
            return key
    if emission_nm:
        for lo, hi, key in FLUOR_EMISSION_BANDS:
            if lo <= emission_nm < hi:
                return key
    return re.sub(r"\W+", "_", name).strip("_") or "fluor"


def _is_transmitted(name: str, modalities) -> bool:
    if "fluorescence" in modalities:
        return False
    if TRANSMITTED_MODALITIES & set(modalities):
        return True
    return bool(re.search(r"\bph(ase)?\d*\b|\bbf\b|brightfield|trans", name.lower()))


def classify_channels(f: nd2.ND2File, phase_override=None) -> list[dict]:
    """Assign every channel a role and an output key.

    Returns one dict per channel, in file order::

        {"index": 0, "name": "New", "role": "phase", "key": "phase", ...}
        {"index": 1, "name": "GFP_CRB", "role": "fluor", "key": "GFP", ...}

    ``phase_override`` (a channel name or index) forces which channel is phase.
    Exits if no single phase channel can be identified.
    """
    channels = []
    for ch in (f.metadata.channels or []):
        modalities = list(ch.microscope.modalityFlags or [])
        channels.append({
            "index": ch.channel.index,
            "name": ch.channel.name,
            "modalities": modalities,
            "emission_nm": ch.channel.emissionLambdaNm,
            "excitation_nm": ch.channel.excitationLambdaNm,
            "transmitted": _is_transmitted(ch.channel.name, modalities),
        })
    if not channels:
        sys.exit("No channel metadata in this nd2 - can't identify phase/fluorescence.")

    if phase_override is not None:
        matches = [c for c in channels
                   if c["name"] == phase_override or str(c["index"]) == str(phase_override)]
        if not matches:
            names = ", ".join(f"{c['index']}={c['name']!r}" for c in channels)
            sys.exit(f"--phase-channel {phase_override!r} matches no channel ({names}).")
        phase = matches[0]
    else:
        candidates = [c for c in channels if c["transmitted"]]
        if len(candidates) != 1:
            names = ", ".join(f"{c['index']}={c['name']!r}" for c in channels)
            found = "no" if not candidates else "multiple"
            sys.exit(f"Found {found} phase contrast channel candidates among ({names}); "
                     "pick one with --phase-channel.")
        phase = candidates[0]

    used_keys = {PHASE_KEY}
    for c in channels:
        if c is phase:
            c["role"], c["key"] = "phase", PHASE_KEY
            continue
        c["role"] = "fluor"
        key, n = _fluor_key(c["name"], c["emission_nm"]), 1
        base = key
        while key in used_keys:
            n += 1
            key = f"{base}_{n}"
        used_keys.add(key)
        c["key"] = key
    return channels


def fluor_channels(channels) -> list[dict]:
    return [c for c in channels if c["role"] == "fluor"]


# --------------------------------------------------------------------------- #
# 3. Positions and time axis
# --------------------------------------------------------------------------- #

def list_positions(f: nd2.ND2File) -> list[dict]:
    """One dict per stage position.

    ``index`` is the 0-based P index in the file and ``scope_number`` the
    1-based number NIS shows for it. ``nis_name`` is whatever name the point
    was given in NIS (often blank); ``name`` is filled in by the trap UI.
    """
    n = f.sizes.get("P", 1)
    points = []
    for loop in f.experiment:
        if loop.type == "XYPosLoop":
            points = loop.parameters.points
            break
    positions = []
    for i in range(n):
        pt = points[i] if i < len(points) else None
        stage = pt.stagePositionUm if pt else None
        positions.append({
            "index": i,
            "scope_number": i + 1,
            "nis_name": (pt.name or "") if pt else "",
            "name": None,
            "x_um": stage.x if stage else np.nan,
            "y_um": stage.y if stage else np.nan,
            "z_um": stage.z if stage else np.nan,
        })
    return positions


def parse_index_spec(spec: str, n: int) -> list[int]:
    """'1-4,7' (scope numbers, 1-based) -> [0, 1, 2, 3, 6] (0-based indices)."""
    picked: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            sys.exit(f"Can't parse position spec {part!r} (expected e.g. '1-4,7').")
        lo = int(match.group(1))
        hi = int(match.group(2)) if match.group(2) else lo
        for number in range(lo, hi + 1):
            if not 1 <= number <= n:
                sys.exit(f"Position #{number} out of range (file has #1-#{n}).")
            if number - 1 not in picked:
                picked.append(number - 1)
    return picked


def _nominal_period_s(f: nd2.ND2File):
    for loop in f.experiment:
        if loop.type == "TimeLoop" and loop.parameters.periodMs:
            return loop.parameters.periodMs / 1000.0
        if loop.type == "NETimeLoop" and loop.parameters.periods:
            return loop.parameters.periods[0].periodMs / 1000.0
    return None


def time_axis(f: nd2.ND2File):
    """Time of each timepoint in seconds since the first one.

    Returns ``(times, source)``. Uses the median over positions of the
    per-frame acquisition times when those look valid; some files carry
    garbage timestamps, in which case the nominal loop period is used.
    """
    n_t = f.sizes.get("T", 1)
    if n_t == 1:
        return np.zeros(1), "single timepoint"

    try:
        events = f.events(orient="list")
        t_idx = np.asarray(events.get("T Index", []), dtype=float)
        times = np.asarray(events.get("Time [s]", []), dtype=float)
    except (AttributeError, KeyError, ValueError):
        t_idx = times = np.array([])
    keep = ~np.isnan(t_idx) & np.isfinite(times) & (np.abs(times) < MAX_SANE_TIME_S)
    if keep.any():
        per_t = np.full(n_t, np.nan)
        for k in range(n_t):
            hits = times[keep & (t_idx == k)]
            if hits.size:
                per_t[k] = np.median(hits)
        if not np.isnan(per_t).any() and np.all(np.diff(per_t) > 0):
            return per_t - per_t[0], "frame timestamps"

    period = _nominal_period_s(f)
    if period:
        return np.arange(n_t) * period, f"nominal {period:g} s period"
    return np.arange(n_t, dtype=float), "timepoint index"


def position_label(p) -> str:
    """'#3' or '#3 (trap-row-A)' once the position has been named."""
    return f"#{p['scope_number']}" + (f" ({p['name']})" if p.get("name") else "")


# --------------------------------------------------------------------------- #
# Pixel access
# --------------------------------------------------------------------------- #

def frame_sequence_map(f: nd2.ND2File) -> dict:
    """``{(t, p): sequence index}`` for reading single frames with ``read_frame``.

    Any other loop (e.g. Z) is pinned to its first plane.
    """
    seqs = {}
    for seq, loops in enumerate(f.loop_indices):
        if any(v for k, v in loops.items() if k not in ("T", "P")):
            continue
        seqs[(loops.get("T", 0), loops.get("P", 0))] = seq
    return seqs


def read_channels(f: nd2.ND2File, seqs, t: int, p: int, channels) -> dict:
    """One timepoint of one position as ``{key: 2-D array}``.

    Much faster than slicing ``position_stack``: one direct read per frame.
    The arrays are copies, so they stay valid after the file is closed
    (``read_frame`` can hand back a view into the memory-mapped file).
    """
    frame = f.read_frame(seqs[(t, p)])
    if frame.ndim == 2:
        frame = frame[np.newaxis]
    return {c["key"]: np.array(frame[c["index"]]) for c in channels}


def position_stack(f: nd2.ND2File, position: int, channels) -> dict:
    """Lazy per-channel image stacks for one stage position.

    Returns ``{key: dask array}`` (e.g. ``stacks["phase"]``, ``stacks["GFP"]``)
    with every axis other than P and C kept in file order, usually (T, Y, X).
    Nothing is read from disk until the array is computed or indexed into numpy.
    """
    data = f.to_dask()
    axes = list(f.sizes)

    def pick(axis, i):
        index = [slice(None)] * len(axes)
        index[axes.index(axis)] = i
        return tuple(index)

    if "P" in axes:
        data = data[pick("P", position)]
        axes.remove("P")
    stacks = {}
    for c in channels:
        stacks[c["key"]] = data[pick("C", c["index"])] if "C" in axes else data
    return stacks


def stack_axes(f: nd2.ND2File) -> str:
    """Axis order of the arrays returned by ``position_stack``, e.g. 'TYX'."""
    return "".join(a for a in f.sizes if a not in ("P", "C"))


def scale_to_unit(image, limits):
    """Linearly map ``limits=(lo, hi)`` to [0, 1], clipped, as float32."""
    lo, hi = limits
    scaled = (image.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled, 0.0, 1.0, out=scaled)


def percentile_limits(images, percentiles):
    """Contrast limits pooled over ``images`` (subsampled for speed)."""
    pooled = np.concatenate([np.ravel(im[::4, ::4]) for im in images])
    lo, hi = np.percentile(pooled, percentiles)
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# 4. Sanity check
# --------------------------------------------------------------------------- #

def read_first_frames(f, seqs, positions, channels) -> dict:
    """Read and report t=0 for every position; returns ``{index: {key: array}}``."""
    first = {}
    for p in positions:
        frames = read_channels(f, seqs, 0, p["index"], channels)
        means = []
        for key, frame in frames.items():
            means.append(f"{key}={frame.mean():.0f}")
            if not frame.any():
                means[-1] += " [EMPTY]"
        print(f"     {position_label(p):>4}: " + "  ".join(means))
        first[p["index"]] = frames
    return first


# --------------------------------------------------------------------------- #
# 5. Trap selection
# --------------------------------------------------------------------------- #

def _orient(a, b, c) -> int:
    """Sign of the turn a -> b -> c (0 when collinear)."""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if abs(cross) < 1e-9 else (1 if cross > 0 else -1)


def _within_box(a, b, c) -> bool:
    """For collinear a, b, c: does c lie on segment ab?"""
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, q1, q2) -> bool:
    """True if closed segments p1p2 and q1q2 share any point."""
    o1, o2 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    o3, o4 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _within_box(p1, p2, q1)) or (o2 == 0 and _within_box(p1, p2, q2))
            or (o3 == 0 and _within_box(q1, q2, p1)) or (o4 == 0 and _within_box(q1, q2, p2)))


def polygon_area(vertices) -> float:
    """Absolute shoelace area."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_problem(vertices, closed=True):
    """Why ``vertices`` isn't a simple path/polygon, or None if it is.

    With ``closed=True`` a None result means the polygon is a Jordan curve:
    at least 3 vertices, non-zero area, and no edge touches another except
    neighbours at their shared vertex. With ``closed=False`` the open path is
    checked the same way (used while a trap is still being drawn).
    """
    pts = [tuple(map(float, v)) for v in vertices]
    if closed and len(pts) < 3:
        return "a trap needs at least 3 corners"
    for a, b in zip(pts, pts[1:] + ([pts[0]] if closed else [])):
        if a == b:
            return "two consecutive corners are at the same spot"
    segs = list(pairwise(pts))
    if closed:
        segs.append((pts[-1], pts[0]))
    n = len(segs)
    for i in range(n):
        for j in range(i + 1, n):
            s, t = segs[i], segs[j]
            if j == i + 1 or (closed and i == 0 and j == n - 1):
                # Neighbours share one corner; they only clash if they fold
                # back over each other along the same line.
                shared = s[1] if j == i + 1 else s[0]
                u = s[0] if j == i + 1 else s[1]
                v = t[1] if j == i + 1 else t[0]
                if _orient(shared, u, v) == 0 and (
                        _within_box(shared, u, v) or _within_box(shared, v, u)):
                    return "an edge doubles back over the previous one"
            elif segments_intersect(*s, *t):
                return "edges cross or touch"
    if closed and polygon_area(pts) < 1.0:
        return "the trap has no area"
    return None


def _check_display():
    """Exit with a hint if the trap UI can't be shown."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not has_display or backend in {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}:
        sys.exit("No interactive display available for the trap UI - rerun with "
                 "--traps <file> to reuse traps drawn earlier.")


class TrapPicker:
    """Matplotlib window for naming one position and drawing its traps.

    Left-click adds a corner. Close the trap by clicking its first corner,
    right-clicking, or pressing Enter. Backspace (or "Undo corner") removes
    the last corner; "Undo trap" removes the last finished trap. The toolbar's
    zoom/pan can be used freely -- clicks made in those modes add nothing.

    Vertices are stored as ``(x, y)`` = (column, row) pixel coordinates.
    """

    CLOSE_RADIUS_PX = 12

    def __init__(self, image, position, n_positions, taken_names=(), pyplot=None):
        if pyplot is None:
            from matplotlib import pyplot
        from matplotlib.widgets import Button, TextBox

        self.plt = pyplot
        self.position = position
        self.taken_names = {n.lower() for n in taken_names}
        self.traps: list[np.ndarray] = []
        self.current: list[tuple[float, float]] = []
        self.finished = False
        self._trap_artists: list = []

        p = position
        self.fig = pyplot.figure(figsize=(10, 11.5))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(f"Traps - scope position #{p['scope_number']}")
        self.fig.text(0.5, 0.975, f"Scope position #{p['scope_number']}",
                      ha="center", va="top", fontsize=22, fontweight="bold")
        stage = f"stage X={p['x_um']:.1f} um, Y={p['y_um']:.1f} um"
        nis = f"   NIS name: {p['nis_name']!r}" if p["nis_name"] else ""
        self.fig.text(0.5, 0.935, f"{p['scope_number']} of {n_positions}   |   {stage}{nis}",
                      ha="center", va="top", fontsize=11)

        self.ax = self.fig.add_axes([0.05, 0.17, 0.9, 0.74])
        lo, hi = percentile_limits([image], PHASE_PERCENTILES)
        self.ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.height, self.width = image.shape
        (self.path_line,) = self.ax.plot([], [], "-o", color="yellow", ms=4, lw=1.5)

        self.name_box = TextBox(self.fig.add_axes([0.25, 0.105, 0.5, 0.04]),
                                f"Name for #{p['scope_number']}: ",
                                initial=p.get("name") or p["nis_name"] or "")
        self._buttons = []
        for i, (label, callback) in enumerate([("Undo corner", self.undo_corner),
                                               ("Undo trap", self.undo_trap),
                                               ("Done", self.done)]):
            button = Button(self.fig.add_axes([0.2 + 0.21 * i, 0.045, 0.18, 0.045]), label)
            button.on_clicked(lambda _event, cb=callback: cb())
            self._buttons.append(button)
        self.status = self.fig.text(0.5, 0.012, "", ha="center", va="bottom", fontsize=11)
        self._say("Click corners of a trap; click the first corner (or right-click / Enter) "
                  "to close it.")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # -- event handlers ------------------------------------------------------ #

    def _toolbar_busy(self) -> bool:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def on_click(self, event):
        if event.inaxes is not self.ax or self._toolbar_busy() or event.xdata is None:
            return
        if event.button == 3:
            self.close_trap()
            return
        if event.button != 1:
            return
        x = float(np.clip(event.xdata, -0.5, self.width - 0.5))
        y = float(np.clip(event.ydata, -0.5, self.height - 0.5))
        if len(self.current) >= 3:
            first = self.ax.transData.transform(self.current[0])
            here = self.ax.transData.transform((x, y))
            if np.hypot(*(first - here)) <= self.CLOSE_RADIUS_PX:
                self.close_trap()
                return
        self.add_corner(x, y)

    def on_key(self, event):
        if self.name_box.capturekeystrokes:
            return
        if event.key == "enter":
            self.close_trap()
        elif event.key == "backspace":
            self.undo_corner()

    # -- actions --------------------------------------------------------------- #

    def add_corner(self, x, y) -> bool:
        problem = polygon_problem(self.current + [(x, y)], closed=False)
        if problem:
            self._say(f"Can't put a corner there: {problem}.", error=True)
            return False
        self.current.append((x, y))
        self._redraw_current()
        self._say(f"Trap {len(self.traps) + 1}: {len(self.current)} corner(s).")
        return True

    def close_trap(self) -> bool:
        if not self.current:
            return False
        problem = polygon_problem(self.current, closed=True)
        if problem:
            self._say(f"Can't close this trap: {problem}.", error=True)
            return False
        vertices = np.array(self.current, dtype=float)
        self.traps.append(vertices)
        self.current = []
        self._redraw_current()
        self._draw_trap(vertices, len(self.traps))
        self._say(f"{len(self.traps)} trap(s) drawn. Draw another, or name the position "
                  "and press Done.")
        return True

    def undo_corner(self):
        if self.current:
            self.current.pop()
            self._redraw_current()
            self._say(f"Trap {len(self.traps) + 1}: {len(self.current)} corner(s).")

    def undo_trap(self):
        if self.traps:
            self.traps.pop()
            for artist in self._trap_artists.pop():
                artist.remove()
            self.fig.canvas.draw_idle()
            self._say(f"{len(self.traps)} trap(s) drawn.")

    def done(self) -> bool:
        name = self.name_box.text.strip()
        if self.current:
            self._say("Finish (close) or undo the trap in progress first.", error=True)
        elif not self.traps:
            self._say("Every position needs at least one trap.", error=True)
        elif not name:
            self._say("Give this position a name first.", error=True)
        elif name.lower() in self.taken_names:
            self._say(f"{name!r} is already used by another position.", error=True)
        else:
            self.name = name
            self.finished = True
            self.plt.close(self.fig)
            return True
        return False

    # -- drawing --------------------------------------------------------------- #

    def _redraw_current(self):
        xs = [v[0] for v in self.current]
        ys = [v[1] for v in self.current]
        self.path_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def _draw_trap(self, vertices, number):
        from matplotlib.patches import Polygon

        color = TRAP_COLORS[(number - 1) % len(TRAP_COLORS)]
        patch = Polygon(vertices, closed=True, facecolor=color, alpha=TRAP_ALPHA,
                        edgecolor=color, lw=2)
        outline = Polygon(vertices, closed=True, fill=False, edgecolor=color, lw=2)
        cx, cy = vertices.mean(axis=0)
        label = self.ax.text(cx, cy, f"T{number}", color="white", ha="center", va="center",
                             fontsize=12, fontweight="bold")
        self.ax.add_patch(patch)
        self.ax.add_patch(outline)
        self._trap_artists.append([patch, outline, label])
        self.fig.canvas.draw_idle()

    def _say(self, message, error=False):
        self.status.set_text(message)
        self.status.set_color("red" if error else "black")
        self.fig.canvas.draw_idle()

    def run(self):
        """Block until Done; returns ``(name, [vertices, ...])``."""
        self.plt.show(block=True)
        if not self.finished:
            sys.exit(f"Trap selection for position #{self.position['scope_number']} "
                     "was closed without pressing Done - nothing saved.")
        return self.name, self.traps


def pick_traps(positions, first_frames, n_positions) -> None:
    """Run the trap UI for each position, filling ``name`` and ``traps`` in place.

    ``n_positions`` is the total in the file, shown as "#3 of 8".
    """
    _check_display()
    import matplotlib.pyplot as plt

    taken = []
    for p in positions:
        print(f"     #{p['scope_number']}: waiting for trap UI...")
        picker = TrapPicker(first_frames[p["index"]][PHASE_KEY], p, n_positions, taken, plt)
        name, traps = picker.run()
        p["name"] = name
        p["traps"] = [{"name": f"T{i + 1}", "vertices": v} for i, v in enumerate(traps)]
        taken.append(name)
        print(f"     #{p['scope_number']}: named {name!r}, {len(traps)} trap(s)")


def load_traps(path: Path, positions, image_shape) -> None:
    """Fill ``name`` and ``traps`` from a ``*_traps.pkl``, validating everything."""
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    by_index = {p["index"]: p for p in saved["positions"]}
    missing = [f"#{p['scope_number']}" for p in positions if p["index"] not in by_index]
    if missing:
        sys.exit(f"{path} has no traps for position(s) {', '.join(missing)}.")
    if tuple(saved.get("image_shape", image_shape)) != tuple(image_shape):
        sys.exit(f"{path} was drawn on {saved['image_shape']} images, "
                 f"but this file's are {image_shape}.")
    for p in positions:
        entry = by_index[p["index"]]
        if not entry["traps"]:
            sys.exit(f"{path}: position #{p['scope_number']} has no traps.")
        for trap in entry["traps"]:
            problem = polygon_problem(trap["vertices"])
            if problem:
                sys.exit(f"{path}: #{p['scope_number']} {trap['name']} is invalid ({problem}).")
        p["name"] = entry["name"]
        p["traps"] = entry["traps"]
        print(f"     #{p['scope_number']}: {p['name']!r}, {len(p['traps'])} trap(s)")
    names = [p["name"].lower() for p in positions]
    if len(set(names)) != len(names):
        sys.exit(f"{path}: position names aren't unique.")


# --------------------------------------------------------------------------- #
# 6. Fluorescence background
# --------------------------------------------------------------------------- #

def trap_mask(shape, traps) -> np.ndarray:
    """Boolean image, True for pixels whose centre lies inside any trap."""
    from skimage.draw import polygon2mask

    mask = np.zeros(shape, dtype=bool)
    for trap in traps:
        vertices = np.asarray(trap["vertices"], dtype=float)
        mask |= polygon2mask(shape, vertices[:, ::-1])  # (x, y) -> (row, col)
    return mask


def fluor_backgrounds(positions, first_frames, channels, shape) -> None:
    """Set ``p["fluor_background"] = {key: level}`` for every position.

    The level is the median of the position's first frame over every pixel
    outside its traps, one value per fluorescence channel. The median, unlike
    the mean, isn't pulled up by fluorescent cells sitting outside the traps.
    """
    for p in positions:
        outside = ~trap_mask(shape, p["traps"])
        if not outside.any():
            sys.exit(f"Position #{p['scope_number']}: the traps cover the whole image, "
                     "so there's no background to measure.")
        frames = first_frames[p["index"]]
        p["fluor_background"] = {
            c["key"]: float(np.median(frames[c["key"]][outside])) for c in fluor_channels(channels)
        }
        levels = ", ".join(f"{k}={v:.1f}" for k, v in p["fluor_background"].items())
        print(f"     {position_label(p)}: {levels or '(no fluorescence)'}  "
              f"(from the {outside.mean():.0%} of the frame outside traps)")


# --------------------------------------------------------------------------- #
# 8. Spatial bandpass
# --------------------------------------------------------------------------- #

def bandpass_transfer(shape, pixel_um, min_um, max_um, order=BANDPASS_ORDER):
    """Bandpass transfer function H(f) on the ``scipy.fft.rfft2`` grid of ``shape``.

    ``f`` is the radial spatial frequency in cycles/um (each axis scaled by
    its own pixel size, ``pixel_um = (x, y)``). H is the product of a
    Butterworth high-pass at ``1 / max_um`` and a Butterworth low-pass at
    ``1 / min_um``::

        H(f) = 1 / sqrt(1 + (f_lo / f) ** (2n))  *  1 / sqrt(1 + (f / f_hi) ** (2n))

    so H(0) = 0 (the mean is removed), H = 1/sqrt(2) at each cutoff, and H
    is ~1 in between. Real and non-negative, so phase is left untouched.
    """
    fy = np.fft.fftfreq(shape[0], d=pixel_um[1])[:, np.newaxis]
    fx = np.fft.rfftfreq(shape[1], d=pixel_um[0])[np.newaxis, :]
    f = np.hypot(fx, fy)
    f_lo, f_hi = 1.0 / max_um, 1.0 / min_um
    with np.errstate(divide="ignore"):
        highpass = 1.0 / np.sqrt(1.0 + (f_lo / f) ** (2 * order))
    lowpass = 1.0 / np.sqrt(1.0 + (f / f_hi) ** (2 * order))
    return (highpass * lowpass).astype(np.float32)


class SpatialBandpass:
    """FFT bandpass for 2-D frames of a fixed shape and pixel size.

    Each frame is mirror-padded, Fourier transformed, multiplied by
    ``bandpass_transfer``, and transformed back; the padding is then cropped.
    Output is float32 and centred on zero (the mean is filtered out).
    """

    def __init__(self, shape, pixel_um, min_um=DEFAULT_BANDPASS_MIN_UM,
                 max_um=DEFAULT_BANDPASS_MAX_UM, order=BANDPASS_ORDER):
        self.shape = tuple(shape)
        self.pixel_um = tuple(pixel_um)
        self.min_um, self.max_um, self.order = min_um, max_um, order
        pad = int(np.ceil(BANDPASS_PAD_WAVELENGTHS * max_um / min(pixel_um)))
        self.pad = min(pad, min(shape) - 1)  # reflect padding can't exceed the image
        padded = [scipy.fft.next_fast_len(n + 2 * self.pad, real=True) for n in shape]
        self.pad_widths = [(self.pad, m - n - self.pad) for n, m in zip(shape, padded)]
        self.transfer = bandpass_transfer(padded, pixel_um, min_um, max_um, order)

    def __call__(self, image):
        padded = np.pad(image.astype(np.float32), self.pad_widths, mode="reflect")
        spectrum = scipy.fft.rfft2(padded, workers=-1)
        spectrum *= self.transfer
        filtered = scipy.fft.irfft2(spectrum, s=padded.shape, workers=-1)
        return filtered[self.pad:self.pad + self.shape[0], self.pad:self.pad + self.shape[1]]

    def nyquist_um(self) -> float:
        """Shortest wavelength the pixel grid can represent (2 pixels)."""
        return 2.0 * max(self.pixel_um)

    def describe(self) -> dict:
        return {"min_um": self.min_um, "max_um": self.max_um, "order": self.order,
                "pixel_um": self.pixel_um, "pad_px": self.pad,
                "transfer": "butterworth bandpass"}


def report_bandpass(bandpass: SpatialBandpass) -> None:
    px = max(bandpass.pixel_um)
    print(f"     keeping structure between {bandpass.min_um:g} and {bandpass.max_um:g} um "
          f"({bandpass.min_um / px:.2f}-{bandpass.max_um / px:.1f} px), "
          f"order-{bandpass.order} Butterworth, {bandpass.pad} px mirror padding")
    nyquist = bandpass.nyquist_um()
    if bandpass.min_um < nyquist:
        print(f"    !! {bandpass.min_um:g} um is finer than this file can resolve "
              f"({nyquist:.2f} um = 2 pixels at {px:g} um/px); the low cutoff has "
              "almost no effect, so this acts as a high-pass")
    if bandpass.max_um <= nyquist:
        print(f"    !! {bandpass.max_um:g} um high cutoff is at or below the "
              f"{nyquist:.2f} um resolution limit; little or nothing will pass")


# --------------------------------------------------------------------------- #
# 9. Videos
# --------------------------------------------------------------------------- #

def _render_rgba(width, height, draw):
    """Render matplotlib artists onto a transparent ``height x width`` RGBA layer.

    ``draw(ax)`` adds artists in pixel coordinates (x right, y down).
    Returns float32 RGBA in [0, 1].
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_alpha(0)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    ax.patch.set_alpha(0)
    draw(ax)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.float32)[:height, :width] / 255.0
    if rgba.shape[:2] != (height, width):
        padded = np.zeros((height, width, 4), np.float32)
        padded[:rgba.shape[0], :rgba.shape[1]] = rgba
        rgba = padded
    return rgba


def trap_layer(width, height, traps, label_size):
    """Static RGBA layer of colored, translucent trap polygons with labels."""
    from matplotlib.patches import Polygon

    def draw(ax):
        for i, trap in enumerate(traps):
            color = TRAP_COLORS[i % len(TRAP_COLORS)]
            ax.add_patch(Polygon(trap["vertices"], closed=True, facecolor=color,
                                 alpha=TRAP_ALPHA, edgecolor="none"))
            ax.add_patch(Polygon(trap["vertices"], closed=True, fill=False,
                                 edgecolor=color, lw=2))
            cx, cy = np.asarray(trap["vertices"]).mean(axis=0)
            ax.text(cx, cy, trap["name"], color="white", ha="center", va="center",
                    fontsize=label_size, fontweight="bold")

    return _render_rgba(width, height, draw)


def text_label(text, font_px):
    """Small RGBA tile with white text on a translucent dark box."""
    from matplotlib.patches import Rectangle

    height = int(font_px * 1.8)
    width = int(font_px * 0.62 * len(text) + font_px)

    def draw(ax):
        ax.add_patch(Rectangle((-0.5, -0.5), width, height, facecolor="black", alpha=0.55))
        ax.text(font_px * 0.5, height / 2, text, color="white", ha="left", va="center",
                fontsize=font_px * 72 / 100, family="monospace")

    return _render_rgba(width, height, draw)


def blend(rgb, layer, x=0, y=0):
    """Alpha-composite RGBA ``layer`` onto float RGB ``rgb`` at (x, y), in place."""
    h = min(layer.shape[0], rgb.shape[0] - y)
    w = min(layer.shape[1], rgb.shape[1] - x)
    if h <= 0 or w <= 0:
        return rgb
    region = rgb[y:y + h, x:x + w]
    alpha = layer[:h, :w, 3:4]
    region *= 1.0 - alpha
    region += layer[:h, :w, :3] * alpha
    return rgb


def format_time(seconds) -> str:
    minutes = round(float(seconds) / 60)
    return f"t = {minutes // 60:02d}:{minutes % 60:02d}"


def fluor_color(key, used):
    base = key.split("_")[0]
    if base in FLUOR_COLORS and FLUOR_COLORS[base] not in used:
        return FLUOR_COLORS[base]
    for color in FLUOR_COLORS.values():
        if color not in used:
            return color
    return (1.0, 1.0, 1.0)


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "position"


class VideoSet:
    """The three mp4 writers for one position."""

    KINDS = ("phase", "composite", "traps")

    def __init__(self, out_dir: Path, stem: str, position, size, fps):
        import imageio_ffmpeg

        self.paths = {}
        self.writers = {}
        base = f"{stem}_pos{position['scope_number']}_{safe_filename(position['name'])}"
        for kind in self.KINDS:
            path = out_dir / f"{base}_{kind}.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(path), size, fps=fps, codec="libx264", quality=None,
                macro_block_size=1, ffmpeg_log_level="error",
                output_params=["-crf", "23", "-preset", "veryfast"])
            writer.send(None)
            self.paths[kind] = path
            self.writers[kind] = writer

    def write(self, kind, rgb):
        frame = np.ascontiguousarray((rgb * 255.0 + 0.5).astype(np.uint8))
        self.writers[kind].send(frame)

    def close(self):
        for writer in self.writers.values():
            writer.close()


def contrast_limits(read, positions, channels, n_t):
    """Per-position, per-channel contrast limits from a few sampled timepoints.

    ``read(t, index)`` returns the processed frames for a timepoint. Limits
    are fixed for the whole video, so brightness changes over time (e.g.
    rising fluorescence) stay visible. Fluorescence arrives background-
    subtracted, so its lower limit is 0: background shows as black.
    """
    sample_ts = sorted({int(t) for t in np.linspace(0, n_t - 1, CONTRAST_SAMPLES).round()})
    limits = {}
    for p in positions:
        frames = [read(t, p["index"]) for t in sample_ts]
        limits[p["index"]] = {}
        for c in channels:
            is_phase = c["role"] == "phase"
            lo, hi = percentile_limits([fr[c["key"]] for fr in frames],
                                       PHASE_PERCENTILES if is_phase else FLUOR_PERCENTILES)
            if not is_phase:
                lo = 0.0
            limits[p["index"]][c["key"]] = (lo, hi)
    return limits


def write_videos(f, seqs, positions, channels, times, out_dir: Path, stem: str, fps,
                 bandpass=None):
    """Write phase / composite / trap-overlay mp4s for every position in one pass.

    Phase frames go through ``bandpass`` (a ``SpatialBandpass``) first, if
    given. Fluorescence frames are left unfiltered and have the position's
    ``fluor_background`` level subtracted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = f.sizes["Y"], f.sizes["X"]
    # libx264 + yuv420p needs even dimensions; drop a row/column if needed.
    height, width = height - height % 2, width - width % 2
    font_px = max(14, height // 40)
    n_t = len(times)

    backgrounds = {p["index"]: p["fluor_background"] for p in positions}

    def read(t, index):
        frames = read_channels(f, seqs, t, index, channels)
        for c in channels:
            key = c["key"]
            if c["role"] == "phase":
                if bandpass is not None:
                    frames[key] = bandpass(frames[key])
            else:
                frames[key] = frames[key].astype(np.float32) - backgrounds[index][key]
        return frames

    print(f"     setting contrast from {CONTRAST_SAMPLES} timepoints per position")
    limits = contrast_limits(read, positions, channels, n_t)

    fluors = fluor_channels(channels)
    colors, used = {}, []
    for c in fluors:
        colors[c["key"]] = np.array(fluor_color(c["key"], used), np.float32)
        used.append(tuple(colors[c["key"]]))
    legend = " + ".join(["phase"] + [c["key"] for c in fluors])

    overlays, captions, videos = {}, {}, {}
    try:
        for p in positions:
            overlays[p["index"]] = trap_layer(width, height, p["traps"], font_px * 0.8)
            captions[p["index"]] = text_label(f"#{p['scope_number']} {p['name']}", font_px)
            captions[(p["index"], "composite")] = text_label(legend, font_px * 0.8)
            videos[p["index"]] = VideoSet(out_dir, stem, p, (width, height), fps)

        start = time.monotonic()
        report_every = max(1, n_t // 10)
        for t in range(n_t):
            stamp = text_label(format_time(times[t]), font_px)
            for p in positions:
                frames = read(t, p["index"])
                lim = limits[p["index"]]
                phase = scale_to_unit(frames[PHASE_KEY][:height, :width], lim[PHASE_KEY])
                gray = np.repeat(phase[:, :, np.newaxis], 3, axis=2)

                def labelled(rgb, extra=None, p=p, stamp=stamp):
                    blend(rgb, stamp, font_px // 2, font_px // 2)
                    caption = captions[p["index"]]
                    blend(rgb, caption, font_px // 2, height - caption.shape[0] - font_px // 2)
                    if extra is not None:
                        blend(rgb, extra, width - extra.shape[1] - font_px // 2, font_px // 2)
                    return rgb

                video = videos[p["index"]]
                video.write("phase", labelled(gray.copy()))

                composite = gray * COMPOSITE_PHASE_WEIGHT
                for c in fluors:
                    signal = scale_to_unit(frames[c["key"]][:height, :width], lim[c["key"]])
                    composite += signal[:, :, np.newaxis] * colors[c["key"]]
                np.clip(composite, 0.0, 1.0, out=composite)
                video.write("composite", labelled(composite, captions[(p["index"], "composite")]))

                video.write("traps", labelled(blend(gray, overlays[p["index"]])))

            if (t + 1) % report_every == 0 or t + 1 == n_t:
                elapsed = time.monotonic() - start
                remaining = elapsed / (t + 1) * (n_t - t - 1)
                print(f"     timepoint {t + 1}/{n_t}  "
                      f"({elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left)")
    finally:
        for video in videos.values():
            video.close()

    for p in positions:
        for path in videos[p["index"]].paths.values():
            print(f"     saved {path}")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def save_pickle(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"     saved {path}")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def default_nd2(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*.nd2"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"No .nd2 files found in {data_dir}/ - pass a path explicitly.")
    return candidates[-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nd2", nargs="?", type=Path,
                        help="nd2 file to process (default: newest .nd2 in data/)")
    parser.add_argument("--objects", type=Path, default=None,
                        help="directory for output pickles (default: objects/)")
    parser.add_argument("--videos", type=Path, default=None,
                        help="directory for output mp4s (default: videos/)")
    parser.add_argument("--positions", default=None,
                        help="scope position numbers to process, e.g. '1-4,7' (default: all)")
    parser.add_argument("--phase-channel", default=None,
                        help="name or index of the phase contrast channel, if "
                             "auto-detection picks wrong")
    parser.add_argument("--traps", type=Path, default=None,
                        help="reuse names/traps from a saved *_traps.pkl instead of the UI")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"video frame rate (default {DEFAULT_FPS})")
    parser.add_argument("--bandpass-min-um", type=float, default=DEFAULT_BANDPASS_MIN_UM,
                        help="phase bandpass: smallest structure kept, in um "
                             f"(default {DEFAULT_BANDPASS_MIN_UM})")
    parser.add_argument("--bandpass-max-um", type=float, default=DEFAULT_BANDPASS_MAX_UM,
                        help="phase bandpass: largest structure kept, in um "
                             f"(default {DEFAULT_BANDPASS_MAX_UM})")
    parser.add_argument("--no-bandpass", action="store_true",
                        help="write videos from unfiltered phase frames")
    parser.add_argument("--skip-videos", action="store_true",
                        help="stop after saving traps and metadata")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = args.nd2 or default_nd2(DATA_DIR)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    if not args.no_bandpass and not 0 < args.bandpass_min_um < args.bandpass_max_um:
        sys.exit("Bandpass cutoffs must satisfy 0 < --bandpass-min-um < --bandpass-max-um.")
    if args.traps and not args.traps.exists():
        sys.exit(f"Traps file not found: {args.traps}")
    objects_dir = args.objects or OBJECTS_DIR
    videos_dir = args.videos or VIDEOS_DIR
    stem = path.stem

    with open_nd2(path) as f:
        info = describe_file(f)
        print(f"[1] Opened {path.name}")
        print("     axes:  " + ", ".join(f"{k}={v}" for k, v in info["sizes"].items()))
        v = info["voxel_um"]
        print(f"     dtype: {info['dtype']}, pixel: {v['x']:.4g} x {v['y']:.4g} um")
        extra = set(info["sizes"]) - set("TPCYX")
        if extra:
            print(f"    !! unexpected axes {sorted(extra)} - only the first plane is used")

        print("\n[2] Identifying channels")
        channels = classify_channels(f, args.phase_channel)
        for c in channels:
            em = f", em {c['emission_nm']:g} nm" if c["emission_nm"] else ""
            print(f"     {c['index']}: {c['name']!r:<12} -> {c['key']:<6} "
                  f"({'/'.join(c['modalities']) or 'no modality'}{em})")
        n_fluor = len(fluor_channels(channels))
        if n_fluor == 0:
            print("    !! no fluorescence channels - phase only")
        elif n_fluor > MAX_FLUOR_CHANNELS:
            print(f"    !! {n_fluor} fluorescence channels (expected at most {MAX_FLUOR_CHANNELS})")

        print("\n[3] Positions and time axis")
        positions = list_positions(f)
        n_total = len(positions)
        if n_total == 1:
            print("    !! only one stage position in this file")
        if args.positions:
            picked = parse_index_spec(args.positions, n_total)
            positions = [positions[i] for i in picked]
        print(f"     {len(positions)} position(s): " + ", ".join(position_label(p) for p in positions))
        times, time_source = time_axis(f)
        if time_source != "frame timestamps" and len(times) > 1:
            print("    !! per-frame timestamps missing or corrupt; falling back")
        print(f"     {len(times)} timepoints over {times[-1] / 3600:.2f} h ({time_source})")

        print("\n[4] Reading first frame of each position")
        seqs = frame_sequence_map(f)
        first_frames = read_first_frames(f, seqs, positions, channels)

        image_shape = (info["sizes"]["Y"], info["sizes"]["X"])
        if args.traps:
            print(f"\n[5] Loading position names and traps from {args.traps}")
            load_traps(args.traps, positions, image_shape)
        else:
            print("\n[5] Name each position and draw its traps")
            pick_traps(positions, first_frames, n_total)
        save_pickle({"file": str(path.resolve()), "image_shape": image_shape,
                     "positions": positions, "Time": times},
                    objects_dir / f"{stem}_traps.pkl")

        print("\n[6] Fluorescence background (median outside traps, first frame)")
        fluor_backgrounds(positions, first_frames, channels, image_shape)

        bandpass = None
        if not args.no_bandpass:
            bandpass = SpatialBandpass(image_shape, (v["x"], v["y"]),
                                       args.bandpass_min_um, args.bandpass_max_um)

        print("\n[7] Saving experiment layout")
        layout = {
            "file": str(path.resolve()),
            **info,
            "stack_axes": stack_axes(f),
            "channels": channels,
            "positions": positions,
            "time_source": time_source,
            "video_bandpass": bandpass.describe() if bandpass else None,
            "Time": times,
        }
        save_pickle(layout, objects_dir / f"{stem}_metadata.pkl")

        if bandpass is None:
            print("\n[8] Spatial bandpass off (--no-bandpass)")
        else:
            print("\n[8] Spatial bandpass (FFT) for phase frames")
            report_bandpass(bandpass)

        if args.skip_videos:
            print("\n[9] Skipping videos (--skip-videos)")
        else:
            print(f"\n[9] Writing videos ({len(positions)} position(s) x 3, {args.fps:g} fps)")
            write_videos(f, seqs, positions, channels, times, videos_dir, stem, args.fps,
                         bandpass)

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
