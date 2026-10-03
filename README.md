# ndInterpreter
Standardized workflow for processing Nikon `.nd2` microscopy files from microfluidics experiments.

Expects multi-position time-lapse nd2s with one phase contrast channel and up to three fluorescence channels

## Setup

```
conda env create -f environment.yml   # first time only
conda activate microscopy
```

## Usage

```
python nd2_workflow.py                          # uses the newest .nd2 in data/
python nd2_workflow.py data/some_experiment.nd2  # or point at a specific file
```

nd2s are never loaded whole. Pixel data is read one frame at a time, so a file can live on an external drive and be passed by path.

A window pops up for each stage position, one after another:

1. **Name the position.** The scope's number for it (e.g. **Scope position #3**, 1-based as NIS shows it) and its stage X/Y are shown at the top, so you can check it against your notes. Type the name in the box at the bottom.
2. **Draw its traps.** Left-click the corners of a trap, then close it by clicking the first corner again, right-clicking, or pressing Enter. Traps can be any simple polygon. Corners that would make edges cross are refused. Repeat for each trap; every position needs at least one.
   - **Undo corner** (or Backspace) removes the last corner; **Undo trap** removes the last finished trap.
   - The toolbar's zoom/pan work as usual. Clicks made while zoom or pan is active don't add corners.
3. Hit **Done**. It won't let you through without a name (unique across positions) and at least one trap.

Then the fluorescence background is measured (see [Fluorescence background](#fluorescence-background)), phase frames are spatially bandpass-filtered (see [Spatial bandpass filter](#spatial-bandpass-filter)), and the videos are written, with progress printed as it goes:

```
[6] Fluorescence background (median outside traps, first frame)
     #1 (ctrl-row): GFP=816.0  (from the 98% of the frame outside traps)
...
[8] Spatial bandpass (FFT) for phase frames
     keeping structure between 0.35 and 4 um (0.64-7.3 px), order-2 Butterworth, 30 px mirror padding
    !! 0.35 um is finer than this file can resolve (1.10 um = 2 pixels at 0.55 um/px); ...

[9] Writing videos (8 position(s) x 3, 10 fps)
     setting contrast from 3 timepoints per position
     timepoint 19/193  (1.4 min elapsed, ~12.6 min left)
```

### Videos

Three mp4s per position land in `videos/`, named `<file>_pos<#>_<name>_<kind>.mp4`:

- `_phase.mp4`: phase contrast alone.
- `_composite.mp4`: phase (dimmed) with every fluorescence channel overlaid in color (GFP green, RFP magenta, YFP yellow, CFP cyan, BFP blue, Cy5 orange).
- `_traps.mp4`: phase with the traps drawn as colored, translucent polygons. The polygons stay where you drew them on the first frame, so if the image drifts the traps visibly slide off their chambers.

Every frame carries the elapsed time (top left) and the position number and name (bottom left). Each channel's contrast is fixed for the whole video, set from a few timepoints spread over the run, so fluorescence rising over time stays visible as brightening. Fluorescence is shown from its background level (black) upward.

The whole file is read once, in on-disk order, feeding all the videos at the same time. Expect roughly 15 minutes for a 16 GB file on an external drive.

`--fps` (default `10`) sets the frame rate. `--videos <dir>` sends the mp4s somewhere other than `videos/`. `--skip-videos` stops after the traps and metadata are saved.

### Fluorescence background

Fluorescence is not filtered. Instead, each position's background is subtracted, one level per fluorescence channel:

1. The traps drawn for the position are rasterized into a mask. A pixel counts as inside a trap if its centre is inside the polygon or on its edge.
2. The background is the **median of the first frame over every pixel outside all of that position's traps**.
3. That level is subtracted from every frame of that channel at that position, for the whole run. Values below background go negative and display as black.

Because the level comes from the first frame and is held fixed, fluorescence rising later in the run still shows as brightening. Each position gets its own level, which corrects for position-to-position differences in illumination and camera offset. The levels are printed at step `[6]` and saved per position in `_metadata.pkl` as `"fluor_background"`, e.g. `{"GFP": 816.0}`.

Everything outside the traps counts: flow channels, walls, and any chambers you didn't draw as traps. The median is used rather than the mean because fluorescent cells already sitting there in the first frame are a small, very bright minority. They would pull a mean up, but barely move the median. If more than about half of the non-trap area were bright, the median would rise too; drawing every chamber that holds cells as a trap avoids that.

### Spatial bandpass filter

Before a phase frame goes into a video, it is filtered in the spatial-frequency domain. Fluorescence channels are not filtered (see above). The filter keeps structure between **0.35 µm and 4 µm** and removes anything finer or broader. 0.35 µm is roughly the diffraction limit of the 20x/0.75 NA objective (λ / (2 NA) ≈ 0.35–0.37 µm for 520–550 nm light). 4 µm is a few bacterial cell lengths, so slow variation like uneven illumination, phase halos, and background gradients is removed while cells and chamber edges are kept. Only the videos are filtered. The trap UI shows the raw first frame, and nothing filtered is saved to `objects/`.

**Transfer function.** The filter multiplies the image's 2-D Fourier transform by a real, radially symmetric gain $H(f)$, where $f = \sqrt{f_x^2 + f_y^2}$ is the spatial frequency in cycles/µm (each axis scaled by its own pixel size). $H$ is a Butterworth high-pass times a Butterworth low-pass:

```math
H(f) \;=\; \underbrace{\frac{1}{\sqrt{1 + \left(f_\text{lo}/f\right)^{2n}}}}_{\text{high-pass}}
\;\cdot\; \underbrace{\frac{1}{\sqrt{1 + \left(f/f_\text{hi}\right)^{2n}}}}_{\text{low-pass}},
\qquad f_\text{lo} = \frac{1}{4\ \mu\text{m}},\quad f_\text{hi} = \frac{1}{0.35\ \mu\text{m}},\quad n = 2
```

- **Gentle rolloff.** A Butterworth filter is maximally flat in its passband and falls off smoothly, with no hard edge. Order $n = 2$ gives a slope of about 12 dB per octave far from each cutoff. That is gentle enough to keep ringing (halos next to sharp edges) small, while still suppressing broad background well. A brick-wall cut would ring badly.
- **At each cutoff** the gain is $1/\sqrt{2} \approx 0.71$ (−3 dB), so the cutoffs mark where the filter starts to bite, not where it stops passing.
- **At $f = 0$** the gain is exactly 0, so each frame's mean is removed and its values centre on zero.
- **Gain is real and non-negative**, so the Fourier phase is untouched and nothing shifts position.

How much of a pattern's amplitude survives, by its wavelength (period). These are the default settings; filtering test sinusoids on a 0.55 µm/px grid reproduces them.

| wavelength | 1.5 µm | 2 µm | 4 µm (cutoff) | 8 µm | 16 µm | 30 µm |
|---|---|---|---|---|---|---|
| gain | 0.99 | 0.97 | 0.71 | 0.24 | 0.06 | 0.02 |

**Edges.** The FFT treats the image as if it tiled the plane, so a mismatch between opposite edges looks like a sharp step and rings. Each frame is therefore mirror-padded before transforming, by 4 × the longest passed wavelength (30 px at 0.55 µm/px; a little more to reach a fast FFT size), and the padding is cropped off afterwards. On a steep test ramp this cuts edge artifacts from about 40% of the image's range to about 0.3%.

**What it does.** Illumination gradients and halos disappear, leaving cells, debris, and chamber walls on a flat background. The filter isn't used on fluorescence because it would remove intensity information. A chamber filled uniformly with fluorescent cells would show mostly its outlines, and brightness rising over the run would be removed along with the mean.

**Resolution limit.** The finest wavelength a pixel grid can hold is 2 pixels, the Nyquist limit: 1.1 µm at 0.55 µm/px. A cutoff finer than that has almost nothing to act on. With 0.55 µm pixels, the 0.35 µm low cutoff passes everything that reaches it (gain 0.99 even at 1.1 µm), so the filter effectively acts as a 4 µm high-pass. The script prints a `!!` warning whenever a cutoff is beyond the resolution limit. The low cutoff starts to matter with smaller pixels (higher magnification or a finer camera).

**Options.**

```
python nd2_workflow.py --bandpass-min-um 0.5 --bandpass-max-um 10   # change the cutoffs
python nd2_workflow.py --no-bandpass                                # unfiltered phase
```

The pixel size comes from the nd2's own calibration (`"voxel_um"`). The settings used are saved in `_metadata.pkl` as `"video_bandpass"`.

### Skipping the UI

Names and traps are saved to `objects/<file>_traps.pkl`. Pass that file back to re-run without drawing again:

```
python nd2_workflow.py --traps objects/soc-exp-14_traps.pkl
```

The UI needs a display (WSLg works). Without one, the script exits and asks for `--traps`.

### Channels

Channels are identified by the microscope modality recorded in the file, not by their name, since NIS lets a channel be named anything (the phase channel in SOC-EXP-14 is called `New`). The transmitted-light channel is keyed `phase`. Fluorescence channels are keyed by fluorophore (`GFP`, `YFP`, `RFP`, `CFP`, `BFP`, `Cy5`), matched from the channel name or, failing that, the emission wavelength.

If auto-detection picks the wrong phase channel, or can't decide, name it yourself:

```
python nd2_workflow.py --phase-channel "New"   # by name
python nd2_workflow.py --phase-channel 0       # or by index
```

### Positions

`--positions` restricts processing to a subset of stage positions, by scope number (1-based, as NIS shows them):

```
python nd2_workflow.py --positions "1-4,7"
```

### Output

Pickles land in `objects/`:

- `<file>_traps.pkl` holds `"positions"`, each with its `"name"`, `"scope_number"`, and `"traps"`. Each trap is `{"name": "T1", "vertices": array}`, where the vertices are an `(N, 2)` array of `(x, y)` = (column, row) pixel coordinates on the full-size image.
- `<file>_metadata.pkl` holds the experiment layout: `"channels"` (index, name, role, and key for each), `"positions"` (as above, plus stage x/y/z), `"sizes"`, `"voxel_um"`, and `"video_bandpass"` (the phase filter settings, or `None` with `--no-bandpass`). Each position also carries its `"fluor_background"` levels.

Both carry a `"Time"` key with the time axis in seconds, e.g. `data["Time"]`. Per-frame timestamps are used when they're valid. Some files carry corrupt timestamps, and then the nominal time-loop period is used instead (`"time_source"` records which).

`--objects <dir>` sends the pickles somewhere other than `objects/` (created if it doesn't exist).

### Using it from Python

```python
import nd2_workflow as w

with w.open_nd2("data/some_experiment.nd2") as f:
    channels = w.classify_channels(f)
    stacks = w.position_stack(f, 0, channels)   # {"phase": dask (T, Y, X), "GFP": ...}
    first_gfp = stacks["GFP"][0].compute()       # reads one frame
```

## Dependencies

- Python 3.10+
- [nd2](https://github.com/tlambert03/nd2)
- [dask](https://www.dask.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/), for the FFT
- [matplotlib](https://matplotlib.org/) with a Qt backend ([PySide6](https://doc.qt.io/qtforpython-6/)), for the trap UI
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), which bundles ffmpeg, for the videos
