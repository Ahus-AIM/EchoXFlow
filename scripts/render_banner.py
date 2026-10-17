"""Render the README banner: one patient's views over the heartbeat, with Doppler spectra and ECG below.

Top row: a parasternal long-axis B-mode, A4C tissue Doppler, colour Doppler and the LV contour, and a parasternal
short axis at the mitral valve, each looping one heartbeat from its QRS trigger to the next. Native frames are
blended to a fixed number of phases per beat, so the panels stay in phase without
judder. Second row: a pulsed-wave tissue Doppler trace of the mitral annulus on its own baseline (s' above, e' and
a' below), then the mitral-inflow pulsed-wave and aortic-outflow continuous-wave spectra on one shared baseline,
mitral rising above it (flow towards the probe) and aortic hanging below it (away). Every strip spans three beats,
each warped to a third of the width; a recording with fewer beats repeats its first ones. Bottom: the ECG of the
aortic recording over the same three beats in its own band. One cursor sweeps the spectra and ECG while the sectors
repeat their beat three times.
Sectors are cropped to 15 cm, colour bars are off, and the spectra fade into the figure background.

    uv run python -m scripts.render_banner --data-root /path/to/EchoXFlow
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from echoxflow import load_croissant, open_recording
from echoxflow.plotting.gating import BloodGateConfig, TissueGateConfig
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline
from echoxflow.plotting.writers import write_video

EXAM = "exam_62a8547f9cde99ec"
BLOOD_GATE = BloodGateConfig(max_bmode_intensity=0.40, min_power_minus_bmode=0.25)
TISSUE_GATE = TissueGateConfig(min_bmode_intensity=0.18, smoothing_px=2)  # blood pool black, not speckly
BLUE_GAIN = 1.35  # brightens the blue-dominant tissue colours of the scanner table
ECG_GAIN = 1.5  # ECG amplitude relative to its band
BACKGROUND = (34, 34, 34)  # the figure background #222222
GATED_TISSUE = (26, 26, 26)  # tissue Doppler pixels the gate hides, a little darker than the background
ECG_TRACE = (74, 196, 74)
ECG_CURSOR = (255, 96, 96)
SPECTRUM_STOPS = (
    (0.0, (34, 34, 34)),
    (0.3, (34, 34, 34)),
    (0.5, (110, 96, 18)),
    (0.82, (199, 178, 38)),
    (1.0, (255, 255, 255)),
)
# The weakest third of the spectrum is the figure background #222222, so noise disappears into it; the ramp then
# climbs through dim yellows and reaches white only at the very top, which keeps most of the envelope in colour.
QRS_PATH = "timestamps/ecg_qrs_trigger_times"


@dataclass(frozen=True)
class Panel:
    recording: str
    modalities: tuple[str, ...]
    strain: bool  # strain panels are aligned to one beat by the renderer and carry no QRS trigger array


@dataclass(frozen=True)
class Strip:
    recording: str
    modality: str  # "1d_pulsed_wave_doppler" or "1d_continuous_wave_doppler"
    side: str  # keep the velocities "above" or "below" the baseline, or "both"
    direction: str  # draw the flow "up" from a baseline at the bottom or "down" from one at the top
    extent: float = 1.0  # fraction of that side's velocity range to keep, from the baseline outwards
    height: int = 90  # pixel height of the strip


PANELS = (
    Panel(f"{EXAM}/recording_d61c5fda3ebaa910", ("2d_brightness_mode",), False),  # PLAX B-mode
    Panel(f"{EXAM}/recording_5b9fe9fd1a42eefb", ("2d_brightness_mode", "tissue_doppler"), False),
    Panel(f"{EXAM}/recording_f12f8f4a2d1343fc", ("2d_brightness_mode", "2d_color_doppler"), False),
    Panel(f"{EXAM}/recording_57891e0354517bc8", ("seq03",), True),
    Panel(f"{EXAM}/recording_3fc5dd481f3b14a4", ("2d_brightness_mode",), False),  # PSAX at the mitral valve
)
STRIPS = (  # stacked top to bottom; the lower two share one baseline
    Strip(f"{EXAM}/recording_2ef831af3c9e704e", "1d_pulsed_wave_doppler", "both", "up", height=120),  # annulus TDI
    Strip(f"{EXAM}/recording_134f79db4ea6df04", "1d_pulsed_wave_doppler", "above", "up", 0.8),  # mitral inflow
    Strip(f"{EXAM}/recording_6256bae4c8e4202f", "1d_continuous_wave_doppler", "below", "down", 0.5),  # aortic
)


def _qrs_beats(store, count: int) -> list[tuple[float, float]]:
    """``count`` consecutive beats between QRS triggers at least 0.3 s apart, cycling when there are fewer."""
    triggers = np.sort(np.asarray(store.load_array(QRS_PATH), dtype=np.float64))
    triggers = triggers[np.r_[True, np.diff(triggers) >= 0.3]]
    beats = [(float(a), float(b)) for a, b in zip(triggers[:-1], triggers[1:])]
    if not beats:
        raise SystemExit(f"{store.path} has fewer than two QRS triggers")
    if len(beats) < count:
        print(f"{store.path.name}: {len(beats)} complete beats, repeating them to {count}")
    return [beats[index % len(beats)] for index in range(count)]


def _blend_beat(renderer, specs, ecg: TraceSpec, window: tuple[float, float], phase: np.ndarray) -> np.ndarray:
    """Native frames of the beat, linearly blended at ``phase`` fractions of the window."""
    start, end = window
    native = np.asarray(select_timeline(specs).timestamps, dtype=np.float64)
    step = float(np.median(np.diff(native))) if native.size > 1 else 0.0
    keep = native[(native >= start - step) & (native <= end + step)]
    if keep.size == 0:
        raise SystemExit(f"No frames between {start:.2f} and {end:.2f} s")
    frames = np.stack(list(renderer.iter_frames(specs, ecg, keep))).astype(np.float32)
    targets = start + phase * (end - start)
    upper = np.clip(np.searchsorted(keep, targets), 1, keep.size - 1)
    lower = upper - 1
    weight = np.clip((targets - keep[lower]) / np.maximum(keep[upper] - keep[lower], 1e-9), 0.0, 1.0)
    weight = np.where(targets <= keep[0], 0.0, weight)
    blended = frames[lower] * (1.0 - weight)[:, None, None, None] + frames[upper] * weight[:, None, None, None]
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _spectral_strip(root: Path, record, strip: Strip, beats: list[tuple[float, float]], size: tuple[int, int]):
    """``(height, width, 3)`` spectrum of ``beats`` warped to equal widths, cropped to one side of the baseline.

    The result is oriented so the flow points in ``strip.direction``, with the baseline at the top or bottom.
    """
    width, height = size
    store = open_recording(record, root=root)
    loaded = store.load_modality(strip.modality)
    metadata = store.spectral_metadata(loaded.data_path)
    data = np.asarray(loaded.data, dtype=np.float32) / 255.0  # [columns, rows], velocity decreasing with row
    data = np.clip((data - 0.1) / 0.8, 0.0, 1.0) ** 0.8  # lift the envelope above the noise floor
    times = np.asarray(loaded.timestamps, dtype=np.float64)
    baseline = int(round(float(metadata.baseline_row)))
    skip = 3  # rows next to the baseline carry wall clutter, not flow
    if strip.side == "both":
        rows = _occupied_rows(data)
    elif strip.side == "below":
        rows = slice(baseline + skip, baseline + max(skip + 1, int(round((data.shape[1] - baseline) * strip.extent))))
    else:
        rows = slice(max(0, baseline - int(round(baseline * strip.extent))), max(1, baseline - skip))
    beat_width = width // len(beats)
    pieces = []
    for index, (start, end) in enumerate(beats):
        columns = (times >= start) & (times < end)
        piece = data[columns][:, rows].T  # [rows, columns]
        piece_width = width - beat_width * (len(beats) - 1) if index == len(beats) - 1 else beat_width
        resized = Image.fromarray((piece * 255).astype(np.uint8)).resize((piece_width, height), Image.BILINEAR)
        pieces.append(np.asarray(resized, dtype=np.float32) / 255.0)
    image = _colourise(np.concatenate(pieces, axis=1))
    if strip.side == "both":
        return image
    rises = strip.side == "above"  # velocities above the baseline sit at the top of the native rows
    if rises == (strip.direction == "down"):
        image = image[::-1].copy()
    return image


def _occupied_rows(data: np.ndarray, threshold: float = 0.35, margin: int = 2) -> slice:
    """The row range that ever carries signal above ``threshold`` (99th percentile over time), plus a margin."""
    strong = np.flatnonzero(np.percentile(data, 99, axis=0) > threshold)
    if strong.size == 0:
        return slice(0, data.shape[1])
    return slice(max(0, int(strong[0]) - margin), min(data.shape[1], int(strong[-1]) + margin + 1))


def _colourise(values: np.ndarray) -> np.ndarray:
    positions = np.asarray([stop for stop, _ in SPECTRUM_STOPS])
    colours = np.asarray([rgb for _, rgb in SPECTRUM_STOPS], dtype=np.float32)
    out = np.stack([np.interp(values, positions, colours[:, channel]) for channel in range(3)], axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _ecg_line(store, beats: list[tuple[float, float]], size: tuple[int, int], top: int) -> list[tuple[float, float]]:
    """Polyline of the ECG over ``beats``, each beat warped to an equal share of the width, ``top`` pixels down."""
    width, height = size
    ecg = store.load_stream("ecg")
    times = np.asarray(ecg.timestamps, dtype=np.float64)
    signal = np.asarray(ecg.data, dtype=np.float64).reshape(-1)
    inside = (times >= beats[0][0]) & (times < beats[-1][1])
    times, signal = times[inside], signal[inside]
    signal = signal - np.median(signal)
    signal = signal / max(float(np.abs(signal).max()), 1e-9)
    x = np.zeros_like(times)
    for index, (start, end) in enumerate(beats):
        beat = (times >= start) & (times < end)
        x[beat] = (index + (times[beat] - start) / (end - start)) / len(beats) * (width - 1)
    y = top + height / 2 - 22 - signal * ECG_GAIN * (height / 2 - 4)  # well above centre: the S wave dips low
    return list(zip(x.tolist(), y.tolist()))


def _brighten_blue(frames: np.ndarray) -> np.ndarray:
    """Scale blue-dominant pixels by ``BLUE_GAIN``, and flatten the gated-away grey B-mode to ``GATED_TISSUE``.

    The table's blues are much darker than its reds. Inside the sector, grey pixels are the B-mode the tissue gate
    left uncoloured; they become one flat tone a little darker than the figure background.
    """
    rgb = frames.astype(np.float32)
    blue = (rgb[..., 2] > rgb[..., 0] * 1.2) & (rgb[..., 2] > rgb[..., 1] * 1.2) & (rgb[..., 2] > 40)
    rgb[blue] = np.minimum(rgb[blue] * BLUE_GAIN, 255.0)
    spread = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    background = np.all(frames == BACKGROUND, axis=-1)
    rgb[(spread <= 3) & ~background] = GATED_TISSUE
    return rgb.astype(np.uint8)


def _strain_ecg(root: Path, record, role: str) -> TraceSpec:
    """The strain panel's own ECG, shifted so the beat the renderer aligns the panel to starts at 0 s."""
    store = open_recording(record, root=root)
    contour = np.asarray(store.load_array(f"timestamps/{role}_contour"), dtype=np.float64).reshape(-1)
    triggers = np.sort(np.asarray(store.load_array(f"timestamps/{role}_ecg_qrs"), dtype=np.float64))
    overlap = [min(b, contour[-1]) - max(a, contour[0]) for a, b in zip(triggers[:-1], triggers[1:])]
    start = float(triggers[int(np.argmax(overlap))])
    signal = np.asarray(store.load_array(f"data/{role}_ecg"), dtype=np.float32).reshape(-1)
    times = np.asarray(store.load_array(f"timestamps/{role}_ecg"), dtype=np.float64).reshape(-1) - start
    return TraceSpec(signal=signal, timestamps=times, label="ECG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/banner.webp"))
    parser.add_argument("--panel-width", type=int, default=400)
    parser.add_argument("--panel-height", type=int, default=300)
    parser.add_argument("--ecg-height", type=int, default=110, help="height of the ECG band under the spectra")
    parser.add_argument("--trim-sides", type=int, default=14, help="pixels cut from each side of every sector panel")
    parser.add_argument("--trim-top", type=int, default=10, help="pixels cut above the sector panels")
    parser.add_argument("--trim-bottom", type=int, default=4, help="pixels cut under the sector panels")
    parser.add_argument("--beats", type=int, default=3, help="heartbeats per loop")
    parser.add_argument("--frames", type=int, default=30, help="frames per heartbeat")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-width", type=int, default=1240, help="downscale the composed frames to this width")
    args = parser.parse_args()

    catalog = load_croissant(root=args.data_root)
    by_id = {f"{r.exam_id}/{r.recording_id}": r for r in catalog.recordings}
    base = replace(
        PlotStyle.from_config(),
        width_px=args.panel_width,
        height_px=args.panel_height,
        show_figure_title=False,
        show_panel_labels=False,
        show_cartesian_depth_ruler=False,
        show_colorbars=False,
        show_color_doppler_extent=False,
        show_ecg=False,
        cartesian_max_depth_m=0.15,
        tissue_doppler_brightness_lift=0.2,
        blood_gate=BLOOD_GATE,
        tissue_gate=TISSUE_GATE,
    )
    phase = np.arange(args.frames, dtype=np.float64) / args.frames
    columns = []
    renderer = RecordingPlotRenderer(style=base)
    for panel in PANELS:
        record = by_id[panel.recording]
        specs, ecg = renderer._load_specs(
            record, root=args.data_root, modalities=panel.modalities, view_mode="cartesian", show_annotations=True
        )
        if panel.strain:
            ecg = _strain_ecg(args.data_root, record, panel.modalities[0])
            times = np.asarray(select_timeline(specs).timestamps, dtype=np.float64)
            window = (float(times[0]), float(times[-1] + np.median(np.diff(times))))
        else:
            window = _qrs_beats(open_recording(record, root=args.data_root), 1)[0]
        frames = _blend_beat(renderer, specs, ecg, window, phase)
        frames = frames[
            :,
            args.trim_top : args.panel_height - args.trim_bottom,
            args.trim_sides : args.panel_width - args.trim_sides,
        ]
        columns.append(_brighten_blue(frames) if "tissue_doppler" in panel.modalities else frames)
        print(f"{panel.recording}: {window[0]:.3f}-{window[1]:.3f} s")
    sectors = np.concatenate([np.concatenate(columns, axis=2)] * args.beats, axis=0)
    width = sectors.shape[2]

    strips = []
    ecg_points = None
    for strip in STRIPS:
        record = by_id[strip.recording]
        store = open_recording(record, root=args.data_root)
        beats = _qrs_beats(store, args.beats)
        strips.append(_spectral_strip(args.data_root, record, strip, beats, (width, strip.height)))
        top = args.panel_height - args.trim_top - args.trim_bottom + sum(item.height for item in STRIPS)
        ecg_points = _ecg_line(store, beats, (width, args.ecg_height), top)
        print(f"{strip.recording}: beats {beats[0][0]:.2f}-{beats[-1][1]:.2f} s, {strip.direction}")
    assert ecg_points is not None
    ecg_band = np.full((args.ecg_height, width, 3), SPECTRUM_STOPS[0][1], dtype=np.uint8)
    strip_row = np.concatenate([*strips, ecg_band], axis=0)

    frames = []
    count = len(sectors)
    for index in range(count):
        image = Image.fromarray(np.concatenate([sectors[index], strip_row], axis=0))
        draw = ImageDraw.Draw(image)
        cursor = index / count * (width - 1)
        draw.line(ecg_points, fill=ECG_TRACE, width=2)
        sector_bottom = args.panel_height - args.trim_top - args.trim_bottom
        draw.line([(cursor, sector_bottom), (cursor, image.height)], fill=ECG_CURSOR, width=2)
        if args.output_width and args.output_width != image.width:
            scaled = (args.output_width, round(image.height * args.output_width / image.width))
            image = image.resize(scaled, Image.LANCZOS)
        frames.append(np.asarray(image))
    path = write_video(args.output, np.stack(frames), fps=args.fps)
    print(f"{path}: {frames[0].shape[1]}x{frames[0].shape[0]}, {count} frames, {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
