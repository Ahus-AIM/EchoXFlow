"""Render the Task 1-3 README clips: one scan-converted ground-truth recording per task as a looping animation.

Each clip starts on a QRS trigger of the scanner's ECG and ends just before the next one, so it loops without a jump
(strain panels are already aligned to one beat by the renderer). The recordings come from the held-out fold:

    uv run python -m scripts.render_task_media --data-root /path/to/EchoXFlow

Colour Doppler is drawn through the recording's own two-dimensional colour table and gated harder than the library
default so little colour sits on bright tissue;
tissue Doppler is scan-converted with aliasing-aware interpolation and coloured through its table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from echoxflow import load_croissant, open_recording, render_recording_video
from echoxflow.plotting.gating import BloodGateConfig
from echoxflow.plotting.style import PlotStyle

README_BLOOD_GATE = BloodGateConfig(max_bmode_intensity=0.40, min_power_minus_bmode=0.25)


@dataclass(frozen=True)
class Clip:
    name: str
    recording: str  # exam_id/recording_id
    modalities: tuple[str, ...]
    qrs_path: str | None  # QRS trigger timestamps that bound the clip; None renders the whole timeline
    beats: int


CLIPS = (
    Clip(
        "task1_tissue_doppler",
        "exam_70e269e91e104df4/recording_e9932d0a9c0bd8b3",
        ("2d_brightness_mode", "tissue_doppler"),
        "timestamps/ecg_qrs_trigger_times",
        beats=1,
    ),
    Clip(
        "task2_color_doppler",
        "exam_017367abb6d99d8d/recording_11cdd7e1d283f6c2",
        ("2d_brightness_mode", "2d_color_doppler"),
        "timestamps/ecg_qrs_trigger_times",
        beats=1,
    ),
    Clip(
        "task3_lv_segmentation",
        "exam_017367abb6d99d8d/recording_286428c197b7831d",
        ("seq03",),
        None,
        beats=1,
    ),
)


def _window(root: Path, record, clip: Clip) -> tuple[float, float] | None:
    """First ``clip.beats`` consecutive QRS triggers that are at least 0.3 s apart, as ``(start, end)`` seconds."""
    if clip.qrs_path is None:
        return None
    triggers = np.sort(np.asarray(open_recording(record, root=root).load_array(clip.qrs_path), dtype=np.float64))
    for start in range(len(triggers) - clip.beats):
        window = triggers[start : start + clip.beats + 1]
        if np.all(np.diff(window) >= 0.3):
            return float(window[0]), float(window[-1])
    raise SystemExit(f"{clip.recording} has fewer than {clip.beats + 1} usable QRS triggers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--width", type=int, default=540)
    parser.add_argument("--height", type=int, default=468)
    parser.add_argument("--max-fps", type=float, default=30.0)
    parser.add_argument("--only", nargs="*", default=None, help="clip names to render (default: all)")
    parser.add_argument(
        "--format", default="webp", choices=("webp", "webm", "mp4"), help="animated WebP plays on GitHub (default)"
    )
    args = parser.parse_args()

    catalog = load_croissant(root=args.data_root)
    by_id = {f"{r.exam_id}/{r.recording_id}": r for r in catalog.recordings}
    style = replace(
        PlotStyle.from_config(),
        width_px=args.width,
        height_px=args.height,
        show_figure_title=False,
        show_panel_labels=False,
        blood_gate=README_BLOOD_GATE,
    )
    for clip in CLIPS:
        if args.only is not None and clip.name not in args.only:
            continue
        record = by_id[clip.recording]
        window = _window(args.data_root, record, clip)
        path = render_recording_video(
            record,
            args.output_dir / f"{clip.name}.{args.format}",
            root=args.data_root,
            modalities=clip.modalities,
            view_mode="cartesian",
            style=style,
            max_fps=args.max_fps,
            time_range=window,
        )
        span = "whole timeline" if window is None else f"{window[0]:.3f}-{window[1]:.3f} s"
        print(f"{path}: {clip.recording} {span}, {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
