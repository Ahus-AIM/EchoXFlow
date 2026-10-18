"""Render the Task 4 README media: a 4:3 overview still and one real-vs-generated animated WebP per apical view.

By default each clip comes from the held-out evaluation fold and spans exactly one heartbeat: it starts on an R peak,
and its 32 frames end where the next R peak begins, so the animations loop without a jump.
Recordings use the evaluation's clip order, noise and sampling steps:

    uv run python -m scripts.render_task4_media \
      --beamspace OUT/video_diffusion/fold4/beamspace_0128/ema.pt \
      --cartesian OUT/video_diffusion/fold4/cartesian_0128/ema.pt

With --saved-evaluation evaluation_epoch010, render the existing evaluated 32-frame clips on CPU instead.
These clips are not constrained to heartbeat boundaries. --pick selects a recording from that evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from echoxflow import open_recording
from tasks.video_diffusion.dataset import BMODE_PATH, ECG_PATH, ClipRecording, build_datasets
from tasks.video_diffusion.evaluate import (
    _hash_key,
    clip_id,
    generate_clip,
    load_checkpoint,
    materialize,
    recording_geometry,
    select_clips,
)

BACKGROUND = (34, 34, 34)
TEXT = (220, 220, 220)
REFERENCE = (110, 235, 200)
ECG_COLOR = (70, 205, 70)
GRID = (120, 120, 120)
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def _clip(root: Path, recording: ClipRecording, start: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Native B-mode frames ``[T, R, A]`` (uint8) and their timestamps for the clip."""
    store = open_recording(root / recording.zarr_path, root=root)
    span = 1 + (frames - 1) * recording.frame_stride
    stream = store.load_stream_slice(BMODE_PATH, start, start + span)
    positions = np.arange(frames) * recording.frame_stride
    return np.asarray(stream.data)[positions], np.asarray(stream.timestamps, dtype=np.float64)[positions]


def _ecg(root: Path, recording: ClipRecording) -> tuple[np.ndarray, np.ndarray]:
    """Raw ECG; for display it is scaled to the shown window (the model input clips the QRS peaks)."""
    store = open_recording(root / recording.zarr_path, root=root)
    ecg = store.load_stream(ECG_PATH)
    return np.asarray(ecg.timestamps, dtype=np.float64), np.asarray(ecg.data, dtype=np.float64).reshape(-1)


def _r_peaks(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """R-peak times, Pan-Tompkins style: QRS complexes are the steepest part of the ECG, so peaks of the squared slope
    integrated over 80 ms above 40% of its maximum, at least 0.3 s apart, each refined to the largest deflection
    within 50 ms. Amplitude alone would also pick tall T waves."""
    rate = 1.0 / float(np.median(np.diff(times)))
    x = signal - np.median(signal)
    slope = np.gradient(x) ** 2
    width = max(1, int(0.08 * rate))
    energy = np.convolve(slope, np.ones(width) / width, mode="same")
    threshold = 0.4 * float(np.percentile(energy, 99.9))
    min_gap = int(0.3 * rate)
    candidates = (
        np.flatnonzero((energy[1:-1] > threshold) & (energy[1:-1] >= energy[:-2]) & (energy[1:-1] >= energy[2:])) + 1
    )
    peaks: list[int] = []
    for index in candidates:
        if peaks and index - peaks[-1] < min_gap:
            if energy[index] > energy[peaks[-1]]:
                peaks[-1] = int(index)
        else:
            peaks.append(int(index))
    reach = int(0.05 * rate)
    refined = [
        max(0, index - reach) + int(np.argmax(np.abs(x[max(0, index - reach) : index + reach + 1]))) for index in peaks
    ]
    return times[refined]


def _one_beat_start(root: Path, recording: ClipRecording, frames: int) -> int | None:
    """A clip start on an R peak whose ``frames`` frames end one frame before the next R peak, or ``None``."""
    store = open_recording(root / recording.zarr_path, root=root)
    frame_times = np.asarray(store.load_timestamps(BMODE_PATH), dtype=np.float64)
    ecg_times, ecg = _ecg(root, recording)
    if len(ecg_times) < 100 or np.any(np.diff(ecg_times) <= 0):
        return None
    peaks = _r_peaks(ecg_times, ecg)
    best: tuple[float, int] | None = None
    for start in recording.candidate_starts:
        times = frame_times[start + recording.frame_stride * np.arange(frames)]
        dt = float(np.median(np.diff(times)))
        first = peaks[np.argmin(np.abs(peaks - times[0]))] if len(peaks) else np.inf
        later = peaks[peaks > first + 0.3]
        if not len(later):
            continue
        error = abs(first - times[0]) + abs(later[0] - (times[-1] + dt))
        if abs(first - times[0]) <= dt / 2 and abs(later[0] - (times[-1] + dt)) <= dt / 2:
            if best is None or error < best[0]:
                best = (error, int(start))
    return None if best is None else best[1]


def _one_beat_length(times: np.ndarray, ecg: tuple[np.ndarray, np.ndarray]) -> int:
    """Trim trailing frames only; require both R-peak boundaries within half a frame."""
    peaks = _r_peaks(*ecg)
    dt = float(np.median(np.diff(times)))
    if len(peaks) < 2:
        raise ValueError("The displayed ECG must contain two R peaks")
    first = int(np.argmin(np.abs(peaks - times[0])))
    if first + 1 >= len(peaks) or abs(peaks[first] - times[0]) > dt / 2:
        raise ValueError("The selected clip must start at an R peak")
    count = round((peaks[first + 1] - times[0]) / dt)
    if not 2 <= count <= len(times) or abs(times[count - 1] + dt - peaks[first + 1]) > dt / 2:
        raise ValueError("The selected clip does not contain a complete heartbeat")
    return count


def _one_beat_candidates(
    dataset, view: str, frames: int, seed: int, limit: int, only: str | None = None
) -> list[tuple[ClipRecording, int]]:
    """Up to ``limit`` recordings of ``view`` with a one-heartbeat clip, in the evaluation's seeded order (or just the
    recording ``only``, given as ``exam_id/recording_id``)."""
    ranked = sorted(dataset.grouped[view], key=lambda r: _hash_key(seed, "recording", r.exam_id, r.recording_id))
    if only is not None:
        ranked = [r for r in ranked if f"{r.exam_id}/{r.recording_id}" == only]
        if not ranked:
            raise SystemExit(f"{only} is not a {view} recording of the evaluation fold")
    found = []
    for recording in ranked:
        start = _one_beat_start(dataset.root, recording, frames)
        if start is not None:
            found.append((recording, start))
            if len(found) >= limit:
                break
    if only is not None and not found:
        raise SystemExit(f"{only} has no one-heartbeat clip; pick another recording with --candidates")
    return found


def _on_background(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """uint8 ``[T, H, W]`` -> RGB with everything outside the imaging sector in the page background colour."""
    rgb = np.repeat(frames[..., None], 3, axis=-1)
    rgb[:, ~mask] = BACKGROUND
    return rgb


def _draw_ecg(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ecg: tuple[np.ndarray, np.ndarray],
    window: tuple[float, float],
    markers: list[tuple[float, tuple[int, int, int]]],
) -> None:
    left, top, right, bottom = box
    times, signal = ecg
    inside = (times >= window[0]) & (times <= window[1])
    t, v = times[inside], signal[inside]
    if len(v):
        v = v - np.median(v)
        v = v / max(float(np.abs(v).max()), 1e-9)
    for time_s, color in markers:
        x = left + (time_s - window[0]) / (window[1] - window[0]) * (right - left)
        draw.line([(x, top), (x, bottom)], fill=color, width=1 if color == GRID else 2)
    if len(t) > 1:
        x = left + (t - window[0]) / (window[1] - window[0]) * (right - left)
        y = (top + bottom) / 2 - v / 1.05 * (bottom - top) / 2
        draw.line(list(zip(x.tolist(), y.tolist())), fill=ECG_COLOR, width=2)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and draw.textlength(f"{lines[-1]} {word}", font=font) <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def _centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.FreeTypeFont, fill) -> None:
    width = draw.textlength(text, font=font)
    draw.text((x - width / 2, y), text, font=font, fill=fill)


VALID = (52, 72, 52)


def _beat_window(times: np.ndarray) -> tuple[float, float]:
    """ECG span of the clip's heartbeat: half a frame before its first frame to half a frame before the next beat."""
    dt = float(np.median(np.diff(times)))
    return times[0] - dt / 2, times[-1] + dt / 2


def overview_still(frames: np.ndarray, times: np.ndarray, ecg, path: Path) -> None:
    """400 x 300 still: the reference frame, two later frames of the same clip and the synchronized ECG."""
    image = Image.new("RGB", (400, 300), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = _font(13)
    elapsed = times - times[0]
    picks = [0, int(np.argmin(np.abs(elapsed - 0.2))), int(np.argmin(np.abs(elapsed - 0.4)))]
    size, gap = 124, 6
    left = (400 - 3 * size - 2 * gap) // 2
    for slot, index in enumerate(picks):
        x = left + slot * (size + gap)
        image.paste(Image.fromarray(frames[index]).resize((size, size), Image.BILINEAR), (x, 22))
        label, color = ("reference", REFERENCE) if index == 0 else (f"t+{elapsed[index]:.2f}s", TEXT)
        _centered(draw, x + size / 2, 4, label, font, color)
    window = _beat_window(times)
    markers = [(times[index], REFERENCE if index == 0 else GRID) for index in picks]
    _draw_ecg(draw, (left, 168, 400 - left, 284), ecg, window, markers)
    _centered(draw, 200, 284, "ECG", _font(11), TEXT)
    image.save(path, optimize=True)


def comparison_animation(
    reference: np.ndarray,
    generated: list[tuple[str, np.ndarray]],
    target: np.ndarray,
    times: np.ndarray,
    ecg,
    ecg_config,
    view: str,
    path: Path,
    size: int = 256,
    window_label: str = "shown heartbeat",
) -> None:
    """Input | Output | Target, separated by lines.

    Input (still): the first frame and the ECG, with the frame's time marked on it. Output: the
    generated clips; Target: the real clip; both in real time above one still ECG strip spanning the same input period,
    where the shaded interval is the part the clips show and the moving line is the current frame.
    ``window_label`` distinguishes one-heartbeat selections from fixed-length evaluation clips.
    """
    gap, separator, top, ecg_height, tick, inner = 12, 28, 50, 70, 16, 0  # the two outputs touch
    input_width = size
    output_left = gap + input_width + separator
    output_right = output_left + len(generated) * size + (len(generated) - 1) * inner
    target_left = output_right + separator
    width = target_left + size + gap
    ecg_top = top + size + 12
    height = ecg_top + ecg_height + tick + 6
    heading, label, small = _font(16), _font(13), _font(11)

    base = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(base)
    for x in (gap + input_width + separator / 2, output_right + separator / 2):
        draw.line([(x, 8), (x, ecg_top - 8)], fill=GRID, width=1)
    _centered(draw, gap + input_width / 2, 6, "Input", heading, REFERENCE)
    _centered(draw, (output_left + output_right) / 2, 6, "Output", heading, TEXT)
    _centered(draw, target_left + size / 2, 6, "Target", heading, TEXT)
    base.paste(Image.fromarray(reference).resize((input_width, input_width), Image.BILINEAR), (gap, top))
    _centered(draw, gap + input_width / 2, 30, f"first frame ({view})", label, TEXT)
    for slot, (name, _frames) in enumerate(generated):
        _centered(draw, output_left + slot * (size + inner) + size / 2, 30, name, label, TEXT)
    _centered(draw, target_left + size / 2, 30, "Real clip", label, TEXT)

    # The period the clip's ECG windows cover, cut to the ECG the recording holds (it often starts with the video).
    period = (
        max(times[0] - float(ecg_config["history_seconds"]), float(ecg[0][0])),
        min(times[-1] + float(ecg_config["future_seconds"]), float(ecg[0][-1])),
    )
    # Input ECG (unclipped, unlike the model's normalized input) with the first frame's time marked, and a caption.
    left_box = (gap, top + size + 6, gap + input_width, top + size + 62)
    _draw_ecg(draw, left_box, ecg, period, [(times[0], REFERENCE)])
    caption_y = left_box[3] + 6
    for line in _wrap(draw, "One frame, corresponding ECG, and their relation", label, input_width):
        _centered(draw, gap + input_width / 2, caption_y, line, label, TEXT)
        caption_y += 16

    # Output/target ECG: the same period, still; the shaded beat is the part the clips show.
    right_box = (output_left, ecg_top, target_left + size, ecg_top + ecg_height)

    def x_at(time_s: float) -> float:
        return right_box[0] + (time_s - period[0]) / (period[1] - period[0]) * (right_box[2] - right_box[0])

    beat = _beat_window(times)
    draw.rectangle([x_at(beat[0]), right_box[1], x_at(beat[1]), right_box[3]], fill=VALID)
    _draw_ecg(draw, right_box, ecg, period, [])
    _centered(draw, (x_at(beat[0]) + x_at(beat[1])) / 2, right_box[3] + 3, window_label, small, TEXT)

    images = []
    for index in range(len(times)):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for slot, (_name, frames) in enumerate(generated):
            x = output_left + slot * (size + inner)
            image.paste(Image.fromarray(frames[index]).resize((size, size), Image.BILINEAR), (x, top))
        image.paste(Image.fromarray(target[index]).resize((size, size), Image.BILINEAR), (target_left, top))
        x = x_at(times[index])
        draw.line([(x, right_box[1] - 3), (x, right_box[3] + 3)], fill=REFERENCE, width=2)
        stamp = f"t = {times[index] - times[0]:.2f} s"
        draw.text((right_box[2] - draw.textlength(stamp, font=small), right_box[3] + 3), stamp, font=small, fill=TEXT)
        images.append(image)
    durations = np.diff(times, append=times[-1] + np.median(np.diff(times))) * 1000.0
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=[round(d) for d in durations],
        loop=0,
        quality=90,
        method=6,
    )


#: Recordings shown per view, picked from ``--candidates`` sheets of the first one-heartbeat clips in the evaluation's
#: order for clearly visible anatomy (the GIFs illustrate the task; they are not a random sample).
PICKS: dict[str, str] = {
    "A2C": "exam_3ec59d2f86c8e129/recording_e9dbbd4215419807",
    "A3C": "exam_5e4f10383942305a/recording_124842d657450ef7",
    "A4C": "exam_bff626dafea635ec/recording_294fef10bdbcb56b",
}


def _candidate_sheet(root: Path, found: list[tuple[ClipRecording, int]], path: Path, size: int = 180) -> None:
    columns = 6
    rows = (len(found) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size, rows * (size + 28)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for index, (recording, start) in enumerate(found):
        frame, _times = _clip(root, recording, start, 1)
        image = materialize(frame, space="beamspace", geometry=recording_geometry(recording), size=size)[0]
        x, y = (index % columns) * size, (index // columns) * (size + 28)
        sheet.paste(Image.fromarray(image), (x, y + 28))
        draw.text(
            (x + 4, y + 2),
            f"{index}: {recording.exam_id[-6:]}/{recording.recording_id[-6:]}",
            font=_font(11),
            fill=TEXT,
        )
        draw.text((x + 4, y + 14), f"{recording.exam_id}/{recording.recording_id}"[:34], font=_font(8), fill=GRID)
    sheet.save(path)
    with open(path.with_suffix(".txt"), "w", encoding="utf-8") as handle:
        for index, (recording, start) in enumerate(found):
            handle.write(f"{index}\t{recording.exam_id}/{recording.recording_id}\t{start}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beamspace", type=Path, required=True, help="ema.pt of a beamspace Task 4 model")
    parser.add_argument("--cartesian", type=Path, required=True, help="ema.pt of the matching Cartesian model")
    parser.add_argument("--data-root", default=None, help="EchoXFlow root (default: from the training config)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("assets"), help="Directory for the images and animations."
    )
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--saved-evaluation",
        help="Read samples.npz from this evaluation subdirectory of each run, without model inference.",
    )
    parser.add_argument(
        "--trim-to-beat",
        action="append",
        choices=("A2C", "A3C", "A4C"),
        default=[],
        help="Trim extra end frames for this view; the clip must already start at an R peak and contain a full beat.",
    )
    parser.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="VIEW=EXAM/RECORDING",
        help="recording to show for a view (default: PICKS); see --candidates",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help="instead of rendering, write a sheet of the first frames of this many candidates per view",
    )
    args = parser.parse_args()
    if args.saved_evaluation and args.candidates:
        parser.error("--candidates selects one-heartbeat clips; use --pick for saved evaluation clips")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.saved_evaluation else args.device)
    models = {}
    saved = {}
    for space, path in (("beamspace", args.beamspace), ("cartesian", args.cartesian)):
        if args.saved_evaluation:
            directory = path.parent / args.saved_evaluation
            metadata = json.loads((directory / "metrics.json").read_text())
            if Path(metadata["checkpoint"]).resolve() != path.resolve():
                raise ValueError(f"Saved evaluation does not belong to {path}")
            if metadata["sampling_steps"] != args.sampling_steps or metadata["coordinate_space"] != space:
                raise ValueError("Saved evaluation sampling steps or coordinate space disagree")
            config = torch.load(path, map_location="cpu", weights_only=False)["config"]
            models[space] = (None, config)
            with np.load(directory / "samples.npz", allow_pickle=False) as arrays:
                if str(arrays["coordinate_space"]) != space:
                    raise ValueError("Saved samples have the wrong coordinate space")
                saved[space] = dict(zip(arrays["clip_ids"].tolist(), arrays["generated"], strict=True))
        else:
            models[space] = load_checkpoint(path, device)
    datasets = {space: build_datasets(config, data_root=args.data_root)[1] for space, (_, config) in models.items()}
    config = models["cartesian"][1]
    seed = int(config.get("evaluation", {}).get("seed", 0))
    clip_frames = int(config["data"].get("clip_frames", 32))
    reference = datasets["cartesian"]
    display = max(192, int(config["data"]["spatial_size"]))  # show the models' own resolution, at least 192 px
    picks = {**PICKS, **dict(item.split("=", 1) for item in args.pick)}
    if args.candidates:
        for view in reference.views:
            found = _one_beat_candidates(reference, view, clip_frames, seed, args.candidates)
            _candidate_sheet(reference.root, found, args.output_dir / f"task4_candidates_{view.lower()}.png")
        return 0
    if args.saved_evaluation:
        picks = dict(item.split("=", 1) for item in args.pick)
        evaluated = select_clips(reference, clips_per_view=int(config["evaluation"]["clips_per_view"]), seed=seed)
        expected_ids = {clip_id(r, start, clip_frames) for r, start in evaluated}
        if any(set(samples) != expected_ids for samples in saved.values()):
            raise ValueError("Saved clips do not match the current held-out evaluation selection")
        clips = []
        for view in reference.views:
            matches = [
                (r, s)
                for r, s in evaluated
                if r.view == view and (view not in picks or f"{r.exam_id}/{r.recording_id}" == picks[view])
            ]
            if not matches:
                raise ValueError(f"No saved evaluation clip matches {view}")
            clips.append(matches[0])
    else:
        clips = [
            _one_beat_candidates(reference, view, clip_frames, seed, 1, only=picks.get(view))[0]
            for view in reference.views
        ]
    for recording, start in clips:
        geometry = recording_geometry(recording)
        native, times = _clip(reference.root, recording, start, clip_frames)
        real = materialize(native, space="beamspace", geometry=geometry, size=display)
        mask = materialize(np.full_like(native[:1], 255), space="beamspace", geometry=geometry, size=display)[0] > 127
        panels = []
        for space, (model, model_config) in models.items():
            generated = (
                saved[space][clip_id(recording, start, clip_frames)]
                if args.saved_evaluation
                else generate_clip(
                    model,
                    model_config,
                    datasets[space].load_clip(recording, start),
                    noise_key=clip_id(recording, start, clip_frames),
                    seed=seed,
                    sampling_steps=args.sampling_steps,
                    device=device,
                )
            )
            future = materialize(generated, space=space, geometry=geometry, size=display)
            panels.append((f"{space.capitalize()} model", _on_background(np.concatenate([real[:1], future]), mask)))
        real_rgb = _on_background(real, mask)
        ecg = _ecg(reference.root, recording)
        view = recording.view.upper()
        if view in args.trim_to_beat:
            count = _one_beat_length(times, ecg)
            times, real_rgb = times[:count], real_rgb[:count]
            panels = [(name, frames[:count]) for name, frames in panels]
        animation = args.output_dir / f"task4_{view.lower()}.webp"
        comparison_animation(
            real_rgb[0],
            panels,
            real_rgb,
            times,
            ecg,
            config["ecg"],
            view,
            path=animation,
            size=display,
            window_label="shown clip" if args.saved_evaluation and view not in args.trim_to_beat else "shown heartbeat",
        )
        print(f"wrote {view}: {clip_id(recording, start, clip_frames)}", flush=True)
        if view == "A4C":
            overview_still(real_rgb, times, ecg, args.output_dir / "task4_video_diffusion_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
