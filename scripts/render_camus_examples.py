"""Render the CAMUS README media: the first five CAMUS patients animated by an EchoXFlow ECG.

The animation shows the patients (A2C above A4C) whose end-diastolic frame is extended by the beamspace Task 4 model,
conditioned on the ECG and frame timing of one EchoXFlow clip. Frames are generated like the CAMUS evaluation (same
preparation, noise and sampling steps) and shown scan converted. The ECG clip is a one-heartbeat A4C clip of the
evaluation fold (R peak to R peak), so the animation loops without a jump; ``--ecg-clips`` renders one animation per
given recording (default ``PICKS``, a regular sinus rhythm).

    uv run python -m scripts.render_camus_examples \
      --checkpoint OUT/video_diffusion/fold4/beamspace_0256/ema.pt --camus-root /path/to/CAMUS-beamspace
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from echoxflow import find_recordings, load_croissant, open_recording
from echoxflow.scan import SectorGeometry
from scripts.render_task4_media import (
    BACKGROUND,
    REFERENCE,
    TEXT,
    _beat_window,
    _centered,
    _draw_ecg,
    _ecg,
    _font,
    _on_background,
    _one_beat_candidates,
    _one_beat_length,
)
from tasks.video_diffusion.dataset import build_datasets, prepare_frames
from tasks.video_diffusion.evaluate import generate_clip, load_checkpoint, materialize

VIEWS = ("A2C", "A4C")
#: Conditioning recording (``exam_id/recording_id`` of the evaluation fold): a regular sinus rhythm, picked from the
#: fold's A4C recordings that hold a one-heartbeat clip.
PICKS = ("exam_f9c77b631303f89c/recording_0d916e1a782047e6",)


def camus_frame(camus_root: Path, records: dict, patient: str, view: str) -> tuple[np.ndarray, SectorGeometry]:
    """The end-diastolic beamspace frame ``[R, A]`` (uint8) of one CAMUS recording and its sector geometry."""
    record = records[f"recording_camus_{patient}_{view.lower()}"]
    stream = open_recording(record, root=camus_root).load_stream("2d_brightness_mode")
    attrs = json.loads((camus_root / record.zarr_path / ".zattrs").read_text(encoding="utf-8"))
    ed = int(attrs["recording_manifest"]["metadata"]["camus"]["ed_frame_index"])
    geometry = stream.metadata.geometry
    if not isinstance(geometry, SectorGeometry):
        raise ValueError(f"{record.recording_id}: missing sector geometry")
    return np.asarray(stream.data)[ed], geometry


def animation(
    clips: dict[tuple[str, str], np.ndarray], patients: list[str], times: np.ndarray, ecg, path: Path
) -> None:
    """Patients as columns, A2C over A4C, above one still ECG strip with the moving frame marker."""
    size, gap, top, ecg_height = 160, 6, 26, 64
    width = gap + len(patients) * (size + gap)
    ecg_top = top + 2 * size + gap + 8
    height = ecg_top + ecg_height + 22
    label, small = _font(13), _font(11)
    base = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(base)
    for column, patient in enumerate(patients):
        _centered(draw, gap + column * (size + gap) + size / 2, 6, patient.replace("patient", "CAMUS "), label, TEXT)
    for row, view in enumerate(VIEWS):
        draw.text((gap + 4, top + row * (size + gap) + 4), view, font=small, fill=TEXT)
    # More ECG context than the model's windows, so the rhythm around the shown beat is visible.
    period = (max(times[0] - 3.0, float(ecg[0][0])), min(times[-1] + 1.5, float(ecg[0][-1])))
    box = (gap, ecg_top, width - gap, ecg_top + ecg_height)

    def x_at(time_s: float) -> float:
        return box[0] + (time_s - period[0]) / (period[1] - period[0]) * (box[2] - box[0])

    beat = _beat_window(times)
    draw.rectangle([x_at(beat[0]), box[1], x_at(beat[1]), box[3]], fill=(52, 72, 52))
    _draw_ecg(draw, box, ecg, period, [])
    _centered(draw, (x_at(beat[0]) + x_at(beat[1])) / 2, box[3] + 3, "conditioning ECG (EchoXFlow)", small, TEXT)

    images = []
    for index in range(len(times)):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for column, patient in enumerate(patients):
            for row, view in enumerate(VIEWS):
                frame = Image.fromarray(clips[(patient, view)][index]).resize((size, size), Image.BILINEAR)
                image.paste(frame, (gap + column * (size + gap), top + row * (size + gap)))
        for row in range(2):
            draw.text((gap + 4, top + row * (size + gap) + 4), VIEWS[row], font=small, fill=TEXT)
        x = x_at(times[index])
        draw.line([(x, box[1] - 3), (x, box[3] + 3)], fill=REFERENCE, width=2)
        stamp = f"t = {times[index] - times[0]:.2f} s"
        draw.text((box[2] - draw.textlength(stamp, font=small), box[3] + 3), stamp, font=small, fill=TEXT)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="ema.pt of a beamspace Task 4 model")
    parser.add_argument("--camus-root", type=Path, required=True, help="CAMUS EchoXFlow package (has croissant.json)")
    parser.add_argument("--data-root", default=None, help="EchoXFlow root (default: ECHOXFLOW_DATA_ROOT)")
    parser.add_argument("--patients", type=int, default=5, help="CAMUS patients shown (the first ones by id)")
    parser.add_argument(
        "--ecg-clips",
        nargs="+",
        default=list(PICKS),
        help="EchoXFlow A4C recordings (exam/recording), one animation each",
    )
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    if config["data"]["coordinate_space"] != "beamspace":
        parser.error("--checkpoint must be a beamspace model")
    seed = int(config.get("evaluation", {}).get("seed", 0))
    size = int(config["data"]["spatial_size"])
    future = int(config["data"].get("clip_frames", 32)) - 1
    data_root = Path(args.data_root or os.environ["ECHOXFLOW_DATA_ROOT"])

    camus_root = args.camus_root
    records = {
        r.recording_id: r
        for r in find_recordings(
            croissant=load_croissant(camus_root / "croissant.json"), array_paths=("2d_brightness_mode",)
        )
    }
    both = {rid.removeprefix("recording_camus_").rsplit("_", 1)[0] for rid in records}
    both = {p for p in both if all(f"recording_camus_{p}_{v.lower()}" in records for v in VIEWS)}
    patients = sorted(both)[: args.patients]

    _, echoxflow = build_datasets({**config, "data": {**config["data"], "views": list(VIEWS)}}, data_root=data_root)
    templates = [_one_beat_candidates(echoxflow, "A4C", future + 1, seed, 1, only=pick)[0] for pick in args.ecg_clips]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, (recording, start) in enumerate(templates, start=1):
        template = echoxflow.load_clip(recording, start)
        times = start + np.arange(future + 1) * recording.frame_stride
        store = open_recording(data_root / recording.zarr_path, root=data_root)
        stamps = np.asarray(store.load_stream("2d_brightness_mode").timestamps, dtype=np.float64)[times]
        ecg = _ecg(data_root, recording)
        beat = _one_beat_length(stamps, ecg)  # frames of the heartbeat; the rest is dropped so the loop is seamless
        clips = {}
        for patient in patients:
            for view in VIEWS:
                frame, geometry = camus_frame(camus_root, records, patient, view)
                first = prepare_frames(frame[None], coordinate_space="beamspace", size=size, geometry=geometry)
                sample = {
                    **template,
                    "frames": torch.from_numpy(first).float()[None].expand(future + 1, -1, -1, -1).clone(),
                    "depth_start_cm": torch.tensor(geometry.depth_start_m * 100.0, dtype=torch.float32),
                    "depth_end_cm": torch.tensor(geometry.depth_end_m * 100.0, dtype=torch.float32),
                    "sector_width_rad": torch.tensor(geometry.angle_span_rad, dtype=torch.float32),
                    "sector_tilt_rad": torch.tensor(
                        0.5 * (geometry.angle_start_rad + geometry.angle_end_rad), dtype=torch.float32
                    ),
                }
                key = f"camus/{patient}/{view}/ecg{number}"
                generated = generate_clip(
                    model, config, sample, noise_key=key, seed=seed, sampling_steps=args.sampling_steps, device=device
                )
                full = np.concatenate([(first[0] * 255).round().astype(np.uint8)[None], generated])[:beat]
                mask = (
                    materialize(np.full_like(full[:1], 255), space="beamspace", geometry=geometry, size=size)[0] > 127
                )
                clips[(patient, view)] = _on_background(
                    materialize(full, space="beamspace", geometry=geometry, size=size), mask
                )
                print(f"sampled {key}", flush=True)
        path = args.output_dir / f"camus_ecg{number}.webp"
        animation(clips, patients, stamps[:beat], ecg, path)
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
