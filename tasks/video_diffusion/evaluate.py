"""Sample held-out clips and score them in pretrained feature spaces.

For one trained run (``ema.pt``) this

1. selects ``clips_per_view`` evaluation-fold clips per view (one per recording, hash-ordered with ``seed``),
2. generates the 31 future frames of each clip from its first frame and ECG with 32 deterministic DDIM steps
   (per-clip initial noise seeded from the clip identity, so the selection and noise are shared by all runs),
3. scan converts generated and real future frames onto each extractor's square canvas (beamspace outputs are scan
   converted; Cartesian outputs are resized), and real frames are re-read from the native-resolution beamspace data,
4. extracts features and computes per-view Frechet distance (frame spaces) and paired KID (clip spaces), and the
   equal-weight mean over views.

Outputs in ``--output-dir`` (default ``<run>/evaluation``): ``features/<extractor>.npz`` (per-row sample, exam and
view identities, used by :mod:`tasks.video_diffusion.tables`), ``metrics.csv`` and, written last, ``metrics.json``.

The evaluation is resumable: generated clips are stored in chunks of 16 under ``generated/`` while it runs, and a
CLI reruns first verify provenance, including checkpoint, settings and extractor weights, then reuse matching
chunks and feature files. Once ``metrics.json`` is written the
chunks are deleted; ``--keep-samples`` keeps the generated clips as ``samples.npz`` instead. All files are written
atomically.

``--camus-root PATH`` additionally evaluates CAMUS-conditioned generation in ``<output-dir>/camus``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import shutil
import signal
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from echoxflow import find_recordings, load_croissant, open_recording
from echoxflow.scan import SectorGeometry
from tasks.utils.atomic import atomic_open, atomic_save, remove_stale_temporaries
from tasks.video_diffusion.dataset import (
    BMODE_PATH,
    ClipRecording,
    VideoDiffusionDataset,
    build_datasets,
    prepare_frames,
    scan_convert_stack,
)
from tasks.video_diffusion.features import (
    CLIP_EXTRACTORS,
    EXTRACTOR_LABELS,
    FRAME_EXTRACTORS,
    MissingExtractorError,
    build_extractor,
)
from tasks.video_diffusion.metrics import frechet_distance, kid_subsets, paired_kid, stable_seed, unpaired_kid
from tasks.video_diffusion.model import VideoDiffusionUNet, build_model, pixel_unpack_video, sample_future_clip
from tasks.video_diffusion.provenance import (
    bind_output,
    dataset_identity,
    file_digest,
    implementation_identity,
    inventory,
)
from tasks.video_diffusion.task import diffusion_schedule, prepare_batch


def requested_extractors(
    names: Sequence[str], device: torch.device, *, allow_partial: bool = False
) -> tuple[dict[str, Any], dict[str, str]]:
    models, skipped = {}, {}
    for name in names:
        try:
            models[name] = build_extractor(name, device)
        except MissingExtractorError as error:
            if not allow_partial:
                raise
            skipped[name] = str(error)
    return models, skipped


def extractor_identity(extractor: Any) -> str:
    """Fingerprint the actual loaded weights, including automatically downloaded models."""
    digest = hashlib.sha256()
    digest.update(f"{type(extractor).__name__}:{extractor.input_size}:{extractor.kind}".encode())
    for name, tensor in sorted(extractor.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_evaluation(args: argparse.Namespace, config: Mapping[str, Any], models: Mapping[str, Any]) -> None:
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output_dir", "device", "batch_size", "extractor_models", "extractor_skips", "keep_samples"}
    }
    identity = {
        "settings": settings,
        "config": config,
        "checkpoint": file_digest(args.checkpoint),
        "dataset": dataset_identity(config),
        "implementation": implementation_identity(),
        "extractors": {name: extractor_identity(model) for name, model in models.items()},
        "camus": None if args.camus_root is None else inventory(args.camus_root),
    }
    bind_output(args.output_dir, identity, artifacts=("generated", "features", "samples.npz", "metrics.json", "camus"))


def evaluation_complete(directory: Path, names: Sequence[str], *, keep_samples: bool = False) -> bool:
    path = directory / "metrics.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        hashes = payload["feature_sha256"]
        present = {row["extractor"] for row in payload["metrics"] if row["view"] == "mean"}
        return (
            not payload["skipped_extractors"]
            and set(names) <= present
            and all(hashes[name] == file_digest(directory / "features" / f"{name}.npz") for name in names)
            and (not keep_samples or (directory / "samples.npz").is_file())
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


ALL_EXTRACTORS = FRAME_EXTRACTORS + CLIP_EXTRACTORS
GENERATION_CHUNK_CLIPS = 16


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[VideoDiffusionUNet, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(config, activation_checkpointing=False)
    model.load_state_dict(state["ema_model"])
    return model.eval().requires_grad_(False).to(device), config


def _hash_key(*parts: object) -> int:
    return stable_seed(*parts)


def select_clips(dataset: VideoDiffusionDataset, *, clips_per_view: int, seed: int) -> list[tuple[ClipRecording, int]]:
    """One clip per recording: recordings and their clip starts are ranked by a seeded hash; views interleaved."""
    per_view: dict[str, list[tuple[ClipRecording, int]]] = {}
    for view, recordings in dataset.grouped.items():
        ranked = sorted(recordings, key=lambda r: _hash_key(seed, "recording", r.exam_id, r.recording_id))
        chosen = []
        for recording in ranked[:clips_per_view]:
            start = min(
                recording.candidate_starts,
                key=lambda s: _hash_key(
                    seed, "candidate", recording.exam_id, recording.recording_id, s, recording.frame_stride
                ),
            )
            chosen.append((recording, int(start)))
        per_view[view] = chosen
    return [
        per_view[view][slot] for slot in range(clips_per_view) for view in dataset.views if slot < len(per_view[view])
    ]


def clip_id(recording: ClipRecording, start: int, clip_frames: int) -> str:
    end = start + (clip_frames - 1) * recording.frame_stride
    return f"{recording.exam_id}/{recording.recording_id}:{start}-{end}:stride{recording.frame_stride}"


def to_uint8(values: np.ndarray) -> np.ndarray:
    clipped: np.ndarray = np.clip(np.rint(np.asarray(values, dtype=np.float32) * 255.0), 0, 255)
    return clipped.astype(np.uint8)


def generate_resumable(
    ids: Sequence[str],
    load: Callable[[Sequence[int]], Any],
    generate: Callable[[Any, Sequence[int]], np.ndarray],
    directory: Path,
    *,
    batch: int,
) -> np.ndarray:
    """Generated clips ``generate(load(indices), indices)`` (``batch`` clips at a time) for every id, reusing chunks of
    an interrupted run whose clip ids match. The conditioning of the next batch is loaded while the GPU samples."""
    directory.mkdir(parents=True, exist_ok=True)
    remove_stale_temporaries(directory)
    todo: list[tuple[Path, list[str], list[int]]] = []
    batches: list[list[int]] = []
    for first in range(0, len(ids), GENERATION_CHUNK_CLIPS):
        chunk_ids = list(ids[first : first + GENERATION_CHUNK_CLIPS])
        path = directory / f"chunk_{first // GENERATION_CHUNK_CLIPS:04d}.npz"
        if _load_chunk(path, chunk_ids) is None:
            indices = list(range(first, first + len(chunk_ids)))
            todo.append((path, chunk_ids, indices))
            batches.extend(indices[i : i + batch] for i in range(0, len(indices), batch))
        else:
            print(f"reusing {path} ({len(chunk_ids)} clips)", flush=True)
    generated: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = [pool.submit(load, b) for b in batches[:2]]
        for number, indices in enumerate(batches):
            if number + 2 < len(batches):
                pending.append(pool.submit(load, batches[number + 2]))
            frames = generate(pending.pop(0).result(), indices)
            generated.update(zip(indices, frames))
            if indices[-1] == todo[0][2][-1]:
                path, chunk_ids, chunk_indices = todo.pop(0)
                stacked = np.stack([generated.pop(index) for index in chunk_indices])
                atomic_save(
                    path, lambda handle: np.savez_compressed(handle, generated=stacked, clip_ids=np.asarray(chunk_ids))
                )
    chunks = []
    for first in range(0, len(ids), GENERATION_CHUNK_CLIPS):
        chunk_ids = list(ids[first : first + GENERATION_CHUNK_CLIPS])
        chunk = _load_chunk(directory / f"chunk_{first // GENERATION_CHUNK_CLIPS:04d}.npz", chunk_ids)
        if chunk is None:
            raise RuntimeError(f"missing generated chunk for clips {chunk_ids[0]}..{chunk_ids[-1]}")
        chunks.append(chunk)
    return np.concatenate(chunks)


def _load_chunk(path: Path, ids: Sequence[str]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            if data["clip_ids"].tolist() != list(ids):
                return None
            return np.asarray(data["generated"])
    except (OSError, ValueError, KeyError):
        return None


def finish_samples(directory: Path, output: Path, *, keep: bool, metadata: Mapping[str, np.ndarray]) -> None:
    """After the metrics are written: keep the generated clips as ``samples.npz`` or delete them."""
    if keep:
        chunks = sorted(directory.glob("chunk_*.npz"))
        if chunks:
            arrays: dict[str, Any] = {
                "generated": np.concatenate([np.load(chunk)["generated"] for chunk in chunks]),
                **metadata,
            }
            atomic_save(output, lambda handle: np.savez_compressed(handle, **arrays))
    shutil.rmtree(directory, ignore_errors=True)


@torch.no_grad()
def generate_clips(
    model: VideoDiffusionUNet,
    config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    noise_keys: Sequence[str],
    seed: int,
    sampling_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Return the generated future frames ``[N, T-1, S, S]`` as uint8 in the model's coordinate space.

    The initial noise of every clip is seeded by its own key, so the result does not depend on the batching."""
    prepared = prepare_batch(default_collate(list(samples)), device=device, config=config)
    shape = (1, model.data_channels, model.clip_frames - 1, *prepared.first_frame.shape[-2:])
    generators = [torch.Generator(device=device).manual_seed(stable_seed(seed, key)) for key in noise_keys]
    noise = torch.cat([torch.randn(shape, generator=generator, device=device) for generator in generators])
    packed = sample_future_clip(
        model,
        diffusion_schedule(config, device),
        prepared.first_frame,
        prepared.coordinate_context,
        prepared.ecg_waveform,
        sampling_steps=sampling_steps,
        initial_noise=noise,
        use_autocast=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )
    image = pixel_unpack_video(packed, prepared.pixel_pack_factor, output_size=prepared.image_size)
    return to_uint8(image[:, 0].add(1.0).mul(0.5).clamp(0.0, 1.0).cpu().numpy())


def generate_clip(
    model: VideoDiffusionUNet,
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    noise_key: str,
    seed: int,
    sampling_steps: int,
    device: torch.device,
) -> np.ndarray:
    """One clip ``[T-1, S, S]``; see :func:`generate_clips`."""
    clips = generate_clips(
        model, config, [sample], noise_keys=[noise_key], seed=seed, sampling_steps=sampling_steps, device=device
    )
    return np.asarray(clips[0])


def generation_batch(args: argparse.Namespace, size: int) -> int:
    """Clips sampled per forward pass: ``--generation-batch``, or by default 16 at 128 px scaled by the pixel count."""
    if args.generation_batch:
        return int(args.generation_batch)
    return max(1, int(16 * (128 / size) ** 2))


def materialize(frames: np.ndarray, *, space: str, geometry: SectorGeometry, size: int) -> np.ndarray:
    """uint8 ``[T, H, W]`` in ``space`` -> uint8 ``[T, size, size]`` on the square Cartesian canvas."""
    values = np.asarray(frames, dtype=np.float32) / 255.0
    if space == "beamspace":
        return to_uint8(scan_convert_stack(values, geometry, size))
    resized = F.interpolate(torch.from_numpy(values)[:, None], size=(size, size), mode="bilinear", align_corners=False)
    return to_uint8(resized[:, 0].numpy())


def recording_geometry(recording: ClipRecording) -> SectorGeometry:
    return SectorGeometry.from_center_width(
        depth_start_m=recording.depth_start_m,
        depth_end_m=recording.depth_end_m,
        tilt_rad=recording.tilt_rad,
        width_rad=recording.width_rad,
    )


def native_future_frames(root: Path, recording: ClipRecording, start: int, clip_frames: int) -> np.ndarray:
    store = open_recording(root / recording.zarr_path, root=root)
    span = 1 + (clip_frames - 1) * recording.frame_stride
    frames = np.asarray(store.load_array_slice(BMODE_PATH, start, start + span))
    return frames[np.arange(1, clip_frames) * recording.frame_stride]


# ---------------------------------------------------------------------------------------------------------------------
# Feature extraction and metrics


def extract_features(
    names: Sequence[str],
    *,
    real: Callable[[int], tuple[np.ndarray, np.ndarray]],
    generated: Callable[[int], np.ndarray],
    output_dir: Path,
    metadata: Mapping[str, Mapping[str, np.ndarray]],
    device: torch.device,
    batch_size: int,
    extractors: Mapping[str, Any] | None = None,
    skipped_extractors: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Run each extractor, grouped by input size. ``real(size)`` returns (frames, clips); returns skip reasons."""
    if extractors is None:
        extractors, skipped = requested_extractors(names, device)
    else:
        skipped = dict(skipped_extractors or {})
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_temporaries(feature_dir)
    done = {
        name
        for name, extractor in extractors.items()
        if _features_match(feature_dir / f"{name}.npz", metadata["frame" if extractor.kind == "frame" else "clip"])
    }
    for name in sorted(done):
        print(f"reusing features/{name}.npz", flush=True)
    for size in sorted({extractor.input_size for name, extractor in extractors.items() if name not in done}):
        real_frames, real_clips = real(size)
        generated_clips = generated(size)
        for name, extractor in extractors.items():
            if extractor.input_size != size or name in done:
                continue
            extractor.model.to(device)
            extractor.device = device
            if extractor.kind == "frame":
                real_features = extractor.extract(real_frames, batch_size=batch_size)
                generated_features = extractor.extract(generated_clips.reshape(-1, size, size), batch_size=batch_size)
                rows = metadata["frame"]
            else:
                real_features = extractor.extract(real_clips, batch_size=max(1, batch_size // 8))
                generated_features = extractor.extract(generated_clips, batch_size=max(1, batch_size // 8))
                rows = metadata["clip"]
            extractor.model.cpu()
            extractor.device = torch.device("cpu")
            arrays: dict[str, Any] = {
                "real": real_features,
                "generated": generated_features,
                "kind": np.asarray(extractor.kind),
                **{key: np.asarray(value) for key, value in rows.items()},
            }
            atomic_save(feature_dir / f"{name}.npz", lambda handle: np.savez(handle, **arrays))
            print(f"extracted {name}: real {real_features.shape}, generated {generated_features.shape}", flush=True)
        del real_frames, real_clips, generated_clips
    return skipped


def _features_match(path: Path, rows: Mapping[str, np.ndarray]) -> bool:
    """True when ``path`` holds features for exactly these real and generated rows (a finished extractor)."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            return all(key in data and np.array_equal(data[key], np.asarray(value)) for key, value in rows.items())
    except (OSError, ValueError):
        return False


def compute_metrics(
    output_dir: Path,
    names: Sequence[str],
    *,
    paired: bool,
    kid_subset_count: int,
    kid_subset_size: int,
    kid_seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Per-view FD (frame features) or KID (clip features) and the equal-weight mean over views."""
    rows: list[dict[str, Any]] = []
    for name in names:
        path = output_dir / "features" / f"{name}.npz"
        data = np.load(path)
        kind = str(data["kind"])
        values = {}
        real_views, generated_views = data["real_views"], data["generated_views"]
        for view in sorted(set(generated_views.tolist())):
            real = data["real"][real_views == view]
            generated = data["generated"][generated_views == view]
            if kind == "frame":
                values[view] = frechet_distance(real, generated, device=device)
            elif paired:
                subsets = kid_subsets(
                    len(real),
                    subsets=kid_subset_count,
                    subset_size=kid_subset_size,
                    seed=kid_seed,
                    namespace=f"{view}:paired:kid",
                )
                values[view] = paired_kid(real, generated, subsets, device=device)
            else:
                values[view] = unpaired_kid(
                    real,
                    generated,
                    subsets=kid_subset_count,
                    subset_size=kid_subset_size,
                    seed=kid_seed,
                    namespace=f"{view}:kid",
                    device=device,
                )
        metric = "fd" if kind == "frame" else ("paired_kid" if paired else "kid")
        for view, value in values.items():
            rows.append(
                {"extractor": name, "label": EXTRACTOR_LABELS[name], "metric": metric, "view": view, "value": value}
            )
        rows.append(
            {
                "extractor": name,
                "label": EXTRACTOR_LABELS[name],
                "metric": metric,
                "view": "mean",
                "value": float(np.mean(list(values.values()))),
            }
        )
    return rows


def write_metrics(
    output_dir: Path, rows: list[dict[str, Any]], *, skipped: Mapping[str, str], info: Mapping[str, Any]
) -> None:
    with atomic_open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["extractor", "label", "metric", "view", "value"])
        writer.writeheader()
        writer.writerows(rows)
    # Partial reports never act as a completion marker.
    filename = "metrics.partial.json" if skipped or not rows else "metrics.json"
    if filename != "metrics.json":
        (output_dir / "metrics.json").unlink(missing_ok=True)
    hashes = {name: file_digest(output_dir / "features" / f"{name}.npz") for name in {row["extractor"] for row in rows}}
    with atomic_open(output_dir / filename, "w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {**info, "metrics": rows, "skipped_extractors": dict(skipped), "feature_sha256": hashes}, indent=2
            )
        )
    for row in rows:
        if row["view"] == "mean":
            print(f"{row['label']:<28} {row['value']:.4g}")


# ---------------------------------------------------------------------------------------------------------------------
# EchoXFlow evaluation


def evaluate_echoxflow(
    args: argparse.Namespace, model: VideoDiffusionUNet, config: dict[str, Any], device: torch.device
) -> Path:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = config.get("evaluation", {})
    seed = int(args.seed if args.seed is not None else evaluation.get("seed", 0))
    space = str(config["data"]["coordinate_space"])
    clip_frames = int(config["data"].get("clip_frames", 32))
    _, dataset = build_datasets(config, data_root=args.data_root)
    clips = select_clips(dataset, clips_per_view=args.clips_per_view, seed=seed)
    ids = [clip_id(recording, start, clip_frames) for recording, start in clips]
    samples_path = output_dir / "samples.npz"
    generated_dir = output_dir / "generated"
    if samples_path.is_file() and np.load(samples_path)["clip_ids"].tolist() == ids:
        generated_all = np.load(samples_path)["generated"]
        print(f"reusing {samples_path}")
    else:

        def load(indices: Sequence[int]) -> list[dict[str, Any]]:
            return [dataset.load_clip(*clips[index]) for index in indices]

        def generate(samples: list[dict[str, Any]], indices: Sequence[int]) -> np.ndarray:
            frames = generate_clips(
                model,
                config,
                samples,
                noise_keys=[ids[index] for index in indices],
                seed=seed,
                sampling_steps=args.sampling_steps,
                device=device,
            )
            print(f"sampled {indices[-1] + 1}/{len(clips)} {ids[indices[-1]]}", flush=True)
            return frames

        size = int(config["data"]["spatial_size"])
        generated_all = generate_resumable(ids, load, generate, generated_dir, batch=generation_batch(args, size))
    geometries = [recording_geometry(recording) for recording, _ in clips]

    @functools.cache
    def native_frames() -> list[np.ndarray]:
        return [native_future_frames(dataset.root, recording, start, clip_frames) for recording, start in clips]

    future = clip_frames - 1
    clip_meta = {"exam_ids": [r.exam_id for r, _ in clips], "views": [r.view for r, _ in clips], "ids": ids}
    metadata = {
        "clip": {
            f"{side}_{key}": np.asarray(value) for side in ("real", "generated") for key, value in clip_meta.items()
        },
        "frame": {
            f"{side}_{key}": np.repeat(np.asarray(value), future)
            for side in ("real", "generated")
            for key, value in clip_meta.items()
        },
    }

    def real(size: int) -> tuple[np.ndarray, np.ndarray]:
        stack = np.stack(
            [materialize(f, space="beamspace", geometry=g, size=size) for f, g in zip(native_frames(), geometries)]
        )
        return stack.reshape(-1, size, size), stack

    def generated_at(size: int) -> np.ndarray:
        return np.stack([materialize(f, space=space, geometry=g, size=size) for f, g in zip(generated_all, geometries)])

    skipped = extract_features(
        args.extractors,
        real=real,
        generated=generated_at,
        output_dir=output_dir,
        metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        extractors=getattr(args, "extractor_models", None),
        skipped_extractors=getattr(args, "extractor_skips", None),
    )
    rows = compute_metrics(
        output_dir,
        [name for name in args.extractors if name not in skipped],
        paired=True,
        kid_subset_count=int(evaluation.get("kid_subsets", 100)),
        kid_subset_size=int(evaluation.get("kid_subset_size", 32)),
        kid_seed=int(evaluation.get("kid_seed", 17)),
        device=device,
    )
    info = {
        "protocol": "echoxflow",
        "checkpoint": str(args.checkpoint),
        "coordinate_space": space,
        "spatial_size": int(config["data"]["spatial_size"]),
        "evaluation_fold": int(config["data"].get("evaluation_fold", 4)),
        "clips": len(clips),
        "clips_per_view": {view: sum(r.view == view for r, _ in clips) for view in dataset.views},
        "sampling_steps": args.sampling_steps,
        "seed": seed,
    }
    write_metrics(output_dir, rows, skipped=skipped, info=info)
    finish_samples(
        generated_dir,
        samples_path,
        keep=args.keep_samples,
        metadata={
            "clip_ids": np.asarray(ids),
            "exam_ids": np.asarray([r.exam_id for r, _ in clips]),
            "views": np.asarray([r.view for r, _ in clips]),
            "coordinate_space": np.asarray(space),
        },
    )
    return output_dir


# ---------------------------------------------------------------------------------------------------------------------
# CAMUS evaluation


def evaluate_camus(
    args: argparse.Namespace, model: VideoDiffusionUNet, config: dict[str, Any], device: torch.device
) -> Path:
    """Condition on CAMUS ED frames paired with EchoXFlow ECG/timing; compare with the patients' own later frames.

    CAMUS is read from the CAMUS EchoXFlow package (``--camus-root``, the folder holding ``croissant.json``): B-mode on
    the fitted fan in beamspace with its sector geometry, and the ED/ES frame indices in the recording metadata. Every
    patient with both views is used (``--camus-patients`` limits them, for smoke runs); the reference is the non-ED
    frames and the ED-to-ES clips of the same patients, and the EchoXFlow ECG clips are cycled over the patients.
    """
    camus_root = Path(args.camus_root)
    views = ("A2C", "A4C")
    records = {
        record.recording_id: record
        for record in find_recordings(
            croissant=load_croissant(camus_root / "croissant.json"), array_paths=("2d_brightness_mode",)
        )
    }

    def recording_id(patient: str, view: str) -> str:
        return f"recording_camus_{patient}_{view.lower()}"

    def load_from_ed(patient: str, view: str) -> tuple[np.ndarray, SectorGeometry]:
        """Native beamspace uint8 frames ``[T, R, A]`` ordered from ED to ES, and their sector geometry."""
        record = records[recording_id(patient, view)]
        stream = open_recording(record, root=camus_root).load_stream("2d_brightness_mode")
        attrs = json.loads((camus_root / record.zarr_path / ".zattrs").read_text(encoding="utf-8"))
        phases = attrs["recording_manifest"]["metadata"]["camus"]
        frames = np.asarray(stream.data)
        if int(phases["ed_frame_index"]) > int(phases["es_frame_index"]):
            frames = frames[::-1]
        geometry = stream.metadata.geometry
        if not isinstance(geometry, SectorGeometry):
            raise ValueError(f"{record.recording_id}: missing sector geometry")
        return np.ascontiguousarray(frames), geometry

    output_dir: Path = args.output_dir / "camus"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else config.get("evaluation", {}).get("seed", 0))
    space = str(config["data"]["coordinate_space"])
    size = int(config["data"]["spatial_size"])
    future = int(config["data"].get("clip_frames", 32)) - 1
    camus_patients = {rid.removeprefix("recording_camus_").rsplit("_", 1)[0] for rid in records}
    camus_patients = {p for p in camus_patients if all(recording_id(p, view) in records for view in views)}
    patients = sorted(camus_patients, key=lambda p: _hash_key(seed, "camus-generation-patient", p))
    if args.camus_patients is not None:
        patients = patients[: int(args.camus_patients)]
    conditioning = reference = patients
    _, echoxflow = build_datasets(
        {**config, "data": {**config["data"], "views": list(views)}}, data_root=args.data_root
    )
    templates = select_clips(echoxflow, clips_per_view=len(patients), seed=seed)

    # Generation: CAMUS end-diastolic frame + ECG and frame timing of an EchoXFlow clip of the same view.
    items: list[dict[str, Any]] = []
    jobs: list[tuple[str, str, Any]] = []
    for view in views:
        view_templates = [clip for clip in templates if clip[0].view == view]
        for index, patient in enumerate(conditioning):
            template_clip = view_templates[index % len(view_templates)]
            jobs.append((view, patient, template_clip))
            _, geometry = load_from_ed(patient, view)
            # Version the identities so resumable chunks/features from other CAMUS sources cannot be reused.
            items.append(
                {"id": f"camus/{patient}/{view}/package-v2", "patient": patient, "view": view, "geometry": geometry}
            )

    def conditioning_sample(index: int) -> dict[str, Any]:
        view, patient, template_clip = jobs[index]
        sequence, geometry = load_from_ed(patient, view)
        # The first frame is prepared exactly like a training frame of this coordinate space.
        first = prepare_frames(sequence[:1], coordinate_space=space, size=size, geometry=geometry)
        template = echoxflow.load_clip(*template_clip)
        first_frames = torch.from_numpy(first).float()[None]
        return {
            **template,
            "frames": first_frames.expand(future + 1, -1, -1, -1).clone(),
            "depth_start_cm": torch.tensor(geometry.depth_start_m * 100.0, dtype=torch.float32),
            "depth_end_cm": torch.tensor(geometry.depth_end_m * 100.0, dtype=torch.float32),
            "sector_width_rad": torch.tensor(geometry.angle_span_rad, dtype=torch.float32),
            "sector_tilt_rad": torch.tensor(
                0.5 * (geometry.angle_start_rad + geometry.angle_end_rad), dtype=torch.float32
            ),
        }

    def load(indices: Sequence[int]) -> list[dict[str, Any]]:
        return [conditioning_sample(index) for index in indices]

    def generate(samples: list[dict[str, Any]], indices: Sequence[int]) -> np.ndarray:
        keys = [f"camus/{jobs[index][1]}/{jobs[index][0]}" for index in indices]
        generated = generate_clips(
            model, config, samples, noise_keys=keys, seed=seed, sampling_steps=args.sampling_steps, device=device
        )
        print(f"sampled {indices[-1] + 1}/{len(jobs)} {keys[-1]}", flush=True)
        return generated

    generated_dir = output_dir / "generated"
    generated_all = generate_resumable(
        [item["id"] for item in items], load, generate, generated_dir, batch=generation_batch(args, size)
    )

    # Reference: every non-ED frame of the same patients, and one ED-to-ES clip per patient with enough frames.
    sequences: dict[tuple[str, str], tuple[np.ndarray, Any]] = {}
    real_frames: list[tuple[str, str, int]] = []
    real_clips: list[tuple[str, str, np.ndarray]] = []
    for view in views:
        for patient in reference:
            sequences[(view, patient)] = load_from_ed(patient, view)
        lengths = {patient: len(sequences[(view, patient)][0]) for patient in reference}
        real_frames.extend((view, p, i) for p in reference for i in range(1, lengths[p]))
        eligible = [p for p in reference if lengths[p] - 1 >= 16]
        real_clips.extend((view, p, np.rint(np.linspace(1, lengths[p] - 1, future)).astype(int)) for p in eligible)

    def real(size_: int) -> tuple[np.ndarray, np.ndarray]:
        def square(view: str, patient: str, indices: np.ndarray) -> np.ndarray:
            sequence, geometry = sequences[(view, patient)]
            return materialize(sequence[indices], space="beamspace", geometry=geometry, size=size_)

        frames = np.stack([square(v, p, np.asarray([i]))[0] for v, p, i in real_frames])
        return frames, np.stack([square(v, p, indices) for v, p, indices in real_clips])

    def generated_at(size_: int) -> np.ndarray:
        return np.stack(
            [materialize(f, space=space, geometry=i["geometry"], size=size_) for f, i in zip(generated_all, items)]
        )

    generated_meta = {
        "generated_views": [i["view"] for i in items],
        "generated_exam_ids": [i["patient"] for i in items],
        "generated_ids": [i["id"] for i in items],
    }
    metadata = {
        kind: {
            **{
                key: np.repeat(np.asarray(value), future if kind == "frame" else 1)
                for key, value in generated_meta.items()
            },
            "real_views": np.asarray([row[0] for row in rows]),
            "real_exam_ids": np.asarray([row[1] for row in rows]),
        }
        for kind, rows in (("frame", real_frames), ("clip", real_clips))
    }
    skipped = extract_features(
        args.extractors,
        real=real,
        generated=generated_at,
        output_dir=output_dir,
        metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        extractors=getattr(args, "extractor_models", None),
        skipped_extractors=getattr(args, "extractor_skips", None),
    )
    rows = compute_metrics(
        output_dir,
        [name for name in args.extractors if name not in skipped],
        paired=False,
        kid_subset_count=100,
        kid_subset_size=32,
        kid_seed=seed,
        device=device,
    )
    info = {
        "protocol": "camus",
        "phase_order": "ed-to-es",
        "camus_croissant_sha256": hashlib.sha256((camus_root / "croissant.json").read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "coordinate_space": space,
        "spatial_size": size,
        "clips": len(items),
        "seed": seed,
    }
    write_metrics(output_dir, rows, skipped=skipped, info=info)
    finish_samples(
        generated_dir,
        output_dir / "samples.npz",
        keep=args.keep_samples,
        metadata={"clip_ids": np.asarray([item["id"] for item in items])},
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="ema.pt written by tasks.video_diffusion.train")
    parser.add_argument("--data-root", default=None, help="EchoXFlow root (default: from the training config)")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <checkpoint dir>/evaluation")
    parser.add_argument("--clips-per-view", type=int, default=68)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument(
        "--generation-batch", type=int, default=0, help="clips sampled per forward pass (default: 16 at 128 px, scaled)"
    )
    parser.add_argument("--seed", type=int, default=None, help="clip selection and noise seed (default 0)")
    parser.add_argument("--extractors", nargs="+", default=list(ALL_EXTRACTORS), choices=ALL_EXTRACTORS)
    parser.add_argument(
        "--allow-partial", action="store_true", help="write partial reports for unavailable extractors; exit nonzero"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="frame-extractor batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--camus-root", type=Path, default=None, help="CAMUS EchoXFlow package root (contains croissant.json)"
    )
    parser.add_argument("--camus-patients", type=int, default=None, help="limit the CAMUS patients (smoke runs)")
    parser.add_argument(
        "--skip-echoxflow", action="store_true", help="with --camus-root: only run the CAMUS evaluation"
    )
    parser.add_argument(
        "--keep-samples",
        action="store_true",
        help="keep the generated clips as samples.npz (by default they are deleted once the metrics are written)",
    )
    return parser


def _stop_on_sigterm(signum: int, frame: object) -> None:
    del frame
    print(f"received signal {signum}; stopping (rerun the same command to resume)", flush=True)
    raise SystemExit(128 + signum)  # unwind: in-progress atomic writes remove their temporary files


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_echoxflow and args.camus_root is None:
        parser.error("request EchoXFlow evaluation or supply --camus-root")
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    config["data"]["max_recordings"] = None  # always evaluate on the complete held-out fold
    if args.data_root is not None:
        config["data"]["root_dir"] = args.data_root
    args.output_dir = args.output_dir or args.checkpoint.parent / "evaluation"
    args.extractor_models, args.extractor_skips = requested_extractors(
        args.extractors, torch.device("cpu"), allow_partial=args.allow_partial
    )
    prepare_evaluation(args, config, args.extractor_models)
    if not args.skip_echoxflow and not evaluation_complete(
        args.output_dir, args.extractors, keep_samples=args.keep_samples
    ):
        print(f"evaluation written to {evaluate_echoxflow(args, model, config, device)}")
    if args.camus_root is not None and not evaluation_complete(
        args.output_dir / "camus", args.extractors, keep_samples=args.keep_samples
    ):
        print(f"CAMUS evaluation written to {evaluate_camus(args, model, config, device)}")
    directories = ([] if args.skip_echoxflow else [args.output_dir]) + (
        [args.output_dir / "camus"] if args.camus_root else []
    )
    return 0 if directories and all(evaluation_complete(path, args.extractors) for path in directories) else 1


if __name__ == "__main__":
    raise SystemExit(main())
