from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from echoxflow import RecordingStore, open_recording
from echoxflow.manifest import manifest_documents
from echoxflow.scan import SectorGeometry

BMODE_PATH = "data/2d_brightness_mode"


@dataclass(frozen=True)
class RoiMask:
    mask: Tensor
    spatial_pixel_count: int
    metadata: Mapping[str, object]


def tissue_doppler_roi_mask_for_sample(sample: object) -> RoiMask | None:
    """Return a beamspace sampling-gate ROI mask for one tissue Doppler sample."""
    record = getattr(sample, "record", None)
    if record is None:
        return None
    if str(getattr(sample, "coordinate_space", "")).strip().lower() != "beamspace":
        return None

    target = getattr(sample, "doppler_target", None)
    if not isinstance(target, Tensor) or target.ndim < 5:
        return None
    target_shape = (int(target.shape[-2]), int(target.shape[-1]))

    store = open_recording(record, root=getattr(sample, "data_root", None))
    gate = tissue_doppler_sampling_gate_metadata(store)
    if gate is None:
        return None
    geometry = _bmode_geometry_for_sample(store, sample)
    if geometry is None:
        return None
    mask = sampling_gate_roi_mask(gate, geometry=geometry, target_shape=target_shape)
    if mask is None:
        return None
    return RoiMask(
        mask=torch.from_numpy(mask[None, None, None, :, :].astype(np.float32)),
        spatial_pixel_count=int(np.count_nonzero(mask)),
        metadata=dict(gate),
    )


def tissue_doppler_sampling_gate_metadata(store: RecordingStore) -> Mapping[str, object] | None:
    """Find tissue Doppler sampling-gate metadata in a recording manifest."""
    for document in manifest_documents(dict(getattr(store.group, "attrs", {}))):
        gate = _sampling_gate_from_tracks(document)
        if gate is not None:
            return gate
        gate = _mapping(document.get("sampling_gate_metadata"))
        if gate is not None:
            return gate
        for sector in _mapping_items(document.get("sectors")):
            role = _semantic_id(sector)
            if role in {"tissue_doppler", "predicted_tissue_doppler"}:
                gate = _mapping(sector.get("sampling_gate_metadata"))
                if gate is not None:
                    return gate
    return None


def sampling_gate_roi_mask(
    gate: Mapping[str, object],
    *,
    geometry: SectorGeometry,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Rasterize a tissue Doppler sample-volume gate into beamspace target pixels."""
    center_depth = _finite_float(gate.get("gate_center_depth_m"))
    tilt_rad = _finite_float(gate.get("gate_tilt_rad"))
    sample_volume = _finite_float(gate.get("gate_sample_volume_m"))
    if center_depth is None or tilt_rad is None or sample_volume is None:
        return None
    if center_depth <= 0.0 or sample_volume <= 0.0:
        return None

    height, width = int(target_shape[0]), int(target_shape[1])
    if height <= 0 or width <= 0:
        return None

    depth0 = max(float(geometry.depth_start_m), float(center_depth) - 0.5 * float(sample_volume))
    depth1 = min(float(geometry.depth_end_m), float(center_depth) + 0.5 * float(sample_volume))
    half_angle = 0.5 * float(sample_volume) / max(1e-6, float(center_depth))
    angle_min = min(float(geometry.angle_start_rad), float(geometry.angle_end_rad))
    angle_max = max(float(geometry.angle_start_rad), float(geometry.angle_end_rad))
    angle0 = max(angle_min, float(tilt_rad) - half_angle)
    angle1 = min(angle_max, float(tilt_rad) + half_angle)
    if depth1 < depth0 or angle1 < angle0:
        return None

    rows = np.linspace(float(geometry.depth_start_m), float(geometry.depth_end_m), height, dtype=np.float32)
    cols = np.linspace(float(geometry.angle_start_rad), float(geometry.angle_end_rad), width, dtype=np.float32)
    row_mask = (rows >= depth0) & (rows <= depth1)
    col_mask = (cols >= angle0) & (cols <= angle1)
    mask = row_mask[:, None] & col_mask[None, :]
    return np.asarray(mask, dtype=bool) if bool(np.any(mask)) else None


def _bmode_geometry_for_sample(store: RecordingStore, sample: object) -> SectorGeometry | None:
    clip_start = getattr(sample, "clip_start", None)
    clip_stop = getattr(sample, "clip_stop", None)
    bmode = store.load_stream_slice(
        BMODE_PATH,
        None if clip_start is None else int(clip_start),
        None if clip_stop is None else int(clip_stop),
    )
    geometry = getattr(getattr(bmode, "metadata", None), "geometry", None)
    return geometry if isinstance(geometry, SectorGeometry) else None


def _sampling_gate_from_tracks(document: Mapping[str, Any]) -> Mapping[str, object] | None:
    for track in _mapping_items(document.get("tracks")):
        derived = _mapping(track.get("derived_from"))
        if _semantic_id(track) == "tissue_doppler_gate":
            return derived or track
        if derived is not None and str(derived.get("kind", "")).strip() == "tissue_doppler_gate":
            return derived
    return None


def _mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else None


def _mapping_items(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping))


def _semantic_id(value: Mapping[str, object]) -> str:
    return str(value.get("semantic_id") or value.get("sector_role_id") or value.get("track_role_id") or "").strip()


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
