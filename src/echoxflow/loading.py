"""Zarr loading helpers for EchoXFlow recordings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from threading import get_ident
from typing import Any, Mapping, Sequence

import numpy as np

from echoxflow.config import data_root
from echoxflow.croissant import RecordingRecord
from echoxflow.manifest import manifest_documents
from echoxflow.mesh import MeshFrame as MeshFrame  # noqa: F401
from echoxflow.mesh import PackedMeshAnnotation, load_packed_mesh_annotation
from echoxflow.objects import ArrayRef, RecordingObject, RecordingRef, recording_object_from_metadata
from echoxflow.scan.geometry import SectorGeometry, sector_geometry_from_mapping
from echoxflow.spectral import SpectralMetadata, spectral_metadata_from_attrs
from echoxflow.streams import (
    EchoStream,
    StreamMetadata,
    TissueDopplerRawStream,
    stream_from_arrays,
    temporal_axis_for_path,
)


@dataclass(frozen=True)
class LoadedArray:
    """Raw array and optional timestamps loaded from one recording store."""

    name: str
    data_path: str
    data: np.ndarray
    timestamps_path: str | None
    timestamps: np.ndarray | None
    sample_rate_hz: float | None
    attrs: Mapping[str, Any]
    stream: EchoStream | None = None


@dataclass(frozen=True)
class RecordingStore:
    """Opened Zarr store plus optional Croissant metadata for one recording."""

    path: Path
    group: Any
    record: RecordingRecord | None = None
    root: Path | None = None
    cache_dir: Path | None = None
    cache_read_only: bool = False
    cache_include: tuple[str, ...] = ()
    cache_exclude: tuple[str, ...] = ()

    @property
    def array_paths(self) -> tuple[str, ...]:
        if self.record is not None:
            return self.record.array_paths
        return tuple(_walk_array_paths(self.group))

    def load_array(self, path: str) -> np.ndarray:
        """Load one complete raw Zarr array by path, for example `data/2d_brightness_mode`."""
        array_path = _normalize_array_path(path)
        return self._load_complete_array(array_path)

    def load_array_ref(self, ref: ArrayRef, recording: RecordingRef | None = None) -> np.ndarray:
        """Load an array reference, optionally from a linked recording."""
        store = self if recording is None else self.open_reference(recording)
        return store.load_array(ref.path)

    def open_reference(self, ref: RecordingRef) -> RecordingStore:
        """Open a linked recording reference, returning this store for self references."""
        if ref.is_self:
            return self
        path = self._reference_path(ref)
        return _open_recording_path(
            path,
            record=None,
            root=self.root,
            cache_dir=self.cache_dir,
            cache_read_only=self.cache_read_only,
            cache_include=self.cache_include,
            cache_exclude=self.cache_exclude,
        )

    def _reference_path(self, ref: RecordingRef) -> Path:
        for raw in (ref.zarr_path, ref.relative_zarr_path):
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_absolute():
                return path
            if self.root is not None and ref.zarr_path == raw:
                return self.root / path
            return self.path.parent / path
        raise ValueError("Recording reference has no zarr path")

    def load_array_slice(self, path: str, start: int | None, stop: int | None) -> np.ndarray:
        """Load a temporal slice from one raw Zarr array, optionally via the complete-array cache."""
        array_path = _normalize_array_path(path)
        array = self.group[array_path]
        selector = _temporal_slice(array_path, array.ndim, int(start or 0), stop)
        if self.cache_dir is None:
            return np.asarray(array[selector])
        return np.asarray(self._load_complete_array(array_path)[selector])

    def load_timestamps(self, name: str) -> np.ndarray | None:
        """Load timestamps for a data path or modality name when present."""
        timestamp_path = self.timestamp_path(name)
        if timestamp_path is None:
            return None
        return self.load_array(timestamp_path)

    def timestamp_path(self, name: str) -> str | None:
        """Return the best matching `timestamps/...` path for a data path or modality name."""
        manifest_path = _manifest_timestamp_path(_normalize_data_path(name), getattr(self.group, "attrs", {}))
        if manifest_path is not None and manifest_path in self.group:
            return manifest_path
        for candidate in _timestamp_candidates(name):
            if candidate in self.group:
                return candidate
        return None

    def load_modality(self, name: str) -> LoadedArray:
        """Load a data array plus matching timestamps and inferred sample rate."""
        stream = self._display_stream(self.load_stream(name))
        return LoadedArray(
            name=stream.name,
            data_path=stream.data_path,
            data=stream.data,
            timestamps_path=stream.timestamps_path,
            timestamps=stream.timestamps,
            sample_rate_hz=stream.sample_rate_hz,
            attrs=_stream_attrs(stream.metadata),
            stream=stream,
        )

    def load_modality_slice(self, name: str, start: int | None, stop: int | None) -> LoadedArray:
        """Load a temporal slice of a modality array plus matching timestamp slice."""
        stream = self._display_stream(self.load_stream_slice(name, start, stop))
        return LoadedArray(
            name=stream.name,
            data_path=stream.data_path,
            data=stream.data,
            timestamps_path=stream.timestamps_path,
            timestamps=stream.timestamps,
            sample_rate_hz=stream.sample_rate_hz,
            attrs=_stream_attrs(stream.metadata),
            stream=stream,
        )

    def _display_stream(self, stream: EchoStream) -> EchoStream:
        if not isinstance(stream, TissueDopplerRawStream):
            return stream
        fenc_table_path = stream.metadata.fenc_table_path
        if not fenc_table_path:
            raise ValueError(f"{stream.data_path} is native TDI but has no FencTable metadata")
        return stream.to_float(self.load_array(fenc_table_path))

    def load_stream(self, name: str) -> EchoStream:
        """Load, validate, and type one data stream."""
        data_path = _normalize_data_path(name)
        timestamps_path = self.timestamp_path(data_path)
        timestamps = None if timestamps_path is None else self.load_array(timestamps_path)
        data = self.load_array(data_path)
        return self._stream_from_data(data_path, data, timestamps_path, timestamps)

    def load_stream_slice(self, name: str, start: int | None, stop: int | None) -> EchoStream:
        """Load, validate, and type a temporal slice of one data stream."""
        data_path = _normalize_data_path(name)
        start_index = int(start or 0)
        timestamps_path = self.timestamp_path(data_path)
        timestamps = None
        if timestamps_path is not None:
            timestamps = self.load_array_slice(timestamps_path, start_index, stop)
        data = self.load_array_slice(data_path, start_index, stop)
        return self._stream_from_data(data_path, data, timestamps_path, timestamps)

    def spectral_metadata(self, name: str) -> SpectralMetadata:
        """Return focused typed metadata for a spectral Doppler stream."""
        data_path = _normalize_data_path(name)
        row_count = None
        if data_path in self.group:
            shape = tuple(getattr(self.group[data_path], "shape", ()))
            if len(shape) >= 2:
                row_count = int(shape[1])
        return spectral_metadata_from_attrs(data_path, dict(getattr(self.group, "attrs", {})), row_count=row_count)

    def load_object(self) -> RecordingObject:
        """Return semantic object references parsed from this recording's metadata."""
        return recording_object_from_metadata(
            self.record,
            dict(getattr(self.group, "attrs", {})),
            store_path=self.path,
        )

    def load_packed_mesh_annotation(self, prefix: str = "3d_left_ventricle_mesh") -> PackedMeshAnnotation:
        """Load typed packed LV mesh arrays using conventional EchoXFlow mesh paths."""
        return load_packed_mesh_annotation(
            group=self.group,
            store_path=self.path,
            timestamp_path=self.timestamp_path,
            prefix=prefix,
        )

    def _stream_from_data(
        self,
        data_path: str,
        data: np.ndarray,
        timestamps_path: str | None,
        timestamps: np.ndarray | None,
    ) -> EchoStream:
        return stream_from_arrays(
            data_path=data_path,
            data=data,
            timestamps_path=timestamps_path,
            timestamps=timestamps,
            sample_rate_hz=_sample_rate(self.record, data_path),
            metadata=_stream_metadata(data_path, dict(getattr(self.group, "attrs", {})), record=self.record),
        )

    def _load_complete_array(self, array_path: str) -> np.ndarray:
        array = self.group[array_path]
        cache_path = self._array_cache_path(array_path, array)
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as loaded:
                    return np.asarray(loaded["data"])
            except Exception:
                if self.cache_read_only:
                    raise
                logging.getLogger("echoxflow.loading.cache").warning(
                    "could not read recording cache %s; reloading source array",
                    cache_path,
                    exc_info=True,
                )
        values = np.asarray(array[:])
        if cache_path is not None and not self.cache_read_only:
            self._write_array_cache(cache_path, values)
        return values

    def _array_cache_path(self, array_path: str, array: Any) -> Path | None:
        if self.cache_dir is None:
            return None
        if not _cache_path_allowed(array_path, include=self.cache_include, exclude=self.cache_exclude):
            return None
        payload = {
            "version": 1,
            "store": str(self.path.expanduser().resolve()),
            "array_path": str(array_path).strip("/"),
            "shape": tuple(int(dim) for dim in getattr(array, "shape", ())),
            "dtype": str(getattr(array, "dtype", "")),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()[:24]
        return self.cache_dir / "arrays" / key[:2] / f"{key}.npz"

    def _write_array_cache(self, path: Path, values: np.ndarray) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{get_ident()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, data=np.asarray(values))
            os.replace(temporary, path)
        except OSError as exc:
            logging.getLogger("echoxflow.loading.cache").warning(
                "could not write recording cache %s: %s",
                path,
                exc,
            )
        finally:
            temporary.unlink(missing_ok=True)


def open_recording(
    record: RecordingRecord | str | Path,
    *,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_read_only: bool = False,
    cache_include: Sequence[str] | None = None,
    cache_exclude: Sequence[str] | None = None,
) -> RecordingStore:
    """Open a recording Zarr store from a Croissant record or direct path."""
    root_path: Path | None
    if isinstance(record, RecordingRecord):
        root_path = data_root(root)
        path = record.path(root_path)
        metadata: RecordingRecord | None = record
    else:
        path = Path(record).expanduser()
        root_path = Path(root).expanduser() if root is not None else None
        metadata = None
    return _open_recording_path(
        path,
        record=metadata,
        root=root_path,
        cache_dir=None if cache_dir is None else Path(cache_dir).expanduser(),
        cache_read_only=cache_read_only,
        cache_include=_cache_patterns(cache_include),
        cache_exclude=_cache_patterns(cache_exclude),
    )


def _open_recording_path(
    path: Path,
    *,
    record: RecordingRecord | None,
    root: Path | None,
    cache_dir: Path | None,
    cache_read_only: bool,
    cache_include: tuple[str, ...],
    cache_exclude: tuple[str, ...],
) -> RecordingStore:
    zarr_module = import_module("zarr")
    if path.is_file():
        zip_store = getattr(import_module("zarr.storage"), "ZipStore")
        store = zip_store(path, mode="r")
        group = zarr_module.open_group(store=store, mode="r")
    else:
        group = zarr_module.open_group(path, mode="r")
    return RecordingStore(
        path=path,
        group=group,
        record=record,
        root=root,
        cache_dir=cache_dir,
        cache_read_only=cache_read_only,
        cache_include=cache_include,
        cache_exclude=cache_exclude,
    )


def load_modality(
    record: RecordingRecord | str | Path,
    name: str,
    *,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_read_only: bool = False,
    cache_include: Sequence[str] | None = None,
    cache_exclude: Sequence[str] | None = None,
) -> LoadedArray:
    """Open a recording and load one modality array with timestamps."""
    return open_recording(
        record,
        root=root,
        cache_dir=cache_dir,
        cache_read_only=cache_read_only,
        cache_include=cache_include,
        cache_exclude=cache_exclude,
    ).load_modality(name)


def load_modality_slice(
    record: RecordingRecord | str | Path,
    name: str,
    start: int | None,
    stop: int | None,
    *,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_read_only: bool = False,
    cache_include: Sequence[str] | None = None,
    cache_exclude: Sequence[str] | None = None,
) -> LoadedArray:
    """Open a recording and load a sliced modality array with sliced timestamps."""
    return open_recording(
        record,
        root=root,
        cache_dir=cache_dir,
        cache_read_only=cache_read_only,
        cache_include=cache_include,
        cache_exclude=cache_exclude,
    ).load_modality_slice(name, start, stop)


def load_stream(
    record: RecordingRecord | str | Path,
    name: str,
    *,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_read_only: bool = False,
    cache_include: Sequence[str] | None = None,
    cache_exclude: Sequence[str] | None = None,
) -> EchoStream:
    """Open a recording and load one validated typed stream."""
    return open_recording(
        record,
        root=root,
        cache_dir=cache_dir,
        cache_read_only=cache_read_only,
        cache_include=cache_include,
        cache_exclude=cache_exclude,
    ).load_stream(name)


def load_stream_slice(
    record: RecordingRecord | str | Path,
    name: str,
    start: int | None,
    stop: int | None,
    *,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_read_only: bool = False,
    cache_include: Sequence[str] | None = None,
    cache_exclude: Sequence[str] | None = None,
) -> EchoStream:
    """Open a recording and load one validated typed stream slice."""
    return open_recording(
        record,
        root=root,
        cache_dir=cache_dir,
        cache_read_only=cache_read_only,
        cache_include=cache_include,
        cache_exclude=cache_exclude,
    ).load_stream_slice(name, start, stop)


def _cache_patterns(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values.strip("/"),) if values.strip() else ()
    return tuple(str(value).strip("/") for value in values if str(value).strip())


def _cache_path_allowed(array_path: str, *, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    path = str(array_path).strip("/")
    if include and not any(fnmatch(path, pattern) for pattern in include):
        return False
    return not any(fnmatch(path, pattern) for pattern in exclude)


def _normalize_array_path(path: str) -> str:
    text = str(path).strip("/")
    if "/" in text:
        return text
    return _normalize_data_path(text)


def _normalize_data_path(path: str) -> str:
    path = str(path).strip("/")
    if path.startswith("data/"):
        return path
    if path.startswith("timestamps/"):
        raise ValueError(f"Expected a data array path, got timestamp path `{path}`")
    return f"data/{path}"


def _timestamp_candidates(name: str) -> tuple[str, ...]:
    name = _normalize_data_path(name).removeprefix("data/")
    candidates = [f"timestamps/{name}"]
    for suffix in ("_velocity", "_power"):
        if name.endswith(suffix):
            candidates.append(f"timestamps/{name[: -len(suffix)]}")
    for annotation in ("_annotation_", "_overlay_physical_points_"):
        if annotation in name:
            candidates.append(f"timestamps/{name.split(annotation, 1)[0]}")
    return tuple(dict.fromkeys(candidates))


def _temporal_slice(data_path: str, data_ndim: int, start: int, stop: int | None) -> tuple[slice | int, ...] | slice:
    if start < 0:
        raise ValueError("Slice start must be non-negative")
    axis = temporal_axis_for_path(data_path, data_ndim)
    if axis == 0:
        return slice(start, stop)
    selector = [slice(None)] * data_ndim
    selector[axis] = slice(start, stop)
    return tuple(selector)


def _sample_rate(record: RecordingRecord | None, data_path: str) -> float | None:
    if record is None:
        return None
    name = data_path.removeprefix("data/")
    candidates = [name]
    for suffix in ("_velocity", "_power"):
        if name.endswith(suffix):
            candidates.append(name[: -len(suffix)])
    if name == "tissue_doppler":
        candidates.append("2d_color_doppler")
    return next((rate for candidate in candidates if (rate := record.sample_rate_hz(candidate)) is not None), None)


def _stream_metadata(
    data_path: str,
    group_attrs: Mapping[str, Any],
    *,
    record: RecordingRecord | None = None,
) -> StreamMetadata:
    if _normalize_data_path(data_path) == "data/3d_brightness_mode":
        raw = _three_dimensional_document(group_attrs)
        stitch_beat_count = _stitch_beat_count(record=record, raw=raw)
        return StreamMetadata(
            data_path=data_path,
            value_range=(0.0, 255.0),
            stitch_beat_count=stitch_beat_count,
            raw=_with_stitch_beat_count(raw, stitch_beat_count),
        )
    sector = _sector_for_data_path(data_path, group_attrs)
    if sector is None:
        return StreamMetadata(data_path=data_path)
    return StreamMetadata(
        data_path=data_path,
        value_range=_optional_float_pair(sector.get("value_range")),
        velocity_limit_mps=_optional_float(sector.get("velocity_limit_mps"))
        or _optional_float(sector.get("velocity_scale_mps")),
        velocity_limit_source=_optional_str(sector.get("velocity_limit_source"))
        or _optional_str(sector.get("velocity_scale_source")),
        storage_encoding=_optional_str(sector.get("storage_encoding")),
        fenc_table_path=_array_ref_path(_lookup_table_ref(sector)),
        colormap_path=_array_ref_path(sector.get("colormap")),
        geometry=_sector_geometry(sector.get("geometry")),
        raw=sector,
    )


def _stream_attrs(metadata: StreamMetadata) -> dict[str, Any]:
    return {
        "value_range": metadata.value_range,
        "velocity_limit_mps": metadata.velocity_limit_mps,
        "velocity_limit_source": metadata.velocity_limit_source,
        "storage_encoding": metadata.storage_encoding,
        "fenc_table_path": metadata.fenc_table_path,
        "colormap_path": metadata.colormap_path,
        "geometry": metadata.geometry,
        "stitch_beat_count": metadata.stitch_beat_count,
    }


def _stitch_beat_count(*, record: RecordingRecord | None, raw: Mapping[str, Any] | None) -> int | None:
    if record is not None and record.stitch_beat_count is not None:
        return record.stitch_beat_count
    if raw is None:
        return None
    for source in (raw, raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else None):
        if source is None:
            continue
        count = _optional_positive_int(source.get("stitch_beat_count"))
        if count is not None:
            return count
    return None


def _with_stitch_beat_count(raw: Mapping[str, Any] | None, count: int | None) -> Mapping[str, Any] | None:
    if count is None:
        return raw
    restored = dict(raw or {})
    restored.setdefault("stitch_beat_count", int(count))
    return restored


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _sector_for_data_path(data_path: str, group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    roles = _sector_role_candidates(data_path)
    if not roles:
        return None
    for document in _manifest_documents(group_attrs):
        sectors = document.get("sectors")
        if not isinstance(sectors, list):
            continue
        for sector in sectors:
            if not isinstance(sector, dict):
                continue
            if _semantic_id(sector) in roles:
                return sector
    return None


def _sector_role_candidates(data_path: str) -> tuple[str, ...]:
    name = _normalize_data_path(data_path).removeprefix("data/")
    if name == "3d_brightness_mode":
        return ("bmode", "3d_brightness_mode")
    if name.startswith("2d_brightness_mode"):
        return ("bmode",)
    if name == "2d_color_doppler_velocity":
        return ("2d_color_doppler_velocity", "2d_color_doppler", "color_doppler")
    if name == "2d_color_doppler_power":
        return ("2d_color_doppler_power", "2d_color_doppler", "color_doppler")
    if name == "tissue_doppler":
        return ("tissue_doppler", "predicted_tissue_doppler")
    if name == "segmentation_overlay":
        return ("segmentation_overlay",)
    return ()


def _manifest_documents(group_attrs: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return manifest_documents(group_attrs)


def _three_dimensional_document(group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for document in _manifest_documents(group_attrs):
        sector = _three_dimensional_sector(document)
        if sector is not None:
            return _with_internal_3d_metadata(document, sector)
    return None


def _with_internal_3d_metadata(document: Mapping[str, Any], sector: Mapping[str, Any]) -> Mapping[str, Any]:
    restored = dict(document)
    geometry = sector.get("geometry")
    if isinstance(geometry, Mapping):
        restored["render_metadata"] = dict(geometry)
    public_metadata = restored.get("metadata")
    if isinstance(public_metadata, Mapping):
        time_reference = public_metadata.get("time_reference")
        qrs_trigger_times = public_metadata.get("qrs_trigger_times")
        if time_reference is not None:
            restored["volume_time_reference"] = time_reference
        if qrs_trigger_times is not None:
            restored["volume_qrs_trigger_times"] = qrs_trigger_times
    if "output_timeline" in restored and "timeline" not in restored:
        restored["timeline"] = restored["output_timeline"]
    return restored


def _three_dimensional_sector(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    sectors = document.get("sectors")
    if not isinstance(sectors, list):
        return None
    for sector in sectors:
        if not isinstance(sector, Mapping):
            continue
        geometry = sector.get("geometry")
        if not isinstance(geometry, Mapping):
            continue
        if geometry.get("coordinate_system") == "spherical_sector_3d" and {"DepthStart", "DepthEnd", "Width"} <= set(
            geometry
        ):
            return sector
        frames = sector.get("frames")
        frame_path = _array_ref_path(frames)
        if frame_path == "data/3d_brightness_mode" and {"DepthStart", "DepthEnd", "Width"} <= set(geometry):
            return sector
    return None


def _lookup_table_ref(sector: Mapping[str, Any]) -> Any | None:
    native_tables = sector.get("native_tdi_lookup_tables")
    if isinstance(native_tables, dict):
        return native_tables.get("fenc_lookup_table")
    tdi_tables = sector.get("tdi_tables")
    if isinstance(tdi_tables, dict):
        return tdi_tables.get("fenc_table")
    return None


def _semantic_id(value: Mapping[str, Any]) -> str:
    return str(value.get("semantic_id") or value.get("sector_role_id") or value.get("track_role_id") or "").strip()


def _array_ref_path(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    path = value.get("array_path") or value.get("zarr_path") or value.get("path")
    return str(path).strip("/") if path else None


def _manifest_timestamp_path(data_path: str, group_attrs: Mapping[str, Any]) -> str | None:
    sector = _sector_for_data_path(data_path, group_attrs)
    if sector is not None:
        timestamp_path = _array_ref_path(sector.get("timestamps"))
        if timestamp_path is not None:
            return timestamp_path
    if data_path != "data/3d_brightness_mode":
        return None
    for document in _manifest_documents(group_attrs):
        timelines = document.get("timelines")
        if isinstance(timelines, Mapping):
            timestamp_path = _array_ref_path(timelines.get("frame_timestamps"))
            if timestamp_path is not None:
                return timestamp_path
    return None


def _sector_geometry(value: Any) -> SectorGeometry | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return sector_geometry_from_mapping(value)
    except (KeyError, TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _optional_float_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    low, high = float(value[0]), float(value[1])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return low, high


def _walk_array_paths(group: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    members = group.items() if hasattr(group, "items") else group.members()
    for key, value in members:
        path = f"{prefix}/{key}" if prefix else str(key)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            paths.append(path)
        else:
            paths.extend(_walk_array_paths(value, path))
    return tuple(paths)
