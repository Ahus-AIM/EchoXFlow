"""Zarr loading helpers for EchoXFlow recordings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from threading import get_ident
from typing import Any, Mapping, Sequence

import numpy as np

from echoxflow.config import data_root
from echoxflow.croissant import RecordingRecord, StreamSummary, check_schema_generation, stream_summaries_from_attrs
from echoxflow.manifest import annotation_documents, manifest_documents
from echoxflow.mesh import MeshFrame as MeshFrame  # noqa: F401
from echoxflow.mesh import PackedMeshAnnotation, load_packed_mesh_annotation
from echoxflow.objects import ArrayRef, RecordingObject, RecordingRef, recording_object_from_metadata
from echoxflow.scan.geometry import SectorGeometry, sector_geometry_from_mapping
from echoxflow.schema import (
    ARRAY_GROUPS,
    FRAME_LOCATIONS_PATH,
    LOOKUP_TABLE_GROUP,
    TDI_PACKED_STORAGE_ENCODING,
    VOLUME_QRS_TIMES_PATH,
    ArrayInfo,
    array_infos_from_attrs,
    current_array_candidates,
    is_lookup_table_path,
    lookup_table_role,
    normalize_array_path,
    semantic_name,
    tdi_storage_encoding,
)
from echoxflow.spectral import (
    SpectralCalibration,
    SpectralMetadata,
    spectral_calibration_refs,
    spectral_metadata_from_attrs,
    spectral_segment_offsets_ref,
)
from echoxflow.streams import (
    ColorFlowRawStream,
    EchoStream,
    StreamMetadata,
    TissueDopplerRawStream,
    constant_velocity_scale,
    stream_from_arrays,
    validated_stream_timestamps,
)


def resolve_array_path_in(group: Any, path: str) -> str | None:
    """Return the path a group actually stores a requested array under, if any.

    A request for `data/2d_color_doppler_velocity` resolves to `derived/2d_color_doppler_velocity`.
    """
    return next((candidate for candidate in current_array_candidates(path) if candidate in group), None)


def lookup_table_paths_at(store_path: Path) -> tuple[str, ...]:
    """Return the content-addressed lookup tables a directory store carries."""
    root = Path(store_path) / LOOKUP_TABLE_GROUP
    if not root.is_dir():
        return ()
    return tuple(
        sorted(f"{LOOKUP_TABLE_GROUP}/{entry.name}" for entry in root.iterdir() if (entry / ".zarray").is_file())
    )


def lookup_table_path_at(store_path: Path, role: str) -> str | None:
    """Return the store's only lookup table whose shape matches a role.

    This is the fallback for stores whose manifest does not scope lookup tables to their
    sectors: early stores recorded only the generic `lookup_table` role, so the
    stored shape is the only signal (see `echoxflow.schema.lookup_table_role`). When several
    tables share a role, shape cannot say which sector each belongs to, so none is returned.
    """
    matches = [
        path
        for path in lookup_table_paths_at(store_path)
        if lookup_table_role(
            json.loads((Path(store_path) / path / ".zarray").read_text(encoding="utf-8")).get("shape", ())
        )
        == role
    ]
    if len(matches) > 1:
        logging.getLogger("echoxflow.loading").warning(
            "%s has %d unscoped %s lookup tables; refusing to pick one by shape", store_path, len(matches), role
        )
        return None
    return matches[0] if matches else None


def stitch_beat_count_at(store_path: Path, group: Any, record: RecordingRecord | None) -> int | None:
    """Return how many cardiac cycles a stitched 3-D volume was assembled from.

    The Croissant column wins when present; otherwise the per-frame sub-volume index
    `coordinates/3d_brightness_mode_frame_locations` carries the count and also says which
    frames came from which beat. Single-beat acquisitions ship no such array.
    """
    if record is not None and record.stitch_beat_count is not None:
        return record.stitch_beat_count
    if FRAME_LOCATIONS_PATH not in group:
        return 1 if "data/3d_brightness_mode" in group else None
    locations = np.asarray(group[FRAME_LOCATIONS_PATH][:])
    return int(np.unique(locations).size) or None


def volume_qrs_trigger_times_at(group: Any) -> np.ndarray | None:
    """Return the R-peak times a 3-D acquisition was gated on, in store seconds.

    They are published as `annotations/3d_source_qrs_times` on the same clock as
    `timestamps/3d_brightness_mode`, so volumes and triggers share one axis.
    """
    if VOLUME_QRS_TIMES_PATH not in group:
        return None
    return np.asarray(group[VOLUME_QRS_TIMES_PATH][:], dtype=np.float64).reshape(-1)


def augmented_stream_metadata(
    metadata: StreamMetadata,
    *,
    store_path: Path,
    group: Any,
    record: RecordingRecord | None,
) -> StreamMetadata:
    """Restore stream metadata that the store keeps in arrays rather than the manifest.

    Colour tables, the Tissue Doppler FencTable, the 3-D stitch beat count, and the 3-D
    R-peak times live in arrays rather than the manifest. Both the library reader and the dashboard
    fast reader route through here so they agree.
    """
    return _with_store_volume_metadata(
        _with_store_lookup_tables(metadata, store_path=store_path, group_attrs=group.attrs),
        store_path=store_path,
        group=group,
        record=record,
    )


def _with_store_lookup_tables(
    metadata: StreamMetadata, *, store_path: Path, group_attrs: Mapping[str, Any]
) -> StreamMetadata:
    name = semantic_name(metadata.data_path)
    if name in {"1d_pulsed_wave_doppler", "1d_continuous_wave_doppler"}:
        return _with_spectral_colormap(metadata, store_path=store_path, group_attrs=group_attrs)
    is_tissue_doppler = name.startswith(("tissue_doppler", "2d_tissue_doppler"))
    if not is_tissue_doppler and not name.startswith("2d_color_doppler"):
        return metadata
    if isinstance(metadata.raw, Mapping) and isinstance(metadata.raw.get("lookup_tables"), list):
        # The sector scopes its own tables; a store-wide shape match could belong to another sector.
        return metadata
    # Only native Tissue Doppler decodes through a FencTable. Claiming one for colour flow
    # would attach a table the renderer must not apply.
    fenc_table_path = metadata.fenc_table_path
    if is_tissue_doppler and not fenc_table_path:
        fenc_table_path = lookup_table_path_at(store_path, "fenc_table")
    colormap_path = metadata.colormap_path or lookup_table_path_at(store_path, "flow_color_rgb_table")
    if fenc_table_path == metadata.fenc_table_path and colormap_path == metadata.colormap_path:
        return metadata
    return replace(metadata, fenc_table_path=fenc_table_path, colormap_path=colormap_path)


def _valid_spectral_table(store_path: Path, path: str | None) -> bool:
    if path is None:
        return False
    try:
        descriptor = json.loads((store_path / path / ".zarray").read_text())
    except (OSError, ValueError):
        return False
    return lookup_table_role(descriptor.get("shape", ())) == "color_rgb_table"


def _with_spectral_colormap(
    metadata: StreamMetadata, *, store_path: Path, group_attrs: Mapping[str, Any]
) -> StreamMetadata:
    """Use an explicit spectral RGB table, or the only RGB table no other stream claims.

    Legacy exports keep the spectral ColorTable without a track reference. Sector tables
    (including the tissue grey table) belong to those images and must be excluded. Multiple
    remaining tables are ambiguous; retain the configured rendering fallback in that case.
    """
    if metadata.colormap_path is not None:
        path: str | None = metadata.colormap_path
        return replace(
            metadata,
            colormap_path=path if _valid_spectral_table(store_path, path) else None,
            colormap_selection="explicit" if _valid_spectral_table(store_path, path) else "explicit_unresolved",
            colormap_candidates=(metadata.colormap_path,),
        )
    claimed: set[str] = set()
    for _, document in annotation_documents(group_attrs):
        for collection in ("sectors", "tracks"):
            entries = document.get(collection, ())
            if not isinstance(entries, (list, tuple)):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                tables = _scoped_lookup_tables(entry)
                if _sector_frames_path(entry) == normalize_array_path(metadata.data_path):
                    if isinstance(entry.get("lookup_tables"), list):
                        path = _unique_lookup_table(tables, "color_rgb_table", entry)
                        if not _valid_spectral_table(store_path, path):
                            path = None
                        return replace(
                            metadata,
                            colormap_path=path,
                            colormap_selection="explicit" if path else "explicit_unresolved",
                            colormap_candidates=tuple(tables.get("color_rgb_table", ())),
                        )
                claimed.update(path for paths in tables.values() for path in paths)
    candidates = [
        path
        for path in lookup_table_paths_at(store_path)
        if path not in claimed
        and lookup_table_role(json.loads((store_path / path / ".zarray").read_text()).get("shape", ()))
        == "color_rgb_table"
    ]
    if len(candidates) == 1:
        return replace(
            metadata,
            colormap_path=candidates[0],
            colormap_selection="unique_unclaimed",
            colormap_candidates=tuple(candidates),
        )
    if len(candidates) > 1:
        logging.getLogger("echoxflow.loading").warning(
            "%s has %d unclaimed spectral RGB tables; refusing to pick one", store_path, len(candidates)
        )
    return replace(
        metadata,
        colormap_selection="ambiguous_fallback" if candidates else "missing_fallback",
        colormap_candidates=tuple(candidates),
    )


def _with_store_volume_metadata(
    metadata: StreamMetadata,
    *,
    store_path: Path,
    group: Any,
    record: RecordingRecord | None,
) -> StreamMetadata:
    if semantic_name(metadata.data_path) != "3d_brightness_mode":
        return metadata
    raw = metadata.raw
    count = metadata.stitch_beat_count
    if count is None:
        count = stitch_beat_count_at(store_path, group, record)
        if count is not None:
            raw = _with_stitch_beat_count(raw, count)
    raw = _with_volume_qrs_trigger_times(raw, volume_qrs_trigger_times_at(group))
    if count == metadata.stitch_beat_count and raw is metadata.raw:
        return metadata
    return replace(metadata, stitch_beat_count=count, raw=raw)


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
class StreamHeader:
    """What a full stream load knows about a stream, read without its samples.

    ``stream`` is the typed stream of the first temporal sample only: its type, metadata and sector geometry are those
    of the full stream. ``shape`` is the stored array shape, ``frame_count`` its temporal length and ``timestamps`` the
    complete timestamps, typed and checked against that length as a full load checks them.
    ``velocity_scale_mps_per_frame`` is the complete per-frame Doppler calibration, when the stream declares one.
    """

    stream: EchoStream
    shape: tuple[int, ...]
    frame_count: int
    timestamps: np.ndarray | None
    velocity_scale_mps_per_frame: np.ndarray | None = None


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
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def close(self) -> None:
        """Release the underlying store. Loaded NumPy arrays remain usable; safe to call twice."""
        if not self._closed:
            close = getattr(self.group.store, "close", None)
            if close is not None:
                close()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> RecordingStore:
        if self._closed:
            raise ValueError("RecordingStore is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def array_paths(self) -> tuple[str, ...]:
        """Every array this recording is known to hold.

        A Croissant record can publish a reduced inventory (lookup tables, calibration vectors and segment
        coordinates have been left out of some catalogs), so the store's own array index is merged in.
        """
        declared = _attribute_array_paths(getattr(self.group, "attrs", {}))
        if self.record is not None:
            return tuple(dict.fromkeys((*self.record.array_paths, *declared)))
        # Stores without consolidated metadata use implicit sub-groups that cannot be listed,
        # so fall back to the array index every EchoXFlow store carries in its attributes.
        return _walk_array_paths(self.group) or declared

    def array_info(self, path: str) -> ArrayInfo | None:
        """Return what the store declares about an array: role, units, dimensions, derivation inputs."""
        resolved = self.resolve_array_path(path) or normalize_array_path(path)
        return array_infos_from_attrs(getattr(self.group, "attrs", {})).get(resolved)

    def stream_metadata(self, name: str) -> StreamMetadata:
        """Return the typed interpretation metadata of a stream without reading any arrays."""
        data_path = self.resolve_array_path(name) or _normalize_data_path(name)
        return _stream_metadata(data_path, dict(getattr(self.group, "attrs", {})), record=self.record)

    def velocity_scales(self, name: str) -> np.ndarray | None:
        """Return a Doppler stream's complete per-frame velocity scale in m/s (NaN where uncalibrated)."""
        data_path = self.resolve_array_path(name) or _normalize_data_path(name)
        return self._velocity_scale_per_frame(self.stream_metadata(data_path), data_path)

    def load_validity_mask(self, name: str, start: int | None = None, stop: int | None = None) -> np.ndarray | None:
        """Load the exported validity mask of a stream for the same frames, or ``None`` if none is declared.

        Colour Doppler declares `power_code > 0 and (Rx != 0 or Ry != 0)` per sample; the mask shares the
        stream's ``[time, radial, beam]`` layout and is sliced with the same frame selection.
        """
        metadata = self.stream_metadata(name)
        if not metadata.validity_mask_path or metadata.validity_mask_path not in self.group:
            return None
        mask = np.asarray(self.load_array_slice(metadata.validity_mask_path, int(start or 0), stop))
        return mask.astype(bool, copy=False)

    def heart_rate_bpm(self) -> float | None:
        """Return the heart rate the scanner recorded for this acquisition, if any (for reporting, not a label)."""
        for document in manifest_documents(getattr(self.group, "attrs", {})):
            metadata = document.get("metadata")
            if isinstance(metadata, Mapping):
                value = _optional_float(metadata.get("heart_rate_bpm"))
                if value is not None and value > 0.0:
                    return value
        return None

    def stream_summaries(self) -> tuple[StreamSummary, ...]:
        """Return the exporter's per-stream summaries stored with this recording, if it carries any."""
        attrs = getattr(self.group, "attrs", {})
        owner = None if self.record is None else self.record.recording_id
        return stream_summaries_from_attrs(dict(attrs), recording_id=owner)

    def stream_summary(self, name: str) -> StreamSummary | None:
        """Return the store's own summary of a stream; a derived product reports its source stream's summary."""
        summaries = {summary.array_path: summary for summary in self.stream_summaries()}
        data_path = self.resolve_array_path(name) or _normalize_data_path(name)
        if data_path in summaries:
            return summaries[data_path]
        source = self.stream_metadata(data_path).source_data_path
        return None if source is None else summaries.get(source)

    def segment_offsets(self, name: str) -> np.ndarray | None:
        """Return the source-segment boundaries (frame or sample indices) a stream declares, if any."""
        metadata = self.stream_metadata(name)
        path = metadata.segment_offsets_path or spectral_segment_offsets_ref(
            metadata.data_path, dict(getattr(self.group, "attrs", {}))
        )
        if not path or path not in self.group:
            return None
        return np.asarray(self.load_array(path), dtype=np.int64).reshape(-1)

    def resolve_array_path(self, path: str) -> str | None:
        """Return the path this store actually stores a requested array under, if any."""
        return resolve_array_path_in(self.group, path)

    def has_array(self, path: str) -> bool:
        """Return whether this store can satisfy a requested array path."""
        return self.resolve_array_path(path) is not None

    def array_shape(self, path: str) -> tuple[int, ...]:
        """Return the stored shape of a requested array, or `()` when it is absent."""
        resolved = self.resolve_array_path(path)
        if resolved is None:
            return ()
        return tuple(int(value) for value in getattr(self.group[resolved], "shape", ()))

    def lookup_table_paths(self) -> tuple[str, ...]:
        """Return the content-addressed lookup-table paths carried by this store."""
        return lookup_table_paths_at(self.path)

    def lookup_table_path(self, role: str) -> str | None:
        """Return the first lookup table matching a shape-derived role, if any.

        Roles come from `echoxflow.schema.lookup_table_role`: `flow_color_rgb_table`,
        `fenc_table`, and `color_rgb_table`.
        """
        return lookup_table_path_at(self.path, role)

    def load_lookup_table(self, role: str) -> np.ndarray | None:
        """Load the first lookup table matching a shape-derived role, if any."""
        path = self.lookup_table_path(role)
        return None if path is None else self.load_array(path)

    def load_array(self, path: str) -> np.ndarray:
        """Load one complete raw Zarr array by path, for example `data/2d_brightness_mode`."""
        array_path = self.resolve_array_path(path) or _normalize_array_path(path)
        return self._load_complete_array(array_path)

    def load_array_ref(self, ref: ArrayRef, recording: RecordingRef | None = None) -> np.ndarray:
        """Load an array reference, optionally from a linked recording."""
        store = self if recording is None else self.open_reference(recording)
        try:
            return store.load_array(ref.path)
        finally:
            if store is not self:
                store.close()

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
        array_path = self.resolve_array_path(path) or _normalize_array_path(path)
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
        if manifest_path is not None:
            if manifest_path not in self.group:
                raise ValueError(f"{name} references missing timestamps: {manifest_path}")
            return manifest_path
        for candidate in _timestamp_candidates(name):
            if candidate in self.group:
                return candidate
        return None

    def load_modality(self, name: str) -> LoadedArray:
        """Load a data array plus matching timestamps and inferred sample rate."""
        return self._loaded_array(self._display_stream(self.load_stream(name)))

    def load_modality_slice(self, name: str, start: int | None, stop: int | None) -> LoadedArray:
        """Load a temporal slice of a modality array plus matching timestamp slice."""
        return self._loaded_array(
            self._display_stream(self.load_stream_slice(name, start, stop)), start=int(start or 0), stop=stop
        )

    def _loaded_array(self, stream: EchoStream, *, start: int = 0, stop: int | None = None) -> LoadedArray:
        attrs = _stream_attrs(stream.metadata)
        if semantic_name(stream.data_path).startswith("1d_"):
            attrs["spectral_metadata"] = self.spectral_metadata(stream.data_path, start, stop)
            offsets = self.segment_offsets(stream.data_path)
            if offsets is not None:
                attrs["spectral_segment_offsets"] = np.asarray(offsets) - start
        return LoadedArray(
            name=stream.name,
            data_path=stream.data_path,
            data=stream.data,
            timestamps_path=stream.timestamps_path,
            timestamps=stream.timestamps,
            sample_rate_hz=stream.sample_rate_hz,
            attrs=attrs,
            stream=stream,
        )

    def _display_stream(self, stream: EchoStream) -> EchoStream:
        if isinstance(stream, ColorFlowRawStream):
            return stream.to_velocity()
        if not isinstance(stream, TissueDopplerRawStream):
            return stream
        fenc_table_path = stream.metadata.fenc_table_path
        if not fenc_table_path and stream.metadata.storage_encoding == TDI_PACKED_STORAGE_ENCODING:
            raise ValueError(f"{stream.data_path} is native packed TDI but has no FencTable metadata")
        return stream.to_float(None if not fenc_table_path else self.load_array(fenc_table_path))

    def load_stream(self, name: str) -> EchoStream:
        """Load, validate, and type one data stream."""
        data_path = self.resolve_array_path(name) or _normalize_data_path(name)
        timestamps_path = self.timestamp_path(data_path)
        timestamps = None if timestamps_path is None else self.load_array(timestamps_path)
        data = self.load_array(data_path)
        return self._stream_from_data(data_path, data, timestamps_path, timestamps)

    def load_stream_header(self, name: str, *, display: bool = False) -> StreamHeader:
        """Read a stream's type, metadata, timestamps and length without loading its samples.

        The header matches :meth:`load_stream` (``display=True``: the stream of :meth:`load_modality`, for example
        native tissue Doppler converted to velocities) and raises the same errors for missing arrays, invalid
        metadata and timestamps that do not match the stored length. Only the first temporal sample is read, so
        value checks of the other samples (non-finite floats) are not run.
        """
        data_path = self.resolve_array_path(name) or _normalize_data_path(name)
        timestamps_path = self.timestamp_path(data_path)
        timestamps = None if timestamps_path is None else self.load_array(timestamps_path)
        array = self.group[self.resolve_array_path(data_path) or _normalize_array_path(data_path)]
        shape = tuple(int(size) for size in getattr(array, "shape", ()))
        if not shape:
            stream = self._stream_from_data(data_path, np.asarray(array[...]), timestamps_path, timestamps)
            stream = self._display_stream(stream) if display else stream
            return StreamHeader(stream=stream, shape=shape, frame_count=0, timestamps=stream.timestamps)
        typed_timestamps = validated_stream_timestamps(data_path, shape, timestamps)
        first = np.asarray(array[_temporal_slice(data_path, len(shape), 0, 1)])
        stream = self._stream_from_data(
            data_path,
            first,
            timestamps_path,
            None if typed_timestamps is None else typed_timestamps[: int(first.shape[0])],
            stop=int(first.shape[0]),
        )
        if display:
            stream = self._display_stream(stream)
        return StreamHeader(
            stream=stream,
            shape=shape,
            frame_count=int(shape[0]),
            timestamps=typed_timestamps,
            velocity_scale_mps_per_frame=self._velocity_scale_per_frame(
                self.stream_metadata(data_path), data_path, frame_count=int(shape[0])
            ),
        )

    def load_stream_slice(self, name: str, start: int | None, stop: int | None) -> EchoStream:
        """Load, validate, and type a temporal slice of one data stream."""
        data_path = self.resolve_array_path(name) or _normalize_data_path(name)
        start_index = int(start or 0)
        timestamps_path = self.timestamp_path(data_path)
        timestamps = None
        if timestamps_path is not None:
            timestamps = self.load_array_slice(timestamps_path, start_index, stop)
        data = self.load_array_slice(data_path, start_index, stop)
        return self._stream_from_data(data_path, data, timestamps_path, timestamps, start=start_index, stop=stop)

    def spectral_metadata(self, name: str, start: int | None = None, stop: int | None = None) -> SpectralMetadata:
        """Return typed metadata for a spectral Doppler stream, calibrated for columns ``start:stop``.

        Stores publish baseline, Nyquist and angle correction per time column. They are loaded,
        checked against the column count and sliced like the matrix, so a selection gets its own physical axes.
        """
        data_path = _normalize_data_path(name)
        attrs = dict(getattr(self.group, "attrs", {}))
        row_count = None
        column_count = None
        if data_path in self.group:
            shape = tuple(getattr(self.group[data_path], "shape", ()))
            if len(shape) >= 2:
                row_count = int(shape[1])
            if shape:
                column_count = int(shape[0])
        calibration = self._spectral_calibration(data_path, attrs, column_count=column_count)
        if calibration is not None:
            calibration = calibration.sliced(int(start or 0), stop)
        return spectral_metadata_from_attrs(data_path, attrs, row_count=row_count, calibration=calibration)

    def _spectral_calibration(
        self, data_path: str, attrs: Mapping[str, Any], *, column_count: int | None
    ) -> SpectralCalibration | None:
        refs = spectral_calibration_refs(data_path, attrs)
        if not refs:
            return None
        values: dict[str, np.ndarray] = {}
        for key, path in refs.items():
            if path not in self.group:
                raise ValueError(f"{data_path} references spectral calibration {path}, which the store does not hold")
            expected_units = {"baseline_fraction": "fraction", "nyquist_mps": "m/s", "angle_correction_rad": "radians"}
            info = self.array_info(path)
            for units in (None if info is None else info.units, self.group[path].attrs.get("units")):
                if units is not None and units != expected_units[key]:
                    raise ValueError(f"{path} ({key}) must be in {expected_units[key]}, got {units!r}")
            array = np.asarray(self.load_array(path), dtype=np.float64).reshape(-1)
            if column_count is not None and array.size != column_count:
                raise ValueError(f"{path} has {array.size} entries but {data_path} has {column_count} columns")
            values[key] = array
        return SpectralCalibration(**values)

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
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> EchoStream:
        metadata = _stream_metadata(data_path, dict(getattr(self.group, "attrs", {})), record=self.record)
        metadata = self._with_velocity_scale(metadata, data_path, start=start, stop=stop)
        return stream_from_arrays(
            data_path=data_path,
            data=data,
            timestamps_path=timestamps_path,
            timestamps=timestamps,
            sample_rate_hz=_sample_rate(self.record, data_path),
            metadata=augmented_stream_metadata(
                metadata,
                store_path=self.path,
                group=self.group,
                record=self.record,
            ),
        )

    def _velocity_scale_per_frame(
        self, metadata: StreamMetadata, data_path: str, *, frame_count: int | None = None
    ) -> np.ndarray | None:
        path = metadata.velocity_scale_path
        if not path:
            return None
        if path not in self.group:
            raise ValueError(f"{data_path} declares per-frame velocity scale {path}, which the store does not hold")
        scale = np.asarray(self.load_array(path), dtype=np.float64).reshape(-1)
        info = self.array_info(path)
        if info is not None and info.units not in {None, "m/s"}:
            raise ValueError(f"{path} must be in m/s, got {info.units!r}")
        if frame_count is None:
            resolved = self.resolve_array_path(data_path) or _normalize_array_path(data_path)
            frame_count = int(getattr(self.group[resolved], "shape", (0,))[0])
        if scale.size != int(frame_count):
            raise ValueError(f"{path} has {scale.size} entries but {data_path} has {frame_count} frames")
        if np.any(np.isinf(scale)) or np.any(np.isfinite(scale) & (scale <= 0.0)):
            raise ValueError(f"{path} must be positive where calibrated (NaN marks uncalibrated frames)")
        return scale

    def _with_velocity_scale(
        self, metadata: StreamMetadata, data_path: str, *, start: int, stop: int | None
    ) -> StreamMetadata:
        """Attach the per-frame scale for frames ``start:stop`` and a scalar limit only when it is constant there."""
        scale = self._velocity_scale_per_frame(metadata, data_path)
        if scale is None:
            return metadata
        selected = scale[int(start) : stop]
        return replace(
            metadata,
            velocity_scale_mps_per_frame=selected,
            velocity_limit_mps=constant_velocity_scale(selected),
        )

    def stitch_beat_count(self) -> int | None:
        """Return how many cardiac cycles a stitched 3-D volume was assembled from."""
        return stitch_beat_count_at(self.path, self.group, self.record)

    def volume_qrs_trigger_times(self) -> np.ndarray | None:
        """Return the R-peak times a 3-D acquisition was gated on, in store seconds."""
        return volume_qrs_trigger_times_at(self.group)

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
        root_path = data_root(root if root is not None else record.source_root)
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
        try:
            group = zarr_module.open_group(store=store, mode="r")
        except BaseException:
            store.close()
            raise
    else:
        group = zarr_module.open_group(path, mode="r")
    check_schema_generation(getattr(group, "attrs", {}).get("schema_version"), context=f"Recording store {path}")
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
    if path.startswith("timestamps/"):
        raise ValueError(f"Expected a data array path, got timestamp path `{path}`")
    head = path.split("/", 1)[0]
    return path if head in ARRAY_GROUPS else f"data/{path}"


def _timestamp_candidates(name: str) -> tuple[str, ...]:
    name = semantic_name(name)
    candidates = [f"timestamps/{name}"]
    # The exporter timestamps the colour-flow and TDI sectors once, under the modality
    # name, while the arrays themselves carry a `_raw`, `_velocity`, or `_power` role.
    for suffix in ("_velocity", "_power", "_raw", "_display_velocity", "_lag_one_magnitude"):
        if name.endswith(suffix):
            candidates.append(f"timestamps/{name[: -len(suffix)]}")
    if name.startswith("2d_tissue_doppler"):
        candidates.append("timestamps/2d_tissue_doppler")
    if name.startswith("2d_color_doppler"):
        candidates.append("timestamps/2d_color_doppler")
    for annotation in ("_annotation_", "_overlay_physical_points_"):
        if annotation in name:
            candidates.append(f"timestamps/{name.split(annotation, 1)[0]}")
    return tuple(dict.fromkeys(candidates))


def _temporal_slice(data_path: str, data_ndim: int, start: int, stop: int | None) -> tuple[slice | int, ...] | slice:
    if start < 0:
        raise ValueError("Slice start must be non-negative")
    return slice(start, stop)


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
    normalized = _normalize_data_path(data_path)
    info = array_infos_from_attrs(group_attrs).get(normalized)
    resolved = _resolve_source_sector(normalized, group_attrs)
    if resolved is None:
        return StreamMetadata(data_path=data_path, array_info=info)
    sector, source_path = resolved
    lookup_tables = _scoped_lookup_tables(sector)
    is_tissue_doppler = semantic_name(normalized).startswith(("tissue_doppler", "2d_tissue_doppler"))
    fenc_table_path = None
    if is_tissue_doppler:
        fenc_table_path = _unique_lookup_table(lookup_tables, "fenc_table", sector)
    return StreamMetadata(
        data_path=data_path,
        value_range=_optional_float_pair(sector.get("value_range")),
        velocity_limit_mps=_optional_float(sector.get("velocity_limit_mps"))
        or _optional_float(sector.get("velocity_scale_mps")),
        velocity_limit_source=_optional_str(sector.get("velocity_limit_source"))
        or _optional_str(sector.get("velocity_scale_source")),
        storage_encoding=_optional_str(sector.get("storage_encoding")) or tdi_storage_encoding(sector),
        fenc_table_path=fenc_table_path,
        colormap_path=_unique_lookup_table(lookup_tables, "flow_color_rgb_table", sector),
        geometry=_sector_geometry(sector.get("geometry")),
        raw=sector,
        velocity_scale_path=_velocity_scale_path(sector, info, group_attrs),
        source_data_path=source_path if source_path != normalized else None,
        validity_mask_path=_array_ref_path(sector.get("validity_mask")),
        segment_offsets_path=_array_ref_path(sector.get("segment_frame_offsets"))
        or _array_ref_path(sector.get("segment_sample_offsets")),
        array_info=info,
    )


#: Sector keys that name a derived product of the sector's own frames, so the product inherits the sector.
_DERIVED_PRODUCT_KEYS: tuple[str, ...] = ("wrapped_radial_velocity", "display_velocity", "lag_one_magnitude")


def _resolve_source_sector(
    data_path: str, group_attrs: Mapping[str, Any], *, _seen: frozenset[str] = frozenset()
) -> tuple[Mapping[str, Any], str] | None:
    """Return the manifest sector that describes an array, plus the array path that sector owns.

    A sector's own frames resolve directly. A derived product (for example `derived/2d_color_doppler_velocity`)
    owns no sector; it inherits the sector of the source it was derived from, found through the sector's explicit
    product references and the array's declared `derivation.inputs`. When those point at more than one sector,
    the product is ambiguous and an error is raised rather than a sector being picked. Stores without either
    declaration fall back to the sector with a matching semantic role.
    """
    exact = _exact_sector_for_data_path(data_path, group_attrs)
    if exact is not None:
        return exact, data_path
    candidates: dict[str, tuple[Mapping[str, Any], str]] = {}
    for sector in _manifest_sectors(group_attrs):
        if any(_array_ref_path(sector.get(key)) == data_path for key in _DERIVED_PRODUCT_KEYS):
            source = _sector_frames_path(sector)
            if source is not None:
                candidates[source] = (sector, source)
    info = array_infos_from_attrs(group_attrs).get(data_path)
    for source in () if info is None else info.derivation_inputs:
        if source in _seen or source == data_path or not _is_frame_source_input(source, group_attrs):
            continue
        found = _resolve_source_sector(source, group_attrs, _seen=_seen | {data_path})
        if found is not None:
            candidates[found[1]] = found
    if len(candidates) > 1:
        raise ValueError(
            f"{data_path} derives from several manifest sectors ({', '.join(sorted(candidates))}); "
            "its geometry, clock and calibration are ambiguous"
        )
    if candidates:
        return next(iter(candidates.values()))
    role_sector = _sector_for_role(data_path, group_attrs)
    return None if role_sector is None else (role_sector, _sector_frames_path(role_sector) or data_path)


def _is_frame_source_input(path: str, group_attrs: Mapping[str, Any]) -> bool:
    """Whether a derivation input is a stream of samples (not a lookup table, calibration vector or coordinate)."""
    if is_lookup_table_path(path) or path.split("/", 1)[0] not in {"data", "derived"}:
        return False
    info = array_infos_from_attrs(group_attrs).get(path)
    return info is None or info.role not in {"lookup_table", "calibration", "mask"}


def _manifest_sectors(group_attrs: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    found: list[Mapping[str, Any]] = []
    for document in manifest_documents(group_attrs):
        sectors = document.get("sectors")
        if isinstance(sectors, list):
            found.extend(sector for sector in sectors if isinstance(sector, Mapping))
    return tuple(found)


def _sector_frames_path(sector: Mapping[str, Any]) -> str | None:
    for key in ("frames", "data", "target_data", "value"):
        path = _array_ref_path(sector.get(key))
        if path is not None:
            return normalize_array_path(path)
    return None


def _scoped_lookup_tables(sector: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return a sector's declared lookup tables by role; empty when the sector does not scope its tables."""
    entries = sector.get("lookup_tables")
    if not isinstance(entries, list):
        return {}
    tables: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        role = _optional_str(entry.get("role"))
        path = _array_ref_path(entry.get("data"))
        if role is not None and path is not None:
            tables.setdefault(role, [])
            if path not in tables[role]:
                tables[role].append(path)
    return {role: tuple(paths) for role, paths in tables.items()}


def _unique_lookup_table(tables: Mapping[str, tuple[str, ...]], role: str, sector: Mapping[str, Any]) -> str | None:
    paths = tables.get(role, ())
    if len(paths) > 1:
        raise ValueError(f"Sector {_semantic_id(sector) or '?'} declares {len(paths)} {role} lookup tables")
    return paths[0] if paths else None


def _velocity_scale_path(
    sector: Mapping[str, Any], info: ArrayInfo | None, group_attrs: Mapping[str, Any]
) -> str | None:
    """Return the per-frame velocity scale a stream decodes with: the sector's reference, else a derivation input."""
    declared = _array_ref_path(sector.get("velocity_scale_per_frame"))
    if declared is not None:
        return normalize_array_path(declared)
    if info is None:
        return None
    infos = array_infos_from_attrs(group_attrs)
    scales = [
        path
        for path in info.derivation_inputs
        if (entry := infos.get(path)) is not None
        and entry.role == "calibration"
        and entry.units == "m/s"
        and entry.dimensions == ("time",)
    ]
    if len(scales) > 1:
        raise ValueError(f"{info.path} declares several per-frame velocity scales: {', '.join(scales)}")
    return scales[0] if scales else None


def _stream_attrs(metadata: StreamMetadata) -> dict[str, Any]:
    return {
        "value_range": metadata.value_range,
        "velocity_limit_mps": metadata.velocity_limit_mps,
        "velocity_limit_source": metadata.velocity_limit_source,
        "storage_encoding": metadata.storage_encoding,
        "fenc_table_path": metadata.fenc_table_path,
        "colormap_path": metadata.colormap_path,
        "colormap_selection": metadata.colormap_selection,
        "colormap_candidates": metadata.colormap_candidates,
        "geometry": metadata.geometry,
        "stitch_beat_count": metadata.stitch_beat_count,
        "velocity_scale_mps_per_frame": metadata.velocity_scale_mps_per_frame,
        "source_data_path": metadata.source_data_path,
        "validity_mask_path": metadata.validity_mask_path,
        "units": None if metadata.array_info is None else metadata.array_info.units,
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


def _with_volume_qrs_trigger_times(
    raw: Mapping[str, Any] | None,
    qrs_trigger_times: np.ndarray | None,
) -> Mapping[str, Any] | None:
    """Publish store-held R-peak times where the manifest readers look for them."""
    if qrs_trigger_times is None or np.asarray(qrs_trigger_times).size == 0:
        return raw
    times = [float(value) for value in np.asarray(qrs_trigger_times, dtype=np.float64).reshape(-1)]
    restored = dict(raw or {})
    if "qrs_trigger_times" in restored.get("metadata", {}) or "volume_qrs_trigger_times" in restored:
        return raw
    public_metadata = dict(restored.get("metadata") or {})
    public_metadata.setdefault("qrs_trigger_times", times)
    restored["metadata"] = public_metadata
    restored.setdefault("volume_qrs_trigger_times", times)
    ecg = dict(restored.get("ecg") or {})
    ecg.setdefault("qrs_trigger_times", times)
    restored["ecg"] = ecg
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
    resolved = _resolve_source_sector(_normalize_data_path(data_path), group_attrs)
    return None if resolved is None else resolved[0]


def _sector_for_role(data_path: str, group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Fallback: the first sector whose semantic role matches the array name."""
    roles = _sector_role_candidates(data_path)
    if not roles:
        return None
    for sector in _manifest_sectors(group_attrs):
        if _semantic_id(sector) in roles:
            return sector
    return None


def _exact_sector_for_data_path(data_path: str, group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    normalized = _normalize_data_path(data_path)
    for document in manifest_documents(group_attrs):
        for key in ("sectors", "tracks"):
            for sector in document.get(key) or ():
                if isinstance(sector, Mapping) and _sector_references_data_path(sector, normalized):
                    return sector
        linked_panels = document.get("linked_panels")
        if isinstance(linked_panels, list):
            for panel in linked_panels:
                if isinstance(panel, Mapping) and _linked_panel_references_data_path(panel, normalized):
                    return panel
    return None


def _sector_references_data_path(sector: Mapping[str, Any], data_path: str) -> bool:
    return any(_array_ref_path(sector.get(key)) == data_path for key in ("frames", "data", "target_data", "value"))


def _linked_panel_references_data_path(panel: Mapping[str, Any], data_path: str) -> bool:
    linked = panel.get("linked_recording")
    if not isinstance(linked, Mapping):
        return False
    frame_path = _optional_str(linked.get("frames_array_path"))
    return (
        _normalize_data_path(frame_path) == data_path
        if frame_path is not None
        else data_path == "data/2d_brightness_mode"
    )


def _sector_role_candidates(data_path: str) -> tuple[str, ...]:
    name = semantic_name(data_path)
    if name == "3d_brightness_mode":
        return ("bmode", "3d_brightness_mode")
    if name.startswith("2d_brightness_mode"):
        return ("bmode",)
    if name in {"2d_color_doppler_velocity", "2d_color_doppler_raw", "2d_color_doppler_lag_one_magnitude"}:
        return ("2d_color_doppler_velocity", "2d_color_doppler", "color_doppler")
    if name == "2d_color_doppler_power":
        return ("2d_color_doppler_power", "2d_color_doppler", "color_doppler")
    if name in {"tissue_doppler", "2d_tissue_doppler_raw", "2d_tissue_doppler_display_velocity"}:
        return ("tissue_doppler", "predicted_tissue_doppler")
    if name == "segmentation_overlay":
        return ("segmentation_overlay",)
    return ()


def _three_dimensional_document(group_attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for document in manifest_documents(group_attrs):
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
        timestamp_path = _sector_timestamp_path(sector)
        if timestamp_path is not None:
            return timestamp_path
    if data_path != "data/3d_brightness_mode":
        return None
    for document in manifest_documents(group_attrs):
        timelines = document.get("timelines")
        if isinstance(timelines, Mapping):
            timestamp_path = _array_ref_path(timelines.get("frame_timestamps"))
            if timestamp_path is not None:
                return timestamp_path
    return None


def _sector_timestamp_path(sector: Mapping[str, Any]) -> str | None:
    for key in ("timestamps", "frame_timestamps", "time"):
        timestamp_path = _array_ref_path(sector.get(key))
        if timestamp_path is not None:
            return timestamp_path
    linked = sector.get("linked_recording")
    if isinstance(linked, Mapping):
        timestamp_path = _optional_str(linked.get("timestamps_array_path"))
        if timestamp_path is not None:
            return timestamp_path.strip("/")
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


def _attribute_array_paths(attrs: Mapping[str, Any]) -> tuple[str, ...]:
    entries = attrs.get("arrays")
    if not isinstance(entries, list):
        return ()
    names = (str(entry.get("name") or "").strip("/") for entry in entries if isinstance(entry, Mapping))
    return tuple(dict.fromkeys(name for name in names if name))


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
