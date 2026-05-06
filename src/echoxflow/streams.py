"""Typed and validated EchoXFlow data streams."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from echoxflow.scan.geometry import SectorGeometry

TDI_NATIVE_STORAGE_ENCODING = "ge_tdi_raw_u16"
TDI_LINEAR_UINT8_STORAGE_ENCODING = "linear_velocity_uint8_mps_v1"
TDI_STORAGE_ENCODING = TDI_NATIVE_STORAGE_ENCODING

FloatArray: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]
UInt16Array: TypeAlias = NDArray[np.uint16]
Int32Array: TypeAlias = NDArray[np.int32]
Int64Array: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class StreamMetadata:
    """Validated metadata relevant to stream interpretation."""

    data_path: str
    value_range: tuple[float, float] | None = None
    velocity_limit_mps: float | None = None
    velocity_limit_source: str | None = None
    storage_encoding: str | None = None
    fenc_table_path: str | None = None
    colormap_path: str | None = None
    geometry: SectorGeometry | None = None
    stitch_beat_count: int | None = None
    raw: Mapping[str, Any] | None = None

    def with_value_range(self, value_range: tuple[float, float] | None) -> StreamMetadata:
        return replace(self, value_range=value_range)


@dataclass(frozen=True)
class BaseStream:
    name: str
    data_path: str
    timestamps_path: str | None
    timestamps: FloatArray | None
    sample_rate_hz: float | None
    metadata: StreamMetadata


@dataclass(frozen=True)
class GenericStream(BaseStream):
    data: np.ndarray
    kind: Literal["generic"] = "generic"


@dataclass(frozen=True)
class EcgStream(BaseStream):
    data: FloatArray
    kind: Literal["ecg"] = "ecg"


@dataclass(frozen=True)
class BrightnessModeStream(BaseStream):
    data: UInt8Array
    kind: Literal["brightness_mode"] = "brightness_mode"

    def to_float(self) -> FloatImageStream:
        return FloatImageStream.from_source(self, self.data.astype(np.float32) / 255.0, value_range=(0.0, 1.0))


@dataclass(frozen=True)
class FloatImageStream(BaseStream):
    data: FloatArray
    kind: Literal["float_image"] = "float_image"

    @classmethod
    def from_source(
        cls,
        source: BaseStream,
        data: FloatArray,
        *,
        value_range: tuple[float, float] | None,
    ) -> FloatImageStream:
        return cls(
            name=source.name,
            data_path=source.data_path,
            data=np.asarray(data, dtype=np.float32),
            timestamps_path=source.timestamps_path,
            timestamps=source.timestamps,
            sample_rate_hz=source.sample_rate_hz,
            metadata=source.metadata.with_value_range(value_range),
        )


@dataclass(frozen=True)
class ColorDopplerVelocityStream(BaseStream):
    data: FloatArray
    kind: Literal["color_doppler_velocity"] = "color_doppler_velocity"

    def to_float(self) -> ColorDopplerVelocityStream:
        return self


@dataclass(frozen=True)
class ColorDopplerPowerStream(BaseStream):
    data: FloatArray
    kind: Literal["color_doppler_power"] = "color_doppler_power"

    def to_float(self) -> ColorDopplerPowerStream:
        return self


@dataclass(frozen=True)
class TissueDopplerRawStream(BaseStream):
    data: UInt16Array
    kind: Literal["tissue_doppler_raw"] = "tissue_doppler_raw"

    def to_float(self, fenc_table: np.ndarray) -> TissueDopplerFloatStream:
        velocity_limit = _positive_float(self.metadata.velocity_limit_mps, "Tissue Doppler velocity_limit_mps")
        physical = render_tdi_native_codes(self.data, fenc_table=fenc_table, velocity_scale_mps=velocity_limit)
        return TissueDopplerFloatStream(
            name=self.name,
            data_path=self.data_path,
            data=physical,
            timestamps_path=self.timestamps_path,
            timestamps=self.timestamps,
            sample_rate_hz=self.sample_rate_hz,
            metadata=self.metadata.with_value_range((-velocity_limit, velocity_limit)),
        )


@dataclass(frozen=True)
class TissueDopplerFloatStream(BaseStream):
    data: FloatArray
    kind: Literal["tissue_doppler_float"] = "tissue_doppler_float"

    def to_float(self) -> TissueDopplerFloatStream:
        return self


@dataclass(frozen=True)
class SpectralDopplerStream(BaseStream):
    data: FloatArray
    kind: Literal["spectral_doppler"] = "spectral_doppler"

    def to_float(self) -> SpectralDopplerStream:
        return self


@dataclass(frozen=True)
class AnnotationMaskStream(BaseStream):
    data: FloatArray
    kind: Literal["annotation_mask"] = "annotation_mask"


@dataclass(frozen=True)
class MeshPointStream(BaseStream):
    data: FloatArray
    kind: Literal["mesh_points"] = "mesh_points"


@dataclass(frozen=True)
class MeshFaceStream(BaseStream):
    data: Int32Array
    kind: Literal["mesh_faces"] = "mesh_faces"


@dataclass(frozen=True)
class MeshFrameOffsetsStream(BaseStream):
    data: Int64Array
    kind: Literal["mesh_frame_offsets"] = "mesh_frame_offsets"


EchoStream: TypeAlias = (
    GenericStream
    | EcgStream
    | BrightnessModeStream
    | FloatImageStream
    | ColorDopplerVelocityStream
    | ColorDopplerPowerStream
    | TissueDopplerRawStream
    | TissueDopplerFloatStream
    | SpectralDopplerStream
    | AnnotationMaskStream
    | MeshPointStream
    | MeshFaceStream
    | MeshFrameOffsetsStream
)


def temporal_axis_for_path(data_path: str, data_ndim: int) -> int:
    """Return the axis that carries time for an EchoXFlow data path."""
    return 0


def temporal_sample_count(data_path: str, data: np.ndarray) -> int:
    """Return the number of temporal samples for one data array."""
    arr = np.asarray(data)
    if arr.ndim == 0:
        return 0
    return int(arr.shape[temporal_axis_for_path(data_path, arr.ndim)])


def stream_from_arrays(
    *,
    data_path: str,
    data: np.ndarray,
    timestamps_path: str | None,
    timestamps: np.ndarray | None,
    sample_rate_hz: float | None,
    metadata: StreamMetadata | None = None,
) -> EchoStream:
    """Validate arrays and return the strongest typed stream for a data path."""
    normalized_path = _normalize_data_path(data_path)
    name = normalized_path.removeprefix("data/")
    stream_metadata = metadata or StreamMetadata(data_path=normalized_path)
    typed_timestamps = _timestamps_or_none(timestamps)
    base_kwargs: dict[str, Any] = {
        "name": name,
        "data_path": normalized_path,
        "timestamps_path": timestamps_path,
        "timestamps": typed_timestamps,
        "sample_rate_hz": sample_rate_hz,
    }
    arr = np.asarray(data)
    _validate_temporal_length(normalized_path, arr, typed_timestamps)
    if name == "ecg" or name.endswith("_ecg"):
        return EcgStream(data=_float_array(arr, ndim=1, label=normalized_path), metadata=stream_metadata, **base_kwargs)
    if _is_brightness_mode(name):
        return BrightnessModeStream(
            data=_uint8_array(arr, min_ndim=2, label=normalized_path), metadata=stream_metadata, **base_kwargs
        )
    if name == "2d_color_doppler_velocity":
        limit = _positive_float(stream_metadata.velocity_limit_mps, "Color Doppler velocity_limit_mps")
        metadata_with_range = stream_metadata.with_value_range((-limit, limit))
        return ColorDopplerVelocityStream(
            data=_float_array(arr, min_ndim=2, label=normalized_path), metadata=metadata_with_range, **base_kwargs
        )
    if name == "2d_color_doppler_power":
        return ColorDopplerPowerStream(
            data=_float_array(arr, min_ndim=2, label=normalized_path),
            metadata=stream_metadata.with_value_range((0.0, 1.0)),
            **base_kwargs,
        )
    if name == "tissue_doppler":
        return _tissue_doppler_stream(arr, stream_metadata, base_kwargs)
    if name.startswith("1d_"):
        return SpectralDopplerStream(
            data=_float_array(arr, min_ndim=1, label=normalized_path), metadata=stream_metadata, **base_kwargs
        )
    if _is_annotation_mask(name):
        return AnnotationMaskStream(
            data=_float_array(arr, min_ndim=2, label=normalized_path), metadata=stream_metadata, **base_kwargs
        )
    if _is_mesh_points(name):
        return MeshPointStream(data=_mesh_points(arr, normalized_path), metadata=stream_metadata, **base_kwargs)
    if _is_mesh_faces(name):
        return MeshFaceStream(data=_mesh_faces(arr, normalized_path), metadata=stream_metadata, **base_kwargs)
    if _is_mesh_frame_offsets(name):
        return MeshFrameOffsetsStream(
            data=_mesh_frame_offsets(arr, normalized_path), metadata=stream_metadata, **base_kwargs
        )
    return GenericStream(data=arr, metadata=stream_metadata, **base_kwargs)


def render_tdi_native_codes(
    raw_codes: np.ndarray,
    *,
    fenc_table: np.ndarray,
    velocity_scale_mps: float,
) -> FloatArray:
    """Convert native GE TDI uint16 codes to physical velocities."""
    codes = np.asarray(raw_codes, dtype=np.uint16)
    if codes.ndim != 3:
        raise ValueError(f"TDI native raw codes must be a 3D frame stack, got shape={codes.shape}")
    fenc_flat = np.asarray(fenc_table, dtype=np.uint8).reshape(-1, 4)
    if fenc_flat.shape[0] < 256 * 256:
        raise ValueError(f"TDI FencTable must contain at least 65536 rows, got {fenc_flat.shape[0]}")
    scale = _positive_float(velocity_scale_mps, "TDI velocity_scale_mps")
    mapped = fenc_flat[: 256 * 256][codes]
    col = mapped[..., 0].astype(np.int16)
    col_physical = 255 - ((col + 128) & 0xFF)
    physical = (((col_physical.astype(np.float32) / 255.0) * 2.0) - 1.0) * scale
    return np.asarray(physical, dtype=np.float32)


def encode_tdi_native_codes(
    velocities_mps: np.ndarray,
    *,
    fenc_table: np.ndarray,
    velocity_scale_mps: float,
) -> UInt16Array:
    """Quantize physical TDI velocities back to native GE uint16 codes."""
    scale = _positive_float(velocity_scale_mps, "TDI velocity_scale_mps")
    fenc_flat = np.asarray(fenc_table, dtype=np.uint8).reshape(-1, 4)
    if fenc_flat.shape[0] < 256 * 256:
        raise ValueError(f"TDI FencTable must contain at least 65536 rows, got {fenc_flat.shape[0]}")
    code_by_bucket = _native_tdi_code_by_velocity_bucket(fenc_flat[: 256 * 256])
    values = np.nan_to_num(np.asarray(velocities_mps, dtype=np.float32), nan=0.0, posinf=scale, neginf=-scale)
    normalized = ((np.clip(values, -scale, scale) / scale) + 1.0) * 0.5
    buckets = np.clip(np.rint(normalized * 255.0), 0.0, 255.0).astype(np.uint8)
    return cast(UInt16Array, code_by_bucket[buckets])


def render_tdi_linear_uint8_codes(
    velocity_codes: np.ndarray,
    *,
    velocity_scale_mps: float,
) -> FloatArray:
    """Convert exported linear TDI uint8 velocity codes to physical velocities."""
    codes = np.asarray(velocity_codes)
    if codes.dtype != np.uint8:
        raise ValueError(f"TDI linear velocity codes must use uint8 storage, got {codes.dtype}")
    scale = _positive_float(velocity_scale_mps, "TDI velocity_scale_mps")
    physical = (((codes.astype(np.float32) / 255.0) * 2.0) - 1.0) * scale
    return np.asarray(physical, dtype=np.float32)


def encode_tdi_linear_uint8_codes(
    velocities_mps: np.ndarray,
    *,
    velocity_scale_mps: float,
) -> UInt8Array:
    """Quantize physical TDI velocities to linear uint8 velocity codes."""
    scale = _positive_float(velocity_scale_mps, "TDI velocity_scale_mps")
    values = np.nan_to_num(np.asarray(velocities_mps, dtype=np.float32), nan=0.0, posinf=scale, neginf=-scale)
    normalized = ((np.clip(values, -scale, scale) / scale) + 1.0) * 0.5
    return cast(UInt8Array, np.clip(np.rint(normalized * 255.0), 0.0, 255.0).astype(np.uint8))


def default_value_range_for_path(data_path: str, values: np.ndarray | None = None) -> tuple[float, float] | None:
    """Return a fixed non-adaptive range for streams without metadata-derived ranges."""
    name = _normalize_data_path(data_path).removeprefix("data/")
    if name.startswith("2d_brightness_mode") or name in {
        "2d_biplane_brightness_mode",
        "2d_triplane_brightness_mode",
        "3d_brightness_mode",
        "1d_motion_mode",
    }:
        return (0.0, 255.0)
    if name in {"1d_pulsed_wave_doppler", "1d_continuous_wave_doppler"}:
        return (0.0, 255.0)
    if name == "2d_color_doppler_power":
        return (0.0, 1.0)
    if values is not None:
        return _dtype_value_range(np.asarray(values))
    return None


def _tissue_doppler_stream(
    arr: np.ndarray, metadata: StreamMetadata, base_kwargs: Mapping[str, Any]
) -> TissueDopplerRawStream | TissueDopplerFloatStream:
    limit = _positive_float(metadata.velocity_limit_mps, "Tissue Doppler velocity_limit_mps")
    if metadata.storage_encoding == TDI_NATIVE_STORAGE_ENCODING:
        if not metadata.fenc_table_path:
            raise ValueError("Native TDI stream is missing fenc_table_path metadata")
        return TissueDopplerRawStream(
            data=_uint16_array(arr, ndim=3, label="data/tissue_doppler"),
            metadata=metadata.with_value_range(_dtype_value_range(arr)),
            **base_kwargs,
        )
    if metadata.storage_encoding == TDI_LINEAR_UINT8_STORAGE_ENCODING:
        return TissueDopplerFloatStream(
            data=render_tdi_linear_uint8_codes(arr, velocity_scale_mps=limit),
            metadata=metadata.with_value_range((-limit, limit)),
            **base_kwargs,
        )
    return TissueDopplerFloatStream(
        data=_float_array(arr, min_ndim=2, label="data/tissue_doppler"),
        metadata=metadata.with_value_range((-limit, limit)),
        **base_kwargs,
    )


def _timestamps_or_none(timestamps: np.ndarray | None) -> FloatArray | None:
    if timestamps is None:
        return None
    values = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Timestamps must be finite")
    return values


def _validate_temporal_length(data_path: str, data: np.ndarray, timestamps: FloatArray | None) -> None:
    if timestamps is None or data.ndim == 0:
        return
    expected = temporal_sample_count(data_path, data)
    if timestamps.size != expected:
        raise ValueError(f"{data_path} has {expected} temporal samples but {timestamps.size} timestamps")


def _native_tdi_code_by_velocity_bucket(fenc_table: np.ndarray) -> UInt16Array:
    col = fenc_table[:, 0].astype(np.int16)
    buckets = np.asarray(255 - ((col + 128) & 0xFF), dtype=np.uint8)
    code_by_bucket = np.zeros(256, dtype=np.uint16)
    seen = np.zeros(256, dtype=bool)
    for code, bucket in enumerate(buckets):
        index = int(bucket)
        if not seen[index]:
            code_by_bucket[index] = np.uint16(code)
            seen[index] = True
    if np.all(seen):
        return cast(UInt16Array, code_by_bucket)
    available = np.flatnonzero(seen)
    if available.size == 0:
        raise ValueError("TDI FencTable does not contain any usable velocity buckets")
    for missing in np.flatnonzero(~seen):
        nearest = available[int(np.argmin(np.abs(available - missing)))]
        code_by_bucket[int(missing)] = code_by_bucket[int(nearest)]
    return cast(UInt16Array, code_by_bucket)


def _float_array(
    values: np.ndarray,
    *,
    label: str,
    ndim: int | None = None,
    min_ndim: int | None = None,
) -> FloatArray:
    arr = np.asarray(values, dtype=np.float32)
    _validate_ndim(arr, label=label, ndim=ndim, min_ndim=min_ndim)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values")
    return cast(FloatArray, arr)


def _uint8_array(values: np.ndarray, *, label: str, ndim: int | None = None, min_ndim: int | None = None) -> UInt8Array:
    arr = np.asarray(values)
    _validate_ndim(arr, label=label, ndim=ndim, min_ndim=min_ndim)
    if arr.dtype != np.uint8:
        raise ValueError(f"{label} must use uint8 storage, got {arr.dtype}")
    return cast(UInt8Array, arr)


def _uint16_array(
    values: np.ndarray, *, label: str, ndim: int | None = None, min_ndim: int | None = None
) -> UInt16Array:
    arr = np.asarray(values)
    _validate_ndim(arr, label=label, ndim=ndim, min_ndim=min_ndim)
    if arr.dtype != np.uint16:
        raise ValueError(f"{label} must use uint16 storage, got {arr.dtype}")
    return cast(UInt16Array, arr)


def _mesh_points(values: np.ndarray, label: str) -> FloatArray:
    arr = _float_array(values, label=label, min_ndim=2)
    if arr.shape[-1] != 3:
        raise ValueError(f"{label} mesh points must have xyz coordinates on the last axis, got shape={arr.shape}")
    return arr


def _mesh_faces(values: np.ndarray, label: str) -> Int32Array:
    arr = np.asarray(values)
    _validate_ndim(arr, label=label, ndim=2, min_ndim=None)
    if arr.shape[1] not in {3, 4}:
        raise ValueError(f"{label} mesh faces must have 3 or 4 vertex indices per face, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{label} mesh faces must use integer storage, got {arr.dtype}")
    if arr.size and int(np.min(arr)) < 0:
        raise ValueError(f"{label} mesh faces must not contain negative vertex indices")
    return cast(Int32Array, arr.astype(np.int32, copy=False))


def _mesh_frame_offsets(values: np.ndarray, label: str) -> Int64Array:
    arr = np.asarray(values)
    _validate_ndim(arr, label=label, ndim=1, min_ndim=None)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{label} frame offsets must use integer storage, got {arr.dtype}")
    offsets = arr.astype(np.int64, copy=False)
    if offsets.size and int(offsets[0]) != 0:
        raise ValueError(f"{label} frame offsets must start at 0")
    if offsets.size > 1 and np.any(np.diff(offsets) < 0):
        raise ValueError(f"{label} frame offsets must be monotonically increasing")
    return cast(Int64Array, offsets)


def _validate_ndim(values: np.ndarray, *, label: str, ndim: int | None, min_ndim: int | None) -> None:
    if ndim is not None and values.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions, got shape={values.shape}")
    if min_ndim is not None and values.ndim < min_ndim:
        raise ValueError(f"{label} must have at least {min_ndim} dimensions, got shape={values.shape}")


def _positive_float(value: float | int | None, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite, got {value!r}")
    return result


def _dtype_value_range(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(arr.dtype, np.bool_):
        return 0.0, 1.0
    return 0.0, 1.0


def _is_brightness_mode(name: str) -> bool:
    return name.startswith("2d_brightness_mode") or name in {
        "2d_biplane_brightness_mode",
        "2d_triplane_brightness_mode",
        "3d_brightness_mode",
    }


def _is_annotation_mask(name: str) -> bool:
    return name.endswith("_mask") or "segmentation_mask" in name or "annotation_mask" in name


def _is_mesh_points(name: str) -> bool:
    return name.endswith("_mesh_points") or (name.endswith("_points") and "mesh" in name)


def _is_mesh_faces(name: str) -> bool:
    return name.endswith("_mesh_faces") or (name.endswith("_faces") and "mesh" in name)


def _is_mesh_frame_offsets(name: str) -> bool:
    return name.endswith("_mesh_frame_offsets") or (name.endswith("_frame_offsets") and "mesh" in name)


def _normalize_data_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("data/") else f"data/{text}"
