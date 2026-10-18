"""Typed and validated EchoXFlow data streams."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from echoxflow.scan.geometry import SectorGeometry
from echoxflow.schema import (
    ARRAY_GROUPS,
    TDI_PACKED_STORAGE_ENCODING,
    TDI_SAMPLE_TYPES,
    ArrayInfo,
    color_flow_raw_to_power,
    color_flow_raw_to_velocity,
    semantic_name,
    tdi_raw_to_velocity,
)

FloatArray: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]
Int32Array: TypeAlias = NDArray[np.int32]
Int64Array: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class StreamMetadata:
    """Validated metadata relevant to stream interpretation.

    Doppler calibration has two forms. ``velocity_limit_mps`` is a single scale and is set only when one scale
    applies to every frame. ``velocity_scale_mps_per_frame`` is the exported per-frame scale, sliced together with
    the frames, with NaN marking frames that carry no calibration. A stream whose scale changes therefore has a
    per-frame scale and no scalar limit; no first, median or maximum value stands in for it.

    ``source_data_path`` is the source array a derived product inherits its sector, clock and calibration from, and
    ``array_info`` the store's own declaration (units, dimensions, derivation) of the loaded array.
    """

    data_path: str
    value_range: tuple[float, float] | None = None
    velocity_limit_mps: float | None = None
    velocity_limit_source: str | None = None
    storage_encoding: str | None = None
    fenc_table_path: str | None = None
    colormap_path: str | None = None
    colormap_selection: str | None = None
    colormap_candidates: tuple[str, ...] = ()
    geometry: SectorGeometry | None = None
    stitch_beat_count: int | None = None
    raw: Mapping[str, Any] | None = None
    velocity_scale_mps_per_frame: np.ndarray | None = field(default=None, compare=False, repr=False)
    velocity_scale_path: str | None = None
    source_data_path: str | None = None
    validity_mask_path: str | None = None
    segment_offsets_path: str | None = None
    array_info: ArrayInfo | None = None

    def with_value_range(self, value_range: tuple[float, float] | None) -> StreamMetadata:
        return replace(self, value_range=value_range)

    def with_frame_slice(self, start: int, stop: int | None) -> StreamMetadata:
        """Return this metadata for frames ``start:stop``; the per-frame scale is sliced with them."""
        scale = self.velocity_scale_mps_per_frame
        if scale is None:
            return self
        return replace(self, velocity_scale_mps_per_frame=np.asarray(scale)[int(start) : stop])

    @property
    def acquisition_controls(self) -> Mapping[str, Any]:
        """The sector's scanner controls (gain, PRF, packet size, ...), for reporting and covariates.

        These are the controls as the source recorded them; calibration that changes during the stream is carried
        by ``velocity_scale_mps_per_frame``, not by the control snapshot.
        """
        controls = None if self.raw is None else self.raw.get("stream_controls")
        return controls if isinstance(controls, Mapping) else {}

    @property
    def velocity_scale_status(self) -> str:
        """``constant``, ``varying``, ``partial`` (some frames uncalibrated) or ``missing``."""
        scale = self.velocity_scale_mps_per_frame
        if scale is None:
            return "missing" if self.velocity_limit_mps is None else "constant"
        return velocity_scale_status(scale)

    def velocity_scales_mps(self, frame_count: int) -> np.ndarray | None:
        """Return one scale per frame (NaN where uncalibrated), or ``None`` without any calibration."""
        if self.velocity_scale_mps_per_frame is not None:
            scale = np.asarray(self.velocity_scale_mps_per_frame, dtype=np.float64).reshape(-1)
            if scale.size != int(frame_count):
                raise ValueError(
                    f"{self.data_path} has {frame_count} frames but {scale.size} per-frame velocity scales"
                )
            return scale
        if self.velocity_limit_mps is None:
            return None
        return np.full(int(frame_count), float(self.velocity_limit_mps), dtype=np.float64)


def velocity_scale_status(scale: np.ndarray) -> str:
    """Classify a per-frame velocity scale as ``constant``, ``varying``, ``partial`` or ``missing``."""
    values = np.asarray(scale, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "missing"
    if finite.size < values.size:
        return "partial"
    return "constant" if np.all(finite == finite[0]) else "varying"


def constant_velocity_scale(scale: np.ndarray | None) -> float | None:
    """Return the one scale of a fully calibrated, constant per-frame scale, else ``None``."""
    if scale is None or velocity_scale_status(scale) != "constant":
        return None
    return float(np.asarray(scale, dtype=np.float64).reshape(-1)[0])


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
class ColorFlowRawStream(BaseStream):
    """Native GE colour-flow estimator output, `[time, radial, beam, 4]` uint8.

    Channels are `rx`, `ry`, `power`, `auxiliary`. Velocity is the phase of the lag-one
    autocorrelation `(rx, ry)`; power is the third channel as a fraction of full scale.
    """

    data: UInt8Array
    kind: Literal["color_flow_raw"] = "color_flow_raw"

    def to_velocity(self) -> ColorDopplerVelocityStream:
        scale = _velocity_scale(self.metadata, int(self.data.shape[0]), "Color Doppler velocity_limit_mps")
        return ColorDopplerVelocityStream(
            name=self.name,
            data_path=self.data_path,
            data=color_flow_raw_to_velocity(self.data, velocity_limit_mps=scale),
            timestamps_path=self.timestamps_path,
            timestamps=self.timestamps,
            sample_rate_hz=self.sample_rate_hz,
            metadata=self.metadata.with_value_range(_velocity_value_range(scale)),
        )

    def to_power(self) -> ColorDopplerPowerStream:
        return ColorDopplerPowerStream(
            name=self.name,
            data_path=self.data_path,
            data=color_flow_raw_to_power(self.data),
            timestamps_path=self.timestamps_path,
            timestamps=self.timestamps,
            sample_rate_hz=self.sample_rate_hz,
            metadata=self.metadata.with_value_range((0.0, 1.0)),
        )

    def to_float(self) -> ColorDopplerVelocityStream:
        return self.to_velocity()


@dataclass(frozen=True)
class TissueDopplerRawStream(BaseStream):
    """Native tissue Doppler estimator output, decoded by its declared storage encoding.

    Stores two uint8 channels, ``[time, radial, beam, 2]``, in one of two encodings
    (``metadata.storage_encoding``): the bytes of a packed ``RxRyPower(5,5,6)`` code that the sector's FencTable
    maps to a frequency, or separate ``FreqBw(8,8)`` frequency and bandwidth channels.
    """

    data: UInt8Array
    kind: Literal["tissue_doppler_raw"] = "tissue_doppler_raw"

    def to_float(self, fenc_table: np.ndarray | None = None) -> TissueDopplerFloatStream:
        frame_count = int(self.data.shape[0])
        scale = _velocity_scale(self.metadata, frame_count, "Tissue Doppler velocity_limit_mps")
        physical = tdi_raw_to_velocity(
            self.data,
            storage_encoding=str(self.metadata.storage_encoding),
            velocity_scale_mps=scale,
            fenc_table=fenc_table,
        )
        value_range = _velocity_value_range(scale)
        return TissueDopplerFloatStream(
            name=self.name,
            data_path=self.data_path,
            data=physical,
            timestamps_path=self.timestamps_path,
            timestamps=self.timestamps,
            sample_rate_hz=self.sample_rate_hz,
            metadata=self.metadata.with_value_range(value_range),
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
    | ColorFlowRawStream
    | TissueDopplerRawStream
    | TissueDopplerFloatStream
    | SpectralDopplerStream
    | AnnotationMaskStream
    | MeshPointStream
    | MeshFaceStream
    | MeshFrameOffsetsStream
)


def temporal_sample_count(data_path: str, data: np.ndarray) -> int:
    """Return the number of temporal samples for one data array."""
    arr = np.asarray(data)
    if arr.ndim == 0:
        return 0
    return int(arr.shape[0])


def validated_stream_timestamps(
    data_path: str,
    shape: Sequence[int],
    timestamps: np.ndarray | None,
) -> FloatArray | None:
    """Type and check timestamps for an array of ``shape`` exactly as :func:`stream_from_arrays` does for its data.

    This validates a stream's timestamps against the stored array length without reading the samples.
    """
    normalized_path = _normalize_data_path(data_path)
    typed_timestamps = _timestamps_or_none(timestamps)
    dims = tuple(int(size) for size in shape)
    if typed_timestamps is None or not dims:
        return typed_timestamps
    expected = dims[0]
    if typed_timestamps.size != expected:
        raise ValueError(f"{normalized_path} has {expected} temporal samples but {typed_timestamps.size} timestamps")
    return typed_timestamps


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
    name = semantic_name(normalized_path)
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
    if _is_color_doppler_role(name, "velocity"):
        scale = _velocity_scale(
            stream_metadata, temporal_sample_count(normalized_path, arr), "Color Doppler velocity_limit_mps"
        )
        metadata_with_range = stream_metadata.with_value_range(_velocity_value_range(scale))
        return ColorDopplerVelocityStream(
            # Samples with no flow estimate are NaN by construction: the lag-one
            # autocorrelation has no phase where its real and imaginary parts are both zero.
            data=_float_array(arr, min_ndim=2, label=normalized_path, allow_nan=True),
            metadata=metadata_with_range,
            **base_kwargs,
        )
    if _is_color_doppler_role(name, "power"):
        return ColorDopplerPowerStream(
            data=_float_array(arr, min_ndim=2, label=normalized_path),
            metadata=stream_metadata.with_value_range((0.0, 1.0)),
            **base_kwargs,
        )
    if _is_color_doppler_role(name, "raw"):
        return ColorFlowRawStream(
            data=_uint8_array(arr, min_ndim=4, label=normalized_path),
            metadata=stream_metadata.with_value_range(_dtype_value_range(arr)),
            **base_kwargs,
        )
    if _is_tissue_doppler_stream(name):
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


def default_value_range_for_path(data_path: str, values: np.ndarray | None = None) -> tuple[float, float] | None:
    """Return a fixed non-adaptive range for streams without metadata-derived ranges."""
    name = semantic_name(data_path)
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
    values = np.asarray(arr)
    # A sector's storage encoding describes its raw array even when the caller asked for the derived display
    # velocity beside it, so dispatch on the array's own layout first.
    if values.dtype == np.uint8 and values.ndim == 4:
        encoding = metadata.storage_encoding
        if encoding not in TDI_SAMPLE_TYPES:
            raise ValueError(
                f"{metadata.data_path} is two-channel native TDI without a supported storage encoding: {encoding!r}"
            )
        if encoding == TDI_PACKED_STORAGE_ENCODING and not metadata.fenc_table_path:
            raise ValueError("Packed native TDI stream is missing its sector's fenc_table_path metadata")
        _velocity_scale(metadata, int(values.shape[0]), "Tissue Doppler velocity_limit_mps")
        return TissueDopplerRawStream(
            data=_uint8_array(values, ndim=4, label=metadata.data_path),
            metadata=metadata.with_value_range(_dtype_value_range(values)),
            **base_kwargs,
        )
    # Physical velocities (the display product): values are already in m/s, scaled frame by frame.
    data = _float_array(arr, min_ndim=2, label="data/tissue_doppler", allow_nan=True)
    scale = _velocity_scale(metadata, int(data.shape[0]), "Tissue Doppler velocity_limit_mps")
    return TissueDopplerFloatStream(
        data=data,
        metadata=metadata.with_value_range(_velocity_value_range(scale)),
        **base_kwargs,
    )


def _velocity_scale(metadata: StreamMetadata, frame_count: int, label: str) -> float | np.ndarray:
    """Return the calibration a Doppler stream decodes with: one scalar, or one scale per frame."""
    if metadata.velocity_scale_mps_per_frame is not None:
        scale = metadata.velocity_scales_mps(frame_count)
        assert scale is not None
        if np.any(np.isinf(scale)) or np.any(np.isfinite(scale) & (scale <= 0.0)):
            raise ValueError(f"{label} per-frame scale must be positive where calibrated")
        if frame_count and not np.any(np.isfinite(scale)):
            raise ValueError(f"{label} is required: no frame of {metadata.data_path} has a calibrated scale")
        constant = constant_velocity_scale(scale)
        return constant if constant is not None else scale
    return _positive_float(metadata.velocity_limit_mps, label)


def _velocity_value_range(scale: float | np.ndarray) -> tuple[float, float] | None:
    """Symmetric display range covering every calibrated frame; a display bound, not a physical scale."""
    values = np.asarray(scale, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    limit = float(np.max(finite))
    return (-limit, limit)


def _is_color_doppler_role(name: str, role: str) -> bool:
    """Match primary and numbered secondary colour streams, e.g. ``2d_color_doppler_velocity_2``."""
    stem = f"2d_color_doppler_{role}"
    return name == stem or (name.startswith(f"{stem}_") and name.removeprefix(f"{stem}_").isdigit())


def _is_tissue_doppler_stream(name: str) -> bool:
    if name == "tissue_doppler":
        return True
    for stem in ("2d_tissue_doppler_raw", "2d_tissue_doppler_display_velocity"):
        if name == stem or (name.startswith(f"{stem}_") and name.removeprefix(f"{stem}_").isdigit()):
            return True
    return False


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


def _float_array(
    values: np.ndarray,
    *,
    label: str,
    ndim: int | None = None,
    min_ndim: int | None = None,
    allow_nan: bool = False,
) -> FloatArray:
    arr = np.asarray(values, dtype=np.float32)
    _validate_ndim(arr, label=label, ndim=ndim, min_ndim=min_ndim)
    if arr.size:
        invalid = np.isinf(arr) if allow_nan else ~np.isfinite(arr)
        if np.any(invalid):
            raise ValueError(f"{label} contains non-finite values")
    return cast(FloatArray, arr)


def _uint8_array(values: np.ndarray, *, label: str, ndim: int | None = None, min_ndim: int | None = None) -> UInt8Array:
    arr = np.asarray(values)
    _validate_ndim(arr, label=label, ndim=ndim, min_ndim=min_ndim)
    if arr.dtype != np.uint8:
        raise ValueError(f"{label} must use uint8 storage, got {arr.dtype}")
    return cast(UInt8Array, arr)


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
    head = text.split("/", 1)[0]
    return text if head in ARRAY_GROUPS else f"data/{text}"
