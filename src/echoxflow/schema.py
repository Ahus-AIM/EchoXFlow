"""EchoXFlow store array locations and lookup-table roles.

Recording stores arrange their arrays in groups according to how they were produced:
`data/`, `derived/`, `annotations/`, `coordinates/` and `timestamps/`. Colour Doppler velocity
and power are `derived/` products of a `data/2d_color_doppler_raw` estimator array, and Tissue
Doppler is stored as its native two-channel estimator output (`data/2d_tissue_doppler_raw`)
beside a `derived/` display velocity. Colour tables live in content-addressed lookup tables under
`data/lookup_tables/<digest>`; the store records only `role: lookup_table` for these, so a reader
has to recognise them by shape.

This module holds that knowledge in one place: the short names the pipelines address derived
products by, and the shape-based lookup-table classification used to render Doppler with the
scanner's own colour tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

#: Every top-level group an array can live in.
ARRAY_GROUPS: tuple[str, ...] = ("data", "derived", "annotations", "coordinates", "timestamps")

#: Groups that hold array values rather than timestamps or index coordinates.
VALUE_GROUPS: tuple[str, ...] = ("data", "derived", "annotations")

LOOKUP_TABLE_GROUP = "data/lookup_tables"

#: Per-frame sub-volume index of a stitched 3-D acquisition. Absent for single-beat scans.
FRAME_LOCATIONS_PATH = "coordinates/3d_brightness_mode_frame_locations"

#: R-peak times a 3-D acquisition was gated on, on the volume timestamp clock.
VOLUME_QRS_TIMES_PATH = "annotations/3d_source_qrs_times"

COLOR_FLOW_STORAGE_ENCODING = "ge_color_flow_rx_ry_power_aux_u8x4"
#: Native TDI whose two channels are signed frequency and bandwidth, source `FreqBw(8,8)`.
TDI_FREQ_BW_STORAGE_ENCODING = "ge_tdi_freq_bw_u8x2"
#: Native TDI whose two channels are the bytes of one packed `RxRyPower(5,5,6)` code.
TDI_PACKED_STORAGE_ENCODING = "ge_tdi_rx_ry_power_5_5_6_u8x2"

#: Source sample type of each two-channel TDI storage encoding.
TDI_SAMPLE_TYPES: Mapping[str, str] = {
    TDI_FREQ_BW_STORAGE_ENCODING: "FreqBw(8,8)",
    TDI_PACKED_STORAGE_ENCODING: "RxRyPower(5,5,6)",
}

#: Channel order of `data/2d_color_doppler_raw`, from the exporter's channel contract.
COLOR_FLOW_CHANNELS: tuple[str, ...] = ("rx", "ry", "power", "auxiliary")

#: Channel order of `data/2d_tissue_doppler_raw`.
TDI_CHANNELS: tuple[str, ...] = ("frequency_code", "bandwidth_confidence_code")

#: Short array name -> the stored arrays that satisfy it, most faithful first.
ARRAY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "data/2d_color_doppler_velocity": ("derived/2d_color_doppler_velocity",),
    "data/2d_color_doppler_power": ("derived/2d_color_doppler_power",),
    # `data/tissue_doppler` is the display velocity; the raw estimator array is the fallback
    # for the few recordings that ship no derived velocity.
    "data/tissue_doppler": (
        "derived/2d_tissue_doppler_display_velocity",
        "data/2d_tissue_doppler_raw",
    ),
    "timestamps/tissue_doppler": ("timestamps/2d_tissue_doppler",),
}

#: The short name each stored array is reached by. The first alias candidate wins, so
#: `derived/2d_tissue_doppler_display_velocity` maps back to `data/tissue_doppler` and the
#: raw fallback does not claim that name.
_SHORT_PATH_BY_STORED: Mapping[str, str] = {
    candidates[0]: short for short, candidates in ARRAY_ALIASES.items() if candidates
}

_PLANE_NAME_RE = re.compile(r"^(?P<group>data|timestamps)/2d_brightness_mode_(?P<plane>plane_)?(?P<index>\d+)$")


@dataclass(frozen=True)
class ArrayInfo:
    """What a store declares about one of its arrays in its root ``arrays`` index.

    Stores list every array with its role, representation, units, dimensions, coordinate system and,
    for derived products, the inputs it was derived from. Absent fields are ``None``.
    """

    path: str
    role: str | None = None
    representation: str | None = None
    units: str | None = None
    dimensions: tuple[str, ...] = ()
    coordinate_system: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    derivation_name: str | None = None
    derivation_inputs: tuple[str, ...] = ()

    @property
    def is_frame_array(self) -> bool:
        """Whether the array is a time series of images rather than a table, coordinate or calibration vector."""
        if is_lookup_table_path(self.path) or self.role in NON_FRAME_ARRAY_ROLES:
            return False
        if self.dimensions:
            return self.dimensions[0] == "time" and len(self.dimensions) >= 3
        return array_group(self.path) in VALUE_GROUPS and len(self.shape) >= 3


#: Array roles that interpret other arrays rather than hold frames of their own.
NON_FRAME_ARRAY_ROLES: frozenset[str] = frozenset(
    {"lookup_table", "calibration", "segment_frame_offsets", "segment_sample_offsets", "timestamps", "coordinates"}
)


def array_infos_from_attrs(attrs: Mapping[str, Any] | None) -> dict[str, ArrayInfo]:
    """Index a store's root ``arrays`` list by normalized array path."""
    entries = (attrs or {}).get("arrays")
    if not isinstance(entries, list):
        return {}
    infos: dict[str, ArrayInfo] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("array_path") or "").strip("/")
        if not name:
            continue
        path = normalize_array_path(name)
        derivation = entry.get("derivation")
        inputs = derivation.get("inputs") if isinstance(derivation, Mapping) else None
        infos[path] = ArrayInfo(
            path=path,
            role=_optional_text(entry.get("role")),
            representation=_optional_text(entry.get("representation")),
            units=_optional_text(entry.get("units")),
            dimensions=tuple(str(value) for value in entry.get("dimensions") or ()),
            coordinate_system=_optional_text(entry.get("coordinate_system")),
            dtype=_optional_text(entry.get("dtype")),
            shape=tuple(int(value) for value in entry.get("shape") or ()),
            derivation_name=_optional_text(derivation.get("name")) if isinstance(derivation, Mapping) else None,
            derivation_inputs=tuple(
                normalize_array_path(str(value)) for value in inputs or () if str(value).strip("/")
            ),
        )
    return infos


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def normalize_array_path(path: str) -> str:
    """Return a store-relative array path, defaulting a bare name into `data/`."""
    text = str(path).strip("/")
    if not text:
        raise ValueError("Array path must not be empty")
    head = text.split("/", 1)[0]
    return text if head in ARRAY_GROUPS else f"data/{text}"


def array_group(path: str) -> str:
    """Return the top-level group of an array path, for example `derived`."""
    return normalize_array_path(path).split("/", 1)[0]


def semantic_name(path: str) -> str:
    """Return an array path with its group prefix removed.

    `derived/2d_color_doppler_velocity` and its short name `data/2d_color_doppler_velocity`
    both reduce to `2d_color_doppler_velocity`, so name-driven dispatch works on either.
    """
    normalized = normalize_array_path(path)
    group, _, remainder = normalized.partition("/")
    return remainder if group in ARRAY_GROUPS else normalized


def current_array_candidates(path: str) -> tuple[str, ...]:
    """Return the stored paths that can satisfy a requested array path, the requested path first."""
    normalized = normalize_array_path(path)
    plane = _PLANE_NAME_RE.fullmatch(normalized)
    if plane is not None:
        prefix = "" if plane["plane"] else "plane_"
        return normalized, f"{plane['group']}/2d_brightness_mode_{prefix}{plane['index']}"
    return (normalized, *ARRAY_ALIASES.get(normalized, ()))


#: Array name -> the stable stream name the dashboard and models address it by.
#: Only display products are renamed; raw estimator arrays keep their own names so the two
#: never collapse into one stream.
CANONICAL_STREAM_NAMES: Mapping[str, str] = {
    "2d_tissue_doppler_display_velocity": "tissue_doppler",
}


def canonical_stream_name(path: str) -> str:
    """Return the stable stream name for an array path.

    `derived/2d_tissue_doppler_display_velocity` is stored under that path, but callers select the
    stream by the name `tissue_doppler`.
    """
    name = semantic_name(path)
    return CANONICAL_STREAM_NAMES.get(name, name)


def display_array_path(path: str) -> str:
    """Return the name an array is displayed and dispatched under.

    The plotting pipeline addresses streams by their short paths, for example
    `data/2d_color_doppler_velocity` rather than `derived/2d_color_doppler_velocity`, and
    `data/2d_brightness_mode_0` rather than `data/2d_brightness_mode_plane_0`.
    """
    normalized = normalize_array_path(path)
    short = _SHORT_PATH_BY_STORED.get(normalized)
    if short is not None:
        return short
    plane = _PLANE_NAME_RE.match(normalized)
    if plane is not None:
        return f"{plane['group']}/2d_brightness_mode_{plane['index']}"
    return normalized


def resolve_array_path(path: str, available: Iterable[str]) -> str | None:
    """Return the first candidate for `path` that exists in `available`, if any."""
    present = available if isinstance(available, (set, frozenset)) else set(available)
    return next((candidate for candidate in current_array_candidates(path) if candidate in present), None)


def is_lookup_table_path(path: str) -> bool:
    """Return whether an array path addresses a content-addressed lookup table."""
    return normalize_array_path(path).startswith(f"{LOOKUP_TABLE_GROUP}/")


ECHOFINE_PREFIX_GROUPS = (
    ("a2ch-", "A2C"),
    ("a3ch-", "A3C"),
    ("a4ch-", "A4C"),
    ("plax-", "PLAX"),
    ("mmode-plax-", "PLAX"),
    ("psax-", "PSAX"),
)
#: EchoPrime model output names that the catalog renames to the common view names.
ECHOPRIME_LABELS = {"Parasternal_Long": "PLAX", "Parasternal_Short": "PSAX"}
ECHOPRIME_GROUPS = {"A2C": "A2C", "A3C": "A3C", "A4C": "A4C", "PLAX": "PLAX", "PSAX": "PSAX"}


def consensus_view(echofine_label: str | None, echoprime_label: str | None) -> str | None:
    """Coarse class when EchoFine and EchoPrime agree (any agreed non-named class is Other), else None."""
    if not echofine_label or not echoprime_label:
        return None
    fine = next((group for prefix, group in ECHOFINE_PREFIX_GROUPS if echofine_label.startswith(prefix)), "Other")
    prime = ECHOPRIME_GROUPS.get(echoprime_label, "Other")
    return fine if fine == prime else None


#: Annotation names that together mean "has a strain contour"; the generic `strain_contour` also matches any strain.
STRAIN_CONTOUR_ANNOTATIONS: tuple[str, ...] = ("strain_contour",)

_STRAIN_CONTOUR_NAMES: Mapping[str, str] = {
    "2d_left_ventricular_strain": "lv_strain_contour",
    "2d_right_ventricular_strain": "rv_strain_contour",
    "2d_left_atrial_strain": "la_strain_contour",
}
_MESH_NAMES: Mapping[str, str] = {
    "3d_left_ventricle_mesh": "lv_mesh",
    "3d_right_ventricle_mesh": "rv_mesh",
    "3d_left_atrium_mesh": "la_mesh",
}
_MEASUREMENT_POINTS_RE = re.compile(r"(^|/)measurement_\d+_points$")
_DOPPLER_ANNOTATION_NAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^1d_pulsed_wave_doppler_annotation_\d+$"), "pw_doppler_annotation"),
    (re.compile(r"^1d_continuous_wave_doppler_annotation_\d+$"), "cw_doppler_annotation"),
    (re.compile(r"^1d_motion_mode_annotation_\d+$"), "mmode_annotation"),
)


def annotation_names(array_paths: Iterable[str], content_types: Iterable[str] = ()) -> tuple[str, ...]:
    """Name the kinds of annotation a recording carries, from its `annotations/` array paths.

    Returns a sorted tuple of: `lv/rv/la_strain_contour` (and the generic `strain_contour`), `lv/rv/la_mesh`,
    `measurement_points`, `pw_doppler_annotation`, `cw_doppler_annotation`, `mmode_annotation`, `ecg_qrs` and
    `la_trace`.
    """
    types = set(content_types)
    found: set[str] = set()
    for path in array_paths:
        if array_group(path) != "annotations":
            continue
        name = semantic_name(path)
        tail = name.rsplit("/", 1)[-1]
        if name.endswith("_contour") and any("strain" in value for value in types):
            found.add("strain_contour")
            found.update(label for content_type, label in _STRAIN_CONTOUR_NAMES.items() if content_type in types)
        found.update(label for key, label in _MESH_NAMES.items() if key in name)
        if _MEASUREMENT_POINTS_RE.search(name):
            found.add("measurement_points")
        found.update(label for pattern, label in _DOPPLER_ANNOTATION_NAMES if pattern.match(tail))
        if name in ("bmode_ecg_qrs", "ecg_qrs"):
            found.add("ecg_qrs")
        if name.startswith("la_trace/"):
            found.add("la_trace")
    return tuple(sorted(found))


def lookup_table_role(shape: Sequence[int]) -> str:
    """Classify a lookup table by its stored shape.

    Lookup tables are addressed by content digest and records only the generic
    `lookup_table` role, so shape is the only signal a store carries. The exporter
    reshapes exactly three roles and leaves everything else flat, which makes the
    reshaped ones unambiguous:

    * `(256, 256, 3)` is the GE FlowColorTable, indexed `[power_code, velocity_code]`.
    * `(65536, 4)` is the FencTable used to decode native Tissue Doppler codes.
    * `(256, 3)` is a plain RGB ColorTable. A recording can carry more than one.

    Flat tables keep the `unknown_lookup_table` role because their source names are not
    recoverable from the published store.
    """
    dimensions = tuple(int(extent) for extent in shape)
    if dimensions == (256, 256, 3):
        return "flow_color_rgb_table"
    if dimensions == (65536, 4):
        return "fenc_table"
    if dimensions == (256, 3):
        return "color_rgb_table"
    return "unknown_lookup_table"


def color_flow_velocity_codes(
    velocities_mps: np.ndarray,
    *,
    velocity_limit_mps: float,
) -> NDArray[np.uint8]:
    """Quantise wrapped colour-flow velocities to GE FlowColorTable velocity codes.

    The table's velocity axis is a phase axis, not a linear ramp from `-limit` to `+limit`.
    Code 0 is zero flow and reads black; codes rise through red and orange to bright yellow
    at `+limit` (code 127); code 128 is `-limit` and reads bright cyan, fading back through
    blue to black at code 255. Aliasing is therefore the hard 127/128 boundary, which is
    exactly the yellow-to-cyan wrap a scanner shows.

    Samples with no flow estimate are NaN and take code 0, the same no-flow black.
    """
    limit = float(velocity_limit_mps)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Colour flow velocity limit must be positive and finite, got {velocity_limit_mps!r}")
    values = np.asarray(velocities_mps, dtype=np.float32)
    fraction = np.clip(values / limit, -1.0, 1.0)
    codes = np.where(fraction >= 0.0, np.rint(fraction * 127.0), 256.0 + np.rint(fraction * 128.0))
    codes = np.where(np.isfinite(values), codes, 0.0)
    return np.asarray(np.mod(codes, 256.0), dtype=np.uint8)


def render_flow_color_table(
    *,
    velocity_codes: np.ndarray,
    secondary_codes: np.ndarray,
    flow_color_table: np.ndarray,
) -> NDArray[np.uint8]:
    """Map colour-flow codes to RGB through a GE FlowColorTable, indexed `[secondary, velocity]`.

    The second axis is always velocity. What the first axis means depends on how the
    stream was acquired: for a `flow_variance: 0` stream it scales brightness with power,
    and for a `flow_variance: 1` stream it blends toward green with turbulence. Use
    `flow_color_first_axis_is_brightness` to tell the two apart.
    """
    table = _flow_color_table(flow_color_table)
    velocity = np.asarray(velocity_codes, dtype=np.uint8)
    secondary = np.asarray(secondary_codes, dtype=np.uint8)
    if velocity.shape != secondary.shape:
        raise ValueError(
            f"Velocity codes {velocity.shape} and secondary codes {secondary.shape} must have the same shape"
        )
    return np.asarray(table[secondary, velocity], dtype=np.uint8)


def flow_color_first_axis_is_brightness(flow_color_table: np.ndarray, *, tolerance: float = 8.0) -> bool:
    """Return whether a FlowColorTable's first axis only scales brightness.

    A power-style table holds one hue ramp scaled by the first index, so every row is a
    scalar multiple of the brightest one. A variance-style table shifts hue toward green
    instead and fails that test.
    """
    table = _flow_color_table(flow_color_table).astype(np.float32)
    reference = table[255]
    peak = float(reference.max())
    if peak <= 0.0:
        return False
    scale = table.reshape(256, -1).max(axis=1) / peak
    predicted = scale[:, None, None] * reference[None, :, :]
    return float(np.abs(predicted - table).max()) <= float(tolerance)


def flow_color_velocity_ramp(flow_color_table: np.ndarray, *, phase_ordered: bool = False) -> NDArray[np.uint8]:
    """Return a FlowColorTable's 256-entry RGB velocity ramp.

    Colour bars and renderers that carry only velocity need one row of the table rather
    than the whole two-dimensional map. The reference row is the one where the first axis
    contributes nothing of its own: full power for a brightness-style table, zero
    turbulence for a variance-style one.

    Set `phase_ordered` for colour flow, whose velocity axis is the estimator phase: black
    zero flow at code 0, bright yellow at `+limit` (code 127), bright cyan at `-limit`
    (code 128), fading back to black at code 255. Renderers index a ramp linearly over
    `[-limit, +limit]`, so the entries are reordered onto that axis and the yellow-to-cyan
    aliasing wrap lands at the two ends where those velocities actually are. Tissue Doppler
    uses the same ramp unreversed, so velocity toward the probe (positive) is warm and velocity
    away from it is cool, as in colour flow; leave `phase_ordered` false for the raw row.
    """
    table = _flow_color_table(flow_color_table)
    row = np.asarray(table[255 if flow_color_first_axis_is_brightness(table) else 0], dtype=np.uint8)
    if not phase_ordered:
        return row
    return np.asarray(row[_linear_velocity_to_phase_code()], dtype=np.uint8)


def _linear_velocity_to_phase_code() -> NDArray[np.intp]:
    """Map each linear `[-limit, +limit]` ramp index onto its GE phase code."""
    fraction = np.linspace(-1.0, 1.0, 256, dtype=np.float64)
    codes = np.where(fraction >= 0.0, np.rint(fraction * 127.0), 256.0 + np.rint(fraction * 128.0))
    return np.mod(codes, 256.0).astype(np.intp)


def _flow_color_table(flow_color_table: np.ndarray) -> NDArray[np.uint8]:
    table = np.asarray(flow_color_table, dtype=np.uint8)
    if table.shape != (256, 256, 3):
        raise ValueError(f"FlowColorTable must have shape (256, 256, 3), got {table.shape}")
    return table


def color_flow_raw_to_velocity(
    raw_channels: np.ndarray,
    *,
    velocity_limit_mps: float | np.ndarray,
) -> NDArray[np.float32]:
    """Decode `data/2d_color_doppler_raw` to wrapped radial velocities in m/s.

    This reproduces the exporter's own derivation bit for bit, so it agrees with the
    shipped `derived/2d_color_doppler_velocity` array. Samples whose real and imaginary
    lag-one components are both zero carry no flow estimate and become NaN.

    `velocity_limit_mps` is either one scale for every frame or one scale per frame, as
    published in `derived/*_velocity_scale_per_frame`. Frames without a calibrated scale
    (NaN) decode to NaN rather than to a borrowed scale.
    """
    channels = _color_flow_channels(raw_channels)
    multiplier = doppler_scale_multiplier(velocity_limit_mps, frame_count=channels.shape[0], divisor=np.pi)
    real = channels[..., 0].view(np.int8).astype(np.float32)
    imaginary = channels[..., 1].view(np.int8).astype(np.float32)
    velocity = np.arctan2(imaginary, real).astype(np.float32) * multiplier
    velocity[(real == 0.0) & (imaginary == 0.0)] = np.nan
    return np.asarray(velocity, dtype=np.float32)


def doppler_scale_multiplier(
    velocity_scale_mps: float | np.ndarray,
    *,
    frame_count: int,
    divisor: float,
) -> NDArray[np.float32]:
    """Return `scale / divisor`, broadcastable against `[time, radial, beam]` frames.

    A scalar scale applies to every frame. A per-frame scale must have one entry per frame;
    NaN entries mark uncalibrated frames and propagate as NaN. Infinite or non-positive
    calibrated entries are rejected rather than silently repaired.
    """
    values = np.asarray(velocity_scale_mps, dtype=np.float64)
    if values.ndim > 1 or (values.ndim == 1 and values.size != int(frame_count)):
        raise ValueError(
            f"Doppler velocity scale must be scalar or one entry per frame ({frame_count}), got shape {values.shape}"
        )
    if values.ndim == 0 and not np.isfinite(values):
        raise ValueError(f"Doppler velocity scale must be positive and finite, got {velocity_scale_mps!r}")
    if np.any(np.isinf(values)) or np.any(np.isfinite(values) & (values <= 0.0)):
        raise ValueError("Doppler velocity scale must be positive where calibrated")
    multiplier = (values / float(divisor)).astype(np.float32)
    return multiplier.reshape(-1, 1, 1) if values.ndim else multiplier


def color_flow_raw_to_power(raw_channels: np.ndarray) -> NDArray[np.float32]:
    """Decode `data/2d_color_doppler_raw` to the power fraction in `[0, 1]`."""
    channels = _color_flow_channels(raw_channels)
    return np.asarray(channels[..., 2].astype(np.float32) / np.float32(255.0), dtype=np.float32)


def tdi_raw_channels_to_native_codes(raw_channels: np.ndarray) -> NDArray[np.uint16]:
    """Pack `data/2d_tissue_doppler_raw` channel pairs into native GE uint16 codes.

    Only the packed `RxRyPower(5,5,6)` encoding stores one code split over two bytes;
    `FreqBw(8,8)` channels are two separate quantities and must not be packed.
    """
    channels = _tdi_channels(raw_channels)
    low = channels[..., 0].astype(np.uint16)
    high = channels[..., 1].astype(np.uint16)
    return np.asarray(low | (high << np.uint16(8)), dtype=np.uint16)


def tdi_storage_encoding(sector: Mapping[str, object] | None) -> str | None:
    """Return the two-channel TDI encoding a manifest sector declares, if it declares one.

    The sector's `storage_encoding` is authoritative. Older sectors declared
    only the source `geometry.sample_type`, which identifies the same two layouts.
    """
    if not sector:
        return None
    encoding = sector.get("storage_encoding")
    if isinstance(encoding, str) and encoding in TDI_SAMPLE_TYPES:
        return encoding
    geometry = sector.get("geometry")
    sample_type = geometry.get("sample_type") if isinstance(geometry, Mapping) else None
    for candidate, declared in TDI_SAMPLE_TYPES.items():
        if sample_type == declared:
            return candidate
    return None


def tdi_raw_signed_frequency(
    raw_channels: np.ndarray,
    *,
    storage_encoding: str,
    fenc_table: np.ndarray | None = None,
) -> NDArray[np.int8]:
    """Return the signed frequency code of each native TDI sample.

    `FreqBw(8,8)` stores the frequency byte directly in channel 0. Packed
    `RxRyPower(5,5,6)` stores a 16-bit code that the sector's own FencTable maps to the
    frequency byte in its first column.
    """
    channels = _tdi_channels(raw_channels)
    if storage_encoding == TDI_FREQ_BW_STORAGE_ENCODING:
        return np.asarray(channels[..., 0].view(np.int8))
    if storage_encoding == TDI_PACKED_STORAGE_ENCODING:
        if fenc_table is None:
            raise ValueError("Packed RxRyPower(5,5,6) tissue Doppler needs its sector's FencTable")
        table = np.asarray(fenc_table, dtype=np.uint8).reshape(-1, 4)
        if table.shape[0] != 256 * 256:
            raise ValueError(f"TDI FencTable must have 65536 rows, got {table.shape[0]}")
        return np.asarray(table[tdi_raw_channels_to_native_codes(channels), 0].view(np.int8))
    raise ValueError(f"Unsupported two-channel tissue Doppler storage encoding {storage_encoding!r}")


def tdi_raw_to_velocity(
    raw_channels: np.ndarray,
    *,
    storage_encoding: str,
    velocity_scale_mps: float | np.ndarray,
    fenc_table: np.ndarray | None = None,
) -> NDArray[np.float32]:
    """Decode `data/2d_tissue_doppler_raw` to radial velocities in m/s.

    The store declares `velocity = signed_int8(frequency) / 128 * scale`, with one scale
    per frame (`velocity_scale_per_frame_or_scalar`). This matches the shipped
    `derived/2d_tissue_doppler_display_velocity` bit for bit. Uncalibrated frames are NaN.
    """
    frequency = tdi_raw_signed_frequency(raw_channels, storage_encoding=storage_encoding, fenc_table=fenc_table)
    multiplier = doppler_scale_multiplier(velocity_scale_mps, frame_count=frequency.shape[0], divisor=128.0)
    return np.asarray(frequency.astype(np.float32) * multiplier, dtype=np.float32)


def _tdi_channels(raw_channels: np.ndarray) -> NDArray[np.uint8]:
    channels = np.asarray(raw_channels)
    if channels.dtype != np.uint8 or channels.ndim != 4 or channels.shape[-1] != 2:
        raise ValueError(
            f"TDI raw channels must be uint8 with shape [time, radial, beam, 2], got {channels.dtype} {channels.shape}"
        )
    return channels


def _color_flow_channels(raw_channels: np.ndarray) -> NDArray[np.uint8]:
    channels = np.asarray(raw_channels, dtype=np.uint8)
    if channels.ndim != 4 or channels.shape[-1] != 4:
        raise ValueError(f"Colour flow raw channels must have shape [time, radial, beam, 4], got {channels.shape}")
    return channels
