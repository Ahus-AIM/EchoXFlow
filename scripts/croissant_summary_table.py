#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_OUTPUT = Path("outputs/tables/croissant_summary_table.tex")


@dataclass(frozen=True)
class ModalityRow:
    key: str
    label: str
    content_types: tuple[str, ...]
    timing_content_types: tuple[str, ...]
    path_markers: tuple[str, ...] = ()
    path_exclusions: tuple[str, ...] = ()
    frame_paths: tuple[str, ...] = ()
    timestamp_paths: tuple[str, ...] = ()
    byte_paths: tuple[str, ...] = ()
    include_bytes: bool = True
    min_frame_count: int = 2
    requires_sampling_gate_metadata: bool = False


TABLE_ROWS = (
    ModalityRow(
        key="electrocardiogram",
        label=r"\textbf{Electrocardiogram}",
        content_types=(),
        timing_content_types=(),
        path_markers=("data/ecg",),
        frame_paths=("data/ecg",),
        timestamp_paths=("timestamps/ecg",),
        byte_paths=("data/ecg", "timestamps/ecg"),
    ),
    ModalityRow(
        key="1d_motion_mode",
        label=r"\quad 1D (M-mode)",
        content_types=("1d_motion_mode",),
        timing_content_types=("1d_motion_mode",),
    ),
    ModalityRow(
        key="2d_brightness_mode",
        label=r"\quad 2D (single plane)",
        content_types=("2d_brightness_mode",),
        timing_content_types=("2d_brightness_mode",),
    ),
    ModalityRow(
        key="2d_biplane_brightness_mode",
        label=r"\quad 2D (dual plane)",
        content_types=("2d_biplane_brightness_mode",),
        timing_content_types=("2d_biplane_brightness_mode",),
    ),
    ModalityRow(
        key="2d_triplane_brightness_mode",
        label=r"\quad 2D (triplane)",
        content_types=("2d_triplane_brightness_mode",),
        timing_content_types=("2d_triplane_brightness_mode",),
    ),
    ModalityRow(
        key="3d_brightness_mode",
        label=r"\quad 3D\textsuperscript{1}",
        content_types=("3d_brightness_mode",),
        timing_content_types=("3d_brightness_mode",),
    ),
    ModalityRow(
        key="2d_tissue_doppler",
        label=r"\quad 2D Tissue",
        content_types=("2d_tissue_doppler",),
        timing_content_types=("2d_tissue_doppler", "2d_color_doppler"),
        path_markers=("tissue_doppler", "tdi"),
    ),
    ModalityRow(
        key="2d_color_doppler",
        label=r"\quad 2D Color",
        content_types=("2d_color_doppler",),
        timing_content_types=("2d_color_doppler",),
        path_markers=("2d_color_doppler", "color_doppler", "color_flow", "flow_color"),
        path_exclusions=("tissue_doppler",),
    ),
    ModalityRow(
        key="1d_pulsed_wave_doppler",
        label=r"\quad 1D Pulsed Wave",
        content_types=("1d_pulsed_wave_doppler",),
        timing_content_types=("1d_pulsed_wave_doppler",),
    ),
    ModalityRow(
        key="1d_continuous_wave_doppler",
        label=r"\quad 1D Continuous Wave",
        content_types=("1d_continuous_wave_doppler",),
        timing_content_types=("1d_continuous_wave_doppler",),
    ),
    ModalityRow(
        key="3d_left_ventricle_mesh",
        label=r"\qquad Left ventricle",
        content_types=(),
        timing_content_types=("3d_brightness_mode",),
        path_markers=("data/3d_left_ventricle_mesh",),
        frame_paths=("timestamps/3d_left_ventricle_mesh", "data/3d_left_ventricle_mesh"),
        timestamp_paths=("timestamps/3d_left_ventricle_mesh",),
        include_bytes=False,
    ),
    ModalityRow(
        key="lv_a2c_strain",
        label=r"\qquad Left ventricle, A2C",
        content_types=("2d_left_ventricular_strain",),
        timing_content_types=("2d_left_ventricular_strain",),
        path_markers=("data/2ch_contour", "data/2ch_curve"),
        frame_paths=("data/2ch_contour", "data/2ch_curve"),
        timestamp_paths=("timestamps/2ch_contour", "timestamps/2ch_curve"),
        include_bytes=False,
    ),
    ModalityRow(
        key="lv_alax_strain",
        label=r"\qquad Left ventricle, ALAX",
        content_types=("2d_left_ventricular_strain",),
        timing_content_types=("2d_left_ventricular_strain",),
        path_markers=("data/3ch_contour", "data/3ch_curve"),
        frame_paths=("data/3ch_contour", "data/3ch_curve"),
        timestamp_paths=("timestamps/3ch_contour", "timestamps/3ch_curve"),
        include_bytes=False,
    ),
    ModalityRow(
        key="lv_a4c_strain",
        label=r"\qquad Left ventricle, A4C",
        content_types=("2d_left_ventricular_strain",),
        timing_content_types=("2d_left_ventricular_strain",),
        path_markers=("data/4ch_contour", "data/4ch_curve"),
        frame_paths=("data/4ch_contour", "data/4ch_curve"),
        timestamp_paths=("timestamps/4ch_contour", "timestamps/4ch_curve"),
        include_bytes=False,
    ),
    ModalityRow(
        key="la_a2c_strain",
        label=r"\qquad Left atrium, A2C",
        content_types=("2d_left_atrial_strain",),
        timing_content_types=("2d_left_atrial_strain",),
        path_markers=("data/2ch_contour", "data/2ch_curve"),
        frame_paths=("data/2ch_contour", "data/2ch_curve"),
        timestamp_paths=("timestamps/2ch_contour", "timestamps/2ch_curve"),
        include_bytes=False,
    ),
    ModalityRow(
        key="la_a4c_strain",
        label=r"\qquad Left atrium, A4C",
        content_types=("2d_left_atrial_strain",),
        timing_content_types=("2d_left_atrial_strain",),
        path_markers=("data/4ch_contour", "data/4ch_curve"),
        frame_paths=("data/4ch_contour", "data/4ch_curve"),
        timestamp_paths=("timestamps/4ch_contour", "timestamps/4ch_curve"),
        include_bytes=False,
    ),
    ModalityRow(
        key="rv_strain",
        label=r"\qquad Right ventricle, A4C",
        content_types=("2d_right_ventricular_strain",),
        timing_content_types=("2d_right_ventricular_strain",),
        path_markers=("data/rv_contour", "data/rv_curve", "data/4ch_contour", "data/4ch_curve"),
        frame_paths=("data/rv_contour", "data/rv_curve", "data/4ch_contour", "data/4ch_curve"),
        timestamp_paths=(
            "timestamps/rv_contour",
            "timestamps/rv_curve",
            "timestamps/4ch_contour",
            "timestamps/4ch_curve",
        ),
        include_bytes=False,
    ),
    ModalityRow(
        key="sample_volume_pulsed_wave",
        label=r"\qquad 1D Pulsed Wave",
        content_types=("1d_pulsed_wave_doppler",),
        timing_content_types=(),
        include_bytes=False,
        min_frame_count=1,
        requires_sampling_gate_metadata=True,
    ),
    ModalityRow(
        key="sample_volume_continuous_wave",
        label=r"\qquad 1D Continuous Wave",
        content_types=("1d_continuous_wave_doppler",),
        timing_content_types=(),
        include_bytes=False,
        min_frame_count=1,
        requires_sampling_gate_metadata=True,
    ),
    ModalityRow(
        key="sparse_points_and_events",
        label=r"\qquad Points and markers\textsuperscript{3}",
        content_types=(),
        timing_content_types=(),
        path_markers=(
            "data/1d_pulsed_wave_doppler_annotation",
            "data/1d_continuous_wave_doppler_annotation",
            "data/1d_motion_mode_annotation",
            "overlay_physical_points",
            "data/2ch_ecg_qrs",
            "data/3ch_ecg_qrs",
            "data/4ch_ecg_qrs",
            "data/rv_ecg_qrs",
        ),
        include_bytes=False,
        min_frame_count=1,
    ),
)


SECTION_BREAKS = {
    "1d_motion_mode": (r"\textbf{Brightness mode}",),
    "2d_tissue_doppler": (r"\textbf{Doppler}",),
    "3d_left_ventricle_mesh": (r"\textbf{Dense annotations}", r"\quad 3D segmentation"),
    "lv_a2c_strain": (r"\quad 2D strain/segmentation",),
    "sample_volume_pulsed_wave": (r"\textbf{Sparse annotations}", r"\quad Sample volumes\textsuperscript{2}"),
    "sparse_points_and_events": (r"\quad Other annotations",),
}

ANNOTATION_ARRAY_MARKERS = (
    "annotation",
    "3d_left_ventricle_mesh",
    "2ch_contour",
    "2ch_curve",
    "3ch_contour",
    "3ch_curve",
    "4ch_contour",
    "4ch_curve",
    "rv_contour",
    "rv_curve",
)

PRIME_LABEL_PATTERNS = (
    (re.compile(r"\bSprime\b", re.IGNORECASE), "S'"),
    (re.compile(r"\bEprime\b", re.IGNORECASE), "E'"),
    (re.compile(r"\bAprime\b", re.IGNORECASE), "A'"),
)

MARKER_STYLE_PATTERNS = (
    (re.compile("'"), "’"),
    (re.compile(r"\bSeptal\b"), "septal"),
    (re.compile(r"\bLateral\b"), "lateral"),
    (re.compile(r"\bTrace\b"), "trace"),
)


@dataclass
class RowStats:
    file_count: int = 0
    item_ids: set[str] | None = None
    subject_ids: set[str] | None = None
    bytes_mib: list[float] | None = None
    sampling_hz: list[float] | None = None
    duration_s: list[float] | None = None
    marker_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.item_ids = set() if self.item_ids is None else self.item_ids
        self.subject_ids = set() if self.subject_ids is None else self.subject_ids
        self.bytes_mib = [] if self.bytes_mib is None else self.bytes_mib
        self.sampling_hz = [] if self.sampling_hz is None else self.sampling_hz
        self.duration_s = [] if self.duration_s is None else self.duration_s
        self.marker_counts = {} if self.marker_counts is None else self.marker_counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a LaTeX per-modality statistics table from an EchoXFlow Croissant JSON file.",
    )
    parser.add_argument("croissant", type=Path, help="Path to croissant.json.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Write LaTeX to this file. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Per-modality statistics. Values are reported as medians and interquartile ranges. "
            "The dense annotations are defined over one heart cycle, but can be expanded to the full underlying "
            "recording using beat-stitching via the ECG under the (verifiable) assumption of regular heart rhythm."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument("--label", default="tab:modality-stats", help="LaTeX table label.")
    parser.add_argument(
        "--subject-field",
        default="exam_id",
        help="Recording field used for the unique count in the exam column. Defaults to exam_id.",
    )
    parser.add_argument(
        "--subject-label",
        default="Exams",
        help="Header label for the unique subject/exam column.",
    )
    args = parser.parse_args()

    table = build_table(
        croissant_path=args.croissant,
        caption=args.caption,
        label=args.label,
        subject_field=args.subject_field,
        subject_label=args.subject_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table + "\n", encoding="utf-8")
    print(args.output)
    return 0


def build_table(*, croissant_path: Path, caption: str, label: str, subject_field: str, subject_label: str) -> str:
    croissant_path = croissant_path.expanduser()
    metadata = load_json(croissant_path)
    recordings = normalized_rows(metadata, record_set_name="recordings")
    arrays = normalized_rows(metadata, record_set_name="arrays", required=False)
    arrays_by_recording = group_arrays_by_recording(arrays)
    stats = collect_stats(
        recordings=recordings,
        arrays_by_recording=arrays_by_recording,
        croissant_dir=croissant_path.parent,
        subject_field=subject_field,
    )
    return render_latex_table(stats=stats, caption=caption, label=label, subject_label=subject_label)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def normalized_rows(metadata: dict[str, Any], *, record_set_name: str, required: bool = True) -> list[dict[str, Any]]:
    for record_set in as_dicts(metadata.get("recordSet")):
        if (
            str(record_set.get("@id", "")).strip() != record_set_name
            and str(record_set.get("name", "")).strip() != record_set_name
        ):
            continue
        field_names = {
            str(field.get("@id", "")).strip(): str(field.get("name") or "").strip()
            for field in as_dicts(record_set.get("field"))
        }
        rows = []
        for row in as_dicts(record_set.get("data")):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                key_text = str(key)
                name = field_names.get(key_text) or key_text.removeprefix(f"{record_set_name}/")
                normalized[name] = value
            rows.append(normalized)
        return rows
    if required:
        raise ValueError(f"Croissant metadata is missing recordSet {record_set_name!r}")
    return []


def as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def render_documents(zarr_attrs: dict[str, Any]) -> list[dict[str, Any]]:
    recording_manifest = zarr_attrs.get("recording_manifest")
    if isinstance(recording_manifest, dict):
        documents = recording_manifest.get("documents")
        if isinstance(documents, list):
            return as_dicts(documents)
        return [recording_manifest]
    documents = []
    for render_input in as_dicts(zarr_attrs.get("render_inputs")):
        document = render_input.get("document")
        if isinstance(document, dict):
            documents.append(document)
    return documents


def group_arrays_by_recording(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        recording_id = clean_text(row.get("recording_id"))
        if recording_id:
            grouped.setdefault(recording_id, []).append(row)
    return grouped


def collect_stats(
    *,
    recordings: list[dict[str, Any]],
    arrays_by_recording: dict[str, list[dict[str, Any]]],
    croissant_dir: Path,
    subject_field: str,
) -> dict[str, RowStats]:
    stats = {row.key: RowStats() for row in TABLE_ROWS}
    zarr_attrs_cache: dict[Path, dict[str, Any]] = {}
    zarr_array_size_cache: dict[tuple[Path, str], int | None] = {}

    for recording in recordings:
        recording_id = clean_text(recording.get("recording_id"))
        array_rows = arrays_by_recording.get(recording_id, [])
        zarr_path = croissant_dir / clean_text(recording.get("zarr_path"))
        zarr_attrs = zarr_attrs_cache.setdefault(zarr_path, read_zarr_attrs(zarr_path))
        content_types = string_list(recording.get("content_types")) or string_list(zarr_attrs.get("content_types"))
        frame_counts = dict_value(recording.get("frame_counts_by_content_type")) or dict_value(
            zarr_attrs.get("frame_counts_by_content_type")
        )
        median_delta = (
            dict_value(recording.get("median_delta_time_by_content_type"))
            or dict_value(recording.get("median_delta_time"))
            or dict_value(zarr_attrs.get("median_delta_time"))
        )
        array_paths = set(string_list(recording.get("array_paths")))
        array_paths.update(clean_text(row.get("array_path")) for row in array_rows)
        array_paths.discard("")
        if not array_rows:
            array_rows = as_dicts(zarr_attrs.get("arrays"))
            array_paths.update(clean_text(row.get("name")) for row in array_rows)
            array_paths.discard("")

        for modality in TABLE_ROWS:
            if not recording_has_modality(
                modality=modality,
                content_types=content_types,
                array_paths=array_paths,
                zarr_attrs=zarr_attrs,
            ):
                continue
            frame_count = modality_frame_count(
                modality=modality,
                frame_counts=frame_counts,
                array_rows=array_rows,
                zarr_attrs=zarr_attrs,
            )
            if frame_count is None or frame_count < modality.min_frame_count:
                continue
            row_stats = stats[modality.key]
            item_id = modality_item_id(modality=modality, recording=recording, array_rows=array_rows)
            if item_id in row_stats.item_ids:
                continue
            row_stats.item_ids.add(item_id)
            row_stats.file_count += 1
            subject_id = clean_text(recording.get(subject_field))
            if subject_id:
                row_stats.subject_ids.add(subject_id)
            byte_array_paths = modality_byte_array_paths(modality=modality, array_rows=array_rows)
            if modality.key == "3d_brightness_mode":
                stitch_beat_count = explicit_stitch_beat_count(recording=recording, zarr_attrs=zarr_attrs)
                if stitch_beat_count is not None:
                    count_key = str(stitch_beat_count)
                    row_stats.marker_counts[count_key] = row_stats.marker_counts.get(count_key, 0) + 1
            if modality.key == "sparse_points_and_events":
                spectral_labels_by_path = spectral_annotation_labels_by_path(zarr_attrs)
                for array_path in byte_array_paths:
                    if not is_spectral_annotation_array_path(array_path):
                        continue
                    marker_name = spectral_labels_by_path.get(array_path)
                    if marker_name:
                        row_stats.marker_counts[marker_name] = row_stats.marker_counts.get(marker_name, 0) + 1
            if modality.include_bytes:
                byte_count = modality_array_size_bytes(
                    zarr_path=zarr_path,
                    array_paths=byte_array_paths,
                    size_cache=zarr_array_size_cache,
                )
                if byte_count is not None:
                    row_stats.bytes_mib.append(float(byte_count) / (1024.0 * 1024.0))

            delta_s = modality_delta_s(modality=modality, median_delta=median_delta, zarr_path=zarr_path)
            if delta_s is not None and delta_s > 0.0:
                row_stats.sampling_hz.append(1.0 / delta_s)
                row_stats.duration_s.append(float(frame_count - 1) * delta_s)

    return stats


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def modality_item_id(*, modality: ModalityRow, recording: dict[str, Any], array_rows: list[dict[str, Any]]) -> str:
    if modality.requires_sampling_gate_metadata:
        recording_id = clean_text(recording.get("recording_id"))
        zarr_path = clean_text(recording.get("zarr_path"))
        return f"sampling_gate:{recording_id or zarr_path}:{modality.key}"
    data_rows = modality_data_array_rows(modality=modality, array_rows=array_rows)
    data_hashes = [clean_text(row.get("data_sha256")) for row in data_rows]
    if data_hashes and all(data_hashes):
        return "sha256:" + "|".join(sorted(set(data_hashes)))
    recording_id = clean_text(recording.get("recording_id"))
    zarr_path = clean_text(recording.get("zarr_path"))
    return f"recording:{recording_id or zarr_path}"


def modality_data_array_rows(*, modality: ModalityRow, array_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    primary_markers = modality_primary_data_path_markers(modality)
    for row in array_rows:
        array_path = clean_text(row.get("array_path") or row.get("name"))
        if not array_path.startswith("data/"):
            continue
        if primary_markers and not path_matches_any_marker(array_path.lower(), primary_markers):
            continue
        if array_row_matches_modality_item(array_path=array_path, row=row, modality=modality):
            rows.append(row)
    return rows


def modality_primary_data_path_markers(modality: ModalityRow) -> tuple[str, ...]:
    if modality.key == "2d_tissue_doppler":
        return ("data/tissue_doppler", "data/tdi")
    if modality.key == "2d_color_doppler":
        return ("data/2d_color_doppler", "data/color_doppler", "data/color_flow", "data/flow_color")
    if modality.frame_paths:
        return tuple(path for path in modality.frame_paths if path.startswith("data/"))
    if is_sparse_annotation_row(modality):
        return modality.path_markers
    if modality.path_markers and not modality.content_types:
        return modality.path_markers
    return tuple(f"data/{content_type}" for content_type in modality.content_types)


def path_matches_any_marker(path: str, markers: tuple[str, ...]) -> bool:
    return any(
        path == marker or path.startswith(f"{marker.rstrip('/')}/") or path.startswith(marker) for marker in markers
    )


def array_row_matches_modality_item(*, array_path: str, row: dict[str, Any], modality: ModalityRow) -> bool:
    lowered_path = array_path.lower()
    if paths_contain({lowered_path}, modality.path_exclusions):
        return False
    row_content_types = set(string_list(row.get("content_types")))
    if modality.key == "2d_tissue_doppler":
        return bool(row_content_types & set(modality.content_types)) or paths_contain(
            {lowered_path}, modality.path_markers
        )
    if modality.key == "2d_color_doppler":
        if paths_contain({lowered_path}, modality.path_markers):
            return True
        return "2d_color_doppler" in row_content_types
    if modality.path_markers and modality.content_types:
        return bool(row_content_types & set(modality.content_types)) and paths_contain(
            {lowered_path}, modality.path_markers
        )
    if modality.path_markers:
        return paths_contain({lowered_path}, modality.path_markers)
    return bool(row_content_types & set(modality.content_types))


def recording_has_modality(
    *,
    modality: ModalityRow,
    content_types: list[str],
    array_paths: set[str],
    zarr_attrs: dict[str, Any],
) -> bool:
    content_type_set = set(content_types)
    if modality.requires_sampling_gate_metadata:
        return bool(content_type_set & set(modality.content_types)) and has_sampling_gate_metadata(zarr_attrs)
    if modality.key == "2d_tissue_doppler":
        has_content_type = bool(content_type_set & set(modality.content_types))
        return has_content_type or paths_contain(array_paths, modality.path_markers)
    if modality.key == "2d_color_doppler":
        if paths_contain(array_paths, modality.path_markers):
            return True
        return "2d_color_doppler" in content_type_set and not paths_contain(array_paths, modality.path_exclusions)
    if modality.path_markers and modality.content_types:
        has_content_type = bool(content_type_set & set(modality.content_types))
        return has_content_type and paths_contain(array_paths, modality.path_markers)
    if modality.path_markers:
        return paths_contain(array_paths, modality.path_markers)
    return bool(content_type_set & set(modality.content_types))


def paths_contain(paths: set[str], markers: tuple[str, ...]) -> bool:
    return any(marker in path.lower() for path in paths for marker in markers)


def has_sampling_gate_metadata(zarr_attrs: dict[str, Any]) -> bool:
    recording_manifest = zarr_attrs.get("recording_manifest")
    if not isinstance(recording_manifest, dict):
        return False
    gate = recording_manifest.get("sampling_gate_metadata")
    if not isinstance(gate, dict):
        return False
    return (
        finite_float(gate.get("gate_center_depth_m")) is not None
        and finite_float(gate.get("gate_tilt_rad")) is not None
        and finite_float(gate.get("gate_sample_volume_m")) is not None
    )


def modality_frame_count(
    *,
    modality: ModalityRow,
    frame_counts: dict[str, Any],
    array_rows: list[dict[str, Any]],
    zarr_attrs: dict[str, Any],
) -> int | None:
    if modality.requires_sampling_gate_metadata:
        return 1 if has_sampling_gate_metadata(zarr_attrs) else None
    if modality.frame_paths:
        path_count = modality_shape_frame_count(modality=modality, array_rows=array_rows)
        if path_count is not None and path_count > 0:
            return path_count
    for content_type in modality.timing_content_types:
        count = positive_int(frame_counts.get(content_type))
        if count is not None:
            return count
    shape_count = modality_shape_frame_count(modality=modality, array_rows=array_rows)
    return shape_count if shape_count is not None and shape_count > 0 else None


def modality_shape_frame_count(*, modality: ModalityRow, array_rows: list[dict[str, Any]]) -> int | None:
    counts: list[int] = []
    for row in array_rows:
        array_path = clean_text(row.get("array_path") or row.get("name")).lower()
        if not array_path or not array_path_matches_modality(array_path=array_path, modality=modality):
            continue
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape:
            continue
        count = positive_int(shape[0])
        if count is not None:
            counts.append(count)
    return max(counts) if counts else None


def modality_byte_array_paths(*, modality: ModalityRow, array_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if modality.byte_paths:
        return modality.byte_paths
    paths = []
    for row in array_rows:
        array_path = clean_text(row.get("array_path") or row.get("name"))
        if array_path and array_row_matches_modality_bytes(array_path=array_path, row=row, modality=modality):
            paths.append(array_path)
    return tuple(dict.fromkeys(paths))


def array_row_matches_modality_bytes(*, array_path: str, row: dict[str, Any], modality: ModalityRow) -> bool:
    lowered_path = array_path.lower()
    if paths_contain({lowered_path}, modality.path_exclusions):
        return False
    if is_sparse_annotation_row(modality):
        return paths_contain({lowered_path}, modality.path_markers)
    if is_dense_annotation_array_path(lowered_path):
        return False
    if modality.path_markers and not paths_contain({lowered_path}, modality.path_markers):
        return False
    if modality.key == "2d_tissue_doppler":
        return paths_contain({lowered_path}, modality.path_markers)
    row_content_types = set(string_list(row.get("content_types")))
    if modality.content_types:
        return bool(row_content_types & set(modality.content_types))
    return bool(modality.path_markers and paths_contain({lowered_path}, modality.path_markers))


def array_path_matches_modality(*, array_path: str, modality: ModalityRow) -> bool:
    if modality.frame_paths:
        return array_path in modality.frame_paths
    if is_sparse_annotation_row(modality):
        return paths_contain({array_path}, modality.path_markers)
    if is_dense_annotation_array_path(array_path):
        return False
    if modality.path_markers and paths_contain({array_path}, modality.path_markers):
        return True
    return any(content_type in array_path for content_type in modality.content_types)


def is_dense_annotation_array_path(array_path: str) -> bool:
    return paths_contain({array_path.lower()}, ANNOTATION_ARRAY_MARKERS)


def is_sparse_annotation_row(modality: ModalityRow) -> bool:
    return modality.min_frame_count <= 1 and bool(modality.path_markers)


def is_spectral_annotation_array_path(array_path: str) -> bool:
    lowered = array_path.lower()
    return (
        "data/1d_pulsed_wave_doppler_annotation" in lowered or "data/1d_continuous_wave_doppler_annotation" in lowered
    )


def explicit_stitch_beat_count(*, recording: dict[str, Any], zarr_attrs: dict[str, Any]) -> int | None:
    recording_count = positive_int(recording.get("stitch_beat_count"))
    if recording_count is not None:
        return recording_count
    counts = []
    for document in render_documents(zarr_attrs):
        count = positive_int(document.get("stitch_beat_count"))
        if count is not None:
            counts.append(count)
    return max(counts) if counts else None


def spectral_annotation_labels_by_path(zarr_attrs: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for document in render_documents(zarr_attrs):
        for annotation in as_dicts(document.get("annotations")):
            value = annotation.get("value")
            if not isinstance(value, dict):
                continue
            array_path = clean_text(value.get("zarr_path"))
            if not array_path:
                continue
            label = normalize_annotation_label(annotation.get("label"))
            if label:
                labels[array_path] = label
    return labels


def normalize_annotation_label(label: object) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in re.split(r"[\\/]+", text) if str(part).strip()]
    if parts:
        text = parts[-1]
        if text.lower() == "manual" and len(parts) > 1:
            text = parts[-2]
    text = re.sub(r"\s+CAD\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+Velocity\b", "", text, flags=re.IGNORECASE)
    for pattern, replacement in PRIME_LABEL_PATTERNS:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


def format_marker_label(label: str) -> str:
    text = label
    for pattern, replacement in MARKER_STYLE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def modality_delta_s(*, modality: ModalityRow, median_delta: dict[str, Any], zarr_path: Path) -> float | None:
    for timestamp_path in modality.timestamp_paths:
        delta = zarr_timestamp_delta_s(zarr_path=zarr_path, array_path=timestamp_path)
        if delta is not None and delta > 0.0:
            return delta
    for content_type in modality.timing_content_types:
        delta = finite_float(median_delta.get(content_type))
        if delta is not None and delta > 0.0:
            return delta
    return None


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def zarr_timestamp_delta_s(*, zarr_path: Path, array_path: str) -> float | None:
    try:
        import numpy as np
        from numcodecs import get_codec
    except ImportError:
        return None

    zarray = read_zarr_array_metadata(zarr_path=zarr_path, array_path=array_path)
    if not zarray:
        return None
    shape = [positive_int(item) for item in zarray.get("shape", [])]
    chunks = [positive_int(item) for item in zarray.get("chunks", [])]
    if len(shape) != 1 or len(chunks) != 1 or shape[0] is None or chunks[0] is None:
        return None
    try:
        dtype = np.dtype(str(zarray.get("dtype", "")))
    except TypeError:
        return None
    if dtype.kind not in {"f", "i", "u"}:
        return None

    arrays = []
    chunk_count = int(math.ceil(shape[0] / chunks[0]))
    for chunk_index in range(chunk_count):
        chunk_bytes = read_zarr_chunk(zarr_path=zarr_path, array_path=array_path, chunk_name=str(chunk_index))
        if chunk_bytes is None:
            return None
        compressor = zarray.get("compressor")
        if isinstance(compressor, dict):
            try:
                chunk_bytes = get_codec(compressor).decode(chunk_bytes)
            except Exception:
                return None
        arrays.append(np.frombuffer(chunk_bytes, dtype=dtype))
    if not arrays:
        return None
    values = np.concatenate(arrays)[: shape[0]].astype(np.float64, copy=False)
    if values.ndim != 1 or values.size < 2:
        return None
    deltas = np.diff(values)
    valid = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if valid.size < 1:
        return None
    return float(np.median(valid))


def read_zarr_array_metadata(*, zarr_path: Path, array_path: str) -> dict[str, Any]:
    metadata_path = f"{array_path}/.zarray"
    if zarr_path.is_dir():
        path = zarr_path / metadata_path
        if not path.is_file():
            return {}
        return load_json(path)
    if not zarr_path.is_file():
        return {}
    try:
        with zipfile.ZipFile(zarr_path) as archive:
            with archive.open(metadata_path) as metadata_file:
                value = json.loads(metadata_file.read().decode("utf-8"))
    except (KeyError, OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_zarr_chunk(*, zarr_path: Path, array_path: str, chunk_name: str) -> bytes | None:
    chunk_path = f"{array_path}/{chunk_name}"
    if zarr_path.is_dir():
        path = zarr_path / chunk_path
        if not path.is_file():
            return None
        return path.read_bytes()
    if not zarr_path.is_file():
        return None
    try:
        with zipfile.ZipFile(zarr_path) as archive:
            with archive.open(chunk_path) as chunk_file:
                return chunk_file.read()
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def read_zarr_attrs(path: Path) -> dict[str, Any]:
    if path.is_dir():
        attrs_path = path / ".zattrs"
        if attrs_path.is_file():
            return load_json(attrs_path)
        return {}
    if not path.is_file():
        return {}
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(".zattrs") as attrs_file:
                value = json.loads(attrs_file.read().decode("utf-8"))
    except (KeyError, OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def path_size_bytes(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return None
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def modality_array_size_bytes(
    *,
    zarr_path: Path,
    array_paths: tuple[str, ...],
    size_cache: dict[tuple[Path, str], int | None],
) -> int | None:
    sizes = []
    for array_path in array_paths:
        key = (zarr_path, array_path)
        if key not in size_cache:
            size_cache[key] = zarr_array_size_bytes(zarr_path=zarr_path, array_path=array_path)
        sizes.append(size_cache[key])
    valid_sizes = [size for size in sizes if size is not None]
    return sum(valid_sizes) if valid_sizes else None


def zarr_array_size_bytes(*, zarr_path: Path, array_path: str) -> int | None:
    if zarr_path.is_dir():
        return path_size_bytes(zarr_path / array_path)
    if not zarr_path.is_file():
        return None
    prefix = f"{array_path.rstrip('/')}/"
    total = 0
    matched = False
    try:
        with zipfile.ZipFile(zarr_path) as archive:
            for info in archive.infolist():
                if info.filename == array_path or info.filename.startswith(prefix):
                    total += int(info.file_size)
                    matched = True
    except (OSError, zipfile.BadZipFile):
        return None
    return total if matched else None


def render_latex_table(*, stats: dict[str, RowStats], caption: str, label: str, subject_label: str) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\footnotesize",
        r"\sisetup{",
        r"  group-separator={,},",
        r"  group-minimum-digits=4",
        r"}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{",
        r"@{}l",
        r"r",
        r"r",
        r"S[table-format=4.1]",
        r"S[table-format=4.1]",
        r"S[table-format=4.1]",
        r"S[table-format=3.0]",
        r"S[table-format=3.0]",
        r"S[table-format=3.0]",
        r"S[table-format=4.1]",
        r"S[table-format=4.1]",
        r"S[table-format=4.1]",
        r"@{}",
        r"}",
        r"\toprule",
        r"{}",
        r"& \multicolumn{1}{c}{\textbf{\#\,Items}}",
        rf"& \multicolumn{{1}}{{c}}{{\textbf{{\#\,{escape_latex_text(subject_label)}}}}}",
        r"& \multicolumn{3}{c}{\textbf{Bytes / item (MiB)}}",
        r"& \multicolumn{3}{c}{\textbf{Sampling rate (Hz)}}",
        r"& \multicolumn{3}{c}{\textbf{Duration/item (s)}} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}",
        r"& & {}",
        r"& \multicolumn{1}{r}{\textbf{P25}} & \multicolumn{1}{r}{\textbf{Median}} & \multicolumn{1}{r}{\textbf{P75}}",
        r"& \multicolumn{1}{r}{\textbf{P25}} & \multicolumn{1}{r}{\textbf{Median}} & \multicolumn{1}{r}{\textbf{P75}}",
        r"& \multicolumn{1}{r}{\textbf{P25}} "
        r"& \multicolumn{1}{r}{\textbf{Median}} "
        r"& \multicolumn{1}{r}{\textbf{P75}} \\",
        r"\midrule",
        "",
    ]
    for modality in TABLE_ROWS:
        sections = SECTION_BREAKS.get(modality.key, ())
        for section in sections:
            lines.extend([render_section_row(section), ""])
        lines.append(render_modality_row(modality=modality, row_stats=stats[modality.key]))
    lines.extend(
        [
            "",
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{\linewidth}",
            r"\footnotesize",
            render_notes(stats),
            r"\end{minipage}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def render_modality_row(*, modality: ModalityRow, row_stats: RowStats) -> str:
    bytes_stats = formatted_quantiles(row_stats.bytes_mib, decimals=1)
    sampling_stats = formatted_quantiles(row_stats.sampling_hz)
    duration_stats = formatted_quantiles(row_stats.duration_s, decimals=1)
    cells = [
        modality.label,
        format_count(row_stats.file_count),
        format_plain_int(len(row_stats.subject_ids)),
        *bytes_stats,
        *sampling_stats,
        *duration_stats,
    ]
    return " & ".join(cells) + r" \\"


def render_section_row(label: str) -> str:
    return " & ".join([label] + ["{}"] * 11) + r" \\"


def render_notes(stats: dict[str, RowStats]) -> str:
    lines = [
        r"\textit{Abbreviations.} A2C: Apical Two-Chamber View; A4C: Apical Four-Chamber View; "
        r"ALAX: Apical Long-Axis View.",
    ]
    beat_counts = stats["3d_brightness_mode"].marker_counts
    beat_parts = []
    if (single_count := int(beat_counts.get("1", 0))) > 0:
        beat_parts.append(f"{single_count} single-beat/unstitched")
    beat_parts.extend(
        f"{count} over {beat} beats" for beat in range(2, 7) if (count := int(beat_counts.get(str(beat), 0))) > 0
    )
    extra_beats = sorted(
        int(beat) for beat in beat_counts if beat.isdigit() and int(beat) > 6 and int(beat_counts.get(beat, 0)) > 0
    )
    beat_parts.extend(f"{int(beat_counts[str(beat)])} over {beat} beats" for beat in extra_beats)
    beat_text = "; ".join(beat_parts)
    if beat_text:
        lines.append(rf"\textsuperscript{{1}} {beat_text}.")
    else:
        lines.append(r"\textsuperscript{1} ECG-gated stitch beat counts were not recorded in this export metadata.")
    lines.append(
        r"\textsuperscript{2} Sample volumes denote guideline-based marker locations "
        r"for the 1D Doppler acquisitions."
    )
    marker_counts = stats["sparse_points_and_events"].marker_counts
    if marker_counts:
        top_markers = sorted(marker_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
        marker_text = "; ".join(
            f"{escape_latex_text(format_marker_label(name))} ({format_plain_int(count)})" for name, count in top_markers
        )
    else:
        marker_text = "none"
    lines.append(
        rf"\textsuperscript{{3}} Most common annotation labels, counted across marker arrays "
        rf"(one item can contain multiple labels): {marker_text}."
    )
    return "\\\\\n".join(lines)


def formatted_quantiles(values: list[float], *, decimals: int = 0) -> list[str]:
    if not values:
        return ["{}", "{}", "{}"]
    return [
        format_number(percentile(values, 0.25), decimals=decimals),
        format_number(percentile(values, 0.50), decimals=decimals),
        format_number(percentile(values, 0.75), decimals=decimals),
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def format_count(value: int) -> str:
    return "{}" if value <= 0 else rf"\num{{{value}}}"


def format_plain_int(value: float | int) -> str:
    number = int(math.floor(float(value) + 0.5))
    return "{}" if number <= 0 else str(number)


def format_number(value: float | int, *, decimals: int) -> str:
    if decimals <= 0:
        return format_plain_int(value)
    return f"{float(value):.{decimals}f}"


def escape_latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
