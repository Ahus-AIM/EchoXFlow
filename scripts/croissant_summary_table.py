#!/usr/bin/env python3
"""Dataset statistics tables from an EchoXFlow export: tab:modality-stats and (with --view-csv) tab:view-stats.

Reads the Croissant record sets (sizes are uncompressed shape x dtype; stream timing from the streams record
set) and, where the catalog cannot resolve a row, the recordings' .zattrs and a few small timestamp arrays.

    uv run python scripts/croissant_summary_table.py "$DATA/croissant.json" \
        --view-csv "$DATA/view_classifier_probabilities.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from echoxflow import open_recording
from echoxflow.schema import ECHOPRIME_LABELS, consensus_view

DEFAULT_OUTPUT_DIR = Path("outputs/tables")


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
        label=r"\qquad Left ventricle, A3C",
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for croissant_summary_table.tex (and the _views files).",
    )
    parser.add_argument(
        "--caption",
        default="Per-modality statistics.",
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
    parser.add_argument(
        "--view-csv",
        type=Path,
        default=None,
        help=(
            "view_classifier_probabilities.csv from build_view_classifier_csv.py. Writes the per-view table next "
            "to the main one (croissant_summary_table_views.tex and .csv) and assigns views to 2D annotation "
            "panels that carry no view of their own."
        ),
    )
    args = parser.parse_args()

    output = args.output_dir / "croissant_summary_table.tex"
    if args.view_csv is not None:
        write_view_tables(
            croissant_path=args.croissant,
            view_csv=args.view_csv,
            output=output.with_name(f"{output.stem}_views.tex"),
        )
    table, report = build_table(
        croissant_path=args.croissant,
        caption=args.caption,
        label=args.label,
        subject_field=args.subject_field,
        subject_label=args.subject_label,
        view_csv=args.view_csv,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table + "\n", encoding="utf-8")
    output.with_name(f"{output.stem}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def build_table(
    *,
    croissant_path: Path,
    caption: str,
    label: str,
    subject_field: str,
    subject_label: str,
    view_csv: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """The tab:modality-stats LaTeX (published layout) and a summary of how document rows were resolved."""
    croissant_path = croissant_path.expanduser()
    metadata = load_json(croissant_path)
    recordings = normalized_rows(metadata, record_set_name="recordings")
    arrays = normalized_rows(metadata, record_set_name="arrays", required=False)
    arrays_by_recording = group_arrays_by_recording(arrays)
    index_croissant(croissant_path.parent, recordings, arrays, normalized_rows(metadata, record_set_name="streams"))
    stats = collect_stats(
        recordings=recordings,
        arrays_by_recording=arrays_by_recording,
        croissant_dir=croissant_path.parent,
        subject_field=subject_field,
    )
    report = resolve_document_rows(
        stats=stats, recordings=recordings, arrays=arrays, root=croissant_path.parent, view_csv=view_csv
    )
    table = render_latex_table(stats=stats, caption=caption, label=label, subject_label=subject_label)
    return paper_layout(table, caption=caption, subject_label=subject_label), report


VIEW_GROUPS = ("A2C", "A3C", "A4C", "PLAX", "PSAX", "Other", "Apical bi-/triplane", "Uncertain")
VIEW_COLUMNS = ("All", *VIEW_GROUPS)
VIEW_HEADERS = ("All", "A2C", "A3C", "A4C", "PLAX", "PSAX", "Other", "Apical bi-/triplane", "Uncertain/ambiguous")
VIEW_SUBSETS = (
    ("All source recordings", None),
    ("Color Doppler subset", "2d_color_doppler"),
    ("Tissue Doppler subset", "2d_tissue_doppler"),
)


def write_view_tables(*, croissant_path: Path, view_csv: Path, output: Path) -> None:
    """Recordings per view (all source recordings, colour and tissue Doppler subsets) as LaTeX and CSV.

    Source recordings are those with an ECG of at least two samples (the electrocardiogram row). The view is the coarse
    class on which EchoPrime and EchoFine agree (``echoxflow.schema.consensus_view``), else uncertain/ambiguous;
    bi-/triplane recordings form their own group. The Doppler subsets hold the items of the 2D Color and 2D Tissue
    rows: the content type with at least two Doppler frames. Per-examination counts include examinations with zero.
    """
    metadata = load_json(croissant_path.expanduser())
    recordings = normalized_rows(metadata, record_set_name="recordings")
    arrays = normalized_rows(metadata, record_set_name="arrays", required=False)
    with_ecg = {
        clean_text(row.get("recording_id"))
        for row in arrays
        if clean_text(row.get("array_path")) == "data/ecg"
        and isinstance(row.get("shape"), list)
        and row["shape"]
        and int(row["shape"][0]) >= 2
    }
    labels: dict[str, str] = {}
    with view_csv.expanduser().open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row.get("skip_reason") or "").strip():
                continue
            prime = (row.get("echoprime_top1") or "").strip()
            view = consensus_view((row.get("echofine_top1") or "").strip(), ECHOPRIME_LABELS.get(prime, prime))
            labels[clean_text(row.get("recording_id"))] = view or "Uncertain"
    exam_ids = sorted({clean_text(r.get("exam_id")) for r in recordings} - {""})
    counts: dict[str, dict[str, Counter[str]]] = {
        title: {exam_id: Counter() for exam_id in exam_ids} for title, _ in VIEW_SUBSETS
    }
    for recording in recordings:
        recording_id, exam_id = clean_text(recording.get("recording_id")), clean_text(recording.get("exam_id"))
        if not exam_id or recording_id not in with_ecg:
            continue
        array_paths = string_list(recording.get("array_paths"))
        if any("2d_brightness_mode_plane_" in path for path in array_paths):
            group = "Apical bi-/triplane"
        else:
            group = labels.get(recording_id, "Uncertain")
        content_types = set(string_list(recording.get("content_types")))
        frame_counts = dict_value(recording.get("frame_counts_by_content_type"))
        for title, content_type in VIEW_SUBSETS:
            if content_type is not None and (
                content_type not in content_types or (positive_int(frame_counts.get(content_type)) or 0) < 2
            ):
                continue
            counts[title][exam_id].update(["All", group])

    def quartiles(values: list[int]) -> tuple[str, str, str]:
        """Median, lower and upper quartile; whole counts stay integers."""
        q25, q50, q75 = (float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75))
        text = [f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}" for v in (q50, q25, q75)]
        return text[0], text[1], text[2]

    csv_rows, blocks = [], []
    for title, _ in VIEW_SUBSETS:
        per_exam = counts[title]
        totals = [sum(c[column] for c in per_exam.values()) for column in VIEW_COLUMNS]
        stats = [quartiles([c[column] for c in per_exam.values()]) for column in VIEW_COLUMNS]
        medians = [f"{median} ({low}--{high})" for median, low, high in stats]
        csv_rows.append({"subset": title, "statistic": "total", **dict(zip(VIEW_HEADERS, totals, strict=True))})
        csv_rows.append(
            {"subset": title, "statistic": "per_exam_median_iqr", **dict(zip(VIEW_HEADERS, medians, strict=True))}
        )
        blocks.append(
            "\n".join(
                [
                    rf"\multicolumn{{{len(VIEW_HEADERS) + 1}}}{{@{{}}l}}{{\textbf{{{title}}}}} \\",
                    r"\quad Total recordings, $n$ & " + " & ".join(rf"\num{{{t}}}" for t in totals) + r" \\",
                    r"\quad Per examination, median & " + " & ".join(median for median, _, _ in stats) + r" \\",
                    r"\quad Per examination, IQR & " + " & ".join(f"{low}--{high}" for _, low, high in stats) + r" \\",
                ]
            )
        )
    table = "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Recordings per view.}",
            r"\label{tab:view-stats}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}l" + "r" * len(VIEW_HEADERS) + r"@{}}",
            r"\toprule",
            " & ".join([r"\textbf{Statistic}"] + [rf"\textbf{{{h}}}" for h in VIEW_HEADERS]) + r" \\",
            r"\midrule",
            "\n\\addlinespace\n".join(blocks),
            r"\bottomrule",
            r"\end{tabular}}",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{\linewidth}",
            r"\footnotesize",
            r"Views are the coarse classes on which the EchoPrime and EchoFine view classifiers agree; "
            r"recordings where they disagree are uncertain/ambiguous.\\",
            r"Doppler subsets count recordings with at least two Doppler frames.\\",
            r"Per-examination counts are medians and interquartile ranges over all examinations.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subset", "statistic", *VIEW_HEADERS])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output)


def format_plain_zero(value: float) -> str:
    return str(int(math.floor(float(value) + 0.5)))


# Croissant-only lookups (filled by index_croissant): sizes are uncompressed (shape x dtype); stream timing comes from
# the "streams" record set. Zarr is only touched for .zattrs (gate metadata, stitch counts, marker labels) and for
# timestamp arrays that the streams record set does not cover (dense annotation contours and meshes).
ARRAY_BYTES: dict[tuple[Path, str], int] = {}
STREAM_DELTA: dict[tuple[Path, str], float] = {}


def index_croissant(
    root: Path, recordings: list[dict[str, Any]], arrays: list[dict[str, Any]], streams: list[dict[str, Any]]
) -> None:
    zarr_by_recording = {clean_text(r.get("recording_id")): root / clean_text(r.get("zarr_path")) for r in recordings}
    path_by_array_id: dict[str, tuple[Path, str]] = {}
    for row in arrays:
        key = (zarr_by_recording.get(clean_text(row.get("recording_id")), root), clean_text(row.get("array_path")))
        path_by_array_id[clean_text(row.get("array_id"))] = key
        shape = row.get("shape")
        if isinstance(shape, list) and row.get("dtype"):
            ARRAY_BYTES[key] = int(np.prod([int(n) for n in shape], dtype=np.int64)) * np.dtype(row["dtype"]).itemsize
    for row in streams:
        key = path_by_array_id.get(clean_text(row.get("timestamp_array_id")))
        delta = finite_float(row.get("median_delta_s"))
        if key is not None and delta is not None and delta > 0.0:
            STREAM_DELTA[key] = delta


class LazyAttrs(dict):
    """Reads a store's .zattrs only when a value is actually requested."""

    def __init__(self, zarr_path: Path) -> None:
        super().__init__()
        self._path, self._loaded = zarr_path, False

    def _load(self) -> None:
        if not self._loaded:
            self._loaded = True
            try:
                self.update(json.loads((self._path / ".zattrs").read_text(encoding="utf-8")))
            except OSError:
                pass

    def get(self, key: Any, default: Any = None) -> Any:
        self._load()
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        self._load()
        return super().__getitem__(key)


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
        zarr_attrs = zarr_attrs_cache.setdefault(zarr_path, LazyAttrs(zarr_path))
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
        # Content type only: B-mode recordings can carry an empty colour segment-offset coordinate array.
        return "2d_color_doppler" in content_type_set
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
        if (zarr_path, timestamp_path) in STREAM_DELTA:
            return STREAM_DELTA[(zarr_path, timestamp_path)]
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


@contextmanager
def open_group(zarr_path: Path) -> Iterator[Any]:
    """Keep a recording store open only while extracting summary metadata."""
    group = open_recording(zarr_path).group
    try:
        yield group
    finally:
        group.store.close()


def zarr_timestamp_delta_s(*, zarr_path: Path, array_path: str) -> float | None:
    try:
        with open_group(zarr_path) as group:
            values = np.asarray(group[array_path][...], dtype=np.float64)
    except Exception:
        return None
    if values.ndim != 1 or values.size < 2:
        return None
    deltas = np.diff(values)
    valid = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    return float(np.median(valid)) if valid.size else None


def read_zarr_attrs(path: Path) -> dict[str, Any]:
    try:
        with open_group(path) as group:
            return dict(group.attrs)
    except Exception:
        return {}


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
    size = ARRAY_BYTES.get((zarr_path, array_path))
    if size is not None:
        return size
    try:
        with open_group(zarr_path) as group:
            return int(group[array_path].nbytes_stored())
    except Exception:
        return None


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
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r">{\raggedleft\arraybackslash}p{3.0em}",
        r"@{}",
        r"}",
        r"\toprule",
        r"{}",
        r"& \multicolumn{1}{c}{\textbf{Items}}",
        rf"& \multicolumn{{1}}{{c}}{{\textbf{{{escape_latex_text(subject_label)}}}}}",
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
    # Whole MiB; sub-MiB items (the ECG) are shown as "<1" rather than a blank zero.
    bytes_stats = [
        "$<$1" if text == "{}" and row_stats.bytes_mib else text
        for text in formatted_quantiles(row_stats.bytes_mib, decimals=0)
    ]
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
        r"Values are medians and interquartile ranges; bytes are uncompressed array sizes.",
        r"Dense annotations are defined over one heart cycle, but can be expanded to the full underlying recording "
        r"using beat-stitching via the ECG under the (verifiable) assumption of regular heart rhythm.",
        r"\textit{Abbreviations.} A2C: Apical Two-Chamber View; A3C: Apical Three-Chamber View; "
        r"A4C: Apical Four-Chamber View.",
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
    quantiles = np.quantile(values, (0.25, 0.5, 0.75))
    return [format_plain_int(value) if decimals <= 0 else f"{value:.{decimals}f}" for value in quantiles]


def format_count(value: int) -> str:
    return "{}" if value <= 0 else rf"\num{{{value}}}"


def format_plain_int(value: float | int) -> str:
    number = int(math.floor(float(value) + 0.5))
    return "{}" if number <= 0 else str(number)


_LATEX_ESCAPES = str.maketrans(
    {
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
)


def escape_latex_text(value: str) -> str:
    return value.translate(_LATEX_ESCAPES)


# ---------------------------------------------------------------------------------------------------------------------
# Rows resolved from annotation documents (merged export layout)
#
# Dense annotations live in annotation documents, possibly nested in merged recordings: one item per annotated panel
# (``aps_contour`` of LV/LA/RV strain and LV volume documents). Views: LV strain panels by exact match of the contour
# timestamps with the document's AHA view timelines (``timestamps/aha_{a2c,a3c,a4c}_native``); otherwise the explicit
# panel role (``2ch``/``3ch``/``4ch``); otherwise, within a document, the EchoPrime view classifier on the linked B-mode
# recording (the panel most likely A4C is A4C, the other A2C). 3D LV meshes are ``aps_mesh_sequence`` documents. 3D
# B-mode items are recordings holding ``data/3d_brightness_mode``; stitch beats are the distinct sub-volume location
# codes in ``coordinates/3d_brightness_mode_frame_locations`` (1 when absent). Points and markers are recordings with a
# named ``measurement`` or a ``spectral_trace`` annotation.

DENSE_ROW_KEYS = {
    ("LV", "A2C"): "lv_a2c_strain",
    ("LV", "A3C"): "lv_alax_strain",
    ("LV", "A4C"): "lv_a4c_strain",
    ("LA", "A2C"): "la_a2c_strain",
    ("LA", "A4C"): "la_a4c_strain",
    ("RV", "A4C"): "rv_strain",
}
DOCUMENT_STRUCTURES = {
    "left_ventricular_strain": "LV",
    "left_ventricular_volume": "LV",
    "left_atrial_strain": "LA",
    "right_ventricular_strain": "RV",
}
ROLE_VIEWS = {"2ch": "A2C", "3ch": "A3C", "4ch": "A4C"}
THREE_D_LOCATIONS = "coordinates/3d_brightness_mode_frame_locations"


def annotation_entries(node: Any, document: dict[str, Any], out: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Collect ``(annotation, enclosing document)`` pairs from a recording manifest, nested documents included."""
    if isinstance(node, dict):
        if "annotation_type" in node or "annotation_document_id" in node:
            document = node
        if "kind" in node and ("value" in node or "target" in node):
            out.append((node, document))
        for value in node.values():
            annotation_entries(value, document, out)
    elif isinstance(node, list):
        for value in node:
            annotation_entries(value, document, out)


def array_values(zarr_path: Path, array_path: str) -> np.ndarray | None:
    try:
        with open_group(zarr_path) as group:
            return np.asarray(group[array_path][...], dtype=np.float64)
    except Exception:
        return None


def rate_and_duration(values: np.ndarray | None) -> tuple[float, float] | None:
    if values is None or values.ndim != 1 or values.size < 2:
        return None
    deltas = np.diff(values)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if not deltas.size:
        return None
    delta = float(np.median(deltas))
    return 1.0 / delta, float(values.size - 1) * delta


def echoprime_apical_probabilities(view_csv: Path | None) -> dict[str, dict[str, float]]:
    if view_csv is None or not view_csv.is_file():
        return {}
    probabilities: dict[str, dict[str, float]] = {}
    with view_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                probabilities[row["recording_id"]] = {v: float(row[f"echoprime_prob_{v}"]) for v in ("A2C", "A4C")}
            except (KeyError, ValueError):
                continue
    return probabilities


def resolve_document_rows(
    *,
    stats: dict[str, RowStats],
    recordings: list[dict[str, Any]],
    arrays: list[dict[str, Any]],
    root: Path,
    view_csv: Path | None,
) -> dict[str, Any]:
    """Replace the 3D B-mode, dense annotation and points/markers rows; return a summary of the resolution."""
    # 3D B-mode and stitch beats.
    with_locations = {a["recording_id"] for a in arrays if a.get("array_path") == THREE_D_LOCATIONS}
    volumes = [r for r in recordings if "data/3d_brightness_mode" in string_list(r.get("array_paths"))]

    def beats(record: dict[str, Any]) -> int:
        if record["recording_id"] not in with_locations:
            return 1
        codes = array_values(root / record["zarr_path"], THREE_D_LOCATIONS)
        return 1 if codes is None else max(1, len(np.unique(codes)))

    with ThreadPoolExecutor(16) as pool:
        beat_counts = list(pool.map(beats, volumes))
    three_d = RowStats()
    stitched: Counter[int] = Counter()
    for record, count in zip(volumes, beat_counts, strict=True):
        zarr_path = root / record["zarr_path"]
        three_d.item_ids.add(record["recording_id"])
        three_d.file_count += 1
        three_d.subject_ids.add(record["exam_id"])
        size = ARRAY_BYTES.get((zarr_path, "data/3d_brightness_mode"))
        if size is not None:
            three_d.bytes_mib.append(size / (1024.0 * 1024.0))
        delta = STREAM_DELTA.get((zarr_path, "timestamps/3d_brightness_mode"))
        frames = positive_int(dict_value(record.get("frame_counts_by_content_type")).get("3d_brightness_mode")) or 0
        if delta and frames >= 2:
            three_d.sampling_hz.append(1.0 / delta)
            three_d.duration_s.append((frames - 1) * delta)
        stitched[count] += 1
    three_d.marker_counts = {str(beats_): n for beats_, n in sorted(stitched.items())}
    stats["3d_brightness_mode"] = three_d

    # Annotation documents.
    annotated = [
        r
        for r in recordings
        if any(
            p.startswith(("annotations/", "derived/")) and not p.endswith("bmode_ecg_qrs")
            for p in string_list(r.get("array_paths"))
        )
    ]

    def load(record: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
        attrs = json.loads((root / record["zarr_path"] / ".zattrs").read_text(encoding="utf-8"))
        entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        annotation_entries(attrs.get("recording_manifest", {}), {}, entries)
        return record, entries

    with ThreadPoolExecutor(32) as pool:
        loaded = list(pool.map(load, annotated))

    panels: list[dict[str, Any]] = []
    meshes, markers = RowStats(), RowStats()
    marker_labels: Counter[str] = Counter()
    la_traces: set[tuple[str, Any]] = set()
    la_trace_exams: set[str] = set()
    for record, entries in loaded:
        zarr_path = root / record["zarr_path"]
        aha: dict[str, str] = {}
        for entry, _document in entries:
            if entry.get("kind") == "aha_segmental_strain":
                view = str(entry.get("links", {}).get("source_view", "")).upper()
                aha.setdefault(view, entry["time"]["zarr_path"])
        has_marker = False
        for entry, document in entries:
            kind = entry.get("kind")
            document_key = (record["recording_id"], document.get("annotation_document_id"))
            if kind == "aps_contour" and document.get("annotation_type") in DOCUMENT_STRUCTURES:
                panels.append(
                    {
                        "structure": DOCUMENT_STRUCTURES[document["annotation_type"]],
                        "lv_strain": document["annotation_type"] == "left_ventricular_strain",
                        "role": str(entry["target"].get("semantic_id")),
                        "zarr_path": zarr_path,
                        "time": entry.get("time", {}).get("zarr_path"),
                        "aha": aha,
                        "exam": record["exam_id"],
                        "document": document_key,
                        "linked": entry.get("target_data", {}).get("recording_id"),
                    }
                )
            elif kind == "recovered_la_trace":
                la_traces.add(document_key)
                la_trace_exams.add(record["exam_id"])
            elif kind == "aps_mesh_sequence" and "left_ventricle" in str(entry["target"].get("semantic_id")):
                meshes.item_ids.add(f"{record['recording_id']}:{entry['value']['zarr_path']}")
                meshes.file_count += 1
                meshes.subject_ids.add(record["exam_id"])
                values = rate_and_duration(array_values(zarr_path, entry["time"]["zarr_path"]))
                if values is not None:
                    meshes.sampling_hz.append(values[0])
                    meshes.duration_s.append(values[1])
            elif kind in {"measurement", "spectral_trace"}:
                measure_id = entry.get("links", {}).get("measure_id")
                if kind == "measurement" and not measure_id:
                    continue
                has_marker = True
                label = normalize_annotation_label(measure_id) if measure_id else ""
                if label:
                    marker_labels[label] += 1
        if has_marker:
            markers.item_ids.add(record["recording_id"])
            markers.file_count += 1
            markers.subject_ids.add(record["exam_id"])
    markers.marker_counts = dict(marker_labels)
    stats["3d_left_ventricle_mesh"] = meshes
    stats["sparse_points_and_events"] = markers

    def resolve_panel(panel: dict[str, Any]) -> None:
        values = array_values(panel["zarr_path"], panel["time"]) if panel["time"] else None
        panel["timing"] = rate_and_duration(values)
        panel["view"] = ROLE_VIEWS.get(panel["role"])
        panel["view_source"] = "role" if panel["view"] else None
        if panel["lv_strain"] and values is not None:
            for view, path in panel["aha"].items():
                reference = array_values(panel["zarr_path"], path)
                if reference is not None and reference.shape == values.shape and np.allclose(reference, values):
                    panel["view"], panel["view_source"] = view, "aha_timeline"
                    break

    with ThreadPoolExecutor(32) as pool:
        list(pool.map(resolve_panel, panels))
    probabilities = echoprime_apical_probabilities(view_csv)
    by_document: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for panel in panels:
        by_document[(panel["structure"], panel["document"])].append(panel)
    for (structure, _), group in by_document.items():
        pending = [p for p in group if p["view"] is None]
        if not pending:
            continue
        if structure == "RV":
            for panel in pending:
                panel["view"], panel["view_source"] = "A4C", "structure"
            continue
        if group[0]["lv_strain"]:
            continue  # LV strain panels without an AHA timeline match stay unassigned
        taken = {p["view"] for p in group if p["view"] is not None}
        options = [view for view in ("A4C", "A2C") if view not in taken]

        def a4c_margin(panel: dict[str, Any]) -> float:
            p = probabilities.get(panel["linked"], {})
            return p.get("A4C", 0.0) - p.get("A2C", 0.0)

        for panel, view in zip(sorted(pending, key=a4c_margin, reverse=True), options, strict=False):
            panel["view"], panel["view_source"] = view, "view_classifier"
    dense = {key: RowStats() for key in DENSE_ROW_KEYS.values()}
    unassigned: Counter[str] = Counter()
    for panel in panels:
        key = DENSE_ROW_KEYS.get((panel["structure"], panel["view"]))
        if key is None:
            unassigned[f"{panel['structure']}|{panel['role']}"] += 1
            continue
        row = dense[key]
        row.item_ids.add(f"{panel['document']}:{panel['role']}")
        row.file_count += 1
        row.subject_ids.add(panel["exam"])
        if panel["timing"] is not None:
            row.sampling_hz.append(panel["timing"][0])
            row.duration_s.append(panel["timing"][1])
    stats.update(dense)
    return {
        "stitch_beats": {str(k): v for k, v in sorted(stitched.items())},
        "panel_view_sources": dict(Counter(str(p["view_source"]) for p in panels if p["view"])),
        "unassigned_panels": dict(unassigned),
        "la_volume_trace_documents_not_tabulated": {"documents": len(la_traces), "exams": len(la_trace_exams)},
        "top_marker_labels": marker_labels.most_common(10),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Paper layout (tab:modality-stats as published: spacer columns; equal-width right-aligned statistic columns)

PAPER_HEADER = r"""\begin{table}[ht]
\centering
\caption{CAPTION}
\footnotesize
\sisetup{
  group-separator={\,},
  group-minimum-digits=4
}
\setlength{\tabcolsep}{1.3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{
l
r
r
p{10pt}@{}
>{\raggedleft\arraybackslash}p{3.0em}
>{\raggedleft\arraybackslash}p{3.0em}
%@{\hspace{2pt}}
>{\raggedleft\arraybackslash}p{3.0em}
@{}p{10pt}@{}
>{\raggedleft\arraybackslash}p{3.0em}
>{\raggedleft\arraybackslash}p{3.0em}
%@{\hspace{2pt}}
>{\raggedleft\arraybackslash}p{3.0em}
@{}p{0pt}@{}
>{\raggedleft\arraybackslash}p{3.0em}
>{\raggedleft\arraybackslash}p{3.0em}
%@{\hspace{2pt}}
>{\raggedleft\arraybackslash}p{3.0em}
}
\toprule
{}
& \multicolumn{1}{c}{\textbf{Items}}
& \multicolumn{1}{c}{\textbf{SUBJECT}}
& {}
& \multicolumn{3}{c}{\textbf{Bytes / item (MiB)}}
& {}
& \multicolumn{3}{c}{\textbf{Sampling rate (Hz)}}
& {}
& \multicolumn{3}{c}{\textbf{Duration/item (s)}} \\
\cmidrule(lr){5-7} \cmidrule(lr){9-11} \cmidrule(lr){13-15}
& & {} & {}
& \multicolumn{1}{r}{\textbf{P25}} & \multicolumn{1}{r}{\textbf{P50}} & \multicolumn{1}{r}{\textbf{P75}}
& {}
& \multicolumn{1}{r}{\textbf{P25}} & \multicolumn{1}{r}{\textbf{P50}} & \multicolumn{1}{r}{\textbf{P75}}
& {}
& \multicolumn{1}{r}{\textbf{P25}} & \multicolumn{1}{r}{\textbf{P50}} & \multicolumn{1}{r}{\textbf{P75}} \\
\midrule"""


def paper_layout(table: str, *, caption: str, subject_label: str) -> str:
    """Re-render ``render_latex_table`` output with the published header and its 15-column row layout."""
    lines = table.split("\n")
    first, last = lines.index(r"\midrule"), lines.index(r"\bottomrule")
    rows = []
    for line in lines[first + 1 : last]:
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.rstrip().removesuffix("\\\\").split("&")]
        cells = [cell or "{}" for cell in cells + [""] * (12 - len(cells))]
        rows.append(" & ".join(cells[:3] + ["{}"] + cells[3:6] + ["{}"] + cells[6:9] + ["{}"] + cells[9:12]) + r" \\")
    header = PAPER_HEADER.replace("CAPTION", caption).replace("SUBJECT", subject_label)
    footer = "\n".join(lines[last:]).replace("\\end{tabular}", "\\end{tabular}}", 1)
    return header + "\n\n" + "\n".join(rows) + "\n\n" + footer


if __name__ == "__main__":
    raise SystemExit(main())
