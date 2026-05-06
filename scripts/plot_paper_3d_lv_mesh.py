#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echoxflow import RecordingRecord, load_croissant, open_recording  # noqa: E402
from echoxflow.config import data_root  # noqa: E402
from echoxflow.loading import PackedMeshAnnotation  # noqa: E402
from echoxflow.plotting import PlotStyle  # noqa: E402
from echoxflow.plotting.annotations import (  # noqa: E402
    _mesh_plane_segments,
    _mesh_points_to_render_frame,
    _segments_to_closed_polygons,
    _valid_mesh_faces,
)
from echoxflow.plotting.panels import renderer_for  # noqa: E402
from echoxflow.plotting.renderer import RecordingPlotRenderer  # noqa: E402
from echoxflow.plotting.specs import PanelSpec  # noqa: E402
from echoxflow.scan import (  # noqa: E402
    mesh_frame_indices_for_volume_timestamps,
    prepare_3d_brightness_for_display,
    spherical_depth_slice_stack,
    spherical_geometry_from_metadata,
    spherical_radial_axis_stack,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = Path("outputs/images/segmentation_masks/paper_3d_lv_mesh_1")
BACKGROUND = "#FFFFFF"
AXIS_COLOR = "#222222"
LV_FILL_COLOR = "#F2A3A3"
MOSAIC_GAP_PX = 8
MOSAIC_PAD_PX = 2
SECTOR_TOP_CROP_CM = 0.0
SECTOR_LINE_LEFT_PAD_PX = 18
SECTOR_LINE_GAP_PX = 5
SECTOR_RULER_LEFT_PAD_PX = 18
SECTOR_RULER_GAP_PX = 5
SECTOR_RULER_MAJOR_TICK_PX = 8
SECTOR_RULER_MINOR_TICK_PX = 4
SOURCE_MOSAIC_ROWS = 3
SOURCE_MOSAIC_COLS = 4
DISPLAY_MOSAIC_ROWS = 2
DISPLAY_MOSAIC_COLS = 3
DISPLAY_PANEL_ORDER = (0, 4, 8, 1, 6, 11)
LONG_AXIS_ANGLES_DEG = (0.0, -60.0, -120.0)
SHORT_AXIS_DEPTH_FRACTIONS = (4.0 / 15.0, 8.0 / 15.0, 10.0 / 15.0)
CUT_LINE_COLORS = ("#0072B2", "#D55E00", "#009E73")
CUT_LINE_ALPHA = 0.52
CUT_LINE_WIDTH = 1.7
SIDE_LINE_WIDTH = 3.4
BOTTOM_OUTLINE_COLORS = ("#CC79A7", "#E69F00", "#56B4E9")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a paper-style 3D LV mesh overlay on a 3D B-mode mosaic.")
    parser.add_argument("source", nargs="?", type=Path, default=None, help="Dataset root or croissant.json.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exam-id", default=None)
    parser.add_argument("--recording-id", default=None)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--frame-policy", choices=("middle", "ed", "es"), default="es")
    parser.add_argument("--mosaic-size", type=int, default=256, help="Per-panel clinical mosaic resolution in pixels.")
    parser.add_argument("--slice-pixels-per-cm", type=float, default=64.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    catalog_path, root = _resolve_source(args.source, root=args.root)
    catalog = load_croissant(catalog_path)
    record = _select_recording(
        catalog.recordings,
        exam_id=args.exam_id,
        recording_id=args.recording_id,
        case_index=int(args.case_index),
    )
    frame_index, time_s, volume_manifest = _select_frame(record, root=root, policy=args.frame_policy)
    figure = _render_mesh_figure(
        record,
        root=root,
        frame_index=frame_index,
        time_s=time_s,
        mosaic_size=int(args.mosaic_size),
        slice_pixels_per_cm=float(args.slice_pixels_per_cm),
    )

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    manifest = output.with_suffix(".manifest.json")
    try:
        figure.savefig(png, dpi=int(args.dpi), facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.0)
        figure.savefig(pdf, dpi=int(args.dpi), facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.0)
    finally:
        plt.close(figure)
    manifest.write_text(
        json.dumps(
            {
                "png": str(png),
                "pdf": str(pdf),
                "exam_id": record.exam_id,
                "recording_id": record.recording_id,
                "frame_policy": args.frame_policy,
                "frame_index": int(frame_index),
                "time_s": None if time_s is None else float(time_s),
                "mosaic_size": int(args.mosaic_size),
                "slice_pixels_per_cm": float(args.slice_pixels_per_cm),
                **volume_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(png)
    return 0


def _resolve_source(source: Path | None, *, root: Path | None) -> tuple[Path, Path]:
    if source is None:
        resolved_root = data_root(root)
        return resolved_root / "croissant.json", resolved_root
    source = source.expanduser()
    if source.is_dir():
        return source / "croissant.json", source
    resolved_root = root.expanduser() if root is not None else source.parent
    return source, resolved_root


def _select_recording(
    records: Sequence[RecordingRecord],
    *,
    exam_id: str | None,
    recording_id: str | None,
    case_index: int,
) -> RecordingRecord:
    candidates = [
        record
        for record in records
        if "data/3d_brightness_mode" in record.array_paths
        and "data/3d_left_ventricle_mesh/point_values" in record.array_paths
        and "data/3d_left_ventricle_mesh/face_values" in record.array_paths
    ]
    if exam_id is not None:
        candidates = [record for record in candidates if record.exam_id == exam_id]
    if recording_id is not None:
        candidates = [record for record in candidates if record.recording_id == recording_id]
    if not candidates:
        raise SystemExit("No 3D B-mode recording with LV mesh was found.")
    index = int(np.clip(int(case_index), 0, len(candidates) - 1))
    return sorted(candidates, key=lambda item: (item.exam_id, item.recording_id))[index]


def _select_frame(
    record: RecordingRecord,
    *,
    root: Path,
    policy: str,
) -> tuple[int, float | None, dict[str, object]]:
    store = open_recording(record, root=root)
    mesh = store.load_packed_mesh_annotation()
    times: list[float] = []
    volumes_ml: list[float] = []
    for index in range(mesh.frame_count):
        frame = mesh.frame(index)
        volume = _closed_surface_volume_m3(frame.points, frame.faces)
        if not np.isfinite(volume):
            continue
        times.append(float(index if frame.timestamp is None else frame.timestamp))
        volumes_ml.append(float(volume) * 1.0e6)
    if not volumes_ml:
        return 0, None, {"mesh_frame_count": int(mesh.frame_count)}
    values = np.asarray(volumes_ml, dtype=np.float64)
    if policy == "ed":
        mesh_index = int(np.argmax(values))
    elif policy == "es":
        mesh_index = int(np.argmin(values))
    else:
        mesh_index = int(values.size // 2)
    mesh_time = float(times[mesh_index])
    volume_timestamps = store.load_timestamps("3d_brightness_mode")
    if volume_timestamps is None:
        return mesh_index, mesh_time, _volume_manifest(mesh, values, mesh_index)
    ts = np.asarray(volume_timestamps, dtype=np.float64).reshape(-1)
    frame_index = int(np.argmin(np.abs(ts - mesh_time))) if ts.size else mesh_index
    return frame_index, float(ts[frame_index]) if ts.size else mesh_time, _volume_manifest(mesh, values, mesh_index)


def _volume_manifest(mesh: object, values: np.ndarray, mesh_index: int) -> dict[str, object]:
    return {
        "mesh_frame_count": int(getattr(mesh, "frame_count", 0)),
        "selected_mesh_index": int(mesh_index),
        "selected_mesh_volume_ml": float(values[int(mesh_index)]),
        "min_mesh_volume_ml": float(np.nanmin(values)),
        "max_mesh_volume_ml": float(np.nanmax(values)),
    }


def _render_mesh_figure(
    record: RecordingRecord,
    *,
    root: Path,
    frame_index: int,
    time_s: float | None,
    mosaic_size: int,
    slice_pixels_per_cm: float,
) -> object:
    style = PlotStyle(
        width_px=1400,
        height_px=1050,
        dpi=140,
        figure_facecolor=BACKGROUND,
        panel_facecolor=BACKGROUND,
        text_color=AXIS_COLOR,
        text_dim_color="#666666",
        annotation_color=LV_FILL_COLOR,
        annotation_edge_color=LV_FILL_COLOR,
        annotation_linewidth=0.0,
        annotation_markersize=2.0,
        panel_title_fontsize=14.0,
        axis_tick_label_fontsize=10.5,
        clinical_depth_ruler_label_fontsize=8.25,
        show_clinical_depth_ruler=False,
    )
    renderer = RecordingPlotRenderer(style=style)
    panels, _ecg = renderer._load_specs(
        record,
        root=root,
        modalities=("3d_brightness_mode",),
        view_mode="clinical",
        show_annotations=True,
    )
    image_panel = next(panel for panel in panels if panel.kind == "image")
    image_panel = _with_filled_mesh_overlay(
        replace(image_panel, label="LV endocardial mesh"),
        frame_index=frame_index,
        mosaic_size=mosaic_size,
        slice_pixels_per_cm=slice_pixels_per_cm,
    )
    timestamps = image_panel.loaded.timestamps
    if time_s is None and timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size:
            time_s = float(ts[min(max(0, int(frame_index)), ts.size - 1)])
    image_shape = np.asarray(image_panel.loaded.data).shape
    aspect = float(image_shape[2]) / max(float(image_shape[1]), 1.0)
    figure_width = 8.2
    figure, ax = plt.subplots(figsize=(figure_width, figure_width / aspect), dpi=140, facecolor=BACKGROUND)
    renderer_for(image_panel.kind).render(
        ax,
        image_panel,
        time_s=float(0.0 if time_s is None else time_s),
        frame_index=int(frame_index),
        style=style,
    )
    _draw_compact_cut_lines(ax, image_panel)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_anchor("C")
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    return figure


def _with_filled_mesh_overlay(
    image_panel: PanelSpec,
    *,
    frame_index: int,
    mosaic_size: int,
    slice_pixels_per_cm: float,
) -> PanelSpec:
    loaded = image_panel.loaded
    attrs = dict(loaded.attrs)
    mesh_annotation = attrs.get("mesh_annotation")
    if not isinstance(mesh_annotation, PackedMeshAnnotation):
        attrs.pop("mosaic_annotation_lines", None)
        return replace(image_panel, loaded=replace(loaded, attrs=attrs))
    raw_metadata = None if loaded.stream is None else loaded.stream.metadata.raw
    geometry = spherical_geometry_from_metadata(raw_metadata)
    compact_data, compact_polygons, cut_lines, selected_timestamp = _physical_scale_six_panel_mosaic(
        loaded,
        mesh_annotation=mesh_annotation,
        frame_index=frame_index,
        geometry=geometry,
        pixels_per_cm=max(float(slice_pixels_per_cm), float(mosaic_size) / 12.0),
    )
    attrs.pop("mosaic_annotation_lines", None)
    attrs["mosaic_annotation_polygons"] = compact_polygons
    attrs["compact_cut_lines"] = cut_lines
    timestamps = None if selected_timestamp is None else np.asarray([selected_timestamp], dtype=np.float64)
    return replace(image_panel, loaded=replace(loaded, data=compact_data, timestamps=timestamps, attrs=attrs))


def _high_resolution_mosaic(loaded: object, *, mosaic_size: int) -> np.ndarray:
    stream = getattr(loaded, "stream", None)
    raw_metadata = None if stream is None else stream.metadata.raw
    if stream is None or not hasattr(stream, "data"):
        return np.asarray(loaded.data)
    from echoxflow.scan import clinical_spherical_mosaic, prepare_3d_brightness_for_display

    geometry = spherical_geometry_from_metadata(raw_metadata)
    prepared = prepare_3d_brightness_for_display(np.asarray(stream.data), loaded.timestamps, raw_metadata)
    return clinical_spherical_mosaic(
        prepared.volumes,
        geometry,
        output_size=(max(16, int(mosaic_size)), max(16, int(mosaic_size))),
    ).frames


def _physical_scale_six_panel_mosaic(
    loaded: object,
    *,
    mesh_annotation: PackedMeshAnnotation,
    frame_index: int,
    geometry: object,
    pixels_per_cm: float,
) -> tuple[np.ndarray, tuple[tuple[np.ndarray, ...], ...], tuple[dict[str, object], ...], float | None]:
    stream = getattr(loaded, "stream", None)
    raw_metadata = None if stream is None else stream.metadata.raw
    if stream is None or not hasattr(stream, "data"):
        return np.asarray(loaded.data), (), (), None

    prepared = prepare_3d_brightness_for_display(np.asarray(stream.data), loaded.timestamps, raw_metadata)
    volumes = np.asarray(prepared.volumes)
    selected_index = min(max(0, int(frame_index)), max(0, int(volumes.shape[0]) - 1))
    selected = volumes[selected_index : selected_index + 1]
    selected_timestamp = None
    if prepared.timestamps is not None:
        timestamps = np.asarray(prepared.timestamps, dtype=np.float64).reshape(-1)
        if timestamps.size:
            selected_timestamp = float(timestamps[min(selected_index, timestamps.size - 1)])

    depth_start_cm = 100.0 * float(getattr(geometry, "depth_start_m", 0.0))
    depth_end_cm = 100.0 * float(getattr(geometry, "depth_end_m", 0.0))
    depth_span_cm = max(depth_end_cm - depth_start_cm, 1.0)
    radial_height = max(32, int(round(depth_span_cm * pixels_per_cm)))
    radial_half_width_cm = max(_radial_axis_half_width_cm(geometry, angle_deg) for angle_deg in LONG_AXIS_ANGLES_DEG)
    radial_width = max(32, int(round(2.0 * radial_half_width_cm * pixels_per_cm)))
    azimuth_half_cm = max(0.1, depth_end_cm * np.tan(0.5 * float(getattr(geometry, "azimuth_width_rad", 0.0))))
    elevation_half_cm = max(0.1, depth_end_cm * np.tan(0.5 * float(getattr(geometry, "elevation_width_rad", 0.0))))
    short_axis_width = max(32, int(round(2.0 * azimuth_half_cm * pixels_per_cm)))
    short_axis_height = max(32, int(round(2.0 * elevation_half_cm * pixels_per_cm)))

    panel_images: list[np.ndarray] = []
    panel_specs: list[dict[str, object]] = []
    for angle_deg in LONG_AXIS_ANGLES_DEG:
        panel = spherical_radial_axis_stack(
            selected,
            geometry,
            angle_deg=float(angle_deg),
            output_size=(radial_height, radial_width),
            half_width_m=radial_half_width_cm / 100.0,
        )[0]
        panel_images.append(panel)
        panel_specs.append(
            {
                "kind": "long_axis",
                "angle_deg": float(angle_deg),
                "x_min": -radial_half_width_cm / 100.0,
                "x_max": radial_half_width_cm / 100.0,
                "y_min": depth_start_cm / 100.0,
                "y_max": depth_end_cm / 100.0,
            }
        )
    for depth_fraction in SHORT_AXIS_DEPTH_FRACTIONS:
        depth_m = float(getattr(geometry, "depth_start_m", 0.0)) + float(depth_fraction) * (
            float(getattr(geometry, "depth_end_m", 0.0)) - float(getattr(geometry, "depth_start_m", 0.0))
        )
        panel = spherical_depth_slice_stack(
            selected,
            geometry,
            depth_fraction=float(depth_fraction),
            output_size=(short_axis_height, short_axis_width),
        )[0]
        panel_images.append(panel)
        panel_specs.append(
            {
                "kind": "short_axis",
                "depth_m": depth_m,
                "x_min": -azimuth_half_cm / 100.0,
                "x_max": azimuth_half_cm / 100.0,
                "y_min": -elevation_half_cm / 100.0,
                "y_max": elevation_half_cm / 100.0,
            }
        )

    cropped_images, crop_boxes = _crop_panel_images(panel_images)
    row_heights = [
        max(image.shape[0] for image in cropped_images[:DISPLAY_MOSAIC_COLS]),
        max(image.shape[0] for image in cropped_images[DISPLAY_MOSAIC_COLS:]),
    ]
    col_widths = [
        max(cropped_images[col].shape[1] + SECTOR_LINE_LEFT_PAD_PX, cropped_images[col + DISPLAY_MOSAIC_COLS].shape[1])
        for col in range(DISPLAY_MOSAIC_COLS)
    ]
    output_h = int(sum(row_heights) + MOSAIC_GAP_PX)
    output_w = int(sum(col_widths) + MOSAIC_GAP_PX * (DISPLAY_MOSAIC_COLS - 1))
    mosaic = np.full((1, output_h, output_w), np.nan, dtype=np.float32)
    placements: list[tuple[int, int]] = []
    for panel_idx, image in enumerate(cropped_images):
        row = panel_idx // DISPLAY_MOSAIC_COLS
        col = panel_idx % DISPLAY_MOSAIC_COLS
        x_cursor = sum(col_widths[:col]) + MOSAIC_GAP_PX * col
        y_cursor = sum(row_heights[:row]) + MOSAIC_GAP_PX * row
        left_pad = SECTOR_LINE_LEFT_PAD_PX if row == 0 else 0
        x_dst = x_cursor + (col_widths[col] - image.shape[1] - left_pad) // 2 + left_pad
        y_dst = y_cursor + row_heights[row] - image.shape[0]
        mosaic[0, y_dst : y_dst + image.shape[0], x_dst : x_dst + image.shape[1]] = image
        placements.append((x_dst, y_dst))

    mesh_polygons = _mesh_polygons_for_panels(
        mesh_annotation,
        geometry=geometry,
        panel_specs=panel_specs,
        crop_boxes=crop_boxes,
        placements=placements,
        output_shapes=[image.shape for image in panel_images],
        selected_index=selected_index,
        volume_timestamps=prepared.timestamps,
        metadata=raw_metadata,
    )
    cut_lines = _cut_lines_for_panels(
        panel_specs,
        crop_boxes=crop_boxes,
        placements=placements,
        output_shapes=[image.shape for image in panel_images],
        cropped_images=cropped_images,
    )
    return mosaic, (mesh_polygons,), cut_lines, selected_timestamp


def _crop_panel_images(images: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    cropped: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(arr)
        if not bool(np.any(finite)):
            boxes.append((0, arr.shape[1], 0, arr.shape[0]))
            cropped.append(arr)
            continue
        yy, xx = np.nonzero(finite)
        x0 = max(0, int(xx.min()) - MOSAIC_PAD_PX)
        x1 = min(arr.shape[1], int(xx.max()) + 1 + MOSAIC_PAD_PX)
        y0 = max(0, int(yy.min()) - MOSAIC_PAD_PX)
        y1 = min(arr.shape[0], int(yy.max()) + 1 + MOSAIC_PAD_PX)
        boxes.append((x0, x1, y0, y1))
        cropped.append(arr[y0:y1, x0:x1])
    return cropped, boxes


def _mesh_polygons_for_panels(
    mesh_annotation: PackedMeshAnnotation,
    *,
    geometry: object,
    panel_specs: list[dict[str, object]],
    crop_boxes: list[tuple[int, int, int, int]],
    placements: list[tuple[int, int]],
    output_shapes: list[tuple[int, ...]],
    selected_index: int,
    volume_timestamps: np.ndarray | None,
    metadata: object,
) -> tuple[np.ndarray, ...]:
    mesh_indices = mesh_frame_indices_for_volume_timestamps(
        mesh_annotation.timestamps,
        volume_timestamps,
        metadata,
        mesh_frame_count=mesh_annotation.frame_count,
        target_count=max(1, int(selected_index) + 1),
    )
    mesh_index = mesh_indices[selected_index] if selected_index < len(mesh_indices) else selected_index
    mesh_frame = mesh_annotation.frame(min(max(0, int(mesh_index)), max(0, mesh_annotation.frame_count - 1)))
    points = _mesh_points_to_render_frame(np.asarray(mesh_frame.points, dtype=np.float32))
    faces = _valid_mesh_faces(np.asarray(mesh_frame.faces, dtype=np.int32), n_points=int(points.shape[0]))
    if points.ndim != 2 or points.shape[0] == 0 or faces.size == 0:
        return ()

    output_polygons: list[np.ndarray] = []
    for spec, crop_box, placement, output_shape in zip(panel_specs, crop_boxes, placements, output_shapes):
        plane_point, plane_normal, basis_u, basis_v = _panel_plane(spec)
        segments = _mesh_plane_segments(points, faces, plane_point, plane_normal, basis_u, basis_v)
        polygons = _segments_to_closed_polygons(segments)
        for polygon in polygons:
            mapped = _panel_polygon_to_global(
                polygon,
                spec=spec,
                crop_box=crop_box,
                placement=placement,
                output_shape=output_shape,
            )
            if mapped.shape[0] >= 3:
                output_polygons.append(mapped)
    return tuple(output_polygons)


def _panel_polygon_to_global(
    polygon: np.ndarray,
    *,
    spec: dict[str, object],
    crop_box: tuple[int, int, int, int],
    placement: tuple[int, int],
    output_shape: tuple[int, ...],
) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x_min = float(spec["x_min"])
    x_max = float(spec["x_max"])
    y_min = float(spec["y_min"])
    y_max = float(spec["y_max"])
    height = int(output_shape[0])
    width = int(output_shape[1])
    x0, _x1, y0, _y1 = crop_box
    x_dst, y_dst = placement
    mapped = np.empty_like(pts)
    mapped[:, 0] = (pts[:, 0] - x_min) / max(x_max - x_min, 1e-8) * float(width - 1) - x0 + x_dst
    mapped[:, 1] = (pts[:, 1] - y_min) / max(y_max - y_min, 1e-8) * float(height - 1) - y0 + y_dst
    return mapped


def _cut_lines_for_panels(
    panel_specs: list[dict[str, object]],
    *,
    crop_boxes: list[tuple[int, int, int, int]],
    placements: list[tuple[int, int]],
    output_shapes: list[tuple[int, ...]],
    cropped_images: list[np.ndarray],
) -> tuple[dict[str, object], ...]:
    lines: list[dict[str, object]] = []
    for panel_idx, (spec, crop_box, placement, output_shape, cropped_image) in enumerate(
        zip(panel_specs, crop_boxes, placements, output_shapes, cropped_images)
    ):
        if spec["kind"] == "long_axis":
            boundary = _left_sector_boundary_line(np.isfinite(cropped_image), placement=placement)
            if boundary.size:
                lines.append({"xy": boundary, "color": CUT_LINE_COLORS[panel_idx], "kind": "side"})
            continue
        for angle_idx, angle_deg in enumerate(LONG_AXIS_ANGLES_DEG):
            line = _short_axis_cut_line(spec, float(angle_deg))
            if line is None:
                continue
            mapped = _panel_polygon_to_global(
                line,
                spec=spec,
                crop_box=crop_box,
                placement=placement,
                output_shape=output_shape,
            )
            mapped = _clip_line_to_rect(
                mapped,
                x_min=float(placement[0]),
                x_max=float(placement[0] + crop_box[1] - crop_box[0]),
                y_min=float(placement[1]),
                y_max=float(placement[1] + crop_box[3] - crop_box[2]),
            )
            if mapped is None:
                continue
            center = _panel_point_to_global(
                np.asarray([[0.0, 0.0]], dtype=np.float32),
                spec=spec,
                crop_box=crop_box,
                placement=placement,
                output_shape=output_shape,
            )[0]
            for segment in _line_segments_inside_mask(mapped, np.isfinite(cropped_image), placement=placement):
                segment = _line_segment_from_center_to_lower_side(segment, center)
                if segment.size:
                    lines.append({"xy": segment, "color": CUT_LINE_COLORS[angle_idx], "kind": "cut"})
    return tuple(lines)


def _panel_point_to_global(
    points: np.ndarray,
    *,
    spec: dict[str, object],
    crop_box: tuple[int, int, int, int],
    placement: tuple[int, int],
    output_shape: tuple[int, ...],
) -> np.ndarray:
    return _panel_polygon_to_global(
        np.asarray(points, dtype=np.float32).reshape(-1, 2),
        spec=spec,
        crop_box=crop_box,
        placement=placement,
        output_shape=output_shape,
    )


def _line_segment_from_center_to_lower_side(segment: np.ndarray, center: np.ndarray) -> np.ndarray:
    pts = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float32)
    center_pt = np.asarray(center, dtype=np.float32).reshape(2)
    endpoints = np.asarray([pts[0], pts[-1]], dtype=np.float32)
    scores = (endpoints[:, 1] - center_pt[1]) - 0.35 * (endpoints[:, 0] - center_pt[0])
    endpoint = endpoints[int(np.argmax(scores))]
    if float(np.linalg.norm(endpoint - center_pt)) < 1.0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([center_pt, endpoint], dtype=np.float32)


def _finite_mask_boundary(mask: np.ndarray, *, placement: tuple[int, int]) -> np.ndarray:
    finite = np.asarray(mask, dtype=bool)
    if finite.ndim != 2 or not bool(np.any(finite)):
        return np.empty((0, 2), dtype=np.float32)
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for y_index in range(finite.shape[0]):
        xs = np.flatnonzero(finite[y_index])
        if xs.size == 0:
            continue
        left.append((float(placement[0] + int(xs.min())), float(placement[1] + y_index)))
        right.append((float(placement[0] + int(xs.max())), float(placement[1] + y_index)))
    top: list[tuple[float, float]] = []
    bottom: list[tuple[float, float]] = []
    for x_index in range(finite.shape[1]):
        ys = np.flatnonzero(finite[:, x_index])
        if ys.size == 0:
            continue
        top.append((float(placement[0] + x_index), float(placement[1] + int(ys.min()))))
        bottom.append((float(placement[0] + x_index), float(placement[1] + int(ys.max()))))
    points = top + right + list(reversed(bottom)) + list(reversed(left))
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float32)
    points.append(points[0])
    return np.asarray(points, dtype=np.float32)


def _left_sector_boundary_line(mask: np.ndarray, *, placement: tuple[int, int]) -> np.ndarray:
    finite = np.asarray(mask, dtype=bool)
    points: list[tuple[float, float]] = []
    for y_index in range(finite.shape[0]):
        xs = np.flatnonzero(finite[y_index])
        if xs.size == 0:
            continue
        points.append((float(placement[0] + int(xs.min())), float(placement[1] + y_index)))
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _line_segments_inside_mask(
    line: np.ndarray,
    mask: np.ndarray,
    *,
    placement: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return ()
    p0 = pts[0]
    p1 = pts[-1]
    length = float(np.linalg.norm(p1 - p0))
    samples = max(8, int(np.ceil(length * 2.0)))
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    sampled = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    local_x = np.rint(sampled[:, 0] - float(placement[0])).astype(np.int64)
    local_y = np.rint(sampled[:, 1] - float(placement[1])).astype(np.int64)
    inside = (
        (local_x >= 0)
        & (local_x < mask.shape[1])
        & (local_y >= 0)
        & (local_y < mask.shape[0])
        & mask[np.clip(local_y, 0, mask.shape[0] - 1), np.clip(local_x, 0, mask.shape[1] - 1)]
    )
    segments: list[np.ndarray] = []
    start: int | None = None
    for idx, value in enumerate(inside):
        if bool(value) and start is None:
            start = idx
        if (not bool(value) or idx == inside.size - 1) and start is not None:
            stop = idx if not bool(value) else idx + 1
            if stop - start >= 2:
                segments.append(np.asarray([sampled[start], sampled[stop - 1]], dtype=np.float32))
            start = None
    return tuple(segments)


def _clip_line_to_rect(
    line: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray | None:
    pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return None
    p0 = pts[0]
    p1 = pts[-1]
    direction = p1 - p0
    t0 = 0.0
    t1 = 1.0
    for edge_p, edge_q in (
        (-direction[0], p0[0] - x_min),
        (direction[0], x_max - p0[0]),
        (-direction[1], p0[1] - y_min),
        (direction[1], y_max - p0[1]),
    ):
        if abs(float(edge_p)) < 1e-12:
            if edge_q < 0.0:
                return None
            continue
        ratio = edge_q / edge_p
        if edge_p < 0.0:
            if ratio > t1:
                return None
            t0 = max(t0, float(ratio))
        else:
            if ratio < t0:
                return None
            t1 = min(t1, float(ratio))
    clipped = np.asarray([p0 + t0 * direction, p0 + t1 * direction], dtype=np.float32)
    return clipped


def _short_axis_cut_line(spec: dict[str, object], angle_deg: float) -> np.ndarray | None:
    x_min = float(spec["x_min"])
    x_max = float(spec["x_max"])
    y_min = float(spec["y_min"])
    y_max = float(spec["y_max"])
    direction = np.asarray([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))], dtype=np.float64)
    candidates: list[float] = []
    if abs(float(direction[0])) > 1e-8:
        candidates.extend([x_min / direction[0], x_max / direction[0]])
    if abs(float(direction[1])) > 1e-8:
        candidates.extend([y_min / direction[1], y_max / direction[1]])
    points: list[np.ndarray] = []
    for value in candidates:
        point = value * direction
        if x_min - 1e-8 <= point[0] <= x_max + 1e-8 and y_min - 1e-8 <= point[1] <= y_max + 1e-8:
            points.append(point)
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda item: float(item[0] * direction[0] + item[1] * direction[1]))
    return np.asarray([ordered[0], ordered[-1]], dtype=np.float32)


def _long_axis_depth_line(spec: dict[str, object], depth_fraction: float) -> np.ndarray:
    y_min = float(spec["y_min"])
    y_max = float(spec["y_max"])
    y = y_min + float(depth_fraction) * (y_max - y_min)
    return np.asarray([[float(spec["x_min"]), y], [0.0, y]], dtype=np.float32)


def _panel_plane(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spec["kind"] == "long_axis":
        angle_rad = np.deg2rad(float(spec["angle_deg"]))
        basis_u = np.asarray([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=np.float64)
        basis_v = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        normal = np.cross(basis_u, basis_v)
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        return np.zeros(3, dtype=np.float64), normal, basis_u, basis_v
    return (
        np.asarray([0.0, 0.0, float(spec["depth_m"])], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )


def _radial_axis_half_width_cm(geometry: object, angle_deg: float) -> float:
    depth_end_cm = 100.0 * float(getattr(geometry, "depth_end_m", 0.0))
    az_half = depth_end_cm * np.tan(0.5 * float(getattr(geometry, "azimuth_width_rad", 0.0)))
    el_half = depth_end_cm * np.tan(0.5 * float(getattr(geometry, "elevation_width_rad", 0.0)))
    angle_rad = np.deg2rad(float(angle_deg))
    return float(az_half * abs(np.cos(angle_rad)) + el_half * abs(np.sin(angle_rad)))


def _compact_mosaic(
    data: np.ndarray,
    polygons: tuple[tuple[np.ndarray, ...], ...],
    *,
    frame_index: int,
    geometry: object,
) -> tuple[np.ndarray, tuple[tuple[np.ndarray, ...], ...], tuple[tuple[dict[str, float | bool], ...], ...]]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 3:
        return arr, polygons, ()
    source_rows, source_cols = SOURCE_MOSAIC_ROWS, SOURCE_MOSAIC_COLS
    display_rows, display_cols = DISPLAY_MOSAIC_ROWS, DISPLAY_MOSAIC_COLS
    cell_h = max(1, int(arr.shape[1]) // source_rows)
    cell_w = max(1, int(arr.shape[2]) // source_cols)
    frame = arr[min(max(0, int(frame_index)), int(arr.shape[0]) - 1)]
    depth_start_cm = 100.0 * float(getattr(geometry, "depth_start_m", 0.0))
    depth_end_cm = 100.0 * float(getattr(geometry, "depth_end_m", 0.0))
    depth_span_cm = max(depth_end_cm - depth_start_cm, 1.0)
    px_per_cm = max(float(cell_h) / depth_span_cm, 1.0)
    sector_crop_px = int(round(SECTOR_TOP_CROP_CM / depth_span_cm * max(cell_h - 1, 1)))
    boxes: list[tuple[int, int, int, int]] = []
    finite_masks: list[np.ndarray] = []
    for panel_idx in range(source_rows * source_cols):
        row = panel_idx // source_cols
        col = panel_idx % source_cols
        tile = frame[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
        finite = np.isfinite(tile)
        finite_masks.append(finite)
        if not bool(np.any(finite)):
            boxes.append((0, cell_w, 0, cell_h))
            continue
        yy, xx = np.nonzero(finite)
        x0 = max(0, int(xx.min()) - MOSAIC_PAD_PX)
        x1 = min(cell_w, int(xx.max()) + 1 + MOSAIC_PAD_PX)
        y0 = max(0, int(yy.min()) - MOSAIC_PAD_PX)
        y1 = min(cell_h, int(yy.max()) + 1 + MOSAIC_PAD_PX)
        if col == 0:
            y0 = min(max(y0, sector_crop_px), max(0, y1 - 2))
        boxes.append((x0, x1, y0, y1))

    display_order = tuple(int(panel_idx) for panel_idx in DISPLAY_PANEL_ORDER)
    display_sizes = {
        panel_idx: _display_panel_size_px(
            boxes[panel_idx],
            panel_idx,
            geometry=geometry,
            px_per_cm=px_per_cm,
            source_cell_shape=(cell_h, cell_w),
        )
        for panel_idx in display_order
    }
    tile_widths = [
        int(display_sizes[panel_idx][1]) + (SECTOR_RULER_LEFT_PAD_PX if _is_sector_panel(panel_idx) else 0)
        for panel_idx in display_order
    ]
    tile_heights = [int(display_sizes[panel_idx][0]) for panel_idx in display_order]
    row_heights = [max(tile_heights[row * display_cols : (row + 1) * display_cols]) for row in range(display_rows)]
    col_widths = [max(tile_widths[col::display_cols]) for col in range(display_cols)]
    output_h = int(sum(row_heights) + MOSAIC_GAP_PX * (display_rows - 1))
    output_w = int(sum(col_widths) + MOSAIC_GAP_PX * (display_cols - 1))
    compact = np.full((arr.shape[0], output_h, output_w), np.nan, dtype=np.float32)

    placements: dict[int, tuple[int, int]] = {}
    y_cursor = 0
    for display_row in range(display_rows):
        x_cursor = 0
        for display_col in range(display_cols):
            panel_idx = display_order[display_row * display_cols + display_col]
            x0, x1, y0, y1 = boxes[panel_idx]
            tile_h, tile_w = display_sizes[panel_idx]
            ruler_pad = SECTOR_RULER_LEFT_PAD_PX if _is_sector_panel(panel_idx) else 0
            tile_and_ruler_w = tile_w + ruler_pad
            x_dst = x_cursor + (col_widths[display_col] - tile_and_ruler_w) // 2 + ruler_pad
            y_dst = y_cursor + row_heights[display_row] - tile_h
            source_row = panel_idx // source_cols
            source_col = panel_idx % source_cols
            src_y0 = source_row * cell_h + y0
            src_y1 = source_row * cell_h + y1
            src_x0 = source_col * cell_w + x0
            src_x1 = source_col * cell_w + x1
            tile = _resize_stack_nearest(arr[:, src_y0:src_y1, src_x0:src_x1], output_shape=(tile_h, tile_w))
            compact[:, y_dst : y_dst + tile_h, x_dst : x_dst + tile_w] = tile
            placements[panel_idx] = (x_dst, y_dst)
            x_cursor += col_widths[display_col] + MOSAIC_GAP_PX
        y_cursor += row_heights[display_row] + MOSAIC_GAP_PX

    compact_polygons: list[tuple[np.ndarray, ...]] = []
    selected_frame_polygons = polygons[min(max(0, int(frame_index)), len(polygons) - 1)] if polygons else ()
    remapped: list[np.ndarray] = []
    for polygon in selected_frame_polygons:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        centroid = np.nanmean(pts, axis=0)
        col = int(np.clip(int(centroid[0] // cell_w), 0, source_cols - 1))
        row = int(np.clip(int(centroid[1] // cell_h), 0, source_rows - 1))
        panel_idx = row * source_cols + col
        if panel_idx not in placements:
            continue
        x0, x1, y0, y1 = boxes[panel_idx]
        x_dst, y_dst = placements[panel_idx]
        tile_h, tile_w = display_sizes[panel_idx]
        scale_x = float(tile_w) / max(float(x1 - x0), 1.0)
        scale_y = float(tile_h) / max(float(y1 - y0), 1.0)
        mapped = pts.copy()
        mapped[:, 0] = (mapped[:, 0] - col * cell_w - x0) * scale_x + x_dst
        mapped[:, 1] = (mapped[:, 1] - row * cell_h - y0) * scale_y + y_dst
        remapped.append(mapped)
    compact_polygons.append(tuple(remapped))

    rulers: list[tuple[dict[str, float | bool], ...]] = []
    for panel_idx in display_order[:display_cols]:
        if not _is_sector_panel(panel_idx):
            continue
        x0, _x1, y0, y1 = boxes[panel_idx]
        x_dst, y_dst = placements[panel_idx]
        finite = finite_masks[panel_idx]
        first_cm = max(
            np.ceil(depth_start_cm + y0 / max(cell_h - 1, 1) * depth_span_cm),
            np.ceil(depth_start_cm + SECTOR_TOP_CROP_CM),
        )
        last_cm = np.floor(depth_end_cm)
        row_ticks: list[dict[str, float | bool]] = []
        for depth_cm in np.arange(first_cm, last_cm + 0.5, 1.0, dtype=np.float64):
            y_panel = (float(depth_cm) - depth_start_cm) * px_per_cm
            tile_h, _tile_w = display_sizes[panel_idx]
            scale_y = float(tile_h) / max(float(y1 - y0), 1.0)
            y = y_dst + (y_panel - y0) * scale_y
            if y < y_dst - 0.5 or y > y_dst + tile_h + 0.5:
                continue
            is_major = bool(np.isclose(float(depth_cm) / 5.0, round(float(depth_cm) / 5.0), atol=1e-6))
            x_base = x_dst - SECTOR_RULER_GAP_PX
            row_ticks.append(
                {
                    "x0": float(x_base),
                    "x1": float(x_base - (SECTOR_RULER_MAJOR_TICK_PX if is_major else SECTOR_RULER_MINOR_TICK_PX)),
                    "y": float(y),
                    "depth_cm": float(depth_cm),
                    "is_major": is_major,
                }
            )
        rulers.append(tuple(row_ticks))
    return compact, tuple(compact_polygons), tuple(rulers)


def _is_sector_panel(panel_idx: int) -> bool:
    return int(panel_idx) % SOURCE_MOSAIC_COLS == 0


def _display_panel_size_px(
    box: tuple[int, int, int, int],
    panel_idx: int,
    *,
    geometry: object,
    px_per_cm: float,
    source_cell_shape: tuple[int, int],
) -> tuple[int, int]:
    x0, x1, y0, y1 = box
    source_h, source_w = source_cell_shape
    if _is_sector_panel(panel_idx):
        return max(2, y1 - y0), max(2, x1 - x0)

    depth_end_cm = 100.0 * float(getattr(geometry, "depth_end_m", 0.0))
    azimuth_width = float(getattr(geometry, "azimuth_width_rad", 0.0))
    elevation_width = float(getattr(geometry, "elevation_width_rad", 0.0))
    full_width_cm = max(0.1, 2.0 * depth_end_cm * np.tan(0.5 * azimuth_width))
    full_height_cm = max(0.1, 2.0 * depth_end_cm * np.tan(0.5 * elevation_width))
    width_cm = full_width_cm * float(x1 - x0) / max(float(source_w), 1.0)
    height_cm = full_height_cm * float(y1 - y0) / max(float(source_h), 1.0)
    return max(2, int(round(height_cm * px_per_cm))), max(2, int(round(width_cm * px_per_cm)))


def _resize_stack_nearest(frames: np.ndarray, *, output_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    height, width = int(output_shape[0]), int(output_shape[1])
    if arr.shape[1:3] == (height, width):
        return arr
    row_idx = _resample_indices(arr.shape[1], height)
    col_idx = _resample_indices(arr.shape[2], width)
    return arr[:, row_idx][:, :, col_idx]


def _resample_indices(source_size: int, target_size: int) -> np.ndarray:
    if target_size <= 1:
        return np.zeros((1,), dtype=np.int64)
    return np.clip(np.rint(np.linspace(0, max(0, source_size - 1), target_size)), 0, max(0, source_size - 1)).astype(
        np.int64
    )


def _draw_compact_cut_lines(ax: object, image_panel: PanelSpec) -> None:
    lines = image_panel.loaded.attrs.get("compact_cut_lines")
    if not isinstance(lines, tuple):
        return
    for line in lines:
        if not isinstance(line, dict):
            continue
        xy = np.asarray(line.get("xy"), dtype=np.float32)
        if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] != 2:
            continue
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=str(line.get("color", AXIS_COLOR)),
            linewidth=SIDE_LINE_WIDTH if line.get("kind") == "side" else CUT_LINE_WIDTH,
            linestyle="-" if line.get("kind") == "side" else (0, (3.0, 2.4)),
            alpha=CUT_LINE_ALPHA,
            solid_capstyle="round",
            zorder=9.6,
            clip_on=False,
        )


def _draw_compact_sector_depth_rulers(ax: object, image_panel: PanelSpec, *, color: str) -> None:
    rulers = image_panel.loaded.attrs.get("compact_sector_depth_rulers")
    if not isinstance(rulers, tuple):
        return
    for row_ticks in rulers:
        if not isinstance(row_ticks, tuple):
            continue
        for tick in row_ticks:
            if not isinstance(tick, dict):
                continue
            x0 = float(tick["x0"])
            x1 = float(tick["x1"])
            y = float(tick["y"])
            is_major = bool(tick["is_major"])
            ax.plot(
                [x0, x1],
                [y, y],
                color=color,
                linewidth=0.7 if is_major else 0.55,
                solid_capstyle="butt",
                zorder=9.5,
                clip_on=False,
            )
            if is_major:
                ax.text(
                    x1 - 2.0,
                    y,
                    f"{int(round(float(tick['depth_cm'])))}",
                    color=color,
                    fontsize=5.8,
                    ha="right",
                    va="center",
                    zorder=9.6,
                    clip_on=False,
                )


def _closed_surface_volume_m3(points: np.ndarray, faces: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    idx = np.asarray(faces, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] < 3 or idx.ndim != 2 or idx.shape[1] < 3:
        return float("nan")
    triangles = idx[:, :3]
    if idx.shape[1] == 4:
        triangles = np.concatenate([triangles, idx[:, [0, 2, 3]]], axis=0)
    valid = np.all((triangles >= 0) & (triangles < pts.shape[0]), axis=1)
    if not bool(np.any(valid)):
        return float("nan")
    tri_points = pts[triangles[valid], :3]
    signed = np.einsum("ij,ij->i", tri_points[:, 0], np.cross(tri_points[:, 1], tri_points[:, 2]))
    return abs(float(np.sum(signed)) / 6.0)


if __name__ == "__main__":
    raise SystemExit(main())
