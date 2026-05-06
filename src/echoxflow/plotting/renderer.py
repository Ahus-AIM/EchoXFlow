"""High-level Matplotlib renderer for EchoXFlow recordings."""

# flake8: noqa: E402

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from echoxflow.croissant import RecordingRecord
from echoxflow.loading import LoadedArray, PackedMeshAnnotation, RecordingStore, open_recording
from echoxflow.objects import AnnotationRef, RecordingObject, StrainPanel
from echoxflow.plotting.annotations import attach_annotation_overlays, mesh_mosaic_annotation_lines
from echoxflow.plotting.clinical import clinical_loaded_arrays
from echoxflow.plotting.layout import spatial_layout, uses_ecg_timescale
from echoxflow.plotting.panels import render_ecg, renderer_for
from echoxflow.plotting.specs import (
    FrameRequest,
    PanelKind,
    PanelSpec,
    PanelView,
    PlotViewMode,
    RenderedFrame,
    TraceSpec,
)
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import nearest_index, resolve_frame_time, select_timeline, temporal_length
from echoxflow.plotting.writers import figure_to_rgb, save_figure, write_video
from echoxflow.scan import (
    clinical_spherical_mosaic,
    preconverted_spherical_mosaic,
    prepare_3d_brightness_for_display,
    spherical_geometry_from_metadata,
)
from echoxflow.scan.geometry import SectorGeometry


@dataclass(frozen=True)
class _StrainImageInput:
    panel: StrainPanel
    loaded: LoadedArray
    contour_overlays: tuple[dict[str, object], ...]
    qrs_trigger_times: np.ndarray


class RecordingPlotRenderer:
    """Render EchoXFlow recordings into Matplotlib figures, RGB frames, and videos."""

    def __init__(self, style: PlotStyle | None = None) -> None:
        self.style = style or PlotStyle.from_config()

    def plot_recording(
        self,
        record: RecordingRecord | str | Path,
        *,
        root: str | Path | None = None,
        modalities: tuple[str, ...] | None = None,
        time_s: float | None = None,
        frame_index: int | None = None,
        view_mode: PlotViewMode | str = "pre_converted",
        show_annotations: bool = True,
    ) -> Figure:
        panels, ecg = self._load_specs(
            record, root=root, modalities=modalities, view_mode=view_mode, show_annotations=show_annotations
        )
        request = FrameRequest(time_s=time_s, frame_index=frame_index)
        current_time = resolve_frame_time(panels, request)
        return self.render_figure_from_specs(
            panels=panels,
            ecg=ecg,
            time_s=current_time,
            frame_index=0 if frame_index is None else int(frame_index),
            dpi=self.style.dpi,
        )

    def render_frame(
        self,
        record: RecordingRecord | str | Path,
        *,
        root: str | Path | None = None,
        modalities: tuple[str, ...] | None = None,
        time_s: float | None = None,
        frame_index: int | None = None,
        view_mode: PlotViewMode | str = "pre_converted",
        show_annotations: bool = True,
    ) -> RenderedFrame:
        figure = self.plot_recording(
            record,
            root=root,
            modalities=modalities,
            time_s=time_s,
            frame_index=frame_index,
            view_mode=view_mode,
            show_annotations=show_annotations,
        )
        try:
            current_time = resolve_frame_time(
                self._load_specs(
                    record,
                    root=root,
                    modalities=modalities,
                    view_mode=view_mode,
                    show_annotations=show_annotations,
                )[0],
                FrameRequest(time_s=time_s, frame_index=frame_index),
            )
            return RenderedFrame(image=figure_to_rgb(figure), time_s=current_time)
        finally:
            plt.close(figure)

    def save_plot(
        self,
        record: RecordingRecord | str | Path,
        output: str | Path,
        *,
        root: str | Path | None = None,
        modalities: tuple[str, ...] | None = None,
        time_s: float | None = None,
        frame_index: int | None = None,
        view_mode: PlotViewMode | str = "pre_converted",
        show_annotations: bool = True,
        dpi: int = 200,
    ) -> Path:
        figure = self.plot_recording(
            record,
            root=root,
            modalities=modalities,
            time_s=time_s,
            frame_index=frame_index,
            view_mode=view_mode,
            show_annotations=show_annotations,
        )
        try:
            return save_figure(figure, output, dpi=dpi)
        finally:
            plt.close(figure)

    def render_video(
        self,
        record: RecordingRecord | str | Path,
        output: str | Path,
        *,
        root: str | Path | None = None,
        modalities: tuple[str, ...] | None = None,
        view_mode: PlotViewMode | str = "pre_converted",
        show_annotations: bool = True,
        max_fps: float | None = None,
        dpi: int = 150,
    ) -> Path:
        panels, ecg = self._load_specs(
            record, root=root, modalities=modalities, view_mode=view_mode, show_annotations=show_annotations
        )
        timeline = select_timeline(panels, max_fps=self.style.max_fps if max_fps is None else float(max_fps))
        frames: list[np.ndarray] = []
        for frame_index, current_time in enumerate(np.asarray(timeline.timestamps, dtype=np.float64).reshape(-1)):
            figure = self.render_figure_from_specs(
                panels=panels,
                ecg=ecg,
                time_s=float(current_time),
                frame_index=frame_index,
                dpi=dpi,
            )
            try:
                frames.append(figure_to_rgb(figure))
            finally:
                plt.close(figure)
        source = record.path(root) if isinstance(record, RecordingRecord) else Path(record).expanduser()
        return write_video(output, np.stack(frames, axis=0), fps=timeline.fps, source=source)

    def render_figure_from_specs(
        self,
        *,
        panels: tuple[PanelSpec, ...],
        ecg: TraceSpec,
        time_s: float,
        frame_index: int,
        dpi: int,
    ) -> Figure:
        if not panels:
            raise ValueError("At least one non-ECG modality panel is required")
        spatial_panels = tuple(panel for panel in panels if not uses_ecg_timescale(panel))
        temporal_panels = tuple(panel for panel in panels if uses_ecg_timescale(panel))
        preserve_single_preconverted_image_aspect = _preserve_single_preconverted_image_aspect(panels)
        layout = spatial_layout(spatial_panels)
        cols = layout.cols
        spatial_rows = layout.rows
        temporal_rows = len(temporal_panels)
        show_ecg = bool(self.style.show_ecg)
        total_rows = spatial_rows + temporal_rows + (1 if show_ecg else 0)
        figure = plt.figure(figsize=self.style.figsize, dpi=int(dpi), facecolor=self.style.figure_facecolor)
        grid = GridSpec(
            total_rows,
            cols,
            figure=figure,
            height_ratios=([1.0] * spatial_rows) + ([0.34] * temporal_rows) + ([0.28] if show_ecg else []),
            width_ratios=layout.width_ratios,
        )
        occupied: set[tuple[int, int]] = set()
        for placement in layout.placements:
            row_end = placement.row + placement.row_span
            col_end = placement.col + placement.col_span
            ax = figure.add_subplot(grid[placement.row : row_end, placement.col : col_end])
            panel = placement.panel
            renderer_for(panel.kind).render(ax, panel, time_s=time_s, frame_index=frame_index, style=self.style)
            if preserve_single_preconverted_image_aspect and panel is panels[0]:
                ax.set_aspect("equal", adjustable="box")
            for row in range(placement.row, row_end):
                for col in range(placement.col, col_end):
                    occupied.add((row, col))
        for row in range(spatial_rows):
            for col in range(cols):
                if (row, col) not in occupied:
                    figure.add_subplot(grid[row, col]).set_axis_off()
        for offset, panel in enumerate(temporal_panels):
            ax = figure.add_subplot(grid[spatial_rows + offset, :])
            renderer_for(panel.kind).render(ax, panel, time_s=time_s, frame_index=frame_index, style=self.style)
            _set_ecg_xlim(ax, ecg)
        if show_ecg:
            ecg_ax = figure.add_subplot(grid[total_rows - 1, :])
            render_ecg(ecg_ax, ecg, time_s=time_s, style=self.style)
            _set_ecg_xlim(ecg_ax, ecg)
        figure.tight_layout(pad=0.8)
        return figure

    def build_panel_specs(
        self, loaded_arrays: tuple[LoadedArray, ...], *, view_mode: PlotViewMode | str = "pre_converted"
    ) -> tuple[PanelSpec, ...]:
        """Build display panel specs from loaded modality arrays."""
        return _panel_specs(loaded_arrays, view_mode=_normalize_view_mode(view_mode))

    def _load_specs(
        self,
        record: RecordingRecord | str | Path,
        *,
        root: str | Path | None,
        modalities: tuple[str, ...] | None,
        view_mode: PlotViewMode | str,
        show_annotations: bool,
    ) -> tuple[tuple[PanelSpec, ...], TraceSpec]:
        store = open_recording(record, root=root)
        obj = store.load_object()
        if _should_plot_strain_object(obj, store, modalities):
            return _strain_specs(
                store, obj, modalities=modalities, view_mode=view_mode, show_annotations=show_annotations
            )
        paths = _resolve_modality_paths(store, modalities)
        loaded_arrays = tuple(store.load_modality(path) for path in paths)
        if show_annotations:
            loaded_arrays = attach_annotation_overlays(store, loaded_arrays)
        panels = self.build_panel_specs(loaded_arrays, view_mode=view_mode)
        ecg = _ecg_trace(store)
        return panels, ecg


def _resolve_modality_paths(store: RecordingStore, modalities: tuple[str, ...] | None) -> tuple[str, ...]:
    available = tuple(path for path in store.array_paths if path.startswith("data/"))
    if modalities is None:
        return tuple(path for path in available if _is_auto_modality(path))
    resolved: list[str] = []
    for modality in modalities:
        expanded = _expand_modality(modality, available)
        if not expanded:
            raise KeyError(f"Requested modality `{modality}` is not available in {store.path}")
        resolved.extend(expanded)
    return tuple(dict.fromkeys(resolved))


def _preserve_single_preconverted_image_aspect(panels: tuple[PanelSpec, ...]) -> bool:
    return len(panels) == 1 and panels[0].view == "pre_converted" and panels[0].kind == "image"


def _should_plot_strain_object(
    obj: RecordingObject,
    store: RecordingStore,
    modalities: tuple[str, ...] | None,
) -> bool:
    if obj.kind != "strain" or not obj.panels:
        return False
    if modalities is None:
        return True
    available = tuple(path for path in store.array_paths if path.startswith("data/"))
    for modality in modalities:
        token = str(modality).strip().lower().strip("/")
        if token in {"strain", "strain_measurement", "2d_strain"}:
            return True
        path = _normalize_data_path(token)
        if path in available and _is_strain_array_path(path):
            return True
        if any(token == panel.role_id.lower() for panel in obj.panels):
            return True
    return False


def _strain_specs(
    store: RecordingStore,
    obj: RecordingObject,
    *,
    modalities: tuple[str, ...] | None,
    view_mode: PlotViewMode | str,
    show_annotations: bool,
) -> tuple[tuple[PanelSpec, ...], TraceSpec]:
    selected_roles = _selected_strain_roles(obj, modalities)
    selected_panels = tuple(panel for panel in obj.panels if selected_roles is None or panel.role_id in selected_roles)
    if not selected_panels:
        raise KeyError(f"Requested strain modalities are not available in {store.path}")
    normalized_view = _normalize_view_mode(view_mode)
    image_view: PanelView = "clinical" if normalized_view == "clinical" else "pre_converted"
    image_inputs = _align_strain_image_inputs(tuple(_strain_image_input(store, panel) for panel in selected_panels))
    image_specs = [
        _strain_image_spec(image_input, view=image_view, show_annotations=show_annotations)
        for image_input in image_inputs
    ]
    curve_specs = list(_strain_curve_specs(store, obj, selected_panels, modalities))
    return (*image_specs, *curve_specs), _strain_ecg_trace(store, selected_panels)


def _selected_strain_roles(obj: RecordingObject, modalities: tuple[str, ...] | None) -> set[str] | None:
    if modalities is None:
        return None
    roles = {panel.role_id for panel in obj.panels}
    selected: set[str] = set()
    for modality in modalities:
        token = str(modality).strip().lower().strip("/")
        if token in {"strain", "strain_measurement", "2d_strain"}:
            return None
        name = token.removeprefix("data/")
        role = name.split("_", 1)[0]
        if role in roles:
            selected.add(role)
        elif token in roles:
            selected.add(token)
    return selected or None


def _strain_image_input(store: RecordingStore, panel: StrainPanel) -> _StrainImageInput:
    try:
        source_store = store.open_reference(panel.bmode.recording)
        loaded = source_store.load_modality(panel.bmode.data_path)
    except Exception as exc:
        recording_id = panel.bmode.recording.recording_id or panel.bmode.recording.zarr_path or "<unknown>"
        raise ValueError(
            f"Could not load linked B-mode for strain panel `{panel.role_id}` from recording `{recording_id}`"
        ) from exc
    loaded = _with_explicit_timestamps(source_store, loaded, panel.bmode.timestamps_path)
    contour_overlays = _strain_contour_overlays(store, panel)
    qrs_trigger_times = _strain_qrs_trigger_times(store, source_store, panel)
    return _StrainImageInput(
        panel=panel,
        loaded=loaded,
        contour_overlays=contour_overlays,
        qrs_trigger_times=qrs_trigger_times,
    )


def _strain_image_spec(
    image_input: _StrainImageInput,
    *,
    view: PanelView,
    show_annotations: bool,
) -> PanelSpec:
    panel = image_input.panel
    loaded = image_input.loaded
    contour_overlays = image_input.contour_overlays
    attrs = {
        **dict(loaded.attrs),
        "strain_role_id": panel.role_id,
        "strain_source_recording_id": panel.bmode.recording.recording_id,
    }
    geometry = _strain_panel_geometry(panel)
    if geometry is not None:
        attrs["annotation_geometry"] = geometry
    if show_annotations and contour_overlays:
        attrs["annotation_overlays"] = (*tuple(attrs.get("annotation_overlays", ())), *contour_overlays)
    loaded = replace(loaded, attrs=attrs)
    if view == "clinical":
        if _loaded_has_sector_geometry(loaded):
            loaded = clinical_loaded_arrays((loaded,))[0]
        else:
            view = "pre_converted"
    return PanelSpec(
        loaded=loaded,
        kind="image",
        label=f"{panel.role_id} b-mode",
        view=view,
    )


def _strain_curve_specs(
    store: RecordingStore,
    obj: RecordingObject,
    selected_panels: tuple[StrainPanel, ...],
    modalities: tuple[str, ...] | None,
) -> tuple[PanelSpec, ...]:
    include_all = _include_all_strain_curves(modalities)
    selected_roles = {panel.role_id for panel in selected_panels}
    requested_paths = _requested_curve_paths(modalities)
    specs: list[PanelSpec] = []
    seen: set[str] = set()
    for annotation in obj.annotations:
        if not _is_curve_annotation(annotation):
            continue
        data_path = _normalize_data_path(annotation.value.path)
        if data_path in seen or not _should_include_strain_curve(
            data_path,
            annotation.target_semantic_id,
            include_all=include_all,
            selected_roles=selected_roles,
            requested_paths=requested_paths,
        ):
            continue
        spec = _strain_curve_spec(store, data_path, role_id=annotation.target_semantic_id, annotation=annotation)
        if spec is not None:
            specs.append(spec)
            seen.add(data_path)
    for data_path in _strain_curve_data_paths(store):
        if data_path in seen or not _should_include_strain_curve(
            data_path,
            _role_from_curve_path(data_path),
            include_all=include_all,
            selected_roles=selected_roles,
            requested_paths=requested_paths,
        ):
            continue
        spec = _strain_curve_spec(store, data_path, role_id=_role_from_curve_path(data_path), annotation=None)
        if spec is not None:
            specs.append(spec)
            seen.add(data_path)
    return tuple(specs)


def _strain_curve_spec(
    store: RecordingStore,
    data_path: str,
    *,
    role_id: str,
    annotation: AnnotationRef | None,
) -> PanelSpec | None:
    data_path = _normalize_data_path(data_path)
    if data_path not in store.group:
        return None
    timestamps_path = None if annotation is None or annotation.time is None else annotation.time.path
    data = np.asarray(store.group[data_path][:])
    timestamps = None
    if timestamps_path is not None and timestamps_path in store.group:
        timestamps = np.asarray(store.group[timestamps_path][:])
    elif (fallback := store.timestamp_path(data_path)) is not None:
        timestamps_path = fallback
        timestamps = np.asarray(store.group[fallback][:])
    label_role = role_id or data_path.removeprefix("data/").removesuffix("_curve")
    loaded = LoadedArray(
        name=f"{label_role}_strain_curve",
        data_path=data_path,
        data=data,
        timestamps_path=timestamps_path,
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={"strain_curve": True, "strain_role_id": role_id},
        stream=None,
    )
    return PanelSpec(
        loaded=loaded,
        kind="line",
        label=f"{label_role} strain",
        view="pre_converted",
    )


def _strain_contour_overlays(store: RecordingStore, panel: StrainPanel) -> tuple[dict[str, object], ...]:
    overlays: list[dict[str, object]] = []
    for annotation in panel.annotations:
        if not _is_contour_annotation(annotation):
            continue
        if annotation.value.path not in store.group:
            continue
        points = np.asarray(store.group[annotation.value.path][:], dtype=np.float32)
        timestamps = None
        if annotation.time is not None and annotation.time.path in store.group:
            timestamps = np.asarray(store.group[annotation.time.path][:], dtype=np.float64)
        overlays.append(
            {
                "kind": "physical_points",
                "points": points,
                "timestamps": timestamps,
                **_annotation_label_metadata(annotation.raw),
                "field": annotation.field,
                "target": {
                    "type": annotation.target_entity or "linked_panel",
                    "semantic_id": panel.role_id,
                    "field": annotation.field,
                },
            }
        )
    return tuple(overlays)


def _annotation_label_metadata(raw: Mapping[str, Any] | None) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        return {}
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        return {}
    return {"label": label.strip()}


def _align_strain_image_inputs(image_inputs: tuple[_StrainImageInput, ...]) -> tuple[_StrainImageInput, ...]:
    qrs_windows = [_strain_panel_qrs_window(image_input) for image_input in image_inputs]
    canonical_duration = _strain_qrs_canonical_duration(qrs_windows)
    if canonical_duration is None:
        return tuple(
            replace(
                image_input,
                loaded=_trim_to_first_strain_frame(image_input.loaded, image_input.contour_overlays),
            )
            for image_input in image_inputs
        )
    aligned: list[_StrainImageInput] = []
    for image_input, qrs_window in zip(image_inputs, qrs_windows):
        if qrs_window is None:
            aligned.append(
                replace(
                    image_input,
                    loaded=_trim_to_first_strain_frame(image_input.loaded, image_input.contour_overlays),
                )
            )
            continue
        start, end = qrs_window
        scale = float(canonical_duration) / float(end - start)
        aligned.append(
            replace(
                image_input,
                loaded=_align_loaded_array_to_qrs_window(
                    image_input.loaded,
                    start=start,
                    end=end,
                    scale=scale,
                ),
                contour_overlays=_align_strain_overlays_to_qrs_window(
                    image_input.contour_overlays,
                    start=start,
                    end=end,
                    scale=scale,
                ),
            )
        )
    return tuple(aligned)


def _strain_qrs_canonical_duration(qrs_windows: list[tuple[float, float] | None]) -> float | None:
    durations = np.asarray(
        [float(window[1] - window[0]) for window in qrs_windows if window is not None],
        dtype=np.float64,
    )
    durations = durations[np.isfinite(durations) & (durations > 0.0)]
    if durations.size == 0:
        return None
    return float(np.median(durations))


def _strain_panel_qrs_window(image_input: _StrainImageInput) -> tuple[float, float] | None:
    qrs = np.asarray(image_input.qrs_trigger_times, dtype=np.float64).reshape(-1)
    qrs = qrs[np.isfinite(qrs)]
    qrs = np.unique(np.sort(qrs))
    if qrs.size < 2:
        return None
    point_timestamps = _strain_point_timestamps(image_input.contour_overlays)
    start_idx = 0
    if point_timestamps.size:
        point_start = float(point_timestamps[0])
        point_end = float(point_timestamps[-1])
        start_idx = min(
            range(qrs.size - 1),
            key=lambda idx: (
                -max(0.0, min(float(qrs[idx + 1]), point_end) - max(float(qrs[idx]), point_start)),
                abs(float(qrs[idx]) - point_start) + abs(float(qrs[idx + 1]) - point_end),
                idx,
            ),
        )
    start = float(qrs[start_idx])
    end = float(qrs[start_idx + 1])
    if not (np.isfinite(start) and np.isfinite(end) and end > start):
        return None
    return (start, end)


def _strain_point_timestamps(overlays: tuple[dict[str, object], ...]) -> np.ndarray:
    timestamp_sets: list[np.ndarray] = []
    for overlay in overlays:
        raw = overlay.get("timestamps")
        if raw is None:
            continue
        timestamps = np.asarray(raw, dtype=np.float64).reshape(-1)
        timestamps = timestamps[np.isfinite(timestamps)]
        if timestamps.size:
            timestamp_sets.append(timestamps)
    if not timestamp_sets:
        return np.array([], dtype=np.float64)
    return np.unique(np.sort(np.concatenate(timestamp_sets, axis=0))).astype(np.float64, copy=False)


def _align_loaded_array_to_qrs_window(
    loaded: LoadedArray,
    *,
    start: float,
    end: float,
    scale: float,
) -> LoadedArray:
    timestamps = loaded.timestamps
    if timestamps is None:
        return loaded
    data = np.asarray(loaded.data)
    count = temporal_length(data)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if count == 0 or ts.size != count:
        return loaded
    clipped_timestamps, clipped_data = _clip_sequence_to_time_window(
        timestamps=ts,
        values=data,
        start=start,
        end=end,
    )
    relative_timestamps = _relative_time_axis_with_origin(clipped_timestamps, origin=start, scale=scale)
    return replace(loaded, data=clipped_data, timestamps=relative_timestamps)


def _align_strain_overlays_to_qrs_window(
    overlays: tuple[dict[str, object], ...],
    *,
    start: float,
    end: float,
    scale: float,
) -> tuple[dict[str, object], ...]:
    aligned: list[dict[str, object]] = []
    for overlay in overlays:
        raw_timestamps = overlay.get("timestamps")
        if raw_timestamps is None:
            aligned.append(overlay)
            continue
        points = np.asarray(overlay.get("points"), dtype=np.float32)
        timestamps = np.asarray(raw_timestamps, dtype=np.float64).reshape(-1)
        if points.ndim < 3 or timestamps.size != int(points.shape[0]):
            aligned.append(
                {
                    **overlay,
                    "timestamps": _relative_time_axis_with_origin(timestamps, origin=start, scale=scale),
                }
            )
            continue
        clipped_timestamps, clipped_points = _clip_sequence_to_time_window(
            timestamps=timestamps,
            values=points,
            start=start,
            end=end,
        )
        aligned.append(
            {
                **overlay,
                "points": clipped_points.astype(np.float32, copy=False),
                "timestamps": _relative_time_axis_with_origin(clipped_timestamps, origin=start, scale=scale),
            }
        )
    return tuple(aligned)


def _relative_time_axis_with_origin(
    values: Any,
    *,
    origin: float | None,
    scale: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    if origin is None:
        relative = arr - float(arr[0])
    else:
        relative = arr - float(origin)
    return (relative * float(scale)).astype(np.float64, copy=False)


def _clip_sequence_to_time_window(
    *,
    timestamps: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    vals = np.asarray(values)
    n = min(ts.shape[0], int(vals.shape[0])) if vals.ndim >= 1 else 0
    if n == 0:
        return np.array([], dtype=np.float64), vals[:0]
    ts = ts[:n]
    vals = vals[:n]
    finite_mask = np.isfinite(ts)
    ts = ts[finite_mask]
    vals = vals[finite_mask]
    if ts.size == 0:
        return np.array([], dtype=np.float64), vals[:0]
    window_mask = (ts >= float(start)) & (ts <= float(end))
    if np.any(window_mask):
        clipped_ts = ts[window_mask]
        clipped_vals = vals[window_mask]
    else:
        clipped_ts = np.array([], dtype=np.float64)
        clipped_vals = vals[:0]
    if clipped_ts.size == 0 or float(clipped_ts[0]) > float(start):
        nearest_idx = nearest_index(ts, float(start), count=ts.size)
        clipped_ts = np.concatenate([np.array([float(start)], dtype=np.float64), clipped_ts], axis=0)
        clipped_vals = np.concatenate([vals[nearest_idx : nearest_idx + 1], clipped_vals], axis=0)
    if float(clipped_ts[-1]) < float(end):
        nearest_idx = nearest_index(ts, float(end), count=ts.size)
        clipped_ts = np.concatenate([clipped_ts, np.array([float(end)], dtype=np.float64)], axis=0)
        clipped_vals = np.concatenate([clipped_vals, vals[nearest_idx : nearest_idx + 1]], axis=0)
    return clipped_ts.astype(np.float64, copy=False), clipped_vals


def _trim_to_first_strain_frame(loaded: LoadedArray, overlays: tuple[dict[str, object], ...]) -> LoadedArray:
    annotation_time, annotation_index = _first_defined_strain_sample(overlays)
    if annotation_time is None and annotation_index is None:
        return loaded
    data = np.asarray(loaded.data)
    count = temporal_length(data)
    if count <= 1:
        return loaded
    start_index = 0
    if annotation_time is not None and loaded.timestamps is not None:
        ts = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
        if ts.size == count and ts.size > 0:
            finite_after = np.flatnonzero(np.isfinite(ts) & (ts >= float(annotation_time)))
            start_index = int(finite_after[0]) if finite_after.size else nearest_index(ts, annotation_time, count=count)
    elif annotation_index is not None:
        start_index = int(np.clip(annotation_index, 0, count - 1))
    if start_index <= 0:
        return loaded
    timestamps = loaded.timestamps
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.reshape(-1).size == count:
            timestamps = ts.reshape(-1)[start_index:]
    return replace(loaded, data=data[start_index:], timestamps=timestamps)


def _first_defined_strain_sample(overlays: tuple[dict[str, object], ...]) -> tuple[float | None, int | None]:
    timed: list[float] = []
    indexed: list[int] = []
    for overlay in overlays:
        points = np.asarray(overlay.get("points"), dtype=np.float32)
        frame_index = _first_defined_points_index(points)
        if frame_index is None:
            continue
        timestamps = overlay.get("timestamps")
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if 0 <= frame_index < ts.size and np.isfinite(ts[frame_index]):
                timed.append(float(ts[frame_index]))
                continue
        indexed.append(frame_index)
    if timed:
        return min(timed), None
    if indexed:
        return None, min(indexed)
    return None, None


def _first_defined_points_index(points: np.ndarray) -> int | None:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim < 2 or arr.shape[-1] < 2:
        return None
    if arr.ndim == 2:
        return 0 if _has_defined_points(arr) else None
    for index, frame in enumerate(arr.reshape(arr.shape[0], -1, arr.shape[-1])):
        if _has_defined_points(frame):
            return index
    return None


def _has_defined_points(points: np.ndarray) -> bool:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim < 2 or arr.shape[-1] < 2:
        return False
    return bool(np.any(np.all(np.isfinite(arr[..., :2]), axis=-1)))


def _include_all_strain_curves(modalities: tuple[str, ...] | None) -> bool:
    if modalities is None:
        return True
    return any(
        str(modality).strip().lower().strip("/") in {"strain", "strain_measurement", "2d_strain"}
        for modality in modalities
    )


def _requested_curve_paths(modalities: tuple[str, ...] | None) -> set[str]:
    if modalities is None:
        return set()
    paths: set[str] = set()
    for modality in modalities:
        path = _normalize_data_path(str(modality).strip().lower().strip("/"))
        if _is_curve_data_path(path):
            paths.add(path)
    return paths


def _strain_curve_data_paths(store: RecordingStore) -> tuple[str, ...]:
    return tuple(path for path in store.array_paths if path.startswith("data/") and _is_curve_data_path(path))


def _should_include_strain_curve(
    data_path: str,
    role_id: str,
    *,
    include_all: bool,
    selected_roles: set[str],
    requested_paths: set[str],
) -> bool:
    return include_all or data_path in requested_paths or bool(role_id and role_id in selected_roles)


def _role_from_curve_path(data_path: str) -> str:
    name = data_path.removeprefix("data/").removesuffix("_curve")
    return name.split("_", 1)[0]


def _strain_qrs_trigger_times(store: RecordingStore, source_store: RecordingStore, panel: StrainPanel) -> np.ndarray:
    values: list[np.ndarray] = []
    for annotation in panel.annotations:
        if not _is_qrs_annotation(annotation):
            continue
        if annotation.time is not None and annotation.time.path in store.group:
            values.append(np.asarray(store.group[annotation.time.path][:], dtype=np.float64))
        elif annotation.value.path in store.group:
            values.append(np.asarray(store.group[annotation.value.path][:], dtype=np.float64))
    values.extend(_strain_qrs_arrays_from_mapping(panel.raw))
    values.extend(_strain_qrs_arrays_from_paths(store, panel.role_id))
    if source_store is not store:
        values.extend(_strain_qrs_arrays_from_paths(source_store, panel.role_id))
    if not values:
        return np.array([], dtype=np.float64)
    qrs = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values], axis=0)
    qrs = qrs[np.isfinite(qrs)]
    return np.unique(np.sort(qrs)).astype(np.float64, copy=False)


def _strain_qrs_arrays_from_mapping(raw: Mapping[str, Any] | None) -> list[np.ndarray]:
    if not isinstance(raw, Mapping):
        return []
    values: list[np.ndarray] = []
    for key in ("ecg_qrs_trigger_times", "qrs_trigger_times"):
        if key in raw:
            value = _optional_float_array(raw[key])
            if value is not None:
                values.append(value)
    linked = raw.get("linked_recording")
    if isinstance(linked, Mapping):
        for key in ("ecg_qrs_trigger_times", "qrs_trigger_times"):
            if key in linked:
                value = _optional_float_array(linked[key])
                if value is not None:
                    values.append(value)
    return values


def _strain_qrs_arrays_from_paths(store: RecordingStore, role_id: str) -> list[np.ndarray]:
    candidate_pairs = (
        (f"timestamps/{role_id}_ecg_qrs", f"data/{role_id}_ecg_qrs"),
        (f"timestamps/{role_id}_qrs", f"data/{role_id}_qrs"),
        ("timestamps/ecg_qrs", "data/ecg_qrs"),
    )
    values: list[np.ndarray] = []
    for timestamp_path, data_path in candidate_pairs:
        path = timestamp_path if timestamp_path in store.group else data_path
        if path in store.group:
            values.append(np.asarray(store.group[path][:], dtype=np.float64))
    return values


def _optional_float_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _is_contour_annotation(annotation: AnnotationRef) -> bool:
    field = annotation.field.lower()
    return "contour" in field or annotation.value.path.endswith("_contour")


def _is_qrs_annotation(annotation: AnnotationRef) -> bool:
    field = annotation.field.lower()
    path = annotation.value.path.lower()
    time_path = "" if annotation.time is None else annotation.time.path.lower()
    return "qrs" in field or path.endswith("_ecg_qrs") or time_path.endswith("_ecg_qrs")


def _is_curve_annotation(annotation: AnnotationRef) -> bool:
    field = annotation.field.lower()
    path = annotation.value.path.lower()
    return "curve" in field or path.endswith("_curve")


def _is_curve_data_path(path: str) -> bool:
    return path.removeprefix("data/").endswith("_curve")


def _is_strain_array_path(path: str) -> bool:
    name = path.removeprefix("data/")
    return name.endswith("_contour") or name.endswith("_curve") or "strain" in name


def _with_explicit_timestamps(store: RecordingStore, loaded: LoadedArray, timestamps_path: str | None) -> LoadedArray:
    if timestamps_path is None or timestamps_path == loaded.timestamps_path or timestamps_path not in store.group:
        return loaded
    return replace(
        loaded,
        timestamps_path=timestamps_path,
        timestamps=np.asarray(store.group[timestamps_path][:]),
    )


def _strain_panel_geometry(panel: StrainPanel) -> SectorGeometry | None:
    raw = panel.geometry
    if raw is None:
        return None
    try:
        return SectorGeometry.from_center_width(
            depth_start_m=float(raw["depth_start_m"]),
            depth_end_m=float(raw["depth_end_m"]),
            tilt_rad=float(raw.get("tilt_rad", 0.0)),
            width_rad=float(raw["width_rad"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _loaded_has_sector_geometry(loaded: LoadedArray) -> bool:
    return loaded.stream is not None and isinstance(loaded.stream.metadata.geometry, SectorGeometry)


def _strain_ecg_trace(store: RecordingStore, panels: tuple[StrainPanel, ...]) -> TraceSpec:
    if _has_ecg(store):
        return _ecg_trace(store)
    for panel in panels:
        try:
            source_store = store.open_reference(panel.bmode.recording)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if _has_ecg(source_store):
            return _ecg_trace(source_store)
    return TraceSpec(
        signal=np.zeros(1, dtype=np.float32),
        timestamps=np.zeros(1, dtype=np.float64),
        label="ECG",
    )


def _has_ecg(store: RecordingStore) -> bool:
    return "data/ecg" in store.array_paths or any(
        path.startswith("data/") and path.endswith("_ecg") for path in store.array_paths
    )


def _expand_modality(modality: str, available: tuple[str, ...]) -> tuple[str, ...]:
    path = _normalize_data_path(modality)
    if path in available:
        return (path,)
    if path == "data/2d_color_doppler":
        return tuple(
            candidate
            for candidate in ("data/2d_color_doppler_velocity", "data/2d_color_doppler_power")
            if candidate in available
        )
    prefix_matches = tuple(candidate for candidate in available if candidate.startswith(f"{path}_"))
    return prefix_matches


def _is_auto_modality(path: str) -> bool:
    name = path.removeprefix("data/")
    if "/" in name or name == "ecg" or name.startswith("line_"):
        return False
    if any(token in name for token in ("annotation", "overlay_physical_points", "colormap")):
        return False
    return name == "tissue_doppler" or name.startswith(("1d_", "2d_", "3d_"))


def _panel_specs(loaded_arrays: tuple[LoadedArray, ...], *, view_mode: PlotViewMode) -> tuple[PanelSpec, ...]:
    if _has_3d_brightness(loaded_arrays):
        return (_three_dimensional_panel(loaded_arrays, view_mode=view_mode),)
    panels: list[PanelSpec] = []
    if view_mode == "pre_converted":
        panels.extend(_panel_spec(loaded, view="pre_converted") for loaded in _preconverted_arrays(loaded_arrays))
    if view_mode == "both":
        panels.extend(
            _panel_spec(loaded, view="pre_converted")
            for loaded in _preconverted_arrays(loaded_arrays)
            if _panel_kind(loaded) == "image"
        )
    if view_mode in {"clinical", "both"}:
        _validate_clinical_color_doppler_pair(loaded_arrays)
        panels.extend(_panel_spec(loaded, view="clinical") for loaded in _clinical_arrays(loaded_arrays))
    return tuple(panels)


def _panel_spec(loaded: LoadedArray, *, view: PanelView) -> PanelSpec:
    return PanelSpec(
        loaded=loaded,
        kind=_panel_kind(loaded),
        label=_label(loaded.data_path),
        view=view,
    )


def _has_3d_brightness(loaded_arrays: tuple[LoadedArray, ...]) -> bool:
    return any(loaded.data_path == "data/3d_brightness_mode" for loaded in loaded_arrays)


def _three_dimensional_panel(loaded_arrays: tuple[LoadedArray, ...], *, view_mode: PlotViewMode) -> PanelSpec:
    loaded = next(loaded for loaded in loaded_arrays if loaded.data_path == "data/3d_brightness_mode")
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    prepared = prepare_3d_brightness_for_display(np.asarray(loaded.data), loaded.timestamps, raw)
    acquisition_stitch_beat_count = (
        None if loaded.stream is None else loaded.stream.metadata.stitch_beat_count
    ) or prepared.stitch_beat_count
    if view_mode == "pre_converted":
        mosaic = preconverted_spherical_mosaic(prepared.volumes).frames
        view: PanelView = "pre_converted"
    else:
        geometry = spherical_geometry_from_metadata(raw)
        mosaic = clinical_spherical_mosaic(prepared.volumes, geometry).frames
        view = "clinical"
    attrs = {
        **dict(loaded.attrs),
        "3d_mosaic_rows": 3,
        "3d_mosaic_cols": 4,
        "3d_stitch_beat_count": acquisition_stitch_beat_count,
        "3d_was_beat_stitched": prepared.was_stitched,
    }
    if view == "pre_converted":
        attrs["data_layout_axes"] = ("T", "E", "A", "R")
        attrs["data_layout_shape"] = tuple(int(size) for size in prepared.volumes.shape)
    mesh_annotation = attrs.get("mesh_annotation")
    if view == "clinical" and isinstance(mesh_annotation, PackedMeshAnnotation):
        try:
            geometry = spherical_geometry_from_metadata(raw)
            attrs["mosaic_annotation_lines"] = mesh_mosaic_annotation_lines(
                mesh_annotation,
                geometry,
                frame_count=int(mosaic.shape[0]),
                mosaic_shape=(int(mosaic.shape[1]), int(mosaic.shape[2])),
                view=view,
                volume_timestamps=prepared.timestamps,
                metadata=raw,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    return _panel_spec(
        replace(
            loaded,
            name="3d_brightness_mode_mosaic",
            data_path="data/3d_brightness_mode",
            data=mosaic,
            timestamps=prepared.timestamps,
            attrs=attrs,
        ),
        view=view,
    )


def _clinical_arrays(loaded_arrays: tuple[LoadedArray, ...]) -> tuple[LoadedArray, ...]:
    visual = tuple(loaded for loaded in loaded_arrays if _panel_kind(loaded) == "image")
    temporal = tuple(loaded for loaded in loaded_arrays if _panel_kind(loaded) != "image")
    if not visual:
        return loaded_arrays
    clinical = clinical_loaded_arrays(visual)
    return (*(clinical or visual), *temporal)


def _preconverted_arrays(loaded_arrays: tuple[LoadedArray, ...]) -> tuple[LoadedArray, ...]:
    color = _first_loaded(loaded_arrays, "data/2d_color_doppler_velocity") or _first_loaded(
        loaded_arrays, "data/2d_color_doppler_power"
    )
    if color is None:
        return loaded_arrays
    return tuple(_with_preconverted_color_extent(loaded, color) for loaded in loaded_arrays)


def _with_preconverted_color_extent(loaded: LoadedArray, color: LoadedArray) -> LoadedArray:
    attrs: dict[str, object] = {}
    color_sector = _loaded_sector_metadata(color)
    if color_sector is None:
        return loaded
    if loaded.data_path.startswith("data/2d_brightness_mode"):
        reference_sector = _loaded_sector_metadata(loaded)
        if reference_sector is not None:
            attrs["preconverted_reference_sector"] = reference_sector
            attrs["preconverted_color_doppler_sector"] = color_sector
    elif loaded.data_path in {"data/2d_color_doppler_velocity", "data/2d_color_doppler_power"}:
        attrs["preconverted_color_doppler_sector"] = color_sector
    if not attrs:
        return loaded
    return replace(loaded, attrs={**dict(loaded.attrs), **attrs})


def _first_loaded(loaded_arrays: tuple[LoadedArray, ...], data_path: str) -> LoadedArray | None:
    return next((loaded for loaded in loaded_arrays if loaded.data_path == data_path), None)


def _loaded_sector_metadata(loaded: LoadedArray) -> object | None:
    if loaded.stream is None:
        return None
    return loaded.stream.metadata.raw


def _validate_clinical_color_doppler_pair(loaded_arrays: tuple[LoadedArray, ...]) -> None:
    paths = {loaded.data_path for loaded in loaded_arrays}
    has_velocity = "data/2d_color_doppler_velocity" in paths
    has_power = "data/2d_color_doppler_power" in paths
    if has_velocity == has_power:
        return
    missing = "data/2d_color_doppler_power" if has_velocity else "data/2d_color_doppler_velocity"
    present = "data/2d_color_doppler_velocity" if has_velocity else "data/2d_color_doppler_power"
    raise ValueError(
        "Clinical color Doppler plotting requires velocity and power arrays together; "
        f"{present} was provided without {missing}."
    )


def _panel_kind(loaded: LoadedArray) -> PanelKind:
    data = np.asarray(loaded.data)
    if data.ndim <= 1:
        return "line"
    if loaded.name.startswith("1d_") or data.ndim == 2:
        return "matrix"
    return "image"


def _set_ecg_xlim(ax: Axes, ecg: TraceSpec) -> None:
    timestamps = np.asarray(ecg.timestamps, dtype=np.float64).reshape(-1)
    valid = timestamps[np.isfinite(timestamps)]
    if valid.size < 2:
        return
    start = float(valid[0])
    end = float(valid[-1])
    if end > start:
        ax.set_xlim(start, end)


def _ecg_trace(store: RecordingStore) -> TraceSpec:
    loaded = store.load_modality(_ecg_data_path(store))
    signal = np.asarray(loaded.data, dtype=np.float32).reshape(-1)
    if loaded.timestamps is None:
        timestamps = np.arange(signal.size, dtype=np.float64)
    else:
        timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    return TraceSpec(signal=signal, timestamps=timestamps, label="ECG")


def _ecg_data_path(store: RecordingStore) -> str:
    if "data/ecg" in store.array_paths:
        return "ecg"
    fallback = next((path for path in store.array_paths if path.startswith("data/") and path.endswith("_ecg")), None)
    if fallback is None:
        raise KeyError(f"No ECG stream is available in {store.path}")
    return fallback


def _normalize_data_path(path: str) -> str:
    text = str(path).strip("/")
    return text if text.startswith("data/") else f"data/{text}"


def _normalize_view_mode(view_mode: PlotViewMode | str) -> PlotViewMode:
    text = str(view_mode).strip().lower().replace("-", "_")
    if text in {"pre_converted", "preconverted", "raw"}:
        return "pre_converted"
    if text == "clinical":
        return "clinical"
    if text == "both":
        return "both"
    raise ValueError(f"Unsupported plot view mode `{view_mode}`")


def _label(path: str) -> str:
    name = path.removeprefix("data/")
    for prefix in ("clinical_", "1d_", "2d_", "3d_"):
        name = name.removeprefix(prefix)
    return name.replace("_", " ")
