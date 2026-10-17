"""High-level Matplotlib renderer for EchoXFlow recordings."""

# flake8: noqa: E402

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.text import Text

from echoxflow.colors import data_modality_colormap_name
from echoxflow.croissant import RecordingRecord
from echoxflow.loading import LoadedArray, PackedMeshAnnotation, RecordingStore, open_recording
from echoxflow.mesh import mesh_volume_curve_ml
from echoxflow.objects import AnnotationRef, RecordingObject, RecordingRef, StrainPanel
from echoxflow.plotting.annotations import (
    attach_annotation_overlays,
    load_mesh_annotation,
    mesh_mosaic_annotation_lines,
)
from echoxflow.plotting.cartesian import cartesian_loaded_arrays
from echoxflow.plotting.gating import BloodGateConfig, TissueGateConfig
from echoxflow.plotting.layout import spatial_layout, uses_ecg_timescale
from echoxflow.plotting.panels import PanelArtists, draw_panel_label, render_ecg, renderer_for
from echoxflow.plotting.specs import (
    FrameRequest,
    PanelKind,
    PanelSpec,
    PanelView,
    PlotViewMode,
    RenderedFrame,
    TraceSpec,
)
from echoxflow.plotting.strain import AHA_MODALITIES, anatomical_panel_views, anatomical_specs, has_anatomical_strain

logger = logging.getLogger(__name__)
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import (
    is_timeline_driver,
    nearest_index,
    resolve_frame_time,
    select_timeline,
    strip_drives_timeline,
    temporal_length,
)
from echoxflow.plotting.writers import figure_to_rgb, save_figure, write_video
from echoxflow.scan import (
    beamspace_spherical_mosaic,
    cartesian_spherical_mosaic,
    mesh_frame_indices_for_volume_timestamps,
    prepare_3d_brightness_for_display,
    spherical_geometry_from_metadata,
)
from echoxflow.scan.contours import contour_group_layout_for_metadata
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.schema import (
    color_flow_velocity_codes,
    display_array_path,
    flow_color_first_axis_is_brightness,
    flow_color_velocity_ramp,
    is_lookup_table_path,
    render_flow_color_table,
)


@dataclass
class _FrameState:
    """A built figure plus what is needed to move it to another frame in place.

    ``panel_artists`` is ``None`` when some panel renderer cannot update its artists safely;
    the caller then rebuilds the whole figure for every frame (decided per figure).
    """

    figure: Figure
    panel_artists: tuple[PanelArtists, ...] | None
    ecg_update: Callable[[float, np.ndarray | None], None] | None
    title: Text | None
    panels: tuple[PanelSpec, ...]
    title_text: Callable[[float, int], str] | None

    @property
    def updatable(self) -> bool:
        return self.panel_artists is not None

    def update(self, *, time_s: float, frame_index: int) -> None:
        if self.panel_artists is None:
            raise RuntimeError("This figure cannot be updated in place")
        for artists in self.panel_artists:
            artists.update(time_s, frame_index)
        if self.ecg_update is not None:
            self.ecg_update(time_s, _ecg_marker_times(self.panels, time_s, frame_index))
        if self.title is not None and self.title_text is not None:
            self.title.set_text(self.title_text(time_s, frame_index))


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
        view_mode: PlotViewMode | str = "beamspace",
        show_annotations: bool = True,
        dpi: int | None = None,
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
            dpi=self.style.dpi if dpi is None else int(dpi),
            title=_recording_title(record),
        )

    def render_frame(
        self,
        record: RecordingRecord | str | Path,
        *,
        root: str | Path | None = None,
        modalities: tuple[str, ...] | None = None,
        time_s: float | None = None,
        frame_index: int | None = None,
        view_mode: PlotViewMode | str = "beamspace",
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
        view_mode: PlotViewMode | str = "beamspace",
        show_annotations: bool = True,
        dpi: int | None = None,
    ) -> Path:
        """Save one frame as an image of exactly ``style.width_px`` x ``style.height_px`` pixels.

        ``dpi`` only scales fonts and line widths relative to the pixel size; ``None`` uses ``style.dpi``.
        """
        figure = self.plot_recording(
            record,
            root=root,
            modalities=modalities,
            time_s=time_s,
            frame_index=frame_index,
            view_mode=view_mode,
            show_annotations=show_annotations,
            dpi=dpi,
        )
        try:
            return save_figure(figure, output, dpi=self.style.dpi if dpi is None else int(dpi))
        finally:
            plt.close(figure)

    def render_video(
        self,
        record: RecordingRecord | str | Path,
        output: str | Path,
        *,
        root: str | Path | None = None,
        modalities: tuple[str, ...] | None = None,
        view_mode: PlotViewMode | str = "beamspace",
        show_annotations: bool = True,
        max_fps: float | None = None,
        dpi: int | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> Path:
        """Render a video of exactly ``style.width_px`` x ``style.height_px`` pixels per frame.

        ``dpi`` only scales fonts and line widths relative to the pixel size; ``None`` uses ``style.dpi``.
        ``time_range`` keeps only the frames with ``start <= time < end`` seconds.
        """
        panels, ecg = self._load_specs(
            record, root=root, modalities=modalities, view_mode=view_mode, show_annotations=show_annotations
        )
        timeline = select_timeline(panels, max_fps=self.style.max_fps if max_fps is None else float(max_fps))
        source = record.path(root) if isinstance(record, RecordingRecord) else Path(record).expanduser()
        times = np.asarray(timeline.timestamps, dtype=np.float64).reshape(-1)
        if time_range is not None:
            start, end = float(time_range[0]), float(time_range[1])
            times = times[(times >= start) & (times < end)]
            if times.size == 0:
                raise ValueError(f"No frames between {start} s and {end} s in {source}")
        frames = self.iter_frames(panels, ecg, times, dpi=dpi, title=_recording_title(record))
        return write_video(output, frames, fps=timeline.fps, source=source)

    def iter_frames(
        self,
        panels: tuple[PanelSpec, ...],
        ecg: TraceSpec,
        times: np.ndarray,
        *,
        dpi: int | None = None,
        title: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Render one RGB frame per timestamp.

        The figure, axes, ticks, labels, colour bars and layout are built once for the first frame;
        later frames only update the per-frame artists. Figures whose panels cannot be updated in
        place are rebuilt for every frame.
        """
        state: _FrameState | None = None
        try:
            for frame_index, current_time in enumerate(times):
                time_s = float(current_time)
                if state is None:
                    state = self._build_figure(
                        panels=panels, ecg=ecg, time_s=time_s, frame_index=frame_index, dpi=dpi, title=title
                    )
                elif state.updatable:
                    state.update(time_s=time_s, frame_index=frame_index)
                else:
                    plt.close(state.figure)
                    state = self._build_figure(
                        panels=panels, ecg=ecg, time_s=time_s, frame_index=frame_index, dpi=dpi, title=title
                    )
                yield figure_to_rgb(state.figure)
        finally:
            if state is not None:
                plt.close(state.figure)

    def render_figure_from_specs(
        self,
        *,
        panels: tuple[PanelSpec, ...],
        ecg: TraceSpec,
        time_s: float,
        frame_index: int,
        dpi: int | None = None,
        title: str | None = None,
    ) -> Figure:
        return self._build_figure(
            panels=panels, ecg=ecg, time_s=time_s, frame_index=frame_index, dpi=dpi, title=title
        ).figure

    def _build_figure(
        self,
        *,
        panels: tuple[PanelSpec, ...],
        ecg: TraceSpec,
        time_s: float,
        frame_index: int,
        dpi: int | None = None,
        title: str | None = None,
    ) -> _FrameState:
        if not panels:
            raise ValueError("At least one non-ECG modality panel is required")
        dpi = self.style.dpi if dpi is None else int(dpi)
        spatial_panels = tuple(panel for panel in panels if not uses_ecg_timescale(panel))
        temporal_panels = tuple(panel for panel in panels if uses_ecg_timescale(panel))
        preserve_single_beamspace_image_aspect = _preserve_single_beamspace_image_aspect(panels)
        layout = spatial_layout(spatial_panels)
        cols = layout.cols
        spatial_rows = layout.rows
        temporal_rows = len(temporal_panels)
        show_ecg = bool(self.style.show_ecg) and not all(
            panel.kind in {"strain_curves", "bullseye"} for panel in panels
        )
        total_rows = spatial_rows + temporal_rows + (1 if show_ecg else 0)
        figure = plt.figure(
            figsize=(self.style.width_px / int(dpi), self.style.height_px / int(dpi)),
            dpi=int(dpi),
            facecolor=self.style.figure_facecolor,
        )
        strip_ratio = (
            self.style.strip_height_ratio_static
            if spatial_panels and all(_is_single_frame(panel) for panel in spatial_panels)
            else self.style.strip_height_ratio
        )
        grid = GridSpec(
            total_rows,
            cols,
            figure=figure,
            height_ratios=([1.0] * spatial_rows)
            + [
                self.style.volume_curve_height_ratio if panel.loaded.name == "lv_volume_curve" else strip_ratio
                for panel in temporal_panels
            ]
            + ([0.28] if show_ecg else []),
            width_ratios=layout.width_ratios,
        )
        occupied: set[tuple[int, int]] = set()
        labelled: list[tuple[Axes, PanelSpec]] = []
        artists: list[PanelArtists | None] = []
        for placement in layout.placements:
            row_end = placement.row + placement.row_span
            col_end = placement.col + placement.col_span
            ax = figure.add_subplot(grid[placement.row : row_end, placement.col : col_end])
            panel = placement.panel
            artists.append(
                renderer_for(panel.kind).render(ax, panel, time_s=time_s, frame_index=frame_index, style=self.style)
            )
            labelled.append((ax, panel))
            if preserve_single_beamspace_image_aspect and panel is panels[0]:
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
            artists.append(
                renderer_for(panel.kind).render(ax, panel, time_s=time_s, frame_index=frame_index, style=self.style)
            )
            labelled.append((ax, panel))
            _set_ecg_xlim(ax, ecg)
        ecg_update = None
        if show_ecg:
            ecg_ax = figure.add_subplot(grid[total_rows - 1, :])
            ecg_update = render_ecg(
                ecg_ax,
                ecg,
                time_s=time_s,
                style=self.style,
                marker_times=_ecg_marker_times(panels, time_s, frame_index),
            )
            _set_ecg_xlim(ecg_ax, ecg)
        top = 1.0
        title_artist: Text | None = None
        if self.style.show_figure_title:
            title_artist, top = _draw_figure_title(
                figure, panels, title=title, time_s=time_s, frame_index=frame_index, style=self.style
            )
        right = 1.0 - (0.04 if temporal_panels else 0.0)  # leave room for the strip's unit tag
        figure.tight_layout(pad=0.8, rect=(0.0, 0.0, right, top))
        for ax, panel in labelled:
            draw_panel_label(ax, panel, self.style)
        return _FrameState(
            figure=figure,
            panel_artists=None if any(item is None for item in artists) else tuple(cast(list[PanelArtists], artists)),
            ecg_update=ecg_update,
            title=title_artist,
            panels=panels,
            title_text=lambda t, i: _figure_title_text(panels, title=title, time_s=t, frame_index=i),
        )

    def build_panel_specs(
        self, loaded_arrays: tuple[LoadedArray, ...], *, view_mode: PlotViewMode | str = "beamspace"
    ) -> tuple[PanelSpec, ...]:
        """Build display panel specs from loaded modality arrays."""
        return _panel_specs(
            loaded_arrays,
            view_mode=_normalize_view_mode(view_mode),
            mosaic_cell_height=_mosaic_cell_height(self.style),
            blood_gate=self.style.blood_gate,
            max_depth_m=self.style.cartesian_max_depth_m,
            tissue_brightness_lift=self.style.tissue_doppler_brightness_lift,
            tissue_gate=self.style.tissue_gate,
        )

    def _load_specs(
        self,
        record: RecordingRecord | str | Path,
        *,
        root: str | Path | None,
        modalities: tuple[str, ...] | None,
        view_mode: PlotViewMode | str,
        show_annotations: bool,
    ) -> tuple[tuple[PanelSpec, ...], TraceSpec]:
        with ExitStack() as stack:
            store = stack.enter_context(open_recording(record, root=root))
            mesh_store: RecordingStore | None = None
            if _is_mesh_only(store):
                mesh_store, store = store, stack.enter_context(_open_linked_volume(store))
            if modalities and all(token in AHA_MODALITIES for token in modalities):
                panels = anatomical_specs(
                    store,
                    curves=any(token != "bullseye" for token in modalities),
                    bullseye=any(token != "aha_curves" for token in modalities),
                )
                return panels, TraceSpec(signal=np.empty(0), timestamps=np.empty(0))
            obj = store.load_object()
            if _should_plot_strain_object(obj, store, modalities):
                return _strain_specs(
                    store,
                    obj,
                    modalities=modalities,
                    view_mode=view_mode,
                    show_annotations=show_annotations,
                    max_depth_m=self.style.cartesian_max_depth_m,
                )
            paths = _resolve_modality_paths(store, modalities)
            loaded_arrays = tuple(
                _with_recording_colormap(store, _with_display_path(store.load_modality(path), path)) for path in paths
            )
            if show_annotations:
                loaded_arrays = attach_annotation_overlays(store, loaded_arrays)
                if mesh_store is not None:
                    loaded_arrays = _with_linked_mesh(loaded_arrays, mesh_store)
            panels = self.build_panel_specs(loaded_arrays, view_mode=view_mode)
            ecg = _ecg_trace(store)
            return panels, ecg


def _recording_title(record: RecordingRecord | str | Path) -> str:
    return record.recording_id if isinstance(record, RecordingRecord) else Path(record).stem


def _figure_title_text(panels: tuple[PanelSpec, ...], *, title: str | None, time_s: float, frame_index: int) -> str:
    parts = [] if not title else [title]
    if panels and all(panel.kind in {"strain_curves", "bullseye"} for panel in panels):
        return title or "Anatomical strain"
    parts.append(f"t = {float(time_s):.3f} s")
    driver = next((panel for panel in panels if is_timeline_driver(panel)), None)
    if driver is not None:
        count = temporal_length(driver.loaded.data)
        index = nearest_index(driver.loaded.timestamps, time_s, count=count, fallback_index=frame_index)
        parts[-1] += f"  frame {index}/{count}"
    return "    ".join(parts)


def _draw_figure_title(
    figure: Figure,
    panels: tuple[PanelSpec, ...],
    *,
    title: str | None,
    time_s: float,
    frame_index: int,
    style: PlotStyle,
) -> tuple[Text, float]:
    """Write the recording id, frame time and frame index on top of the figure; return it and the free top edge."""
    height_pt = figure.get_figheight() * 72.0
    text = figure.text(
        0.008,
        0.998,
        _figure_title_text(panels, title=title, time_s=time_s, frame_index=frame_index),
        ha="left",
        va="top",
        fontsize=style.panel_label_fontsize,
        color=style.text_dim_color,
    )
    return text, 1.0 - 2.0 * style.panel_label_fontsize / height_pt


def _with_recording_colormap(store: RecordingStore, loaded: LoadedArray) -> LoadedArray:
    """Attach the recording's RGB table to spectral, colour and tissue Doppler arrays.

    A colour Doppler array whose table is variance-style also gets the table's two-dimensional colouring, see
    `_flow_color_attrs`, so turbulence tints the picture the way the scanner shows it.
    """
    name = data_modality_colormap_name(loaded.data_path)
    path = None if loaded.stream is None else loaded.stream.metadata.colormap_path
    if name not in {"color_doppler_velocity", "tissue_doppler", "spectral_doppler"} or path is None:
        return loaded
    try:
        table = store.load_array(path)
        if name == "spectral_doppler":
            ramp = np.asarray(table)
            if ramp.shape != (256, 3) or ramp.dtype != np.uint8:
                return loaded
        else:
            ramp = flow_color_velocity_ramp(table, phase_ordered=True)
    except (KeyError, ValueError):
        return loaded
    cmap = ListedColormap(ramp.astype(np.float32) / 255.0, name=f"{name}_recording")
    attrs = {**loaded.attrs, "colormap": cmap}
    if name == "color_doppler_velocity":
        attrs.update(_flow_color_attrs(store, loaded, np.asarray(table)))
    elif name == "tissue_doppler" and flow_color_first_axis_is_brightness(table):
        attrs["tissue_color_table"] = np.asarray(table, dtype=np.uint8)
    return replace(loaded, attrs=attrs)


_LAG_ONE_PATH = "derived/2d_color_doppler_lag_one_magnitude"
_LAG_ONE_FULL_SCALE = 127.0 * float(np.sqrt(2.0))


def _flow_color_attrs(store: RecordingStore, loaded: LoadedArray, table: np.ndarray) -> dict[str, object]:
    """Colour a variance-style colour Doppler stack through the whole FlowColorTable.

    The table is indexed `[secondary, velocity]`. The velocity index is the phase code of the wrapped velocity. The
    exporter stores no turbulence byte (the raw auxiliary channel is always zero), so the secondary index is the
    lag-one magnitude turned around, `255 * (1 - |R1| / full scale)`: coherent flow stays on the zero-variance row and
    incoherent flow blends toward green. Returns nothing when the table is power-style (its first axis only dims) or
    the recording has no lag-one magnitude, and callers then keep the one-dimensional velocity ramp.

    `flow_color_rgb` is the `uint8` `(T, H, W, 3)` stack. `flow_color_codes` holds the `uint16` codes
    `secondary * 256 + velocity`; Cartesian conversion looks up these exact native codes before interpolating RGB.
    """
    try:
        if flow_color_first_axis_is_brightness(table) or _LAG_ONE_PATH not in store.array_paths:
            return {}
        velocity = np.asarray(loaded.data, dtype=np.float32)
        lag_one = np.asarray(store.load_array(_LAG_ONE_PATH), dtype=np.float32)
        value_range = None if loaded.stream is None else loaded.stream.metadata.value_range
        if velocity.ndim != 3 or lag_one.shape != velocity.shape or value_range is None:
            return {}
        velocity_codes = color_flow_velocity_codes(velocity, velocity_limit_mps=float(value_range[1]))
        turbulence = 1.0 - np.clip(lag_one / _LAG_ONE_FULL_SCALE, 0.0, 1.0)
        secondary = np.where(np.isfinite(velocity), np.rint(255.0 * turbulence), 0.0).astype(np.uint8)
        rgb = render_flow_color_table(velocity_codes=velocity_codes, secondary_codes=secondary, flow_color_table=table)
    except (KeyError, ValueError):
        return {}
    codes = secondary.astype(np.uint16) * np.uint16(256) + velocity_codes.astype(np.uint16)
    return {"flow_color_rgb": rgb, "flow_color_codes": codes, "flow_color_table": table.astype(np.uint8)}


def _is_mesh_only(store: RecordingStore) -> bool:
    paths = store.array_paths
    return not any(path.startswith("data/") for path in paths) and any(
        path.startswith("annotations/3d_") and "_mesh" in path for path in paths
    )


def _open_linked_volume(mesh_store: RecordingStore) -> RecordingStore:
    """Open the volume recording a mesh-only recording was derived from."""
    attrs = getattr(mesh_store.group, "attrs", {})
    manifest = attrs.get("recording_manifest") if hasattr(attrs, "get") else None
    linked_volume = manifest.get("linked_volume") if isinstance(manifest, Mapping) else None
    linked = linked_volume.get("linked_recording") if isinstance(linked_volume, Mapping) else None
    recording_id = mesh_store.record.recording_id if mesh_store.record is not None else mesh_store.path.stem
    if not isinstance(linked, Mapping) or not (linked.get("recording_zarr_path") or linked.get("relative_zarr_path")):
        raise ValueError(f"Recording `{recording_id}` holds only a mesh with no linked volume to render")
    ref = RecordingRef(
        recording_id=_text_or_none(linked.get("recording_id")),
        zarr_path=_text_or_none(linked.get("recording_zarr_path")),
        relative_zarr_path=_text_or_none(linked.get("relative_zarr_path")),
        is_self=False,
    )
    try:
        return mesh_store.open_reference(ref)
    except Exception as exc:
        raise ValueError(f"Could not open the volume linked to mesh-only recording `{recording_id}`") from exc


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _with_linked_mesh(loaded_arrays: tuple[LoadedArray, ...], mesh_store: RecordingStore) -> tuple[LoadedArray, ...]:
    if not any(path.startswith("annotations/3d_left_ventricle_mesh/") for path in mesh_store.array_paths):
        logger.warning("LA/RV mesh overlay is unavailable; rendering the linked volume without the mesh")
        return loaded_arrays
    mesh = load_mesh_annotation(mesh_store)
    if mesh is None:
        return loaded_arrays
    return tuple(
        replace(loaded, attrs={**loaded.attrs, "mesh_annotation": mesh})
        if loaded.data_path == "data/3d_brightness_mode"
        else loaded
        for loaded in loaded_arrays
    )


def _resolve_modality_paths(store: RecordingStore, modalities: tuple[str, ...] | None) -> tuple[str, ...]:
    available = _display_data_paths(store)
    if modalities is None:
        return tuple(path for path in available if _is_auto_modality(path))
    resolved: list[str] = []
    for modality in modalities:
        expanded = _expand_modality(modality, available)
        if not expanded:
            raise KeyError(f"Requested modality `{modality}` is not available in {store.path}")
        resolved.extend(expanded)
    return tuple(dict.fromkeys(resolved))


def _is_single_frame(panel: PanelSpec) -> bool:
    """True for a static reference image: a stream holding one frame."""
    shape = np.shape(panel.loaded.data)
    return len(shape) < 3 or int(shape[0]) <= 1


def _preserve_single_beamspace_image_aspect(panels: tuple[PanelSpec, ...]) -> bool:
    return len(panels) == 1 and panels[0].view == "beamspace" and panels[0].kind == "image"


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
    max_depth_m: float | None = None,
) -> tuple[tuple[PanelSpec, ...], TraceSpec]:
    selected_roles = _selected_strain_roles(obj, modalities)
    selected_panels = tuple(panel for panel in obj.panels if selected_roles is None or panel.role_id in selected_roles)
    if not selected_panels:
        raise KeyError(f"Requested strain modalities are not available in {store.path}")
    normalized_view = _normalize_view_mode(view_mode)
    image_views: tuple[PanelView, ...] = (
        ("beamspace", "cartesian")
        if normalized_view == "both"
        else ("cartesian",)
        if normalized_view == "cartesian"
        else ("beamspace",)
    )
    image_inputs = _align_strain_image_inputs(tuple(_strain_image_input(store, panel) for panel in selected_panels))
    image_specs = [
        spec
        for view in image_views
        for image_input in image_inputs
        for spec in (
            _strain_image_spec(image_input, view=view, show_annotations=show_annotations, max_depth_m=max_depth_m),
        )
        if not (normalized_view == "both" and view == "cartesian" and spec.view != "cartesian")
    ]
    curve_specs = (
        anatomical_specs(store)
        if _has_generic_strain_modality(modalities) and has_anatomical_strain(store)
        else _strain_curve_specs(store, obj, selected_panels, modalities)
    )
    if curve_specs and curve_specs[0].kind == "strain_curves":
        views = anatomical_panel_views(store)
        image_specs = [
            replace(
                spec,
                loaded=replace(
                    spec.loaded,
                    attrs={
                        **spec.loaded.attrs,
                        "strain_source_view": views.get(str(spec.loaded.attrs["strain_role_id"])),
                    },
                ),
            )
            for spec in image_specs
        ]
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
    source_store = None
    try:
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
    finally:
        if source_store is not None and source_store is not store:
            source_store.close()


def _strain_image_spec(
    image_input: _StrainImageInput,
    *,
    view: PanelView,
    show_annotations: bool,
    max_depth_m: float | None = None,
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
    if view == "cartesian":
        if _loaded_has_sector_geometry(loaded):
            loaded = cartesian_loaded_arrays((loaded,), max_depth_m=max_depth_m)[0]
        else:
            view = "beamspace"
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
    include_all = _has_generic_strain_modality(modalities)
    selected_roles = {panel.role_id for panel in selected_panels}
    requested_paths = _requested_curve_paths(modalities)
    specs: list[PanelSpec] = []
    seen: set[str] = set()
    for annotation in obj.annotations:
        if annotation.value is None or not _is_curve_annotation(annotation):
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
        role_id = _role_from_curve_path(data_path)
        if data_path in seen or not _should_include_strain_curve(
            data_path,
            role_id,
            include_all=include_all,
            selected_roles=selected_roles,
            requested_paths=requested_paths,
        ):
            continue
        spec = _strain_curve_spec(store, data_path, role_id=role_id, annotation=None)
        if spec is not None:
            specs.append(spec)
            seen.add(data_path)
    return _combined_strain_curve_specs(tuple(specs)) if include_all else tuple(specs)


def _combined_strain_curve_specs(specs: tuple[PanelSpec, ...]) -> tuple[PanelSpec, ...]:
    if len(specs) < 2:
        return specs
    curves = [_mean_curve(np.asarray(spec.loaded.data)) for spec in specs]
    size = max((curve.size for curve in curves), default=0)
    if size == 0:
        return ()
    data = np.full((size, len(curves)), np.nan, dtype=np.float32)
    for index, curve in enumerate(curves):
        data[: curve.size, index] = curve
    timestamps = next((spec.loaded.timestamps for spec in specs if spec.loaded.timestamps is not None), None)
    timestamps = None if timestamps is None or np.asarray(timestamps).size != size else np.asarray(timestamps)
    loaded = replace(
        specs[0].loaded,
        name="strain_curves",
        data_path="data/strain_curves",
        data=data,
        timestamps_path=None,
        timestamps=timestamps,
        attrs={**dict(specs[0].loaded.attrs), "strain_curve_combined": True},
    )
    return (replace(specs[0], loaded=loaded, label="strain"),)


def _mean_curve(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.nanmean(arr.reshape(arr.shape[0], -1), axis=1), dtype=np.float32)


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
        view="beamspace",
    )


def _strain_contour_overlays(store: RecordingStore, panel: StrainPanel) -> tuple[dict[str, object], ...]:
    overlays: list[dict[str, object]] = []
    for annotation in panel.annotations:
        if annotation.value is None or not _is_contour_annotation(annotation):
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
                "group_layout": contour_group_layout_for_metadata(
                    content_type=_strain_content_type(store),
                    point_count=int(points.shape[1]) if points.ndim == 3 else None,
                ),
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


def _strain_content_type(store: RecordingStore) -> str:
    attrs = getattr(store.group, "attrs", {})
    declared = attrs.get("content_types") if hasattr(attrs, "get") else None
    types = [str(value) for value in declared or ()]
    if not types and store.record is not None:
        types = [str(value) for value in store.record.content_types]
    return next((value for value in types if "strain" in value), "")


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


def _has_generic_strain_modality(modalities: tuple[str, ...] | None) -> bool:
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
    return (
        data_path in requested_paths
        or bool(role_id and role_id in selected_roles)
        or (include_all and not selected_roles)
    )


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
        elif annotation.value is not None and annotation.value.path in store.group:
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
    candidate_paths = (f"timestamps/{role_id}_ecg_qrs", f"timestamps/{role_id}_qrs", "timestamps/ecg_qrs")
    return [np.asarray(store.group[path][:], dtype=np.float64) for path in candidate_paths if path in store.group]


def _optional_float_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _is_contour_annotation(annotation: AnnotationRef) -> bool:
    if annotation.value is None:
        return False
    field = annotation.field.lower()
    return "contour" in field or annotation.value.path.endswith("_contour")


def _is_qrs_annotation(annotation: AnnotationRef) -> bool:
    if annotation.value is None:
        return False
    field = annotation.field.lower()
    path = annotation.value.path.lower()
    time_path = "" if annotation.time is None else annotation.time.path.lower()
    return "qrs" in field or path.endswith("_ecg_qrs") or time_path.endswith("_ecg_qrs")


def _is_curve_annotation(annotation: AnnotationRef) -> bool:
    if annotation.value is None:
        return False
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
        try:
            if _has_ecg(source_store):
                return _ecg_trace(source_store)
        finally:
            if source_store is not store:
                source_store.close()
    return TraceSpec(
        signal=np.zeros(1, dtype=np.float32),
        timestamps=np.zeros(1, dtype=np.float64),
        label="ECG",
    )


def _display_data_paths(store: RecordingStore) -> tuple[str, ...]:
    """Return plottable arrays under the short names the panel code dispatches on."""
    paths = (
        display_array_path(path)
        for path in store.array_paths
        if path.startswith(("data/", "derived/")) and not is_lookup_table_path(path)
    )
    return tuple(dict.fromkeys(path for path in paths if path.startswith("data/")))


def _with_display_path(loaded: LoadedArray, display_path: str) -> LoadedArray:
    if loaded.data_path == display_path:
        return loaded
    stream = None if loaded.stream is None else replace(loaded.stream, data_path=display_path)
    return replace(loaded, data_path=display_path, stream=stream)


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
    if "/" in name or name == "ecg" or name.startswith("line_") or name.endswith("_raw"):
        return False
    if any(token in name for token in ("annotation", "overlay_physical_points")):
        return False
    return name == "tissue_doppler" or name.startswith(("1d_", "2d_", "3d_"))


def _mosaic_cell_height(style: PlotStyle) -> int:
    """Pixel height of one 3D mosaic cell: the spatial panel is three cells tall, about 0.84 of the figure height."""
    return max(2, int(round(0.28 * style.height_px)))


def _panel_specs(
    loaded_arrays: tuple[LoadedArray, ...],
    *,
    view_mode: PlotViewMode,
    mosaic_cell_height: int | None = None,
    blood_gate: BloodGateConfig | None = None,
    max_depth_m: float | None = None,
    tissue_brightness_lift: float = 0.0,
    tissue_gate: TissueGateConfig | None = None,
) -> tuple[PanelSpec, ...]:
    if _has_3d_brightness(loaded_arrays):
        mosaic_panel = _three_dimensional_panel(loaded_arrays, view_mode=view_mode, cell_height=mosaic_cell_height)
        volume_panel = _lv_volume_panel(mosaic_panel)
        return (mosaic_panel,) if volume_panel is None else (mosaic_panel, volume_panel)
    panels: list[PanelSpec] = []
    if view_mode == "beamspace":
        panels.extend(_panel_spec(loaded, view="beamspace") for loaded in _beamspace_arrays(loaded_arrays))
    if view_mode == "both":
        panels.extend(
            _panel_spec(loaded, view="beamspace")
            for loaded in _beamspace_arrays(loaded_arrays)
            if _panel_kind(loaded) == "image"
        )
    if view_mode in {"cartesian", "both"}:
        _validate_cartesian_color_doppler_pair(loaded_arrays)
        panels.extend(
            _panel_spec(loaded, view="cartesian")
            for loaded in _cartesian_arrays(
                loaded_arrays,
                blood_gate=blood_gate,
                max_depth_m=max_depth_m,
                tissue_brightness_lift=tissue_brightness_lift,
                tissue_gate=tissue_gate,
            )
        )
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


def _three_dimensional_panel(
    loaded_arrays: tuple[LoadedArray, ...], *, view_mode: PlotViewMode, cell_height: int | None = None
) -> PanelSpec:
    loaded = next(loaded for loaded in loaded_arrays if loaded.data_path == "data/3d_brightness_mode")
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    prepared = prepare_3d_brightness_for_display(np.asarray(loaded.data), loaded.timestamps, raw)
    acquisition_stitch_beat_count = (
        None if loaded.stream is None else loaded.stream.metadata.stitch_beat_count
    ) or prepared.stitch_beat_count
    if view_mode == "beamspace":
        mosaic = beamspace_spherical_mosaic(prepared.volumes, cell_height=cell_height).frames
        mosaic_layout = None
        view: PanelView = "beamspace"
    else:
        geometry = spherical_geometry_from_metadata(raw)
        mosaic_result = cartesian_spherical_mosaic(prepared.volumes, geometry, cell_height=cell_height)
        mosaic = mosaic_result.frames
        mosaic_layout = mosaic_result.layout
        view = "cartesian"
    attrs = {
        **dict(loaded.attrs),
        "3d_mosaic_rows": 3,
        "3d_mosaic_cols": 4,
        "3d_stitch_beat_count": acquisition_stitch_beat_count,
        "3d_was_beat_stitched": prepared.was_stitched,
    }
    if prepared.source_timestamps is not None:
        attrs["ecg_marker_timestamps"] = prepared.source_timestamps
    if view == "cartesian" and mosaic_layout is not None:
        attrs["mosaic_layout"] = mosaic_layout
        attrs["mosaic_geometry"] = geometry
    mesh_annotation = attrs.get("mesh_annotation")
    if view == "cartesian" and mosaic_layout is not None and isinstance(mesh_annotation, PackedMeshAnnotation):
        try:
            geometry = spherical_geometry_from_metadata(raw)
            lines = mesh_mosaic_annotation_lines(
                mesh_annotation,
                frame_count=int(mosaic.shape[0]),
                layout=mosaic_layout,
                view=view,
                volume_timestamps=prepared.timestamps,
                metadata=raw,
            )
            if any(lines):
                attrs["mosaic_annotation_lines"] = lines
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


def _lv_volume_panel(mosaic_panel: PanelSpec) -> PanelSpec | None:
    """Line panel of the LV cavity volume per displayed frame, following the same mesh frames as the overlay."""
    loaded = mosaic_panel.loaded
    mesh = loaded.attrs.get("mesh_annotation")
    if not isinstance(mesh, PackedMeshAnnotation) or loaded.timestamps is None:
        return None
    curve = mesh_volume_curve_ml(mesh)
    if curve is None:
        return None
    raw = None if loaded.stream is None else loaded.stream.metadata.raw
    frame_count = int(np.asarray(loaded.data).shape[0])
    mesh_indices = np.asarray(
        mesh_frame_indices_for_volume_timestamps(
            mesh.timestamps, loaded.timestamps, raw, mesh_frame_count=mesh.frame_count, target_count=frame_count
        ),
        dtype=np.int64,
    )
    timestamps = np.asarray(loaded.timestamps, dtype=np.float64).reshape(-1)
    if mesh_indices.size != timestamps.size or timestamps.size == 0:
        return None
    local = np.clip(mesh_indices - curve.frame_start, 0, curve.volumes_ml.size - 1)
    series = curve.volumes_ml[local].astype(np.float32)
    if not np.any(np.isfinite(series)):
        return None
    markers: list[tuple[float, float, str]] = []
    # the displayed frames can span several beats, each replaying the mesh cycle: mark the extremes of every pass
    breaks = np.flatnonzero(np.diff(local) < 0) + 1
    for segment in np.split(np.arange(local.size), breaks):
        values = series[segment]
        if segment.size < 2 or not np.any(np.isfinite(values)):
            continue
        ed = int(segment[int(np.nanargmax(values))])
        es = int(segment[int(np.nanargmin(values))])
        markers.append((float(timestamps[ed]), float(series[ed]), f"ED {series[ed]:.0f} ml"))
        markers.append((float(timestamps[es]), float(series[es]), f"ES {series[es]:.0f} ml"))
    logger.info("%s", curve.description)
    return PanelSpec(
        loaded=LoadedArray(
            name="lv_volume_curve",
            data_path="data/3d_lv_volume_curve",
            data=series,
            timestamps_path=None,
            timestamps=timestamps,
            sample_rate_hz=None,
            attrs={
                "line_label": "LV volume (ml)",
                "line_markers": tuple(markers),
                "volume_description": curve.description,
            },
            stream=None,
        ),
        kind="line",
        label="LV volume (ml)",
        view="beamspace",
    )


def _cartesian_arrays(
    loaded_arrays: tuple[LoadedArray, ...],
    *,
    blood_gate: BloodGateConfig | None = None,
    max_depth_m: float | None = None,
    tissue_brightness_lift: float = 0.0,
    tissue_gate: TissueGateConfig | None = None,
) -> tuple[LoadedArray, ...]:
    visual = tuple(loaded for loaded in loaded_arrays if _panel_kind(loaded) == "image")
    temporal = tuple(loaded for loaded in loaded_arrays if _panel_kind(loaded) != "image")
    if not visual:
        return loaded_arrays
    cartesian = cartesian_loaded_arrays(
        visual,
        blood_gate=blood_gate,
        max_depth_m=max_depth_m,
        tissue_brightness_lift=tissue_brightness_lift,
        tissue_gate=tissue_gate,
    )
    return (*(cartesian or visual), *temporal)


def _beamspace_arrays(loaded_arrays: tuple[LoadedArray, ...]) -> tuple[LoadedArray, ...]:
    color = _first_loaded(loaded_arrays, "data/2d_color_doppler_velocity") or _first_loaded(
        loaded_arrays, "data/2d_color_doppler_power"
    )
    if color is None:
        return loaded_arrays
    return tuple(_with_beamspace_color_extent(loaded, color) for loaded in loaded_arrays)


def _with_beamspace_color_extent(loaded: LoadedArray, color: LoadedArray) -> LoadedArray:
    attrs: dict[str, object] = {}
    color_sector = _loaded_sector_metadata(color)
    if color_sector is None:
        return loaded
    if loaded.data_path.startswith("data/2d_brightness_mode"):
        reference_sector = _loaded_sector_metadata(loaded)
        if reference_sector is not None:
            attrs["beamspace_reference_sector"] = reference_sector
            attrs["beamspace_color_doppler_sector"] = color_sector
    elif loaded.data_path in {"data/2d_color_doppler_velocity", "data/2d_color_doppler_power"}:
        attrs["beamspace_color_doppler_sector"] = color_sector
    if not attrs:
        return loaded
    return replace(loaded, attrs={**dict(loaded.attrs), **attrs})


def _first_loaded(loaded_arrays: tuple[LoadedArray, ...], data_path: str) -> LoadedArray | None:
    return next((loaded for loaded in loaded_arrays if loaded.data_path == data_path), None)


def _loaded_sector_metadata(loaded: LoadedArray) -> object | None:
    if loaded.stream is None:
        return None
    return loaded.stream.metadata.raw


def _validate_cartesian_color_doppler_pair(loaded_arrays: tuple[LoadedArray, ...]) -> None:
    paths = {loaded.data_path for loaded in loaded_arrays}
    has_velocity = "data/2d_color_doppler_velocity" in paths
    has_power = "data/2d_color_doppler_power" in paths
    if has_velocity == has_power:
        return
    missing = "data/2d_color_doppler_power" if has_velocity else "data/2d_color_doppler_velocity"
    present = "data/2d_color_doppler_velocity" if has_velocity else "data/2d_color_doppler_power"
    raise ValueError(
        "Cartesian color Doppler plotting requires velocity and power arrays together; "
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


def _ecg_marker_times(panels: tuple[PanelSpec, ...], time_s: float, frame_index: int) -> np.ndarray | None:
    if strip_drives_timeline(panels):
        # A single 2D frame has one fixed set of markers; the moving cursor is the timeline.
        return None
    for panel in panels:
        marker_times = panel.loaded.attrs.get("ecg_marker_timestamps")
        if marker_times is None:
            continue
        values = np.asarray(marker_times, dtype=np.float64)
        if values.ndim == 1:
            return values
        if values.ndim != 2 or values.shape[0] == 0:
            continue
        count = temporal_length(panel.loaded.data)
        resolved_index = nearest_index(panel.loaded.timestamps, time_s, count=count, fallback_index=frame_index)
        return cast(np.ndarray, values[min(max(0, resolved_index), values.shape[0] - 1)])
    return None


def _ecg_trace(store: RecordingStore) -> TraceSpec:
    if not _has_ecg(store):
        return TraceSpec(signal=np.zeros(0, dtype=np.float32), timestamps=np.zeros(0, dtype=np.float64), label="ECG")
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
    if text in {"beamspace", "raw"}:
        return "beamspace"
    if text == "cartesian":
        return "cartesian"
    if text == "both":
        return "both"
    raise ValueError(f"Unsupported plot view mode `{view_mode}`")


def _label(path: str) -> str:
    name = path.removeprefix("data/")
    for prefix in ("cartesian_", "1d_", "2d_", "3d_"):
        name = name.removeprefix(prefix)
    return name.replace("_", " ")
