"""Scan-converted echo images keep one centimetre horizontally equal to one centimetre vertically on screen."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from test_plotting import _write_mesh_only_store

from echoxflow.loading import LoadedArray
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.streams import StreamMetadata, stream_from_arrays


def _sector_bmode(shape: tuple[int, int], *, attrs: dict[str, object] | None = None) -> LoadedArray:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.13, tilt_rad=0.0, width_rad=0.9, grid_shape=shape
    )
    data = np.full((2, *shape), 90, dtype=np.uint8)
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    path = "data/2d_brightness_mode"
    stream = stream_from_arrays(
        data_path=path,
        data=data,
        timestamps_path="timestamps/2d_brightness_mode",
        timestamps=timestamps,
        sample_rate_hz=10.0,
        metadata=StreamMetadata(data_path=path, geometry=geometry),
    )
    return LoadedArray(
        name="2d_brightness_mode",
        data_path=path,
        data=data,
        timestamps_path="timestamps/2d_brightness_mode",
        timestamps=timestamps,
        sample_rate_hz=10.0,
        attrs=dict(attrs or {}),
        stream=stream,
    )


def _image_axes(figure) -> list[Axes]:
    return [ax for ax in figure.axes if ax.images and ax.images[0].get_array().ndim >= 2]


def _assert_true_aspect(figure) -> int:
    figure.canvas.draw()
    checked = 0
    for ax in _image_axes(figure):
        box = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        per_x = box.width / abs(x1 - x0)
        per_y = box.height / abs(y1 - y0)
        assert per_x == pytest.approx(per_y, rel=0.01)
        checked += 1
    return checked


def _render(renderer: RecordingPlotRenderer, panels):
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))
    return renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)


@pytest.mark.parametrize("with_overlay", [False, True])
def test_cartesian_2d_panel_has_equal_scale(with_overlay: bool) -> None:
    attrs: dict[str, object] = {}
    if with_overlay:
        contour = np.asarray([[[-0.01, 0.04], [0.0, 0.06], [0.01, 0.04]]] * 2, dtype=np.float32)
        attrs["annotation_overlays"] = (
            {
                "label": "contour",
                "type": "polyline",
                "points": contour,
                "timestamps": np.asarray([0.0, 0.1], dtype=np.float32),
            },
        )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=640, height_px=300, dpi=100))
    panels = renderer.build_panel_specs((_sector_bmode((24, 40), attrs=attrs),), view_mode="cartesian")
    figure = _render(renderer, panels)
    try:
        assert _assert_true_aspect(figure) == 1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("curve_ratio", [0.3, 0.45])
def test_3d_mosaic_with_volume_curve_has_equal_scale(tmp_path: Path, curve_ratio: float) -> None:
    _write_mesh_only_store(tmp_path, linked=True, closed=True)
    renderer = RecordingPlotRenderer(
        style=PlotStyle(width_px=900, height_px=700, dpi=100, volume_curve_height_ratio=curve_ratio)
    )
    panels, ecg = renderer._load_specs(
        tmp_path / "mesh.zarr", root=None, modalities=None, view_mode="cartesian", show_annotations=True
    )
    assert any(panel.kind == "line" for panel in panels)
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        assert _assert_true_aspect(figure) == 1
        mosaic_ax, curve_ax = figure.axes[:2]
        # Measure allocated rows: the mosaic axes can shrink further to retain physical aspect.
        assert curve_ax.get_position(original=True).height / mosaic_ax.get_position(
            original=True
        ).height == pytest.approx(curve_ratio)
    finally:
        plt.close(figure)
