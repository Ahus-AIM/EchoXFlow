import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from echoxflow.colors import data_modality_colormap_name, named_listed_colormap
from echoxflow.croissant import RecordingRecord
from echoxflow.loading import LoadedArray, RecordingStore, open_recording
from echoxflow.mesh import PackedMeshAnnotation
from echoxflow.plotting.annotations import attach_annotation_overlays
from echoxflow.plotting.colorbar import colorbar_spec_for_modality
from echoxflow.plotting.panels import renderer_for
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import FrameRequest, PanelSpec, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import nearest_index, resolve_frame_time, select_timeline
from echoxflow.plotting.writers import figure_to_rgb, write_video
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.schema import color_flow_velocity_codes, flow_color_velocity_ramp
from echoxflow.spectral import SpectralMetadata
from echoxflow.streams import StreamMetadata, stream_from_arrays

zarr = pytest.importorskip("zarr")


@pytest.mark.parametrize("mode", ["pulsed", "continuous"])
@pytest.mark.parametrize("with_table", [True, False])
def test_spectral_renderer_uses_recording_rgb_codes_or_configured_fallback(
    tmp_path: Path, mode: str, with_table: bool
) -> None:
    path = tmp_path / "spectrum.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=2)
    data_path = f"data/1d_{mode}_wave_doppler"
    codes = np.asarray([0, 64, 128, 255], dtype=np.uint8)
    group.create_array(data_path, data=np.tile(codes, (4, 1)))
    group.create_array(f"timestamps/1d_{mode}_wave_doppler", data=np.arange(4, dtype=np.float64))
    table = np.column_stack((np.arange(256), np.arange(256) // 2, np.zeros(256))).astype(np.uint8)
    if with_table:
        group.create_array("data/lookup_tables/gold", data=table)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300))
    panels, ecg = renderer._load_specs(path, root=None, modalities=None, view_mode="beamspace", show_annotations=False)
    assert panels[0].loaded.stream is not None
    assert panels[0].loaded.stream.metadata.colormap_path == ("data/lookup_tables/gold" if with_table else None)
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0)
    try:
        image = figure.axes[0].images[0]
        expected = table[codes] / 255 if with_table else named_listed_colormap(data_path)(image.norm(codes))[:, :3]
        np.testing.assert_allclose(image.cmap(image.norm(codes))[:, :3], expected)
    finally:
        plt.close(figure)


def _loaded(name: str, data: np.ndarray, timestamps: np.ndarray | None) -> LoadedArray:
    data_path = f"data/{name}"
    stream = None
    if name == "tissue_doppler":
        stream = stream_from_arrays(
            data_path=data_path,
            data=data,
            timestamps_path=None if timestamps is None else f"timestamps/{name}",
            timestamps=timestamps,
            sample_rate_hz=None,
            metadata=StreamMetadata(data_path=data_path, velocity_limit_mps=0.2),
        )
    return LoadedArray(
        name=name,
        data_path=data_path,
        data=data,
        timestamps_path=None if timestamps is None else f"timestamps/{name}",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={},
        stream=stream,
    )


def test_timeline_ignores_ecg_and_1d_drivers_and_caps_fps() -> None:
    visual = PanelSpec(
        loaded=_loaded(
            "2d_brightness_mode",
            np.zeros((20, 4, 4), dtype=np.float32),
            np.arange(20, dtype=np.float64) / 120.0,
        ),
        kind="image",
        label="B-mode",
    )
    spectral = PanelSpec(
        loaded=_loaded(
            "1d_pulsed_wave_doppler",
            np.zeros((200, 12), dtype=np.float32),
            np.arange(200, dtype=np.float64) / 400.0,
        ),
        kind="matrix",
        label="PW",
    )

    timeline = select_timeline((spectral, visual), max_fps=60.0)

    assert timeline.fps == 60.0
    assert np.isclose(timeline.timestamps[1] - timeline.timestamps[0], 1.0 / 60.0)


def test_nearest_index_uses_timestamp_distance() -> None:
    assert nearest_index(np.asarray([0.0, 0.1, 0.2]), 0.16, count=3) == 2


def test_layout_uses_configured_background_and_bottom_ecg() -> None:
    style = PlotStyle.from_config()
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    image_panel = PanelSpec(
        loaded=_loaded("2d_brightness_mode", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="B-mode",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(image_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert style.figure_facecolor == "#222222"
        assert style.panel_facecolor == "#222222"
        assert to_hex(figure.get_facecolor(), keep_alpha=False) == "#222222"
        assert len(figure.axes[0].get_xticks()) == 0
        assert len(figure.axes[0].get_yticks()) == 0
        assert all(not spine.get_visible() for spine in figure.axes[0].spines.values())
        assert figure.axes[-1].get_title() == ""
        assert len(figure.axes[-1].get_xticks()) == 0
        assert len(figure.axes[-1].get_yticks()) == 0
        assert all(not spine.get_visible() for spine in figure.axes[-1].spines.values())
    finally:
        plt.close(figure)


def test_single_beamspace_image_preserves_aspect_ratio() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    image_panel = PanelSpec(
        loaded=_loaded("2d_brightness_mode", np.zeros((2, 8, 24), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="B-mode",
        view="beamspace",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(image_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert figure.axes[0].get_aspect() == 1.0
    finally:
        plt.close(figure)


def test_multiple_beamspace_images_keep_automatic_aspect() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = (
        PanelSpec(
            loaded=_loaded("2d_brightness_mode", np.zeros((2, 8, 24), dtype=np.float32), np.asarray([0.0, 0.1])),
            kind="image",
            label="B-mode",
            view="beamspace",
        ),
        PanelSpec(
            loaded=_loaded("tissue_doppler", np.zeros((2, 8, 24), dtype=np.float32), np.asarray([0.0, 0.1])),
            kind="image",
            label="TDI",
            view="beamspace",
        ),
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert figure.axes[0].get_aspect() == "auto"
        assert figure.axes[1].get_aspect() == "auto"
    finally:
        plt.close(figure)


def test_beamspace_doppler_panel_omits_colorbar() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    tissue_panel = PanelSpec(
        loaded=_loaded("tissue_doppler", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="TDI",
        view="beamspace",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(tissue_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].child_axes) == 0
    finally:
        plt.close(figure)


def test_color_doppler_velocity_colorbar_uses_colormap_orientation() -> None:
    spec = colorbar_spec_for_modality("data/2d_color_doppler_velocity", value_range=(-0.8, 0.8))

    assert spec is not None
    assert spec.reverse_cmap is False
    assert spec.tick_labels == ("-0.80", "0.00", "0.80")
    assert spec.bbox_axes[0] + spec.bbox_axes[2] == pytest.approx(0.98)
    assert spec.bbox_axes[2] <= 0.04  # narrow bar
    # The bar leaves headroom above it for the upper limit value.
    assert spec.bbox_axes[1] + spec.bbox_axes[3] <= 0.9


def test_tissue_doppler_colorbar_displays_cm_per_second() -> None:
    spec = colorbar_spec_for_modality("data/tissue_doppler", value_range=(-0.2, 0.2))

    assert spec is not None
    assert spec.label == "Velocity (cm/s)"
    assert spec.value_range == (-20.0, 20.0)
    assert spec.ticks == (-20.0, 0.0, 20.0)
    assert spec.tick_labels == ("-20", "0", "20")


@pytest.mark.parametrize("show_colorbars, show_labels, expected", [(False, True, None), (True, False, [])])
def test_colorbar_and_label_switches(show_colorbars: bool, show_labels: bool, expected) -> None:
    style = PlotStyle(
        width_px=420, height_px=300, dpi=100, show_colorbars=show_colorbars, show_colorbar_labels=show_labels
    )
    renderer = RecordingPlotRenderer(style=style)
    tissue_panel = PanelSpec(
        loaded=_loaded("tissue_doppler", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="TDI",
        view="cartesian",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(tissue_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        if expected is None:
            assert figure.axes[0].child_axes == []  # no colour bar at all
        else:
            assert [text.get_text() for text in figure.axes[0].child_axes[0].texts] == expected  # bar, no limits
    finally:
        plt.close(figure)


def test_cartesian_doppler_panel_adds_fixed_top_right_colorbar() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    tissue_panel = PanelSpec(
        loaded=_loaded("tissue_doppler", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="TDI",
        view="cartesian",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(tissue_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].child_axes) == 1
        colorbar_ax = figure.axes[0].child_axes[0]
        bounds = colorbar_ax.get_position().bounds
        assert bounds[0] > 0.65
        assert bounds[1] > 0.45
        assert [text.get_text() for text in colorbar_ax.texts] == ["20", "-20"]  # no title, limits only
        assert all(text.get_fontsize() == 16.0 for text in colorbar_ax.texts)
        assert [label.get_text() for label in colorbar_ax.get_yticklabels()] == [""]  # unlabelled centre tick
        assert len(colorbar_ax.images) == 1
        assert colorbar_ax.images[0].get_interpolation() == "nearest"
    finally:
        plt.close(figure)


def test_plot_view_modes_build_expected_panel_sets() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    loaded = (
        _loaded("2d_brightness_mode", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        _loaded("1d_pulsed_wave_doppler", np.zeros((3, 6), dtype=np.float32), np.asarray([0.0, 0.05, 0.1])),
    )

    beamspace = renderer.build_panel_specs(loaded, view_mode="beamspace")
    cartesian = renderer.build_panel_specs(loaded, view_mode="cartesian")
    both = renderer.build_panel_specs(loaded, view_mode="both")

    assert [panel.view for panel in beamspace] == ["beamspace", "beamspace"]
    assert [panel.view for panel in cartesian] == ["cartesian", "cartesian"]
    assert [panel.view for panel in both] == ["beamspace", "cartesian", "cartesian"]
    assert [panel.label for panel in both] == [
        "brightness mode",
        "brightness mode",
        "pulsed wave doppler",
    ]


def test_malformed_annotation_overlay_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    class Store:
        array_paths = ("data/overlay_physical_points",)
        group = type("Group", (), {"attrs": {}})()

        def load_array(self, path: str) -> object:
            return [["not-a-number"]]

    loaded = LoadedArray(
        name="2d_brightness_mode",
        data_path="data/2d_brightness_mode",
        data=np.zeros((1, 2, 2), dtype=np.float32),
        timestamps_path=None,
        timestamps=None,
        sample_rate_hz=None,
        attrs={},
        stream=None,
    )

    result = attach_annotation_overlays(Store(), (loaded,))

    assert "annotation_overlays" not in result[0].attrs
    assert any("could not parse annotation overlay" in record.getMessage() for record in caplog.records)


def test_mmode_panel_uses_physical_depth_for_annotations_and_y_axis() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.02,
        depth_end_m=0.10,
        tilt_rad=0.0,
        width_rad=0.6,
        grid_shape=(8, 3),
    )
    data = np.zeros((3, 8), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    stream = stream_from_arrays(
        data_path="data/1d_motion_mode",
        data=data,
        timestamps_path="timestamps/1d_motion_mode",
        timestamps=timestamps,
        sample_rate_hz=None,
        metadata=StreamMetadata(data_path="data/1d_motion_mode", value_range=(0.0, 255.0), geometry=geometry),
    )
    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={
            "annotation_overlays": ({"kind": "spectral_points", "points": np.asarray([[0.1, 0.06]], dtype=np.float32)},)
        },
        stream=stream,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="M-mode")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.isclose(float(markers[0].get_ydata()[0]), 4.0)
        assert ax.get_ylabel() == ""
        assert [text.get_text() for text in ax.texts if text.get_text() == "[cm]"] == ["[cm]"]
        assert np.allclose(ax.get_yticks(), (0.0, 8.0))
        assert [label.get_text() for label in ax.get_yticklabels()] == ["0", "8"]
        assert ax.yaxis.get_visible()
        assert ax.spines["left"].get_visible()
        assert all(label.get_visible() for label in ax.get_yticklabels())
        text_labels = [text.get_text() for text in ax.texts]
        assert "0 cm" not in text_labels
        assert "8 cm" not in text_labels
        assert "4 cm" not in text_labels
        assert np.allclose(ax.get_xlim(), (0.0, 0.2))
        assert np.allclose(ax.get_ylim(), (8.0, 0.0))
    finally:
        plt.close(figure)


def test_mmode_panel_uses_track_y_range_for_centimeter_annotations() -> None:
    data = np.zeros((3, 8), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    stream = stream_from_arrays(
        data_path="data/1d_motion_mode",
        data=data,
        timestamps_path="timestamps/1d_motion_mode",
        timestamps=timestamps,
        sample_rate_hz=None,
        metadata=StreamMetadata(data_path="data/1d_motion_mode", value_range=(0.0, 255.0)),
    )
    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={
            "annotation_overlays": ({"kind": "spectral_points", "points": np.asarray([[0.1, 6.5]], dtype=np.float32)},),
            "spectral_metadata": SpectralMetadata(data_path="data/1d_motion_mode", raw={"y_range": [0.0, 17.0]}),
        },
        stream=stream,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="M-mode")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.isclose(float(markers[0].get_xdata()[0]), 0.1)
        assert np.isclose(float(markers[0].get_ydata()[0]), 6.5)
        assert ax.get_ylabel() == ""
        assert [text.get_text() for text in ax.texts if text.get_text() == "[cm]"] == ["[cm]"]
        assert np.allclose(ax.get_ylim(), (17.0, 0.0))
    finally:
        plt.close(figure)


def test_matrix_annotation_trace_draws_one_label_for_many_points() -> None:
    data = np.zeros((100, 8), dtype=np.float32)
    timestamps = np.linspace(0.0, 0.99, 100, dtype=np.float32)
    trace_points = np.column_stack(
        [
            timestamps,
            np.linspace(1.0, 6.0, 100, dtype=np.float32),
        ]
    ).astype(np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": (
                {"kind": "spectral_points", "points": trace_points, "label": "MV Eprime Septal Velocity"},
            )
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.5, 1.0]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.5, frame_index=0, dpi=100)
    try:
        text_labels = [text.get_text() for text in figure.axes[0].texts]
        assert text_labels.count("MV Eprime Septal Velocity") == 1
    finally:
        plt.close(figure)


def test_annotation_label_strips_known_source_affixes() -> None:
    data = np.zeros((100, 8), dtype=np.float32)
    timestamps = np.linspace(0.0, 0.99, 100, dtype=np.float32)
    trace_points = np.column_stack(
        [
            timestamps,
            np.linspace(1.0, 6.0, 100, dtype=np.float32),
        ]
    ).astype(np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": (
                {"kind": "spectral_points", "points": trace_points, "label": "Cardiac/SD/Aortic/LVOT Trace/Manual"},
            )
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.5, 1.0]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.5, frame_index=0, dpi=100)
    try:
        text_labels = [text.get_text() for text in figure.axes[0].texts]
        assert text_labels.count("Aortic/LVOT Trace") == 1
        assert "Cardiac/SD/Aortic/LVOT Trace/Manual" not in text_labels
    finally:
        plt.close(figure)


def test_mmode_panel_skips_depth_annotations_without_geometry() -> None:
    data = np.zeros((3, 8), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    stream = stream_from_arrays(
        data_path="data/1d_motion_mode",
        data=data,
        timestamps_path="timestamps/1d_motion_mode",
        timestamps=timestamps,
        sample_rate_hz=None,
        metadata=StreamMetadata(data_path="data/1d_motion_mode", value_range=(0.0, 255.0)),
    )
    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={
            "annotation_overlays": ({"kind": "spectral_points", "points": np.asarray([[0.1, 0.06]], dtype=np.float32)},)
        },
        stream=stream,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="M-mode")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers == []
        assert ax.get_ylabel() == ""
        assert [label.get_text() for label in ax.get_yticklabels()] == []
    finally:
        plt.close(figure)


def test_spectral_doppler_panel_adds_velocity_y_axis() -> None:
    data = np.zeros((3, 5), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "spectral_metadata": SpectralMetadata(
                data_path="data/1d_pulsed_wave_doppler",
                row_velocity_mps=np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
            )
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        assert ax.get_ylabel() == ""
        (unit_tag,) = [text for text in ax.texts if text.get_text() == "[m/s]"]
        assert unit_tag.get_position()[0] > 1.0  # just outside the right edge of the strip
        assert np.isclose(unit_tag.get_position()[1], 2.5)  # at the zero-velocity baseline
        assert to_hex(unit_tag.get_color()) == renderer.style.text_dim_color
        assert len(ax.get_yticks()) > 0
        assert any(label.get_text() == "0" for label in ax.get_yticklabels())
        assert np.allclose(ax.get_xlim(), (0.0, 0.2))
        assert np.allclose(ax.get_ylim(), (0.0, 5.0))
    finally:
        plt.close(figure)


def test_spectral_doppler_markers_follow_displayed_row_orientation() -> None:
    data = np.zeros((3, 3), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": (
                {
                    "kind": "spectral_points",
                    "points": np.asarray([[0.1, 1.0], [0.1, 0.0], [0.1, -1.0]], dtype=np.float32),
                },
            ),
            "spectral_metadata": SpectralMetadata(
                data_path="data/1d_pulsed_wave_doppler",
                row_velocity_mps=np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
            ),
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.allclose(markers[0].get_ydata(), [2.5, 1.5, 0.5])
        label_to_tick = {label.get_text(): float(tick) for label, tick in zip(ax.get_yticklabels(), ax.get_yticks())}
        assert label_to_tick["1"] > label_to_tick["0"] > label_to_tick["-1"]
    finally:
        plt.close(figure)


def test_spectral_doppler_markers_follow_shifted_baseline() -> None:
    data = np.zeros((3, 5), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": (
                {
                    "kind": "spectral_points",
                    "points": np.asarray([[0.1, 1.0], [0.1, 0.0], [0.1, -1.0]], dtype=np.float32),
                },
            ),
            "spectral_metadata": SpectralMetadata(
                data_path="data/1d_pulsed_wave_doppler",
                baseline_row=1.0,
                nyquist_limit_mps=1.0,
                row_velocity_mps=np.asarray([1.0, 0.0, -1.0 / 3.0, -2.0 / 3.0, -1.0], dtype=np.float32),
            ),
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.allclose(markers[0].get_ydata(), [4.5, 3.5, 0.5])
        label_to_tick = {label.get_text(): float(tick) for label, tick in zip(ax.get_yticklabels(), ax.get_yticks())}
        assert label_to_tick["1"] == pytest.approx(4.5)
        assert label_to_tick["0"] == pytest.approx(3.5)
    finally:
        plt.close(figure)


def test_spectral_doppler_trace_points_use_velocity_row_centers() -> None:
    data = np.zeros((5, 5), dtype=np.float32)
    timestamps = np.linspace(0.0, 0.4, 5, dtype=np.float32)
    trace_points = np.column_stack(
        [
            timestamps,
            np.asarray([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": ({"kind": "spectral_points", "points": trace_points, "label": "Aortic/AV Trace"},),
            "spectral_metadata": SpectralMetadata(
                data_path="data/1d_pulsed_wave_doppler",
                row_velocity_mps=np.asarray([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32),
            ),
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.2, 0.4]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.2, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.allclose(markers[0].get_xdata(), timestamps)
        assert np.allclose(markers[0].get_ydata(), [4.5, 3.5, 2.5, 1.5, 0.5])
        assert "Aortic/AV Trace" in [text.get_text() for text in ax.texts]
    finally:
        plt.close(figure)


def test_spectral_doppler_trace_points_convert_centimeters_per_second() -> None:
    data = np.zeros((3, 5), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    trace_points = np.column_stack([timestamps, np.asarray([0.0, -40.0, -80.0], dtype=np.float32)]).astype(np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": ({"kind": "spectral_points", "points": trace_points, "label": "LVOT Trace"},),
            "spectral_metadata": SpectralMetadata(
                data_path="data/1d_pulsed_wave_doppler",
                row_velocity_mps=np.asarray([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32),
            ),
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=timestamps)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.allclose(markers[0].get_ydata(), [2.5, 1.7, 0.9], atol=1e-6)
    finally:
        plt.close(figure)


def test_spectral_doppler_trace_points_can_use_top_origin_rows() -> None:
    data = np.zeros((3, 5), dtype=np.float32)
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    trace_points = np.column_stack([timestamps, np.asarray([0.0, 2.0, 4.0], dtype=np.float32)]).astype(np.float32)
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=data,
        timestamps_path="timestamps/1d_pulsed_wave_doppler",
        timestamps=timestamps,
        sample_rate_hz=None,
        attrs={
            "annotation_overlays": ({"kind": "spectral_points", "points": trace_points, "y_coordinate_system": "row"},),
            "spectral_metadata": SpectralMetadata(
                data_path="data/1d_pulsed_wave_doppler",
                row_velocity_mps=np.asarray([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32),
            ),
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=timestamps)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.1, frame_index=0, dpi=100)
    try:
        ax = figure.axes[0]
        markers = [line for line in ax.lines if line.get_marker() == "+"]
        assert markers
        assert np.allclose(markers[0].get_ydata(), [4.5, 2.5, 0.5])
    finally:
        plt.close(figure)


def test_spectral_annotation_overlays_carry_unit_and_coordinate_hints(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((3, 5), dtype=np.float32))
    annotation = group.create_array(
        "data/1d_pulsed_wave_doppler_annotation_lvot_trace",
        data=np.asarray([[0.0, -40.0]], dtype=np.float32),
    )
    annotation.attrs["y_coordinate_system"] = "spectral_velocity"
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "tracks": [{"semantic_id": "1d_pulsed_wave_doppler"}],
        "annotations": [
            {
                "label": "LVOT Trace",
                "target": {"field": "trace_points"},
                "value": {
                    "format": "zarr_array",
                    "zarr_path": "data/1d_pulsed_wave_doppler_annotation_lvot_trace",
                    "y_unit": "cm/s",
                },
            }
        ],
    }
    loaded = LoadedArray(
        name="1d_pulsed_wave_doppler",
        data_path="data/1d_pulsed_wave_doppler",
        data=np.zeros((3, 5), dtype=np.float32),
        timestamps_path=None,
        timestamps=None,
        sample_rate_hz=None,
        attrs={},
        stream=None,
    )

    (loaded_with_overlays,) = attach_annotation_overlays(RecordingStore(tmp_path / "case.zarr", group), (loaded,))

    overlay = loaded_with_overlays.attrs["annotation_overlays"][0]
    assert overlay["kind"] == "spectral_points"
    assert overlay["label"] == "LVOT Trace"
    assert overlay["y_unit"] == "cm/s"
    assert overlay["y_coordinate_system"] == "spectral_velocity"


def test_3d_brightness_mode_builds_single_mosaic_panel() -> None:
    data = np.zeros((2, 5, 7, 9), dtype=np.uint8)
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    stream = stream_from_arrays(
        data_path="data/3d_brightness_mode",
        data=data,
        timestamps_path="timestamps/3d_brightness_mode",
        timestamps=timestamps,
        sample_rate_hz=10.0,
        metadata=StreamMetadata(
            data_path="data/3d_brightness_mode",
            value_range=(0.0, 255.0),
            raw={
                "sectors": [
                    {
                        "semantic_id": "bmode",
                        "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                        "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                        "geometry": {
                            "coordinate_system": "spherical_sector_3d",
                            "DepthStart": 0.01,
                            "DepthEnd": 0.12,
                            "Width": 0.8,
                            "ElevationWidth": 0.7,
                        },
                    }
                ],
                "render_metadata": {
                    "DepthStart": 0.01,
                    "DepthEnd": 0.12,
                    "Width": 0.8,
                    "ElevationWidth": 0.7,
                },
            },
        ),
    )
    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={},
        stream=stream,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    beamspace = renderer.build_panel_specs((loaded,), view_mode="beamspace")
    cartesian = renderer.build_panel_specs((loaded,), view_mode="cartesian")

    assert len(beamspace) == 1
    cell_height = round(0.28 * 300)
    assert beamspace[0].loaded.data.shape[:2] == (2, 3 * cell_height)
    assert beamspace[0].label == "brightness mode"
    assert cartesian[0].loaded.data.shape[0:2] == (2, 3 * cell_height)
    assert cartesian[0].loaded.data.shape[2] > 4 * cell_height
    assert cartesian[0].view == "cartesian"


def test_3d_brightness_mode_panel_uses_beat_stitched_timeline() -> None:
    data = np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1)
    timestamps = np.asarray([10.05, 10.95, 11.05, 11.95, 12.05, 12.95, 13.05, 13.95], dtype=np.float64)
    raw = {
        "stitch_beat_count": 4,
        "metadata": {
            "time_reference": {"volume_time_origin_s": 10.0},
            "qrs_trigger_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        "sectors": [
            {
                "semantic_id": "bmode",
                "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.01,
                    "DepthEnd": 0.12,
                    "Width": 0.8,
                    "ElevationWidth": 0.7,
                },
            }
        ],
        "volume_time_reference": {"volume_time_origin_s": 10.0},
        "volume_qrs_trigger_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        "render_metadata": {
            "DepthStart": 0.01,
            "DepthEnd": 0.12,
            "Width": 0.8,
            "ElevationWidth": 0.7,
        },
    }
    stream = stream_from_arrays(
        data_path="data/3d_brightness_mode",
        data=data,
        timestamps_path="timestamps/3d_brightness_mode",
        timestamps=timestamps,
        sample_rate_hz=2.0,
        metadata=StreamMetadata(data_path="data/3d_brightness_mode", value_range=(0.0, 255.0), raw=raw),
    )
    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={},
        stream=stream,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    (panel,) = renderer.build_panel_specs((loaded,), view_mode="beamspace")

    assert panel.loaded.attrs["3d_was_beat_stitched"] is True
    assert panel.loaded.attrs["3d_stitch_beat_count"] == 4
    assert panel.loaded.data.shape[:2] == (2, 3 * round(0.28 * 300))
    assert np.allclose(panel.loaded.timestamps, [0.05, 0.95])
    np.testing.assert_allclose(
        panel.loaded.attrs["ecg_marker_timestamps"],
        [
            [3.05, 2.05, 1.05, 0.05],
            [3.95, 2.95, 1.95, 0.95],
        ],
        atol=1e-6,
    )


def test_stitched_3d_panel_draws_all_source_times_on_ecg() -> None:
    loaded = LoadedArray(
        name="3d_brightness_mode_mosaic",
        data_path="data/3d_brightness_mode",
        data=np.zeros((2, 6, 6), dtype=np.uint8),
        timestamps_path="timestamps/3d_brightness_mode",
        timestamps=np.asarray([0.1, 0.2], dtype=np.float64),
        sample_rate_hz=None,
        attrs={
            "ecg_marker_timestamps": np.asarray(
                [
                    [0.1, 1.1, 2.1],
                    [0.2, 1.2, 2.2],
                ],
                dtype=np.float64,
            )
        },
        stream=None,
    )
    panel = PanelSpec(loaded=loaded, kind="image", label="brightness mode")
    ecg = TraceSpec(signal=np.zeros(40, dtype=np.float32), timestamps=np.linspace(0.0, 3.0, 40))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.2, frame_index=1, dpi=100)
    try:
        ecg_ax = figure.axes[-1]
        vertical_times = [
            float(np.asarray(line.get_xdata(), dtype=np.float64)[0])
            for line in ecg_ax.lines
            if np.asarray(line.get_xdata()).size == 2 and np.allclose(line.get_xdata(), line.get_xdata()[0])
        ]
        np.testing.assert_allclose(vertical_times, [0.2, 1.2, 2.2])
    finally:
        plt.close(figure)


def test_3d_brightness_mode_panel_uses_croissant_stitch_beat_count_for_display_stitching(tmp_path: Path) -> None:
    timestamps = np.asarray([10.05, 10.95, 11.05, 11.95, 12.05, 12.95, 13.05, 13.95], dtype=np.float64)
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1))
    group.create_array(
        "timestamps/3d_brightness_mode",
        data=timestamps,
    )
    group.attrs["recording_manifest"] = {
        "manifest_type": "3d",
        "sectors": [
            {
                "semantic_id": "bmode",
                "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.01,
                    "DepthEnd": 0.12,
                    "Width": 0.8,
                    "ElevationWidth": 0.7,
                },
            }
        ],
        "metadata": {
            "time_reference": {"volume_time_origin_s": 10.0},
            "qrs_trigger_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
    }
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("3d_brightness_mode",),
        content_types=("3d_brightness_mode",),
        frame_counts_by_content_type={"3d_brightness_mode": 8},
        median_delta_time_by_content_type={"3d_brightness_mode": 1.0},
        array_paths=("data/3d_brightness_mode", "timestamps/3d_brightness_mode"),
        stitch_beat_count=4,
    )
    loaded = open_recording(record, root=tmp_path).load_modality("3d_brightness_mode")
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    (panel,) = renderer.build_panel_specs((loaded,), view_mode="beamspace")

    assert loaded.stream is not None
    assert loaded.stream.metadata.stitch_beat_count == 4
    assert panel.loaded.attrs["3d_stitch_beat_count"] == 4
    assert panel.loaded.attrs["3d_was_beat_stitched"] is True
    assert panel.loaded.data.shape[0] == 2
    assert np.allclose(panel.loaded.timestamps, [0.05, 0.95])


def test_recording_plotter_loads_annotation_overlays_by_default(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    group.create_array(
        "data/bmode_overlay_physical_points_00",
        data=np.asarray([[0.0, 0.04], [0.01, 0.05]], dtype=np.float32),
    )
    group.create_array(
        "data/bmode_overlay_physical_points_01",
        data=np.asarray([[-0.01, 0.03]], dtype=np.float32),
    )
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [
            {
                "semantic_id": "bmode",
                "geometry": {
                    "depth_start_m": 0.0,
                    "depth_end_m": 0.08,
                    "tilt_rad": 0.0,
                    "width_rad": 0.6,
                    "grid_size": [8, 8],
                },
            }
        ],
        "annotations": [
            {
                "kind": "physical_geometry",
                "label": "Aortic point",
                "links": {"geometry_kind": "physical_point"},
                "target": {"type": "sector", "semantic_id": "bmode", "field": "point_coordinates"},
                "value": {"format": "zarr_array", "zarr_path": "data/bmode_overlay_physical_points_00"},
            },
            {
                "kind": "physical_geometry",
                "label": "Mitral point",
                "links": {"geometry_kind": "physical_point"},
                "target": {"type": "sector", "semantic_id": "bmode", "field": "point_coordinates"},
                "value": {"format": "zarr_array", "zarr_path": "data/bmode_overlay_physical_points_01"},
            },
        ],
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode",),
        content_types=("2d_brightness_mode",),
        frame_counts_by_content_type={"2d_brightness_mode": 1},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/ecg",
            "timestamps/ecg",
            "data/bmode_overlay_physical_points_00",
            "data/bmode_overlay_physical_points_01",
        ),
    )

    panels, _ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode",),
        view_mode="beamspace",
        show_annotations=True,
    )
    disabled_panels, _disabled_ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode",),
        view_mode="beamspace",
        show_annotations=False,
    )

    assert panels[0].loaded.attrs["annotation_overlays"][0]["kind"] == "physical_points"
    assert {overlay["label"] for overlay in panels[0].loaded.attrs["annotation_overlays"]} == {
        "Aortic point",
        "Mitral point",
    }
    assert "annotation_overlays" not in disabled_panels[0].loaded.attrs

    figure = renderer.render_figure_from_specs(panels=panels, ecg=_ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        text_labels = [text.get_text() for text in figure.axes[0].texts]
        assert text_labels.count("Aortic point") == 1
        assert text_labels.count("Mitral point") == 1
    finally:
        plt.close(figure)


def test_tissue_doppler_gate_draws_beamspace_vertical_marker(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/tissue_doppler", data=np.full((1, 8, 8), 128, dtype=np.uint8))
    group.create_array("timestamps/tissue_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((1, 8), dtype=np.float32))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    gate = {
        "kind": "tissue_doppler_gate",
        "sector_semantic_id": "tissue_doppler",
        "gate_center_depth_m": 0.04,
        "gate_tilt_rad": 0.1,
        "gate_sample_volume_m": 0.01,
    }
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [
            {
                "semantic_id": "tissue_doppler",
                "velocity_scale_mps": 0.2,
                "sampling_gate_metadata": gate,
                "geometry": {
                    "depth_start_m": 0.0,
                    "depth_end_m": 0.08,
                    "tilt_rad": 0.0,
                    "width_rad": 0.6,
                    "grid_size": [8, 8],
                },
            }
        ],
        "tracks": [{"semantic_id": "tissue_doppler_gate", "kind": "scatter", "derived_from": gate}],
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("tissue_doppler", "1d_pulsed_wave_doppler"),
        content_types=("tissue_doppler", "1d_pulsed_wave_doppler"),
        frame_counts_by_content_type={"tissue_doppler": 1, "1d_pulsed_wave_doppler": 1},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/tissue_doppler",
            "timestamps/tissue_doppler",
            "data/1d_pulsed_wave_doppler",
            "timestamps/1d_pulsed_wave_doppler",
            "data/ecg",
            "timestamps/ecg",
        ),
    )

    panels, ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("tissue_doppler", "1d_pulsed_wave_doppler"),
        view_mode="beamspace",
        show_annotations=True,
    )

    overlays = panels[0].loaded.attrs["annotation_overlays"]
    assert overlays[0]["kind"] == "sampling_gate"
    assert overlays[0]["points"].shape == (2, 2)
    assert overlays[0]["tick_points"].shape == (2, 2)

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        lines = figure.axes[0].lines
        foreground = [line for line in lines if to_hex(line.get_color(), keep_alpha=False) == "#4ac44a"]
        assert any(line.get_marker() == "_" and line.get_linestyle() == "None" for line in foreground)
        assert not any(to_hex(line.get_color(), keep_alpha=False) == "#000000" for line in lines)
        dashed = [line for line in foreground if line.get_linestyle() == "--"]
        assert len(dashed) == 2
        assert all(np.allclose(line.get_xdata(), line.get_xdata()[0]) for line in dashed)
        y_ranges = sorted((float(line.get_ydata()[0]), float(line.get_ydata()[-1])) for line in dashed)
        assert np.isclose(y_ranges[0][0], 0.0)
        assert y_ranges[0][1] < y_ranges[1][0]
        assert np.isclose(y_ranges[1][1], 7.0)
    finally:
        plt.close(figure)


def test_spectral_gate_found_behind_pose_less_scatter_track(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/tissue_doppler", data=np.full((1, 8, 8), 128, dtype=np.uint8))
    group.create_array("timestamps/tissue_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((1, 8), dtype=np.float32))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    geometry = {
        "depth_start_m": 0.0,
        "depth_end_m": 0.08,
        "tilt_rad": 0.0,
        "width_rad": 0.6,
        "grid_size": [8, 8],
    }
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [
            {"semantic_id": "2d_brightness_mode", "geometry": geometry},
            {"semantic_id": "tissue_doppler", "velocity_scale_mps": 0.2, "geometry": geometry},
        ],
        "tracks": [{"semantic_id": "tissue_doppler_gate", "kind": "scatter", "nyquist_mps": 0.2, "num_pixels": 4}],
        "sampling_gate_metadata": {
            "gate_center_depth_m": 0.04,
            "gate_tilt_rad": 0.1,
            "gate_sample_volume_m": 0.01,
        },
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    content_types = ("2d_brightness_mode", "tissue_doppler", "1d_pulsed_wave_doppler")
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=content_types,
        content_types=content_types,
        frame_counts_by_content_type=dict.fromkeys(content_types, 1),
        median_delta_time_by_content_type={},
        array_paths=(
            *(f"data/{name}" for name in content_types),
            *(f"timestamps/{name}" for name in content_types),
            "data/ecg",
            "timestamps/ecg",
        ),
    )

    panels, _ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=content_types,
        view_mode="beamspace",
        show_annotations=True,
    )

    gated = {
        panel.loaded.data_path: [o["kind"] for o in panel.loaded.attrs.get("annotation_overlays", ())]
        for panel in panels
    }
    assert "sampling_gate" in gated["data/2d_brightness_mode"]
    assert any("sampling_gate" in kinds for path, kinds in gated.items() if "tissue_doppler" in path)


def test_mmode_sampling_line_draws_in_beamspace_and_cartesian_views(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_motion_mode", data=np.zeros((3, 8), dtype=np.float32))
    group.create_array("timestamps/1d_motion_mode", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    geometry = {
        "depth_start_m": 0.0,
        "depth_end_m": 0.08,
        "tilt_rad": 0.0,
        "width_rad": 0.6,
        "grid_size": [8, 8],
    }
    cursor_line = np.asarray([[0.0, 0.0], [0.08 * np.sin(0.1), 0.08 * np.cos(0.1)]], dtype=np.float32)
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [{"semantic_id": "bmode", "geometry": geometry}],
        "tracks": [
            {
                "kind": "matrix",
                "semantic_id": "mmode",
                "data": {"zarr_path": "data/1d_motion_mode", "format": "zarr_array"},
            }
        ],
        "overlays": {
            "physical_lines": [{"label": "m_mode_cursor_line", "points": cursor_line.tolist()}],
        },
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode", "1d_motion_mode"),
        content_types=("2d_brightness_mode", "1d_motion_mode"),
        frame_counts_by_content_type={"2d_brightness_mode": 1, "1d_motion_mode": 3},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/1d_motion_mode",
            "timestamps/1d_motion_mode",
            "data/ecg",
            "timestamps/ecg",
        ),
    )

    pre_panels, pre_ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "1d_motion_mode"),
        view_mode="beamspace",
        show_annotations=True,
    )
    assert pre_panels[0].loaded.attrs["annotation_overlays"][0]["kind"] == "sampling_line"

    pre_figure = renderer.render_figure_from_specs(panels=pre_panels, ecg=pre_ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        foreground = [
            line for line in pre_figure.axes[0].lines if to_hex(line.get_color(), keep_alpha=False) == "#4ac44a"
        ]
        solid = [line for line in foreground if line.get_linestyle() == "-"]
        assert len(solid) == 1
        assert solid[0].get_marker() == "None"
        assert np.allclose(solid[0].get_xdata(), solid[0].get_xdata()[0])
    finally:
        plt.close(pre_figure)

    cartesian_panels, cartesian_ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "1d_motion_mode"),
        view_mode="cartesian",
        show_annotations=True,
    )
    cartesian_figure = renderer.render_figure_from_specs(
        panels=cartesian_panels, ecg=cartesian_ecg, time_s=0.0, frame_index=0, dpi=100
    )
    try:
        foreground = [
            line
            for line in cartesian_figure.axes[0].lines
            if to_hex(line.get_color(), keep_alpha=False) == "#4ac44a" and np.isclose(line.get_zorder(), 8.5)
        ]
        solid = [line for line in foreground if line.get_linestyle() == "-"]
        assert len(solid) == 1
        assert np.allclose(np.column_stack([solid[0].get_xdata(), solid[0].get_ydata()]), cursor_line)
    finally:
        plt.close(cartesian_figure)


def test_continuous_wave_sampling_metadata_draws_line_not_box(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_continuous_wave_doppler", data=np.zeros((3, 8), dtype=np.float32))
    group.create_array("timestamps/1d_continuous_wave_doppler", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    geometry = {
        "depth_start_m": 0.0,
        "depth_end_m": 0.08,
        "tilt_rad": 0.0,
        "width_rad": 0.6,
        "grid_size": [8, 8],
    }
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [{"semantic_id": "bmode", "geometry": geometry}],
        "sampling_gate_metadata": {"gate_center_depth_m": 0.04, "gate_tilt_rad": 0.1, "gate_sample_volume_m": 0.01},
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode", "1d_continuous_wave_doppler"),
        content_types=("2d_brightness_mode", "1d_continuous_wave_doppler"),
        frame_counts_by_content_type={"2d_brightness_mode": 1, "1d_continuous_wave_doppler": 3},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/1d_continuous_wave_doppler",
            "timestamps/1d_continuous_wave_doppler",
            "data/ecg",
            "timestamps/ecg",
        ),
    )

    panels, ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "1d_continuous_wave_doppler"),
        view_mode="beamspace",
        show_annotations=True,
    )
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        foreground = [line for line in figure.axes[0].lines if to_hex(line.get_color(), keep_alpha=False) == "#4ac44a"]
        assert [line.get_linestyle() for line in foreground] == ["-"]
        assert not any(line.get_marker() == "_" for line in foreground)
    finally:
        plt.close(figure)


def test_sampling_gate_and_line_overlays_cover_colour_doppler_panels(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((1, 8), dtype=np.float32))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_continuous_wave_doppler", data=np.zeros((3, 8), dtype=np.float32))
    group.create_array("timestamps/1d_continuous_wave_doppler", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    geometry = {"depth_start_m": 0.0, "depth_end_m": 0.08, "tilt_rad": 0.0, "width_rad": 0.6, "grid_size": [8, 8]}
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [{"semantic_id": "bmode", "geometry": geometry}],
        "sampling_gate_metadata": {"gate_center_depth_m": 0.04, "gate_tilt_rad": 0.1, "gate_sample_volume_m": 0.01},
    }
    store = open_recording(tmp_path / "case.zarr")
    bmode = store.load_modality("data/2d_brightness_mode")
    colour_paths = ("data/2d_color_doppler_velocity", "data/2d_color_doppler_power")
    colour = tuple(
        replace(bmode, data_path=path, stream=None if bmode.stream is None else replace(bmode.stream, data_path=path))
        for path in colour_paths
    )
    pw = store.load_modality("data/1d_pulsed_wave_doppler")
    cw = store.load_modality("data/1d_continuous_wave_doppler")

    with_gate = attach_annotation_overlays(store, (bmode, *colour, pw))
    with_line = attach_annotation_overlays(store, (bmode, *colour, cw))

    for loaded in with_gate[:3]:
        assert [o["kind"] for o in loaded.attrs["annotation_overlays"]] == ["sampling_gate"], loaded.data_path
    for loaded in with_line[:3]:
        assert [o["kind"] for o in loaded.attrs["annotation_overlays"]] == ["sampling_line"], loaded.data_path


def test_cartesian_tissue_doppler_carries_gate_overlay(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/tissue_doppler", data=np.full((1, 8, 8), 128, dtype=np.uint8))
    group.create_array("timestamps/tissue_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((1, 8), dtype=np.float32))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.asarray([0.0], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    gate = {
        "kind": "tissue_doppler_gate",
        "sector_semantic_id": "tissue_doppler",
        "gate_center_depth_m": 0.04,
        "gate_tilt_rad": 0.1,
        "gate_sample_volume_m": 0.01,
    }
    geometry = {
        "depth_start_m": 0.0,
        "depth_end_m": 0.08,
        "tilt_rad": 0.0,
        "width_rad": 0.6,
        "grid_size": [8, 8],
    }
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [
            {"semantic_id": "bmode", "geometry": geometry},
            {
                "semantic_id": "tissue_doppler",
                "velocity_scale_mps": 0.2,
                "sampling_gate_metadata": gate,
                "geometry": geometry,
            },
        ],
        "tracks": [{"semantic_id": "tissue_doppler_gate", "kind": "scatter", "derived_from": gate}],
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode", "tissue_doppler", "1d_pulsed_wave_doppler"),
        content_types=("2d_brightness_mode", "tissue_doppler", "1d_pulsed_wave_doppler"),
        frame_counts_by_content_type={"2d_brightness_mode": 1, "tissue_doppler": 1, "1d_pulsed_wave_doppler": 1},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/tissue_doppler",
            "timestamps/tissue_doppler",
            "data/1d_pulsed_wave_doppler",
            "timestamps/1d_pulsed_wave_doppler",
            "data/ecg",
            "timestamps/ecg",
        ),
    )

    no_spectral_panels, _ = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "tissue_doppler"),
        view_mode="beamspace",
        show_annotations=True,
    )
    assert all(
        not any(overlay["kind"] == "sampling_gate" for overlay in panel.loaded.attrs.get("annotation_overlays", ()))
        for panel in no_spectral_panels
    )

    beamspace_panels, _ = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "tissue_doppler", "1d_pulsed_wave_doppler"),
        view_mode="beamspace",
        show_annotations=True,
    )
    sector_panels = tuple(
        panel for panel in beamspace_panels if panel.loaded.data_path != "data/1d_pulsed_wave_doppler"
    )
    assert [panel.loaded.data_path for panel in sector_panels] == ["data/2d_brightness_mode", "data/tissue_doppler"]
    assert all(
        any(overlay["kind"] == "sampling_gate" for overlay in panel.loaded.attrs["annotation_overlays"])
        for panel in sector_panels
    )

    panels, ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "tissue_doppler", "1d_pulsed_wave_doppler"),
        view_mode="cartesian",
        show_annotations=True,
    )

    assert panels[0].view == "cartesian"
    assert sum(overlay["kind"] == "sampling_gate" for overlay in panels[0].loaded.attrs["annotation_overlays"]) == 1

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        foreground = [
            line
            for line in figure.axes[0].lines
            if np.isclose(line.get_zorder(), 8.5) and to_hex(line.get_color(), keep_alpha=False) == "#4ac44a"
        ]
        dashed = [line for line in foreground if line.get_linestyle() == "--"]
        markers = [line for line in foreground if line.get_linestyle() == "-"]
        assert len(dashed) == 2
        assert len(markers) == 2
        assert not any(line.get_marker() == "|" for line in foreground)
    finally:
        plt.close(figure)


def test_3d_mesh_annotation_builds_mosaic_overlay_lines() -> None:
    data = np.zeros((1, 3, 4, 5), dtype=np.uint8)
    timestamps = np.asarray([0.0], dtype=np.float32)
    raw = {
        "sectors": [
            {
                "semantic_id": "bmode",
                "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.01,
                    "DepthEnd": 0.12,
                    "Width": 0.8,
                    "ElevationWidth": 0.7,
                },
            }
        ],
        "render_metadata": {
            "DepthStart": 0.01,
            "DepthEnd": 0.12,
            "Width": 0.8,
            "ElevationWidth": 0.7,
        },
    }
    stream = stream_from_arrays(
        data_path="data/3d_brightness_mode",
        data=data,
        timestamps_path="timestamps/3d_brightness_mode",
        timestamps=timestamps,
        sample_rate_hz=10.0,
        metadata=StreamMetadata(data_path="data/3d_brightness_mode", value_range=(0.0, 255.0), raw=raw),
    )
    from echoxflow.loading import PackedMeshAnnotation

    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={
            "mesh_annotation": PackedMeshAnnotation(
                points_path="data/3d_left_ventricle_mesh/point_values",
                points=np.asarray(
                    [[-0.02, 0.05, -0.01], [0.02, 0.05, 0.01], [0.0, 0.08, 0.0]],
                    dtype=np.float32,
                ),
                faces_path="data/3d_left_ventricle_mesh/face_values",
                faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            )
        },
        stream=stream,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    (pre_panel,) = renderer.build_panel_specs((loaded,), view_mode="beamspace")
    (cartesian_panel,) = renderer.build_panel_specs((loaded,), view_mode="cartesian")

    assert "mosaic_annotation_lines" not in pre_panel.loaded.attrs
    assert "mosaic_annotation_lines" in cartesian_panel.loaded.attrs
    assert any(line.size for line in cartesian_panel.loaded.attrs["mosaic_annotation_lines"][0])


def test_3d_mesh_arrays_without_manifest_link_do_not_attach_overlay(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((2, 3, 4, 5), dtype=np.uint8))
    group.create_array("timestamps/3d_brightness_mode", data=np.asarray([0.0, 1.0], dtype=np.float32))
    mesh_group = group.create_group("annotations/3d_left_ventricle_mesh")
    mesh_group.create_array("point_values", data=np.zeros((3, 3), dtype=np.float32))
    mesh_group.create_array("face_values", data=np.asarray([[0, 1, 2]], dtype=np.int32))
    group.attrs["recording_manifest"] = _three_d_manifest(linked_mesh=False)
    store = open_recording(tmp_path / "case.zarr")
    loaded = attach_annotation_overlays(store, (store.load_modality("3d_brightness_mode"),))[0]

    assert "mesh_annotation" not in loaded.attrs
    (panel,) = RecordingPlotRenderer().build_panel_specs((loaded,), view_mode="cartesian")
    assert "mosaic_annotation_lines" not in panel.loaded.attrs


def test_explicit_3d_mesh_sequence_builds_moving_mosaic_overlay(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((2, 3, 4, 5), dtype=np.uint8))
    group.create_array("timestamps/3d_brightness_mode", data=np.asarray([0.0, 1.0], dtype=np.float32))
    _write_mesh_group(group, frame_count=2)
    group.create_array("mesh_times", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.attrs["recording_manifest"] = _three_d_manifest(mesh_timestamps_path="mesh_times")
    store = open_recording(tmp_path / "case.zarr")
    loaded = attach_annotation_overlays(store, (store.load_modality("3d_brightness_mode"),))[0]

    assert "mesh_annotation" in loaded.attrs
    (panel,) = RecordingPlotRenderer().build_panel_specs((loaded,), view_mode="cartesian")

    assert "mosaic_annotation_lines" in panel.loaded.attrs
    assert len(panel.loaded.attrs["mosaic_annotation_lines"]) == 2
    assert any(line.size for line in panel.loaded.attrs["mosaic_annotation_lines"][0])


def _write_mesh_only_store(tmp_path: Path, *, linked: bool, closed: bool = False) -> None:
    volume = zarr.open_group(tmp_path / "volume.zarr", mode="w")
    volume.create_array("data/3d_brightness_mode", data=np.zeros((2, 3, 4, 5), dtype=np.uint8))
    volume.create_array("timestamps/3d_brightness_mode", data=np.asarray([0.0, 1.0], dtype=np.float32))
    volume.attrs["recording_manifest"] = _three_d_manifest(linked_mesh=False)
    mesh = zarr.open_group(tmp_path / "mesh.zarr", mode="w")
    _write_mesh_group(mesh, frame_count=2, closed=closed)
    manifest: dict[str, object] = {
        "manifest_type": "3d",
        "linked_mesh_sequences": [
            {
                "mesh_data": {"zarr_path": "annotations/3d_left_ventricle_mesh", "format": "zarr_group"},
                "timestamps": {"zarr_path": "timestamps/3d_left_ventricle_mesh", "format": "zarr_array"},
            }
        ],
    }
    if linked:
        manifest["linked_volume"] = {
            "semantic_id": "3d_brightness_mode",
            "linked_recording": {
                "recording_id": "volume",
                "relative_zarr_path": "volume.zarr",
                "frames_array_path": "data/3d_brightness_mode",
            },
        }
    mesh.attrs["recording_manifest"] = manifest


def test_mesh_only_recording_renders_from_linked_volume(tmp_path: Path) -> None:
    _write_mesh_only_store(tmp_path, linked=True)
    renderer = RecordingPlotRenderer()

    panels, _ = renderer._load_specs(
        tmp_path / "mesh.zarr", root=None, modalities=None, view_mode="cartesian", show_annotations=True
    )

    (panel,) = panels
    assert panel.loaded.data_path == "data/3d_brightness_mode"
    assert panel.loaded.data.shape[0] == 2
    assert len(panel.loaded.attrs["mosaic_annotation_lines"]) == 2
    assert any(line.size for line in panel.loaded.attrs["mosaic_annotation_lines"][0])


def test_mesh_only_recording_without_link_names_the_recording(tmp_path: Path) -> None:
    _write_mesh_only_store(tmp_path, linked=False)

    with pytest.raises(ValueError, match="`mesh` holds only a mesh with no linked volume"):
        RecordingPlotRenderer()._load_specs(
            tmp_path / "mesh.zarr", root=None, modalities=None, view_mode="cartesian", show_annotations=True
        )


def test_mesh_mosaic_lines_land_inside_their_layout_cells(tmp_path: Path) -> None:
    from echoxflow.plotting.annotations import mesh_mosaic_annotation_lines
    from echoxflow.scan import SphericalGeometry, cartesian_mosaic_layout

    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    mesh_group = group.create_group("annotations/3d_left_ventricle_mesh")
    # a vertical triangle wall straddling the centre line, deep enough to cross several depth cuts
    points = np.asarray([[-0.01, -0.01, 0.03], [0.01, -0.01, 0.12], [0.0, 0.01, 0.07]], dtype=np.float32)
    # the renderer maps mesh points through a fixed frame transform; invert it to place points in render space
    transform = np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mesh_points = (points @ np.linalg.inv(transform.T)).astype(np.float32)
    mesh_group.create_array("point_values", data=mesh_points)
    mesh_group.create_array("face_values", data=np.asarray([[0, 1, 2]], dtype=np.int32))
    mesh_group.create_array("point_frame_offsets", data=np.asarray([0, 3], dtype=np.int64))
    mesh_group.create_array("face_frame_offsets", data=np.asarray([0, 1], dtype=np.int64))
    group.create_array("timestamps/3d_left_ventricle_mesh", data=np.zeros(1, dtype=np.float32))
    group.attrs["recording_manifest"] = _three_d_manifest()
    annotation = open_recording(tmp_path / "case.zarr").load_packed_mesh_annotation()
    geometry = SphericalGeometry(depth_start_m=0.01, depth_end_m=0.13, azimuth_width_rad=0.8, elevation_width_rad=0.8)
    layout = cartesian_mosaic_layout(geometry, output_size=(60, 60))

    (lines,) = mesh_mosaic_annotation_lines(annotation, frame_count=1, layout=layout, view="cartesian")

    assert lines
    cells_hit = set()
    for line in lines:
        owners = [
            (cell.row, cell.col)
            for cell in layout.cells
            if np.all(line[:, 0] >= cell.image_x0 - 1e-3)
            and np.all(line[:, 0] <= cell.image_x0 + cell.image_width - 1 + 1e-3)
            and np.all(line[:, 1] >= cell.image_y0 - 1e-3)
            and np.all(line[:, 1] <= cell.image_y0 + cell.image_height - 1 + 1e-3)
        ]
        assert owners, "mesh line falls outside every mosaic cell"
        cells_hit.add(owners[0])
    assert any(col > 0 for _, col in cells_hit)
    assert any(col == 0 for _, col in cells_hit)


def test_single_frame_3d_mesh_sequence_does_not_repeat_on_multi_frame_volume(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((2, 3, 4, 5), dtype=np.uint8))
    group.create_array("timestamps/3d_brightness_mode", data=np.asarray([0.0, 1.0], dtype=np.float32))
    _write_mesh_group(group, frame_count=1)
    group.attrs["recording_manifest"] = _three_d_manifest()
    store = open_recording(tmp_path / "case.zarr")
    loaded = attach_annotation_overlays(store, (store.load_modality("3d_brightness_mode"),))[0]
    (panel,) = RecordingPlotRenderer().build_panel_specs((loaded,), view_mode="cartesian")

    assert "mesh_annotation" in loaded.attrs
    assert "mosaic_annotation_lines" not in panel.loaded.attrs


def _three_d_manifest(*, linked_mesh: bool = True, mesh_timestamps_path: str = "timestamps/3d_left_ventricle_mesh"):
    manifest: dict[str, object] = {
        "manifest_type": "3d",
        "sectors": [
            {
                "semantic_id": "bmode",
                "frames": {"array_path": "data/3d_brightness_mode", "format": "zarr_array"},
                "timestamps": {"array_path": "timestamps/3d_brightness_mode", "format": "zarr_array"},
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.01,
                    "DepthEnd": 0.12,
                    "Width": 0.8,
                    "ElevationWidth": 0.7,
                },
            }
        ],
    }
    if linked_mesh:
        manifest["linked_mesh_sequences"] = [
            {
                "mesh_data": {"zarr_path": "data/3d_left_ventricle_mesh", "format": "zarr_group"},
                "timestamps": {"zarr_path": mesh_timestamps_path, "format": "zarr_array"},
                "mesh_key": "LV",
            }
        ]
    return manifest


def _write_mesh_group(group: object, *, frame_count: int, closed: bool = False) -> None:
    points = np.asarray(
        [[-0.02, 0.05, -0.01], [0.02, 0.05, 0.01], [0.0, 0.08, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    if closed:  # a tetrahedron: closed surface with volume 4e-6 m^3 / 6 per unit edge
        points = np.vstack([points, [[0.0, 0.065, 0.03]]]).astype(np.float32)
        faces = np.asarray([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int32)
    mesh_group = group.create_group("annotations/3d_left_ventricle_mesh")
    mesh_group.create_array("point_values", data=np.tile(points, (frame_count, 1)))
    mesh_group.create_array("face_values", data=np.tile(faces, (frame_count, 1)))
    mesh_group.create_array("point_frame_offsets", data=np.arange(frame_count + 1, dtype=np.int64) * points.shape[0])
    mesh_group.create_array("face_frame_offsets", data=np.arange(frame_count + 1, dtype=np.int64) * faces.shape[0])
    group.create_array("timestamps/3d_left_ventricle_mesh", data=np.arange(frame_count, dtype=np.float32))


def test_image_renderer_draws_mosaic_annotation_polygons_without_edges() -> None:
    panel = PanelSpec(
        loaded=LoadedArray(
            name="3d_brightness_mode_mosaic",
            data_path="data/3d_brightness_mode",
            data=np.zeros((1, 16, 16), dtype=np.float32),
            timestamps_path=None,
            timestamps=None,
            sample_rate_hz=None,
            attrs={
                "mosaic_annotation_polygons": ((np.asarray([[2.0, 2.0], [12.0, 2.0], [7.0, 12.0]], dtype=np.float32),),)
            },
        ),
        kind="image",
        label="mesh",
        view="cartesian",
    )
    figure, ax = plt.subplots()
    try:
        renderer_for("image").render(
            ax,
            panel,
            time_s=0.0,
            frame_index=0,
            style=PlotStyle(annotation_color="#F2A3A3"),
        )
        assert len(ax.patches) == 1
        assert to_hex(ax.patches[0].get_facecolor(), keep_alpha=False) == "#f2a3a3"
        assert to_hex(ax.patches[0].get_edgecolor(), keep_alpha=True) == "#00000000"
    finally:
        plt.close(figure)


def test_colormap_resolves_full_croissant_names_and_aliases() -> None:
    full = named_listed_colormap("data/2d_color_doppler_velocity")
    alias = named_listed_colormap("2d_color_doppler_velocity")

    assert data_modality_colormap_name("data/2d_color_doppler_velocity") == "color_doppler_velocity"
    assert data_modality_colormap_name("2d_color_doppler_velocity") == "color_doppler_velocity"
    assert full is not None
    assert alias is not None
    assert np.allclose(full(np.asarray([0.0, 0.5, 1.0])), alias(np.asarray([0.0, 0.5, 1.0])))


def test_color_doppler_velocity_colormap_has_symmetric_black_hold() -> None:
    cmap = named_listed_colormap("data/2d_color_doppler_velocity", size=801)

    assert cmap is not None
    samples = np.asarray([0.2125, 0.425, 0.5, 0.575, 0.7875], dtype=np.float32)
    table = np.asarray(cmap(samples), dtype=np.float32)[:, :3]
    assert table[0, 2] > table[0, 0]
    assert np.allclose(table[1], [0.0, 0.0, 0.0], atol=0.01)
    assert np.allclose(table[2], [0.0, 0.0, 0.0], atol=0.01)
    assert np.allclose(table[3], [0.0, 0.0, 0.0], atol=0.01)
    assert table[4, 0] > table[4, 2]
    assert np.isclose(0.425 - 0.2125, 0.7875 - 0.575)
    assert np.isclose(0.2125 - 0.0, 1.0 - 0.7875)


def test_tissue_doppler_colormap_has_hard_midpoint_cut() -> None:
    cmap = named_listed_colormap("data/tissue_doppler", size=256)

    assert cmap is not None
    table = np.asarray(cmap.colors, dtype=np.float32)
    # Positive velocity (toward the probe) is warm: yellow, then red up to the midpoint.
    early_yellow = table[231]
    assert early_yellow[0] > 0.95
    assert early_yellow[1] > 0.95
    assert early_yellow[2] < 0.1
    assert table[128, 0] > 0.9
    assert table[128, 2] < 0.1
    # Negative velocity is cool: blue just below the midpoint, green at the negative limit.
    assert table[127, 2] > 0.9
    assert table[127, 0] < 0.1
    assert table[0, 1] > 0.9
    assert table[0, 0] < 0.1
    assert table[0, 2] < 0.1
    red_rows = table[128:]
    red_count = int(np.count_nonzero((red_rows[:, 0] > 0.95) & (red_rows[:, 1] < 0.05) & (red_rows[:, 2] < 0.05)))
    assert red_count >= 8


def test_tissue_doppler_colorbar_matches_image_orientation() -> None:
    spec = colorbar_spec_for_modality("data/tissue_doppler", value_range=(-0.2, 0.2))

    assert spec is not None
    assert spec.reverse_cmap is False


def test_write_video_smoke(tmp_path: Path) -> None:
    output = tmp_path / "tiny.mp4"
    frames = np.zeros((2, 15, 17, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255

    result = write_video(output, frames, fps=2.0)

    assert result == output
    assert output.stat().st_size > 0
    av = pytest.importorskip("av")
    with av.open(str(output)) as container:
        assert container.streams.video[0].pix_fmt == "yuv420p"


def test_write_video_webm(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    output = tmp_path / "tiny.webm"
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 255

    result = write_video(output, frames, fps=3.0)

    assert result == output
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name in {"vp9", "vp8"}
        assert sum(1 for _ in container.decode(stream)) == 3


def test_write_video_animated_webp(tmp_path: Path) -> None:
    from PIL import Image

    output = tmp_path / "tiny.webp"
    # Identical consecutive frames are merged by the WebP encoder, so every frame differs.
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 255
    frames[2, :, :, 2] = 255

    assert write_video(output, iter(frames), fps=4.0) == output
    with Image.open(output) as image:
        assert image.n_frames == 3
        assert image.info["loop"] == 0


def test_write_video_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="use .mp4, .webm or .webp"):
        write_video(tmp_path / "tiny.avi", np.zeros((2, 16, 16, 3), dtype=np.uint8), fps=2.0)


def test_write_video_warns_for_single_frame_source(tmp_path: Path) -> None:
    output = tmp_path / "single.mp4"
    source = tmp_path / "source.zarr"
    frames = np.zeros((1, 16, 16, 3), dtype=np.uint8)

    with pytest.warns(UserWarning, match="single-frame MP4.*single\\.mp4.*source\\.zarr"):
        result = write_video(output, frames, fps=2.0, source=source)

    assert result == output
    assert output.stat().st_size > 0


def _write_small_bmode_recording(tmp_path: Path) -> RecordingRecord:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((2, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0, 0.1], dtype=np.float32))
    group.create_array("data/ecg", data=np.asarray([0.0, 1.0], dtype=np.float32))
    group.create_array("timestamps/ecg", data=np.asarray([0.0, 0.1], dtype=np.float32))
    return RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode",),
        content_types=("2d_brightness_mode",),
        frame_counts_by_content_type={"2d_brightness_mode": 2},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/ecg",
            "timestamps/ecg",
        ),
    )


@pytest.mark.parametrize("dpi", [72, 100, 150, 200])
def test_rendered_frame_matches_style_pixel_size(tmp_path: Path, dpi: int) -> None:
    record = _write_small_bmode_recording(tmp_path)
    style = PlotStyle(width_px=501, height_px=377, dpi=dpi)
    renderer = RecordingPlotRenderer(style=style)
    panels, ecg = renderer._load_specs(
        record, root=tmp_path, modalities=None, view_mode="beamspace", show_annotations=True
    )
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=dpi)
    try:
        assert figure_to_rgb(figure).shape == (377, 501, 3)
    finally:
        plt.close(figure)

    frame = renderer.render_frame(record, root=tmp_path)
    assert frame.image.shape == (377, 501, 3)


def test_video_and_still_match_style_pixel_size(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    record = _write_small_bmode_recording(tmp_path)
    style = PlotStyle(width_px=320, height_px=240, dpi=100)
    renderer = RecordingPlotRenderer(style=style)

    video = renderer.render_video(record, tmp_path / "out.mp4", root=tmp_path)
    png = renderer.save_plot(record, tmp_path / "out.png", root=tmp_path, dpi=200)

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (320, 240)
    assert plt.imread(png).shape[:2] == (240, 320)


def test_render_video_time_range_keeps_frames_inside_the_window(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    record = _write_small_bmode_recording(tmp_path)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=320, height_px=240, dpi=100))

    with pytest.warns(UserWarning, match="single-frame MP4"):
        video = renderer.render_video(record, tmp_path / "window.mp4", root=tmp_path, time_range=(0.05, 0.5))
    with av.open(str(video)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 1

    with pytest.raises(ValueError, match="No frames between 1.0 s and 2.0 s"):
        renderer.render_video(record, tmp_path / "empty.mp4", root=tmp_path, time_range=(1.0, 2.0))


def test_default_dpi_is_the_style_dpi(tmp_path: Path) -> None:
    record = _write_small_bmode_recording(tmp_path)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=320, height_px=240, dpi=80))

    default_png = renderer.save_plot(record, tmp_path / "default.png", root=tmp_path)
    explicit_png = renderer.save_plot(record, tmp_path / "explicit.png", root=tmp_path, dpi=80)
    larger_png = renderer.save_plot(record, tmp_path / "larger.png", root=tmp_path, dpi=200)
    assert np.array_equal(plt.imread(default_png), plt.imread(explicit_png))
    assert not np.array_equal(plt.imread(default_png), plt.imread(larger_png))

    panels, ecg = renderer._load_specs(
        record, root=tmp_path, modalities=None, view_mode="beamspace", show_annotations=True
    )
    times = np.asarray([0.0])
    (default_frame,) = renderer.iter_frames(panels, ecg, times)
    (explicit_frame,) = renderer.iter_frames(panels, ecg, times, dpi=80)
    assert np.array_equal(default_frame, explicit_frame)


def test_write_video_dimensions_match_frames(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    output = tmp_path / "dims.mp4"
    write_video(output, np.zeros((2, 20, 30, 3), dtype=np.uint8), fps=2.0)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (30, 20)


def test_write_video_accepts_frame_generator(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    output = tmp_path / "gen.mp4"
    produced = 0

    def frames():
        nonlocal produced
        for value in (0, 128, 255):
            produced += 1
            yield np.full((15, 17, 3), value, dtype=np.uint8)

    write_video(output, frames(), fps=3.0)

    assert produced == 3
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (18, 16)
        assert sum(1 for _ in container.decode(stream)) == 3


def test_write_video_warns_for_single_frame_iterator(tmp_path: Path) -> None:
    output = tmp_path / "single_iter.mp4"
    with pytest.warns(UserWarning, match="single-frame MP4"):
        write_video(output, iter([np.zeros((16, 16, 3), dtype=np.uint8)]), fps=2.0)
    assert output.stat().st_size > 0


def test_write_video_rejects_empty_iterator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        write_video(tmp_path / "empty.mp4", iter(()), fps=2.0)


def _write_flow_table_recording(
    tmp_path: Path, *, with_table: bool, stream: str = "2d_color_doppler_velocity"
) -> tuple[RecordingRecord, np.ndarray]:
    """Write a colour-velocity recording whose FlowColorTable is a recognisable red-to-blue gradient."""
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((2, 8, 8), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0, 0.1], dtype=np.float32))
    group.create_array(f"data/{stream}", data=np.zeros((2, 8, 8), dtype=np.float32))
    group.create_array(f"timestamps/{stream}", data=np.asarray([0.0, 0.1], dtype=np.float32))
    sector: dict[str, object] = {"semantic_id": stream, "velocity_limit_mps": 0.6}
    array_paths = [
        "data/2d_brightness_mode",
        "timestamps/2d_brightness_mode",
        f"data/{stream}",
        f"timestamps/{stream}",
    ]
    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
    table[..., 2] = 255 - np.arange(256, dtype=np.uint8)[None, :]
    if with_table:
        group.create_array("data/lookup_tables/flow_256x256x3", data=table)
        sector["lookup_tables"] = [
            {"role": "flow_color_rgb_table", "data": {"zarr_path": "data/lookup_tables/flow_256x256x3"}}
        ]
        array_paths.append("data/lookup_tables/flow_256x256x3")
    group.attrs["recording_manifest"] = {"manifest_type": "2d", "sectors": [sector]}
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_color_doppler",),
        content_types=("2d_color_doppler",),
        frame_counts_by_content_type={"2d_color_doppler": 2},
        median_delta_time_by_content_type={},
        array_paths=tuple(array_paths),
    )
    return record, table


def _velocity_panel_cmap(tmp_path: Path, record: RecordingRecord, stream: str = "2d_color_doppler_velocity"):
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels, ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=(stream,),
        view_mode="beamspace",
        show_annotations=False,
    )
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        return panels[0].loaded.attrs.get("colormap"), figure.axes[0].images[0].get_cmap()
    finally:
        plt.close(figure)


def test_colour_velocity_panel_uses_the_recordings_flow_colour_table(tmp_path: Path) -> None:
    record, table = _write_flow_table_recording(tmp_path, with_table=True)

    attached, drawn = _velocity_panel_cmap(tmp_path, record)

    expected = flow_color_velocity_ramp(table, phase_ordered=True).astype(np.float32) / 255.0
    assert attached is not None
    assert np.allclose(drawn(np.arange(256))[:, :3], expected)


def test_tissue_velocity_panel_draws_positive_velocity_warm(tmp_path: Path) -> None:
    record, table = _write_flow_table_recording(tmp_path, with_table=True, stream="tissue_doppler")

    attached, drawn = _velocity_panel_cmap(tmp_path, record, stream="tissue_doppler")

    expected = flow_color_velocity_ramp(table, phase_ordered=True).astype(np.float32) / 255.0
    assert attached is not None
    assert np.allclose(drawn(np.arange(256))[:, :3], expected)


def test_colour_velocity_panel_falls_back_to_configured_map_without_a_table(tmp_path: Path) -> None:
    record, _ = _write_flow_table_recording(tmp_path, with_table=False)

    attached, drawn = _velocity_panel_cmap(tmp_path, record)

    configured = named_listed_colormap("color_doppler_velocity")
    assert attached is None
    assert configured is not None
    assert np.allclose(drawn(np.arange(256))[:, :3], configured(np.arange(256))[:, :3])


def _write_variance_flow_recording(tmp_path: Path, *, with_lag_one: bool) -> tuple[RecordingRecord, np.ndarray]:
    """Write a colour-velocity recording with a variance-style table: row 0 is a velocity ramp, rows above add green."""
    record, _ = _write_flow_table_recording(tmp_path, with_table=True)
    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
    table[..., 2] = 255 - np.arange(256, dtype=np.uint8)[None, :]
    table[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    group = zarr.open_group(tmp_path / "case.zarr", mode="r+")
    group["data/lookup_tables/flow_256x256x3"][...] = table
    group["data/2d_color_doppler_velocity"][...] = np.tile(
        np.linspace(-0.55, 0.55, 8, dtype=np.float32)[None, None, :], (2, 8, 1)
    )
    array_paths = record.array_paths
    if with_lag_one:
        lag_one = np.tile(np.linspace(0.0, 150.0, 8, dtype=np.float32)[None, :, None], (2, 1, 8))
        group.create_array("derived/2d_color_doppler_lag_one_magnitude", data=lag_one)
        array_paths = (*array_paths, "derived/2d_color_doppler_lag_one_magnitude")
    return replace(record, array_paths=array_paths), table


def test_colour_velocity_panel_colours_through_the_two_dimensional_flow_table(tmp_path: Path) -> None:
    record, table = _write_variance_flow_recording(tmp_path, with_lag_one=True)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels, ecg = renderer._load_specs(
        record, root=tmp_path, modalities=("2d_color_doppler_velocity",), view_mode="beamspace", show_annotations=False
    )
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        drawn = np.asarray(figure.axes[0].images[0].get_array())
    finally:
        plt.close(figure)

    velocity = np.tile(np.linspace(-0.55, 0.55, 8, dtype=np.float32)[None, :], (8, 1))
    lag_one = np.tile(np.linspace(0.0, 150.0, 8, dtype=np.float32)[:, None], (1, 8))
    secondary = np.rint(255.0 * (1.0 - lag_one / (127.0 * np.sqrt(2.0)))).astype(np.uint8)
    expected = table[secondary, color_flow_velocity_codes(velocity, velocity_limit_mps=0.6)]
    assert drawn.dtype == np.uint8
    assert np.array_equal(drawn, expected)
    assert drawn[..., 1].max() > 200  # turbulent rows blend toward green, which the 1-D ramp never shows


def test_colour_velocity_panel_without_lag_one_keeps_the_velocity_ramp(tmp_path: Path) -> None:
    record, table = _write_variance_flow_recording(tmp_path, with_lag_one=False)

    attached, drawn = _velocity_panel_cmap(tmp_path, record)

    expected = flow_color_velocity_ramp(table, phase_ordered=True).astype(np.float32) / 255.0
    assert attached is not None
    assert np.allclose(drawn(np.arange(256))[:, :3], expected)


def _label_texts(figure) -> list[str]:
    return [text.get_text() for ax in figure.axes for text in ax.texts]


def test_panels_are_labelled_with_stream_and_array_shape(tmp_path: Path) -> None:
    record = _write_small_bmode_recording(tmp_path)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.plot_recording(record, root=tmp_path, frame_index=1)
    try:
        labels = _label_texts(figure)
        assert "2d_brightness_mode\nuint8 2×8×8" in labels
        figure_text = [text.get_text() for text in figure.texts]
        assert figure_text == ["case    t = 0.100 s  frame 1/2"]
    finally:
        plt.close(figure)


def test_cartesian_panel_label_names_the_view(tmp_path: Path) -> None:
    record = _write_small_bmode_recording(tmp_path)
    group = zarr.open_group(tmp_path / "case.zarr", mode="r+")
    group.attrs["recording_manifest"] = {
        "manifest_type": "2d",
        "sectors": [
            {
                "semantic_id": "2d_brightness_mode",
                "geometry": {
                    "depth_start_m": 0.0,
                    "depth_end_m": 0.08,
                    "tilt_rad": 0.0,
                    "width_rad": 0.6,
                    "grid_size": [8, 8],
                },
            }
        ],
    }
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    figure = renderer.plot_recording(record, root=tmp_path, view_mode="cartesian")
    try:
        assert any(label.endswith("\ncartesian") for label in _label_texts(figure))
    finally:
        plt.close(figure)


def test_labels_and_title_can_be_turned_off(tmp_path: Path) -> None:
    record = _write_small_bmode_recording(tmp_path)
    style = PlotStyle(width_px=420, height_px=300, dpi=100, show_panel_labels=False, show_figure_title=False)

    figure = RecordingPlotRenderer(style=style).plot_recording(record, root=tmp_path)
    try:
        assert _label_texts(figure) == []
        assert figure.texts == []
    finally:
        plt.close(figure)


def test_colorbar_limit_labels_stay_inside_the_figure() -> None:
    renderer = RecordingPlotRenderer()
    panel = PanelSpec(
        loaded=_loaded("tissue_doppler", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="TDI",
        view="cartesian",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        figure.canvas.draw()
        bounds = figure.bbox
        labels = [text for ax in figure.axes for child in ax.child_axes for text in child.texts]
        assert [text.get_text() for text in labels] == ["20", "-20"]
        for label in labels:
            extent = label.get_window_extent()
            assert extent.x0 >= bounds.x0 and extent.x1 <= bounds.x1
            assert extent.y0 >= bounds.y0 and extent.y1 <= bounds.y1
    finally:
        plt.close(figure)


def _write_single_frame_with_strip_recording(tmp_path: Path, frames: int = 1) -> RecordingRecord:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=np.zeros((frames, 8, 8), dtype=np.uint8))
    group.create_array(
        "timestamps/2d_brightness_mode",
        data=np.linspace(0.0, 0.5, frames, dtype=np.float32) if frames > 1 else np.asarray([0.5], dtype=np.float32),
    )
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((200, 12), dtype=np.float32))
    group.create_array(
        "timestamps/1d_pulsed_wave_doppler", data=(np.arange(200, dtype=np.float32) / 400.0).astype(np.float32)
    )
    group.create_array("data/ecg", data=np.sin(np.linspace(0.0, 6.0, 300)).astype(np.float32))
    group.create_array("timestamps/ecg", data=np.linspace(0.0, 0.5, 300).astype(np.float32))
    paths = tuple(
        f"{kind}/{name}"
        for name in ("2d_brightness_mode", "1d_pulsed_wave_doppler", "ecg")
        for kind in ("data", "timestamps")
    )
    return RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode", "1d_pulsed_wave_doppler"),
        content_types=("2d_brightness_mode", "1d_pulsed_wave_doppler"),
        frame_counts_by_content_type={"2d_brightness_mode": frames, "1d_pulsed_wave_doppler": 200},
        median_delta_time_by_content_type={},
        array_paths=paths,
    )


def test_single_frame_recording_takes_its_timeline_from_the_spectral_strip(tmp_path: Path) -> None:
    record = _write_single_frame_with_strip_recording(tmp_path)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels, _ = renderer._load_specs(
        record, root=tmp_path, modalities=None, view_mode="beamspace", show_annotations=False
    )
    assert {panel.loaded.name for panel in panels} == {"2d_brightness_mode", "1d_pulsed_wave_doppler"}

    timeline = select_timeline(panels, max_fps=30.0)

    assert timeline.timestamps.size > 1
    assert timeline.fps <= 30.0
    assert timeline.timestamps[0] >= 0.0 and timeline.timestamps[-1] <= 199 / 400.0
    still_time = resolve_frame_time(panels, FrameRequest())
    assert np.isclose(still_time, 0.5 * 199 / 400.0)
    assert resolve_frame_time(panels, FrameRequest(frame_index=3)) == pytest.approx(3 / 400.0)


@pytest.mark.parametrize(("frames", "strip_is_taller"), [(1, True), (6, False)])
def test_strip_height_follows_whether_the_reference_image_is_static(
    tmp_path: Path, frames: int, strip_is_taller: bool
) -> None:
    record = _write_single_frame_with_strip_recording(tmp_path, frames=frames)
    style = PlotStyle(width_px=600, height_px=600, dpi=100, show_ecg=False)
    figure = RecordingPlotRenderer(style=style).plot_recording(record, root=tmp_path, view_mode="beamspace")
    try:
        cells = [ax.get_subplotspec().get_position(figure).height for ax in figure.axes if ax.images]
        assert len(cells) == 2
        image_cell, strip_cell = cells
        assert bool(strip_cell > image_cell) is strip_is_taller
    finally:
        plt.close(figure)


def test_single_frame_recording_video_has_many_frames_without_warning(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    record = _write_single_frame_with_strip_recording(tmp_path)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=320, height_px=240, dpi=100))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        video = renderer.render_video(record, tmp_path / "out.mp4", root=tmp_path, max_fps=30.0)

    with av.open(str(video)) as container:
        assert container.streams.video[0].frames > 1


def test_mesh_points_map_into_the_render_frame() -> None:
    """Mesh x stays lateral, mesh y becomes depth and mesh z becomes negative elevation."""
    from echoxflow.plotting.annotations import _mesh_points_to_render_frame

    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rendered = _mesh_points_to_render_frame(points)
    np.testing.assert_allclose(rendered, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _cartesian_3d_mosaic_figure(style: PlotStyle):
    raw = {
        "sectors": [
            {
                "semantic_id": "bmode",
                "geometry": {
                    "coordinate_system": "spherical_sector_3d",
                    "DepthStart": 0.01,
                    "DepthEnd": 0.12,
                    "Width": 0.8,
                    "ElevationWidth": 0.7,
                },
            }
        ],
    }
    stream = stream_from_arrays(
        data_path="data/3d_brightness_mode",
        data=np.full((2, 5, 7, 9), 120, dtype=np.uint8),
        timestamps_path="timestamps/3d_brightness_mode",
        timestamps=np.asarray([0.0, 0.1], dtype=np.float32),
        sample_rate_hz=10.0,
        metadata=StreamMetadata(data_path="data/3d_brightness_mode", value_range=(0.0, 255.0), raw=raw),
    )
    loaded = LoadedArray(
        name=stream.name,
        data_path=stream.data_path,
        data=stream.data,
        timestamps_path=stream.timestamps_path,
        timestamps=stream.timestamps,
        sample_rate_hz=stream.sample_rate_hz,
        attrs={},
        stream=stream,
    )
    renderer = RecordingPlotRenderer(style=style)
    panels = renderer.build_panel_specs((loaded,), view_mode="cartesian")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))
    return renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.05, frame_index=0, dpi=100)


def test_cartesian_3d_mosaic_draws_level_markers_and_depth_rulers() -> None:
    style = PlotStyle(width_px=900, height_px=700, dpi=100)
    figure = _cartesian_3d_mosaic_figure(style)
    try:
        ax = figure.axes[0]
        colors = {to_hex(c).lower() for c in style.mosaic_level_colors}
        by_color = {c: [] for c in colors}
        for line in ax.lines:
            color = to_hex(line.get_color()).lower()
            if color in by_color:
                by_color[color].append(line)
        for lines in by_color.values():
            # Three horizontal level lines (one per long-axis slice) plus the L-shaped corner bracket.
            assert len(lines) == 4
            assert sum(len(set(np.asarray(line.get_ydata()).tolist())) == 1 for line in lines) == 3
        texts = [t.get_text() for t in ax.texts]
        assert sum(t.endswith(" cm") for t in texts) == 9
        ruler_color = to_hex(style.ecg_trace_color).lower()
        ticks = [
            line for line in ax.lines if to_hex(line.get_color()).lower() == ruler_color and len(line.get_xdata()) == 2
        ]
        assert len(ticks) >= 3 * 5
    finally:
        plt.close(figure)


def test_cartesian_3d_mosaic_levels_can_be_disabled() -> None:
    style = PlotStyle(width_px=900, height_px=700, dpi=100, show_mosaic_levels=False)
    figure = _cartesian_3d_mosaic_figure(style)
    try:
        colors = {to_hex(c).lower() for c in style.mosaic_level_colors}
        assert not [line for line in figure.axes[0].lines if to_hex(line.get_color()).lower() in colors]
        assert not [t for t in figure.axes[0].texts if t.get_text().endswith(" cm")]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("merged_depth", [0, 1, 2])
def test_stored_spectral_doppler_traces_attach_and_draw_as_lines(tmp_path: Path, merged_depth: int) -> None:
    prefix = "annotations/" + "measurement_document_00/" * merged_depth
    trace_path = prefix + "1d_pulsed_wave_doppler_annotation_00"
    measurement_path = prefix + "measurement_000_points"
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    times = np.linspace(0.0, 3.0, 300, dtype=np.float32)
    group.create_array("data/1d_pulsed_wave_doppler", data=np.zeros((300, 64), dtype=np.float32))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=times)
    trace = np.column_stack([np.linspace(1.0, 1.5, 20), -np.sin(np.linspace(0.0, np.pi, 20)) * 0.8]).astype(np.float32)
    for name in (trace_path, measurement_path):
        array = group.create_array(name, data=trace)
        array.attrs.update({"coordinate_system": "spectral_time_velocity", "units": "s_mps"})
    group.attrs["recording_manifest"] = {
        "annotations": [
            {
                "kind": "spectral_trace",
                "target": {"type": "track", "semantic_id": "pw_doppler", "field": "spectral_trace"},
                "target_data": {"zarr_path": "data/1d_pulsed_wave_doppler", "format": "zarr_array"},
                "value": {"zarr_path": trace_path, "format": "zarr_array"},
                "links": {"geometry_kind": "polyline"},
            },
            {
                "kind": "measurement",
                "target": {"type": "recording", "semantic_id": "global", "field": "measurement_points"},
                "value": {"zarr_path": measurement_path, "format": "zarr_array"},
                "links": {
                    "coordinate_type": "spectral",
                    "geometry_kind": "polyline",
                    "measure_id": "Cardiac/SD/Aortic/LVOT Trace/Manual",
                },
            },
        ],
        "tracks": [
            {
                "semantic_id": "pw_doppler",
                "kind": "matrix",
                "data": {"zarr_path": "data/1d_pulsed_wave_doppler"},
                "timestamps": {"zarr_path": "timestamps/1d_pulsed_wave_doppler"},
                "nyquist_mps": 1.0,
                "baseline_frac": 0.5,
            }
        ],
    }
    if merged_depth:
        manifest = dict(group.attrs["recording_manifest"])
        document = {"annotations": manifest.pop("annotations")}
        for _ in range(merged_depth - 1):
            document = {"merged_documents": [document]}
        manifest["merged_documents"] = [document]
        group.attrs["recording_manifest"] = manifest
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d",),
        content_types=("1d_pulsed_wave_doppler",),
        frame_counts_by_content_type={"1d_pulsed_wave_doppler": 300},
        median_delta_time_by_content_type={},
        array_paths=(
            "data/1d_pulsed_wave_doppler",
            "timestamps/1d_pulsed_wave_doppler",
            trace_path,
            measurement_path,
        ),
    )
    store = open_recording(record, root=tmp_path)
    (loaded,) = attach_annotation_overlays(store, (store.load_modality("1d_pulsed_wave_doppler"),))

    overlays = loaded.attrs["annotation_overlays"]
    assert [overlay["kind"] for overlay in overlays] == ["spectral_trace", "spectral_points"]
    assert all(overlay["y_unit"] == "s_mps" for overlay in overlays)
    assert all(overlay["y_coordinate_system"] == "spectral_time_velocity" for overlay in overlays)
    assert overlays[1]["label"] == "Cardiac/SD/Aortic/LVOT Trace/Manual"

    panel = PanelSpec(loaded=loaded, kind="matrix", label="PW")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 1.5, 3.0]))
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    figure = renderer.render_figure_from_specs(panels=(panel,), ecg=ecg, time_s=1.0, frame_index=0, dpi=100)
    try:
        lines = [line for line in figure.axes[0].lines if len(line.get_xdata()) == 20]
        solid = [line for line in lines if line.get_linestyle() == "-"]
        assert solid, "expected a solid trace line on the spectral strip"
        assert np.allclose(solid[0].get_xdata(), trace[:, 0])
        # The strip's y axis is positive up: the negative-velocity dip must lie below the zero-velocity start.
        assert solid[0].get_ydata()[10] < solid[0].get_ydata()[0]
    finally:
        plt.close(figure)


def _cube() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.asarray([tri for q in quads for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))], dtype=np.int32)
    return points, faces


def _packed_mesh(points: np.ndarray, faces: np.ndarray, *, scale: float = 1.0) -> PackedMeshAnnotation:
    return PackedMeshAnnotation(
        points_path="p",
        points=(points * scale).astype(np.float32),
        faces_path="f",
        faces=faces,
        point_frame_offsets=np.asarray([0, points.shape[0]]),
        face_frame_offsets=np.asarray([0, faces.shape[0]]),
    )


def test_mesh_volume_of_a_closed_cube_uses_the_divergence_theorem() -> None:
    from echoxflow.mesh import mesh_volume_curve_ml

    points, faces = _cube()
    curve = mesh_volume_curve_ml(_packed_mesh(points, faces, scale=0.05))  # 5 cm cube: 125 ml

    assert curve is not None and curve.method == "divergence"
    assert curve.volumes_ml[0] == pytest.approx(125.0, rel=0.01)
    flipped = mesh_volume_curve_ml(_packed_mesh(points, faces[:, ::-1].copy(), scale=0.05))
    assert flipped is not None and flipped.volumes_ml[0] == pytest.approx(125.0, rel=0.01)


def test_mesh_volume_of_an_open_surface_falls_back_to_the_convex_hull() -> None:
    from echoxflow.mesh import mesh_volume_curve_ml

    points, faces = _cube()
    curve = mesh_volume_curve_ml(_packed_mesh(points, faces[:-2], scale=0.05))  # one cube face removed

    assert curve is not None and curve.method == "convex_hull"
    assert curve.volumes_ml[0] == pytest.approx(125.0, rel=0.01)
    assert "convex hull" in curve.description


def test_lv_volume_panel_appears_for_a_mesh_record_and_not_for_a_plain_volume(tmp_path: Path) -> None:
    _write_mesh_only_store(tmp_path, linked=True, closed=True)
    renderer = RecordingPlotRenderer()

    panels, _ = renderer._load_specs(
        tmp_path / "mesh.zarr", root=None, modalities=None, view_mode="cartesian", show_annotations=True
    )

    mosaic, curve = panels
    assert mosaic.loaded.data_path == "data/3d_brightness_mode"
    assert curve.kind == "line" and curve.label == "LV volume (ml)"
    assert curve.loaded.data.shape == (2,) and np.all(curve.loaded.data > 0)
    assert curve.loaded.attrs["line_markers"]
    fig = renderer.render_figure_from_specs(
        panels=panels, ecg=TraceSpec(np.zeros(4), np.arange(4.0), "ecg"), time_s=0.0, frame_index=0
    )
    plt.close(fig)

    plain, _ = renderer._load_specs(
        tmp_path / "volume.zarr", root=None, modalities=None, view_mode="cartesian", show_annotations=True
    )
    assert len(plain) == 1


def test_scanner_colorbar_is_two_dimensional_with_scanner_labels() -> None:
    from echoxflow.plotting.colorbar import scanner_colorbar_image, scanner_limit_label

    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[:, :, 0] = np.arange(256)[:, None]
    table[:, :, 1] = np.arange(256)[None, :]
    image = scanner_colorbar_image(table, kind="flow")
    assert image.shape == (184, 30, 3)
    assert image[0, 0, 0] < image[0, -1, 0]  # turbulence axis grows left to right
    assert image[0, 5, 1] != image[-1, 5, 1]  # velocity axis changes down the bar
    assert (scanner_limit_label(0.6), scanner_limit_label(-0.6), scanner_limit_label(0.156)) == (".60", "-.60", ".16")
    spec = colorbar_spec_for_modality("data/2d_color_doppler_velocity", value_range=(-0.6, 0.6), table=table)
    assert spec is not None and spec.tick_labels == ("-.60", ".60")
    assert colorbar_spec_for_modality("data/2d_color_doppler_velocity", value_range=(-0.6, 0.6)).table is None  # type: ignore[union-attr]


@pytest.mark.parametrize("chamber", ["left_atrium", "right_ventricle"])
@pytest.mark.parametrize("suffix", ["", "_1"])
@pytest.mark.parametrize("view", ["beamspace", "cartesian"])
def test_non_lv_mesh_renders_linked_volume_without_overlay(
    tmp_path: Path, caplog, chamber: str, view: str, suffix: str
) -> None:
    _write_mesh_only_store(tmp_path, linked=True)
    mesh = zarr.open_group(tmp_path / "mesh.zarr", mode="a")
    del mesh["annotations/3d_left_ventricle_mesh"]
    mesh.create_array(f"annotations/3d_{chamber}_mesh{suffix}/point_values", data=np.zeros((3, 3)))
    panels, _ = RecordingPlotRenderer()._load_specs(
        tmp_path / "mesh.zarr", root=None, modalities=None, view_mode=view, show_annotations=True
    )
    assert panels
    assert all(panel.loaded.data_path == "data/3d_brightness_mode" for panel in panels)
    assert all("mesh_annotation" not in panel.loaded.attrs for panel in panels)
    assert "without the mesh" in caplog.text
