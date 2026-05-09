from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from echoxflow.colors import data_modality_colormap_name, named_listed_colormap
from echoxflow.croissant import RecordingRecord
from echoxflow.loading import LoadedArray, RecordingStore, open_recording
from echoxflow.plotting.annotations import attach_annotation_overlays
from echoxflow.plotting.colorbar import colorbar_spec_for_modality
from echoxflow.plotting.panels import renderer_for
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import PanelSpec, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import nearest_index, select_timeline
from echoxflow.plotting.writers import write_video
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.spectral import SpectralMetadata
from echoxflow.streams import StreamMetadata, stream_from_arrays

zarr = pytest.importorskip("zarr")


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


def test_single_pre_converted_image_preserves_aspect_ratio() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    image_panel = PanelSpec(
        loaded=_loaded("2d_brightness_mode", np.zeros((2, 8, 24), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="B-mode",
        view="pre_converted",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(image_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert figure.axes[0].get_aspect() == 1.0
    finally:
        plt.close(figure)


def test_multiple_pre_converted_images_keep_automatic_aspect() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = (
        PanelSpec(
            loaded=_loaded("2d_brightness_mode", np.zeros((2, 8, 24), dtype=np.float32), np.asarray([0.0, 0.1])),
            kind="image",
            label="B-mode",
            view="pre_converted",
        ),
        PanelSpec(
            loaded=_loaded("tissue_doppler", np.zeros((2, 8, 24), dtype=np.float32), np.asarray([0.0, 0.1])),
            kind="image",
            label="TDI",
            view="pre_converted",
        ),
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert figure.axes[0].get_aspect() == "auto"
        assert figure.axes[1].get_aspect() == "auto"
    finally:
        plt.close(figure)


def test_pre_converted_doppler_panel_omits_colorbar_by_default() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    tissue_panel = PanelSpec(
        loaded=_loaded("tissue_doppler", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="TDI",
        view="pre_converted",
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
    assert spec.bbox_axes[0] + spec.bbox_axes[2] == 0.98
    assert spec.bbox_axes[1] + spec.bbox_axes[3] == 0.98


def test_tissue_doppler_colorbar_displays_cm_per_second() -> None:
    spec = colorbar_spec_for_modality("data/tissue_doppler", value_range=(-0.2, 0.2))

    assert spec is not None
    assert spec.label == "Velocity (cm/s)"
    assert spec.value_range == (-20.0, 20.0)
    assert spec.ticks == (-20.0, 0.0, 20.0)
    assert spec.tick_labels == ("-20", "0", "20")


def test_clinical_doppler_panel_adds_fixed_top_right_colorbar() -> None:
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    tissue_panel = PanelSpec(
        loaded=_loaded("tissue_doppler", np.zeros((2, 8, 8), dtype=np.float32), np.asarray([0.0, 0.1])),
        kind="image",
        label="TDI",
        view="clinical",
    )
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=(tissue_panel,), ecg=ecg, time_s=0.05, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].child_axes) == 1
        colorbar_ax = figure.axes[0].child_axes[0]
        bounds = colorbar_ax.get_position().bounds
        assert bounds[0] > 0.65
        assert bounds[1] > 0.45
        assert colorbar_ax.get_ylabel() == "Velocity (cm/s)"
        assert colorbar_ax.yaxis.label.get_size() == 16.0
        assert all(label.get_size() == 16.0 for label in colorbar_ax.get_yticklabels())
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

    pre = renderer.build_panel_specs(loaded, view_mode="pre-converted")
    clinical = renderer.build_panel_specs(loaded, view_mode="clinical")
    cartesian = renderer.build_panel_specs(loaded, view_mode="cartesian")
    both = renderer.build_panel_specs(loaded, view_mode="both")

    assert [panel.view for panel in pre] == ["pre_converted", "pre_converted"]
    assert [panel.view for panel in clinical] == ["clinical", "clinical"]
    assert [panel.view for panel in cartesian] == ["clinical", "clinical"]
    assert [panel.view for panel in both] == ["pre_converted", "clinical", "clinical"]
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
        assert ax.get_ylabel() == "Distance (cm)"
        assert np.allclose(ax.get_yticks(), (0.0, 8.0))
        assert [label.get_text() for label in ax.get_yticklabels()] == ["0 cm", "8 cm"]
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
        assert ax.get_ylabel() == "Distance (cm)"
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
        assert ax.get_ylabel() == "Velocity (m/s)"
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

    pre = renderer.build_panel_specs((loaded,), view_mode="pre_converted")
    clinical = renderer.build_panel_specs((loaded,), view_mode="clinical")

    assert len(pre) == 1
    assert pre[0].loaded.data.shape == (2, 360, 480)
    assert pre[0].label == "brightness mode"
    assert clinical[0].loaded.data.shape[0:2] == (2, 360)
    assert clinical[0].loaded.data.shape[2] > 480
    assert clinical[0].view == "clinical"


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

    (panel,) = renderer.build_panel_specs((loaded,), view_mode="pre_converted")

    assert panel.loaded.attrs["3d_was_beat_stitched"] is True
    assert panel.loaded.attrs["3d_stitch_beat_count"] == 4
    assert panel.loaded.data.shape == (2, 360, 480)
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

    (panel,) = renderer.build_panel_specs((loaded,), view_mode="pre_converted")

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
        view_mode="pre_converted",
        show_annotations=True,
    )
    disabled_panels, _disabled_ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode",),
        view_mode="pre_converted",
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


def test_tissue_doppler_gate_draws_preconverted_vertical_marker(tmp_path: Path) -> None:
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
                "storage_encoding": "linear_velocity_uint8_mps_v1",
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
        view_mode="pre_converted",
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


def test_mmode_sampling_line_draws_in_preconverted_and_clinical_views(tmp_path: Path) -> None:
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
        view_mode="pre_converted",
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

    clinical_panels, clinical_ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "1d_motion_mode"),
        view_mode="clinical",
        show_annotations=True,
    )
    clinical_figure = renderer.render_figure_from_specs(
        panels=clinical_panels, ecg=clinical_ecg, time_s=0.0, frame_index=0, dpi=100
    )
    try:
        foreground = [
            line
            for line in clinical_figure.axes[0].lines
            if to_hex(line.get_color(), keep_alpha=False) == "#4ac44a" and np.isclose(line.get_zorder(), 8.5)
        ]
        solid = [line for line in foreground if line.get_linestyle() == "-"]
        assert len(solid) == 1
        assert np.allclose(np.column_stack([solid[0].get_xdata(), solid[0].get_ydata()]), cursor_line)
    finally:
        plt.close(clinical_figure)


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
        view_mode="pre_converted",
        show_annotations=True,
    )
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        foreground = [line for line in figure.axes[0].lines if to_hex(line.get_color(), keep_alpha=False) == "#4ac44a"]
        assert [line.get_linestyle() for line in foreground] == ["-"]
        assert not any(line.get_marker() == "_" for line in foreground)
    finally:
        plt.close(figure)


def test_clinical_tissue_doppler_carries_gate_overlay(tmp_path: Path) -> None:
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
                "storage_encoding": "linear_velocity_uint8_mps_v1",
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
        view_mode="pre_converted",
        show_annotations=True,
    )
    assert all(
        not any(overlay["kind"] == "sampling_gate" for overlay in panel.loaded.attrs.get("annotation_overlays", ()))
        for panel in no_spectral_panels
    )

    preconverted_panels, _ = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("2d_brightness_mode", "tissue_doppler", "1d_pulsed_wave_doppler"),
        view_mode="pre_converted",
        show_annotations=True,
    )
    sector_panels = tuple(
        panel for panel in preconverted_panels if panel.loaded.data_path != "data/1d_pulsed_wave_doppler"
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
        view_mode="clinical",
        show_annotations=True,
    )

    assert panels[0].view == "clinical"
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

    (pre_panel,) = renderer.build_panel_specs((loaded,), view_mode="pre_converted")
    (clinical_panel,) = renderer.build_panel_specs((loaded,), view_mode="clinical")

    assert "mosaic_annotation_lines" not in pre_panel.loaded.attrs
    assert "mosaic_annotation_lines" in clinical_panel.loaded.attrs
    assert any(line.size for line in clinical_panel.loaded.attrs["mosaic_annotation_lines"][0])


def test_3d_mesh_arrays_without_manifest_link_do_not_attach_overlay(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((2, 3, 4, 5), dtype=np.uint8))
    group.create_array("timestamps/3d_brightness_mode", data=np.asarray([0.0, 1.0], dtype=np.float32))
    mesh_group = group.create_group("data/3d_left_ventricle_mesh")
    mesh_group.create_array("point_values", data=np.zeros((3, 3), dtype=np.float32))
    mesh_group.create_array("face_values", data=np.asarray([[0, 1, 2]], dtype=np.int32))
    group.attrs["recording_manifest"] = _three_d_manifest(linked_mesh=False)
    store = open_recording(tmp_path / "case.zarr")
    loaded = attach_annotation_overlays(store, (store.load_modality("3d_brightness_mode"),))[0]

    assert "mesh_annotation" not in loaded.attrs
    (panel,) = RecordingPlotRenderer().build_panel_specs((loaded,), view_mode="clinical")
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
    (panel,) = RecordingPlotRenderer().build_panel_specs((loaded,), view_mode="clinical")

    assert "mosaic_annotation_lines" in panel.loaded.attrs
    assert len(panel.loaded.attrs["mosaic_annotation_lines"]) == 2
    assert any(line.size for line in panel.loaded.attrs["mosaic_annotation_lines"][0])


def test_single_frame_3d_mesh_sequence_does_not_repeat_on_multi_frame_volume(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/3d_brightness_mode", data=np.zeros((2, 3, 4, 5), dtype=np.uint8))
    group.create_array("timestamps/3d_brightness_mode", data=np.asarray([0.0, 1.0], dtype=np.float32))
    _write_mesh_group(group, frame_count=1)
    group.attrs["recording_manifest"] = _three_d_manifest()
    store = open_recording(tmp_path / "case.zarr")
    loaded = attach_annotation_overlays(store, (store.load_modality("3d_brightness_mode"),))[0]
    (panel,) = RecordingPlotRenderer().build_panel_specs((loaded,), view_mode="clinical")

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


def _write_mesh_group(group: object, *, frame_count: int) -> None:
    points = np.asarray(
        [[-0.02, 0.05, -0.01], [0.02, 0.05, 0.01], [0.0, 0.08, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    mesh_group = group.create_group("data/3d_left_ventricle_mesh")
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
        view="clinical",
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
    early_yellow = table[24]
    assert early_yellow[0] > 0.95
    assert early_yellow[1] > 0.95
    assert early_yellow[2] < 0.1
    assert table[127, 0] > 0.9
    assert table[127, 2] < 0.1
    assert table[128, 2] > 0.9
    assert table[128, 0] < 0.1
    red_rows = table[:128]
    red_count = int(np.count_nonzero((red_rows[:, 0] > 0.95) & (red_rows[:, 1] < 0.05) & (red_rows[:, 2] < 0.05)))
    assert red_count >= 8


def test_write_video_smoke(tmp_path: Path) -> None:
    output = tmp_path / "tiny.mp4"
    frames = np.zeros((2, 15, 17, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255

    result = write_video(output, frames, fps=2.0)

    assert result == output
    assert output.stat().st_size > 0


def test_write_video_warns_for_single_frame_source(tmp_path: Path) -> None:
    output = tmp_path / "single.mp4"
    source = tmp_path / "source.zarr"
    frames = np.zeros((1, 16, 16, 3), dtype=np.uint8)

    with pytest.warns(UserWarning, match="single-frame MP4.*single\\.mp4.*source\\.zarr"):
        result = write_video(output, frames, fps=2.0, source=source)

    assert result == output
    assert output.stat().st_size > 0
