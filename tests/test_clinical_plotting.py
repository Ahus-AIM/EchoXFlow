import matplotlib.pyplot as plt
import numpy as np
import pytest

from echoxflow.loading import LoadedArray, _stream_metadata
from echoxflow.plotting import clinical as clinical_module
from echoxflow.plotting.clinical import _tissue_bmode_multiplier, clinical_loaded_arrays
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import PanelSpec, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.streams import StreamMetadata, stream_from_arrays


def _loaded(
    name: str,
    data: np.ndarray,
    *,
    timestamps: np.ndarray,
    geometry: SectorGeometry,
    velocity_limit_mps: float | None = None,
    raw: dict[str, object] | None = None,
    attrs: dict[str, object] | None = None,
) -> LoadedArray:
    data_path = f"data/{name}"
    stream = stream_from_arrays(
        data_path=data_path,
        data=data,
        timestamps_path=f"timestamps/{name}",
        timestamps=timestamps,
        sample_rate_hz=10.0,
        metadata=StreamMetadata(
            data_path=data_path,
            velocity_limit_mps=velocity_limit_mps,
            geometry=geometry,
            raw=raw,
        ),
    )
    return LoadedArray(
        name=name,
        data_path=data_path,
        data=data,
        timestamps_path=f"timestamps/{name}",
        timestamps=timestamps,
        sample_rate_hz=10.0,
        attrs=dict(attrs or {}),
        stream=stream,
    )


def test_clinical_color_doppler_uses_cartesian_composite() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((2, 8, 8), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((2, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )

    clinical = clinical_loaded_arrays((bmode, velocity, power))

    assert len(clinical) == 1
    assert clinical[0].data_path == "data/clinical_color_doppler"
    assert clinical[0].data.ndim == 4
    assert clinical[0].data.shape[-1] == 4
    assert float(np.min(clinical[0].data[..., 3])) == 0.0
    assert float(np.max(clinical[0].data[..., 3])) > 0.0
    assert np.isfinite(clinical[0].data).all()


def test_clinical_color_doppler_uses_native_color_box_geometry() -> None:
    bmode_geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(32, 32),
    )
    color_geometry = SectorGeometry.from_center_width(
        depth_start_m=0.03,
        depth_end_m=0.09,
        tilt_rad=0.20,
        width_rad=0.24,
        grid_shape=(16, 16),
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((1, 32, 32), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=bmode_geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((1, 16, 16), 0.45, dtype=np.float32),
        timestamps=timestamps,
        geometry=color_geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((1, 16, 16), dtype=np.float32),
        timestamps=timestamps,
        geometry=color_geometry,
        velocity_limit_mps=0.8,
    )

    frame = clinical_loaded_arrays((bmode, velocity, power))[0].data[0]
    colorized = (
        (np.abs(frame[..., 0] - frame[..., 1]) > 1e-3)
        | (np.abs(frame[..., 1] - frame[..., 2]) > 1e-3)
        | (np.abs(frame[..., 0] - frame[..., 2]) > 1e-3)
    )
    colored_cols = np.flatnonzero(colorized.any(axis=0))

    assert colored_cols.size > 0
    assert colored_cols.size < frame.shape[1] // 2
    assert int(colored_cols[0]) > 0
    assert float(colored_cols.mean()) > frame.shape[1] / 2.0


def test_clinical_color_doppler_uses_fastest_2d_modality_timeline() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    bmode_timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    doppler_timestamps = np.asarray([0.0, 0.05, 0.1, 0.15], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=bmode_timestamps,
        geometry=geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((4, 8, 8), 0.2, dtype=np.float32),
        timestamps=doppler_timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((4, 8, 8), dtype=np.float32),
        timestamps=doppler_timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )

    clinical = clinical_loaded_arrays((bmode, velocity, power))
    timeline = select_timeline((PanelSpec(loaded=clinical[0], kind="image", label="clinical"),))

    assert clinical[0].data.shape[0] == 4
    np.testing.assert_allclose(clinical[0].timestamps, doppler_timestamps)
    assert clinical[0].stream is bmode.stream
    assert clinical[0].attrs["clinical_source"] == bmode.data_path
    assert timeline.fps == pytest.approx(20.0)
    np.testing.assert_allclose(timeline.timestamps, doppler_timestamps)


def test_clinical_color_doppler_requires_velocity_and_power_pair() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((2, 8, 8), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((2, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    renderer = RecordingPlotRenderer()

    with pytest.raises(ValueError, match="requires velocity and power arrays together"):
        renderer.build_panel_specs((bmode, velocity), view_mode="clinical")
    with pytest.raises(ValueError, match="requires velocity and power arrays together"):
        renderer.build_panel_specs((bmode, power), view_mode="both")

    panels = renderer.build_panel_specs((velocity,), view_mode="pre_converted")
    assert len(panels) == 1


def test_clinical_color_doppler_hides_center_velocity_band() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((1, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.zeros((1, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((1, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )

    frame = clinical_loaded_arrays((bmode, velocity, power))[0].data[0]
    visible = frame[..., 3] > 0.0

    assert np.max(np.abs(frame[visible, 0] - frame[visible, 1])) < 1e-6
    assert np.max(np.abs(frame[visible, 1] - frame[visible, 2])) < 1e-6


def test_clinical_color_doppler_composite_adds_default_colorbar() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((2, 8, 8), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((2, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = renderer.build_panel_specs((bmode, velocity, power), view_mode="clinical")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].child_axes) == 1
        assert figure.axes[0].child_axes[0].get_ylabel() == "Velocity (m/s)"
    finally:
        plt.close(figure)


def test_clinical_color_doppler_preserves_annotation_labels() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((1, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
        attrs={
            "annotation_overlays": (
                {
                    "kind": "physical_points",
                    "points": np.asarray([[0.0, 0.04]], dtype=np.float32),
                    "label": "B-mode point",
                },
            )
        },
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((1, 8, 8), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
        attrs={
            "annotation_overlays": (
                {
                    "kind": "physical_points",
                    "points": np.asarray([[0.01, 0.05]], dtype=np.float32),
                    "label": "Velocity point",
                },
            )
        },
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((1, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = renderer.build_panel_specs((bmode, velocity, power), view_mode="clinical")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0]), timestamps=np.asarray([0.0, 0.1]))

    overlays = panels[0].loaded.attrs["annotation_overlays"]
    assert {overlay["label"] for overlay in overlays if isinstance(overlay, dict) and "label" in overlay} == {
        "B-mode point",
        "Velocity point",
    }

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        text_labels = [text.get_text() for text in figure.axes[0].texts]
        assert text_labels.count("B-mode point") == 1
        assert text_labels.count("Velocity point") == 1
    finally:
        plt.close(figure)


def test_color_doppler_extent_uses_native_sector_metadata() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=0.12,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(12, 12),
    )
    color_geometry = SectorGeometry.from_center_width(
        depth_start_m=0.03,
        depth_end_m=0.09,
        tilt_rad=0.0,
        width_rad=0.4,
        grid_shape=(6, 4),
    )
    bmode_sector = {
        "geometry": {
            "depth_start_m": 0.0,
            "depth_end_m": 0.12,
            "tilt_rad": 0.0,
            "width_rad": 0.8,
        }
    }
    color_sector = {
        "geometry": {
            "depth_start_m": 0.03,
            "depth_end_m": 0.09,
            "tilt_rad": 0.0,
            "width_rad": 0.4,
        },
        "overlays": {
            "physical_polygons": [
                {
                    "label": "color_doppler_extent",
                    "points": [[-0.01, 0.03], [0.01, 0.03], [0.02, 0.09], [-0.02, 0.09]],
                }
            ]
        },
    }
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 12, 12), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
        raw=bmode_sector,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((2, 6, 4), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=color_geometry,
        velocity_limit_mps=0.8,
        raw=color_sector,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((2, 6, 4), dtype=np.float32),
        timestamps=timestamps,
        geometry=color_geometry,
        velocity_limit_mps=0.8,
        raw=color_sector,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=600, height_px=360, dpi=100))
    panels = renderer.build_panel_specs((bmode, velocity, power), view_mode="both")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    assert panels[0].loaded.attrs["preconverted_color_doppler_sector"] == color_sector
    assert panels[0].loaded.attrs["preconverted_reference_sector"] == bmode_sector
    assert "preconverted_color_doppler_rectangle" not in panels[0].loaded.attrs
    assert panels[-1].loaded.attrs["clinical_color_doppler_sector"] == color_sector

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].patches) == 1
        assert len(figure.axes[1].patches) == 0
        assert len(figure.axes[2].patches) == 0
        assert any(line.get_color() == "#FFFFFF" for line in figure.axes[3].lines)
    finally:
        plt.close(figure)


def test_clinical_tissue_doppler_uses_tissue_gate_overlay() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    tissue = _loaded(
        "tissue_doppler",
        np.full((2, 8, 8), 0.05, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.2,
    )

    clinical = clinical_loaded_arrays((bmode, tissue))

    assert len(clinical) == 1
    assert clinical[0].data_path == "data/clinical_tissue_doppler"
    assert clinical[0].data.ndim == 4
    assert clinical[0].data.shape[-1] == 4
    assert float(np.min(clinical[0].data[..., 3])) == 0.0
    assert float(np.max(clinical[0].data[..., 3])) > 0.0
    assert float(np.max(clinical[0].data[..., :3])) <= (64.0 / 255.0 / 0.8) + 0.02


def test_clinical_tissue_doppler_bmode_multiplier_saturates_above_point_eight() -> None:
    bmode = np.asarray([0.0, 102.0, 204.0, 255.0], dtype=np.float32)

    multiplier = _tissue_bmode_multiplier(bmode, value_range=(0.0, 255.0))

    assert np.allclose(multiplier, [0.0, 0.5, 1.0, 1.0])


def test_clinical_tissue_doppler_colormaps_before_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((1, 8, 8), 255, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    tissue = _loaded(
        "tissue_doppler",
        np.linspace(-0.2, 0.2, 64, dtype=np.float32).reshape(1, 8, 8),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.2,
    )
    sampled_shapes: list[tuple[int, ...]] = []
    original = clinical_module.sector_to_cartesian

    def spy_sector_to_cartesian(image: np.ndarray, *args: object, **kwargs: object):
        sampled_shapes.append(np.asarray(image).shape)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(clinical_module, "sector_to_cartesian", spy_sector_to_cartesian)

    clinical = clinical_loaded_arrays((bmode, tissue))

    assert clinical[0].data_path == "data/clinical_tissue_doppler"
    assert (8, 8) in sampled_shapes
    assert (8, 8, 4) in sampled_shapes


@pytest.mark.parametrize("bmode_name", ("2d_brightness_mode_0", "2d_brightness_mode_1", "2d_brightness_mode_2"))
def test_both_mode_places_pre_converted_left_and_clinical_right(bmode_name: str) -> None:
    wrong = {"DepthStart": 0.01, "DepthEnd": 0.04, "Tilt": 0.0, "Width": 0.8, "GridSize": [8, 8]}
    right = {**wrong, "DepthEnd": 0.11}
    geometry = _stream_metadata(
        f"data/{bmode_name}",
        {
            "recording_manifest": {
                "sectors": [
                    {"semantic_id": "bmode", "geometry": wrong},
                    {"semantic_id": "bmode", "frames": {"zarr_path": f"data/{bmode_name}"}, "geometry": right},
                ]
            }
        },
    ).geometry
    assert geometry is not None and geometry.depth_end_m == pytest.approx(0.11)
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        bmode_name,
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = renderer.build_panel_specs((bmode,), view_mode="both")

    assert [panel.view for panel in panels] == ["pre_converted", "clinical"]
    assert panels[1].loaded.data.shape[-1] == 4


def test_clinical_bmode_mmode_keeps_mmode_strip_and_transparent_sector_background() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((3, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    mmode = _loaded(
        "1d_motion_mode",
        np.zeros((3, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = renderer.build_panel_specs((bmode, mmode), view_mode="clinical")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.1, 0.2]))

    assert [panel.loaded.data_path for panel in panels] == ["data/2d_brightness_mode", "data/1d_motion_mode"]
    assert panels[0].kind == "image"
    assert panels[0].loaded.data.shape[-1] == 4
    assert float(np.min(panels[0].loaded.data[..., 3])) == 0.0
    assert float(np.max(panels[0].loaded.data[..., 3])) == 1.0
    assert panels[1].kind == "matrix"

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.1, frame_index=1, dpi=100)
    try:
        assert len(figure.axes) == 3
        assert figure.axes[0].get_position().y0 > figure.axes[1].get_position().y0
    finally:
        plt.close(figure)


def test_both_mode_keeps_single_mmode_strip() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((3, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    mmode = _loaded(
        "1d_motion_mode",
        np.zeros((3, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))

    panels = renderer.build_panel_specs((bmode, mmode), view_mode="both")

    assert [panel.loaded.data_path for panel in panels] == [
        "data/2d_brightness_mode",
        "data/2d_brightness_mode",
        "data/1d_motion_mode",
    ]
    assert [panel.view for panel in panels] == ["pre_converted", "clinical", "clinical"]
    assert sum(panel.kind == "matrix" for panel in panels) == 1


def test_clinical_depth_ruler_is_enabled_by_default() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=360, height_px=260, dpi=100))
    panels = renderer.build_panel_specs((bmode,), view_mode="clinical")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        clinical_ax = figure.axes[0]
        assert len(clinical_ax.lines) > 1
        assert any(text.get_text() == "5" for text in clinical_ax.texts)
    finally:
        plt.close(figure)


def test_clinical_depth_ruler_can_be_disabled() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 8, 8), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    renderer = RecordingPlotRenderer(
        style=PlotStyle(width_px=360, height_px=260, dpi=100, show_clinical_depth_ruler=False)
    )
    panels = renderer.build_panel_specs((bmode,), view_mode="clinical")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        clinical_ax = figure.axes[0]
        assert not clinical_ax.lines
        assert not clinical_ax.texts
    finally:
        plt.close(figure)


def test_both_mode_prefers_horizontal_pre_converted_stack() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((2, 12, 12), 64, dtype=np.uint8),
        timestamps=timestamps,
        geometry=geometry,
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((2, 8, 8), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((2, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=600, height_px=360, dpi=100))
    panels = renderer.build_panel_specs((bmode, velocity, power), view_mode="both")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        assert panels[:3] == tuple(panel for panel in panels if panel.view == "pre_converted")
        large = figure.axes[0].get_position()
        top_small = figure.axes[1].get_position()
        bottom_small = figure.axes[2].get_position()
        clinical = figure.axes[3].get_position()
        assert large.x0 < top_small.x0
        assert np.isclose(top_small.x0, bottom_small.x0)
        assert top_small.y0 > bottom_small.y0
        assert large.height > top_small.height
        assert top_small.x0 < clinical.x0
        pre_width = max(large.x1, top_small.x1, bottom_small.x1) - min(large.x0, top_small.x0, bottom_small.x0)
        assert 0.75 <= pre_width / clinical.width <= 1.35
    finally:
        plt.close(figure)
