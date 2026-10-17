from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import ListedColormap

from echoxflow.loading import LoadedArray, _stream_metadata
from echoxflow.plotting import cartesian as cartesian_module
from echoxflow.plotting.cartesian import (
    _tissue_bmode_multiplier,
    cartesian_loaded_arrays,
    phasor_to_wrapped_velocity,
    wrapped_velocity_to_phasor,
)
from echoxflow.plotting.layout import spatial_layout
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.specs import PanelSpec, TraceSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline
from echoxflow.scan.geometry import SectorGeometry
from echoxflow.schema import flow_color_velocity_ramp
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


def test_cartesian_color_doppler_uses_cartesian_composite() -> None:
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

    cartesian = cartesian_loaded_arrays((bmode, velocity, power))

    assert len(cartesian) == 1
    assert cartesian[0].data_path == "data/cartesian_color_doppler"
    assert cartesian[0].data.ndim == 4
    assert cartesian[0].data.shape[-1] == 4
    assert float(np.min(cartesian[0].data[..., 3])) == 0.0
    assert float(np.max(cartesian[0].data[..., 3])) > 0.0
    assert np.isfinite(cartesian[0].data).all()


def test_cartesian_color_doppler_uses_native_color_box_geometry() -> None:
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

    frame = cartesian_loaded_arrays((bmode, velocity, power))[0].data[0]
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


def test_cartesian_color_doppler_uses_fastest_2d_modality_timeline() -> None:
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

    cartesian = cartesian_loaded_arrays((bmode, velocity, power))
    timeline = select_timeline((PanelSpec(loaded=cartesian[0], kind="image", label="cartesian"),))

    assert cartesian[0].data.shape[0] == 4
    np.testing.assert_allclose(cartesian[0].timestamps, doppler_timestamps)
    assert cartesian[0].stream is bmode.stream
    assert cartesian[0].attrs["cartesian_source"] == bmode.data_path
    assert timeline.fps == pytest.approx(20.0)
    np.testing.assert_allclose(timeline.timestamps, doppler_timestamps)


def test_cartesian_color_doppler_requires_velocity_and_power_pair() -> None:
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
        renderer.build_panel_specs((bmode, velocity), view_mode="cartesian")
    with pytest.raises(ValueError, match="requires velocity and power arrays together"):
        renderer.build_panel_specs((bmode, power), view_mode="both")

    panels = renderer.build_panel_specs((velocity,), view_mode="beamspace")
    assert len(panels) == 1


def test_cartesian_color_doppler_hides_center_velocity_band() -> None:
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

    frame = cartesian_loaded_arrays((bmode, velocity, power))[0].data[0]
    visible = frame[..., 3] > 0.0

    assert np.max(np.abs(frame[visible, 0] - frame[visible, 1])) < 1e-6
    assert np.max(np.abs(frame[visible, 1] - frame[visible, 2])) < 1e-6


def test_cartesian_color_doppler_composite_adds_default_colorbar() -> None:
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
    panels = renderer.build_panel_specs((bmode, velocity, power), view_mode="cartesian")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].child_axes) == 1
        assert [text.get_text() for text in figure.axes[0].child_axes[0].texts] == ["0.80", "-0.80"]
    finally:
        plt.close(figure)


def test_cartesian_color_doppler_preserves_annotation_labels() -> None:
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
    panels = renderer.build_panel_specs((bmode, velocity, power), view_mode="cartesian")
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


def test_grouped_contour_overlay_emphasises_mid_wall_row() -> None:
    from echoxflow.scan.contours import LV_GROUP_LAYOUT

    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    wall = np.asarray(
        [[[-0.02 + 0.04 * g, 0.03 + 0.004 * k] for k in range(5)] for g in range(2)],
        dtype=np.float32,
    ).reshape(-1, 2)
    bmode = _loaded(
        "2d_brightness_mode",
        np.full((1, 8, 8), 64, dtype=np.uint8),
        timestamps=np.asarray([0.0], dtype=np.float32),
        geometry=geometry,
        attrs={
            "annotation_overlays": ({"kind": "physical_points", "points": wall[None], "group_layout": LV_GROUP_LAYOUT},)
        },
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=300, dpi=100))
    panels = renderer.build_panel_specs((bmode,), view_mode="cartesian")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0]), timestamps=np.asarray([0.0, 0.1]))
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        lines = figure.axes[0].lines
        mid = [line for line in lines if line.get_zorder() == 8.3]
        assert len(mid) == 1
        assert mid[0].get_linestyle() == "-" and mid[0].get_marker() == "o"
        assert np.asarray(mid[0].get_xdata()).size == 2
        dots = [line for line in lines if line.get_zorder() == 8.2]
        assert [line.get_alpha() for line in dots] == [0.45, 0.7, 0.7, 0.45]
        assert all(line.get_linestyle() == "None" for line in dots)
        assert mid[0].get_markersize() > dots[1].get_markersize() > dots[0].get_markersize()
        outlines = [line for line in lines if line.get_zorder() == 8.1]
        assert len(outlines) == 2 and all(line.get_alpha() == 0.35 for line in outlines)
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

    assert panels[0].loaded.attrs["beamspace_color_doppler_sector"] == color_sector
    assert panels[0].loaded.attrs["beamspace_reference_sector"] == bmode_sector
    assert "beamspace_color_doppler_rectangle" not in panels[0].loaded.attrs
    assert panels[-1].loaded.attrs["cartesian_color_doppler_sector"] == color_sector

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        assert len(figure.axes[0].patches) == 1
        assert len(figure.axes[1].patches) == 0
        assert len(figure.axes[2].patches) == 0
        assert any(line.get_color() == "#FFFFFF" for line in figure.axes[3].lines)
    finally:
        plt.close(figure)


def test_cartesian_tissue_doppler_uses_tissue_gate_overlay() -> None:
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

    cartesian = cartesian_loaded_arrays((bmode, tissue))

    assert len(cartesian) == 1
    assert cartesian[0].data_path == "data/cartesian_tissue_doppler"
    assert cartesian[0].data.ndim == 4
    assert cartesian[0].data.shape[-1] == 4
    assert float(np.min(cartesian[0].data[..., 3])) == 0.0
    assert float(np.max(cartesian[0].data[..., 3])) > 0.0
    assert float(np.max(cartesian[0].data[..., :3])) <= (64.0 / 255.0 / 0.8) + 0.02


def test_cartesian_tissue_doppler_bmode_multiplier_saturates_above_point_eight() -> None:
    bmode = np.asarray([0.0, 102.0, 204.0, 255.0], dtype=np.float32)

    multiplier = _tissue_bmode_multiplier(bmode, value_range=(0.0, 255.0))

    assert np.allclose(multiplier, [0.0, 0.5, 1.0, 1.0])


def test_cartesian_tissue_doppler_interpolates_phasors_before_colouring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tissue velocity is sampled as a cosine/sine phasor on the aliasing circle and coloured afterwards.

    Interpolating RGBA (or the scalar linearly) blends the two ends of a wrapped velocity into false mid-range
    colours, so the sampler must never see a colour stack or the raw scalar.
    """
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
    calls: list[tuple[tuple[int, ...], object]] = []
    original = cartesian_module.sector_stack_to_cartesian_array

    def spy(frames: np.ndarray, *args: object, **kwargs: object):
        calls.append((np.asarray(frames).shape, kwargs.get("interpolation")))
        return original(frames, *args, **kwargs)

    monkeypatch.setattr(cartesian_module, "sector_stack_to_cartesian_array", spy)

    cartesian = cartesian_loaded_arrays((bmode, tissue))

    assert cartesian[0].data_path == "data/cartesian_tissue_doppler"
    assert calls == [((1, 8, 8), "linear"), ((1, 8, 8, 2), "linear")]
    assert cartesian[0].data.shape[-1] == 4


def test_wrapped_velocity_phasor_blends_across_the_nyquist_boundary() -> None:
    velocity = np.asarray([[0.19, -0.19, 0.05, np.nan]], dtype=np.float32)

    phasor = wrapped_velocity_to_phasor(velocity, limit=0.2)
    restored = phasor_to_wrapped_velocity(phasor, limit=0.2)

    assert phasor.shape == (1, 4, 2)
    assert np.allclose(restored, [[0.19, -0.19, 0.05, 0.0]], atol=1e-6)
    # Linear interpolation between +0.19 and -0.19 passes through the aliasing boundary, not through zero.
    midpoint = phasor_to_wrapped_velocity(phasor[:, :2].mean(axis=1), limit=0.2)
    assert abs(float(midpoint[0])) > 0.19


def test_cartesian_color_doppler_blood_gate_override_hides_colour() -> None:
    from echoxflow.plotting.gating import BloodGateConfig

    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=0.8, grid_shape=(8, 8)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded("2d_brightness_mode", np.full((1, 8, 8), 64, np.uint8), timestamps=timestamps, geometry=geometry)
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((1, 8, 8), 0.6, np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((1, 8, 8), np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )

    default = cartesian_loaded_arrays((bmode, velocity, power))[0].data[0, ..., :3]
    gated = cartesian_loaded_arrays((bmode, velocity, power), blood_gate=BloodGateConfig(max_bmode_intensity=0.0))[
        0
    ].data[0, ..., :3]

    assert not np.allclose(default[..., 0], default[..., 2])  # the default gate lets colour through
    assert np.allclose(gated[..., 0], gated[..., 1]) and np.allclose(gated[..., 1], gated[..., 2])  # grey only


def test_cartesian_tissue_gate_override_hides_colour() -> None:
    from echoxflow.plotting.gating import TissueGateConfig

    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=0.8, grid_shape=(8, 8)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded("2d_brightness_mode", np.full((1, 8, 8), 64, np.uint8), timestamps=timestamps, geometry=geometry)
    tissue = _loaded(
        "tissue_doppler",
        np.full((1, 8, 8), 0.1, np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.2,
    )

    default = cartesian_loaded_arrays((bmode, tissue))[0].data[0, ..., :3]
    gated = cartesian_loaded_arrays((bmode, tissue), tissue_gate=TissueGateConfig(min_bmode_intensity=0.5))[0]
    gated = gated.data[0, ..., :3]

    assert not np.allclose(default[..., 0], default[..., 2])  # B-mode at 0.25 passes the default 0.05 threshold
    assert np.allclose(gated[..., 0], gated[..., 1]) and np.allclose(gated[..., 1], gated[..., 2])  # grey at 0.5


def test_tissue_gate_smoothing_ignores_isolated_speckle() -> None:
    from echoxflow.plotting.gating import TissueGateConfig, box_blur, tissue_gate

    bmode = np.zeros((9, 9), dtype=np.float32)
    bmode[4, 4] = 255.0  # one bright speckle in a dark blood pool

    assert tissue_gate(bmode, config=TissueGateConfig(min_bmode_intensity=0.2))[4, 4]
    assert not tissue_gate(bmode, config=TissueGateConfig(min_bmode_intensity=0.2, smoothing_px=2))[4, 4]
    assert np.allclose(box_blur(np.full((5, 5), 3.0, np.float32), 1), 3.0)  # a flat field is unchanged
    assert np.isclose(float(box_blur(bmode, 1)[4, 4]), 255.0 / 9)


def test_cartesian_max_depth_crops_the_sector_to_an_arc() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.20, tilt_rad=0.0, width_rad=0.8, grid_shape=(32, 32)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded("2d_brightness_mode", np.full((1, 32, 32), 200, np.uint8), timestamps=timestamps, geometry=geometry)

    full = cartesian_loaded_arrays((bmode,))[0].data[0, ..., 3]
    capped = cartesian_loaded_arrays((bmode,), max_depth_m=0.10)[0].data[0, ..., 3]

    assert full[-3, full.shape[1] // 2] > 0.0  # the full sector reaches the deepest rows
    # The ends of the deepest row lie beyond 10 cm from the apex: they are outside the arc, not merely cropped flat.
    assert capped[-1, 0] == 0.0 and capped[-1, -1] == 0.0
    assert capped[capped.shape[0] // 2, capped.shape[1] // 2] > 0.0


def test_tissue_brightness_lift_moves_dark_tissue_up_the_table() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=0.8, grid_shape=(4, 4)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[:, :, 0] = np.arange(256)[:, None]  # red encodes the brightness row
    bmode = _loaded("2d_brightness_mode", np.full((1, 4, 4), 100, np.uint8), timestamps=timestamps, geometry=geometry)
    tissue = _loaded(
        "tissue_doppler",
        np.zeros((1, 4, 4), np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.2,
    )
    tissue = replace(tissue, attrs={**tissue.attrs, "tissue_color_table": table})

    plain = cartesian_loaded_arrays((bmode, tissue))[0].data[0, ..., 0]
    lifted = cartesian_loaded_arrays((bmode, tissue), tissue_brightness_lift=0.2)[0].data[0, ..., 0]

    inside = cartesian_loaded_arrays((bmode, tissue))[0].data[0, ..., 3] > 0
    assert np.all(lifted[inside] > plain[inside])


def test_cartesian_color_doppler_samples_velocity_and_power_nearest(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.11,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )
    timestamps = np.asarray([0.0, 0.1], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode", np.full((2, 8, 8), 64, dtype=np.uint8), timestamps=timestamps, geometry=geometry
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
    modes: list[object] = []
    original = cartesian_module.sector_stack_to_cartesian_array

    def spy(frames: np.ndarray, *args: object, **kwargs: object):
        modes.append(kwargs.get("interpolation"))
        return original(frames, *args, **kwargs)

    monkeypatch.setattr(cartesian_module, "sector_stack_to_cartesian_array", spy)

    cartesian_loaded_arrays((bmode, velocity, power))

    assert modes == ["linear", "nearest", "nearest"]


@pytest.mark.parametrize("bmode_name", ("2d_brightness_mode_0", "2d_brightness_mode_1", "2d_brightness_mode_2"))
def test_both_mode_places_beamspace_left_and_cartesian_right(bmode_name: str) -> None:
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

    assert [panel.view for panel in panels] == ["beamspace", "cartesian"]
    assert panels[1].loaded.data.shape[-1] == 4


def test_cartesian_bmode_mmode_keeps_mmode_strip_and_transparent_sector_background() -> None:
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
    panels = renderer.build_panel_specs((bmode, mmode), view_mode="cartesian")
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
    assert [panel.view for panel in panels] == ["beamspace", "cartesian", "cartesian"]
    assert sum(panel.kind == "matrix" for panel in panels) == 1


def test_cartesian_depth_ruler_is_enabled_by_default() -> None:
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
    panels = renderer.build_panel_specs((bmode,), view_mode="cartesian")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        cartesian_ax = figure.axes[0]
        assert len(cartesian_ax.lines) > 1
        assert any(text.get_text() == "5" for text in cartesian_ax.texts)
    finally:
        plt.close(figure)


def test_cartesian_depth_ruler_can_be_disabled() -> None:
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
        style=PlotStyle(width_px=360, height_px=260, dpi=100, show_cartesian_depth_ruler=False, show_panel_labels=False)
    )
    panels = renderer.build_panel_specs((bmode,), view_mode="cartesian")
    ecg = TraceSpec(signal=np.asarray([0.0, 1.0, 0.0]), timestamps=np.asarray([0.0, 0.05, 0.1]))

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.0, frame_index=0, dpi=100)
    try:
        cartesian_ax = figure.axes[0]
        assert not cartesian_ax.lines
        assert not cartesian_ax.texts
    finally:
        plt.close(figure)


def test_both_mode_prefers_horizontal_beamspace_stack() -> None:
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
        assert panels[:3] == tuple(panel for panel in panels if panel.view == "beamspace")
        large = figure.axes[0].get_position()
        top_small = figure.axes[1].get_position()
        bottom_small = figure.axes[2].get_position()
        cartesian = figure.axes[3].get_position()
        assert large.x0 < top_small.x0
        assert np.isclose(top_small.x0, bottom_small.x0)
        assert top_small.y0 > bottom_small.y0
        assert large.height > top_small.height
        assert top_small.x0 < cartesian.x0
        beamspace_width = max(large.x1, top_small.x1, bottom_small.x1) - min(large.x0, top_small.x0, bottom_small.x0)
        assert 0.75 <= beamspace_width / cartesian.width <= 1.35
    finally:
        plt.close(figure)


def test_spatial_layout_stretches_beamspace_over_cartesian_column() -> None:
    def panel(name: str, view: str) -> PanelSpec:
        loaded = _loaded(
            name,
            np.zeros((1, 8, 8), dtype=np.uint8),
            timestamps=np.asarray([0.0], dtype=np.float32),
            geometry=SectorGeometry(0.0, 0.1, -0.4, 0.4, (8, 8)),
        )
        return PanelSpec(loaded=loaded, kind="image", label=name, view=view)  # type: ignore[arg-type]

    panels = (
        panel("a", "beamspace"),
        panel("b", "beamspace"),
        panel("c", "cartesian"),
        panel("d", "cartesian"),
    )
    layout = spatial_layout(panels)

    assert (layout.rows, layout.cols) == (2, 3)
    covered = {
        (row, col)
        for placement in layout.placements
        for row in range(placement.row, placement.row + placement.row_span)
        for col in range(placement.col, placement.col + placement.col_span)
    }
    assert covered == {(row, col) for row in range(2) for col in range(3)}
    beamspace = [p for p in layout.placements if p.panel.view == "beamspace"]
    assert [(p.row, p.row_span) for p in beamspace] == [(0, 2), (0, 2)]


def _flow_table() -> np.ndarray:
    """Variance-style table: row 0 is a velocity ramp, higher rows add green."""
    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
    table[..., 2] = 255 - np.arange(256, dtype=np.uint8)[None, :]
    table[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    return table


def _flow_color_inputs(secondary: int, *, with_codes: bool) -> tuple[LoadedArray, LoadedArray, LoadedArray]:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=0.8, grid_shape=(8, 8)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    table = _flow_table()
    attrs: dict[str, object] = {}
    if with_codes:
        codes = np.full((1, 8, 8), secondary * 256 + 32, dtype=np.uint16)
        attrs = {"flow_color_codes": codes, "flow_color_table": table}
    # The fallback ramp is table row 0 in phase order, so a zero-turbulence code colours like the 1-D path.
    cmap = ListedColormap(flow_color_velocity_ramp(table, phase_ordered=True) / 255.0)
    bmode = _loaded(
        "2d_brightness_mode", np.full((1, 8, 8), 64, dtype=np.uint8), timestamps=timestamps, geometry=geometry
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((1, 8, 8), 0.2, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
        attrs={"colormap": cmap, **attrs},
    )
    power = _loaded(
        "2d_color_doppler_power",
        np.ones((1, 8, 8), dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )
    return bmode, velocity, power


def test_cartesian_flow_color_stack_looks_up_exact_table_colours() -> None:
    bmode, velocity, _ = _flow_color_inputs(200, with_codes=True)
    grid = cartesian_module.CartesianGrid.from_sector_height(bmode.stream.metadata.geometry, 8)

    colours = cartesian_module._cartesian_flow_color_stack(
        velocity, velocity.stream.metadata.geometry, grid, np.asarray([0.0], dtype=np.float32)
    )

    assert colours is not None
    table = _flow_table() / 255.0
    sampled = np.all(np.isclose(colours, table[200, 32]), axis=-1)
    outside = np.all(np.isclose(colours, 0), axis=-1)  # no sample: RGB masked by scan conversion
    assert sampled.any()
    assert np.all(sampled | outside)


def test_cartesian_flow_interpolates_rgb_across_wrapped_codes_without_false_zero_velocity() -> None:
    from dataclasses import replace

    from echoxflow.scan.conversion import sector_stack_to_cartesian_array

    _, velocity, _ = _flow_color_inputs(0, with_codes=True)
    # Adjacent samples straddle the velocity wrap. Arithmetic interpolation
    # of these packed codes would select an unrelated table cell, rather than mixing their displayed colours.
    codes = np.full((1, 8, 8), 127, dtype=np.uint16)
    codes[:, :, 4:] = 128
    table = _flow_table()
    table[:, 127] = [255, 255, 0]  # yellow at positive Nyquist
    table[:, 128] = [0, 255, 255]  # cyan at negative Nyquist
    values = np.full((1, 8, 8), 0.8, dtype=np.float32)
    values[:, :, 4:] = -0.8
    velocity = replace(velocity, data=values, attrs={"flow_color_codes": codes, "flow_color_table": table})
    geometry = velocity.stream.metadata.geometry
    grid = cartesian_module.CartesianGrid.from_sector_height(geometry, 64)
    native_rgb = table[(codes >> 8).astype(int), (codes & 255).astype(int)]
    expected = sector_stack_to_cartesian_array(native_rgb, geometry, grid=grid, interpolation="linear")
    actual = cartesian_module._cartesian_flow_color_stack(velocity, geometry, grid, np.asarray([0.0]))
    assert actual is not None
    np.testing.assert_allclose(actual, expected.data / 255, atol=1e-6)
    visible = actual[0][expected.mask]
    # Yellow/cyan mixtures become green without increasing the turbulence index.
    assert np.any(
        (visible[:, 0] > 0.1)
        & (visible[:, 0] < 0.8)
        & (visible[:, 2] > 0.1)
        & (visible[:, 2] < 0.8)
        & (visible[:, 1] > 0.99)
    )
    scalar = cartesian_module._cartesian_stack(velocity, geometry, grid, np.asarray([0.0]), interpolation="nearest")
    assert np.all(np.abs(scalar.data[0][scalar.mask]) > 0.79)
    # Native source indices remain exact, and the existing scalar velocity sampling is still nearest.
    np.testing.assert_array_equal(velocity.attrs["flow_color_codes"], codes)


def test_cartesian_color_doppler_tints_turbulence_and_falls_back_without_codes() -> None:
    calm = cartesian_loaded_arrays(_flow_color_inputs(0, with_codes=True))[0].data[0]
    turbulent = cartesian_loaded_arrays(_flow_color_inputs(255, with_codes=True))[0].data[0]
    ramp_only = cartesian_loaded_arrays(_flow_color_inputs(0, with_codes=False))[0].data[0]
    visible = calm[..., 3] > 0.0

    assert np.allclose(calm[visible], ramp_only[visible], atol=0.02)
    assert np.mean(turbulent[visible, 1] - calm[visible, 1]) > 0.1


def test_cartesian_color_doppler_composite_has_no_power_tint() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=0.8, grid_shape=(8, 8)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    bmode = _loaded(
        "2d_brightness_mode", np.full((1, 8, 8), 64, dtype=np.uint8), timestamps=timestamps, geometry=geometry
    )
    velocity = _loaded(
        "2d_color_doppler_velocity",
        np.full((1, 8, 8), 0.5, dtype=np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.8,
    )

    def frame(power_value: float) -> np.ndarray:
        power = _loaded(
            "2d_color_doppler_power",
            np.full((1, 8, 8), power_value, dtype=np.float32),
            timestamps=timestamps,
            geometry=geometry,
            velocity_limit_mps=0.8,
        )
        return cartesian_loaded_arrays((bmode, velocity, power))[0].data[0]

    strong, stronger = frame(0.8), frame(1.0)

    assert (strong[..., 3] > 0.0).any()
    np.testing.assert_array_equal(strong, stronger)


def test_blood_gate_defaults_match_the_scanner_coverage() -> None:
    from echoxflow.plotting.gating import BloodGateConfig, blood_gate

    config = BloodGateConfig()
    assert (config.min_doppler_power, config.min_power_minus_bmode, config.max_bmode_intensity) == (0.20, 0.15, 0.50)
    bmode = np.asarray([[25.0, 25.0, 25.0, 150.0, 250.0]], dtype=np.float32)  # normalised 0.098, 0.588, 0.980
    power = np.asarray([[0.15, 0.30, 0.9, 0.9, 0.9]], dtype=np.float32)
    # weak power fails, moderate power over dark tissue passes, strong power over mid-grey or bright tissue fails
    assert blood_gate(bmode, power).tolist() == [[False, True, True, False, False]]


def test_tissue_doppler_frame_indexes_table_brightness_axis_by_bmode() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=0.8, grid_shape=(4, 4)
    )
    timestamps = np.asarray([0.0], dtype=np.float32)
    table = np.zeros((256, 256, 3), dtype=np.uint8)
    table[:, :, 0] = np.arange(256)[:, None]  # red encodes the brightness row
    table[:, :, 1] = np.arange(256)[None, :]  # green encodes the velocity phase code
    bmode = _loaded("2d_brightness_mode", np.full((1, 4, 4), 100, np.uint8), timestamps=timestamps, geometry=geometry)
    tissue = _loaded(
        "tissue_doppler",
        np.zeros((1, 4, 4), np.float32),
        timestamps=timestamps,
        geometry=geometry,
        velocity_limit_mps=0.2,
        attrs={"tissue_color_table": table},
    )
    tissue = replace(
        tissue, stream=replace(tissue.stream, metadata=replace(tissue.stream.metadata, value_range=(-0.2, 0.2)))
    )
    region = np.ones((4, 4), dtype=bool)
    frame = cartesian_module._tissue_doppler_frame(
        bmode=bmode,
        tissue=tissue,
        bmode_data=np.full((4, 4), 100.0, np.float32),
        bmode_mask=region,
        tissue_data=np.zeros((4, 4), np.float32),
        region=region,
    )
    expected_row = round(255 * (0.85 * 100 / 255 + 0.02))
    np.testing.assert_allclose(frame[0, 0, 0], expected_row / 255, atol=1e-6)
    np.testing.assert_allclose(frame[0, 0, 1], 0.0, atol=1e-6)


@pytest.mark.parametrize("name", ["2d_brightness_mode_plane_0", "2d_brightness_mode_plane_12", "2d_brightness_mode_0"])
def test_cartesian_plane_uses_bmode_colormap(name: str) -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01, depth_end_m=0.11, tilt_rad=0.0, width_rad=1.0, grid_shape=(8, 8)
    )
    data = np.arange(32, dtype=np.uint8).reshape(1, 4, 8)
    kwargs = {"timestamps": np.asarray([0.0]), "geometry": geometry}
    expected = cartesian_loaded_arrays((_loaded("2d_brightness_mode", data, **kwargs),))[0]
    actual = cartesian_loaded_arrays((_loaded(name, data, **kwargs),))[0]
    np.testing.assert_array_equal(actual.data, expected.data)
