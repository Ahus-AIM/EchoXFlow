"""Frames from ``iter_frames`` (one reused figure) must match a figure rebuilt from scratch per frame."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import make_record, strain_document
from test_plotting import _write_mesh_only_store
from test_strain_plotting import _contours

from echoxflow.croissant import RecordingRecord
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline
from echoxflow.plotting.writers import figure_to_rgb

zarr = pytest.importorskip("zarr")


def _assert_reused_figure_matches_fresh(
    renderer: RecordingPlotRenderer,
    record: RecordingRecord | Path,
    *,
    root: Path | None,
    view_mode: str,
    modalities: tuple[str, ...] | None = None,
    min_frames: int = 3,
) -> None:
    panels, ecg = renderer._load_specs(
        record, root=root, modalities=modalities, view_mode=view_mode, show_annotations=True
    )
    times = np.asarray(select_timeline(panels).timestamps, dtype=np.float64).reshape(-1)
    assert times.size >= min_frames
    dpi = 100
    state = renderer._build_figure(panels=panels, ecg=ecg, time_s=float(times[0]), frame_index=0, dpi=dpi)
    plt.close(state.figure)
    assert state.updatable, "the reuse path must be exercised, not the full-rebuild fallback"

    reused = list(renderer.iter_frames(panels, ecg, times, dpi=dpi, title="case"))
    assert len(reused) == times.size
    for index, (time_s, frame) in enumerate(zip(times, reused, strict=True)):
        figure = renderer.render_figure_from_specs(
            panels=panels, ecg=ecg, time_s=float(time_s), frame_index=index, dpi=dpi, title="case"
        )
        try:
            fresh = figure_to_rgb(figure)
        finally:
            plt.close(figure)
        assert frame.shape == fresh.shape
        assert np.abs(frame.astype(int) - fresh.astype(int)).max() <= 2, f"frame {index} differs"
    assert not plt.get_fignums()
    # The frames must actually differ over time, otherwise the comparison proves little.
    assert any(not np.array_equal(reused[0], other) for other in reused[1:])


def _renderer() -> RecordingPlotRenderer:
    return RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=320, dpi=100))


@pytest.mark.parametrize("view_mode", ["beamspace", "both"])
def test_bmode_with_spectral_strip_and_ecg_reuses_figure(tmp_path: Path, view_mode: str) -> None:
    rng = np.random.default_rng(0)
    group = zarr.open_group(tmp_path / "case.zarr", mode="w")
    group.create_array("data/2d_brightness_mode", data=rng.integers(0, 255, (6, 16, 16), dtype=np.uint8))
    group.create_array("timestamps/2d_brightness_mode", data=np.linspace(0.0, 0.5, 6).astype(np.float32))
    group.create_array("data/1d_pulsed_wave_doppler", data=rng.normal(size=(200, 12)).astype(np.float32))
    group.create_array("timestamps/1d_pulsed_wave_doppler", data=np.linspace(0.0, 0.5, 200).astype(np.float32))
    group.create_array("data/ecg", data=np.sin(np.linspace(0.0, 6.0, 300)).astype(np.float32))
    group.create_array("timestamps/ecg", data=np.linspace(0.0, 0.5, 300).astype(np.float32))
    paths = tuple(
        f"{kind}/{name}"
        for name in ("2d_brightness_mode", "1d_pulsed_wave_doppler", "ecg")
        for kind in ("data", "timestamps")
    )
    record = RecordingRecord(
        exam_id="exam",
        recording_id="case",
        zarr_path="case.zarr",
        modes=("2d_brightness_mode", "1d_pulsed_wave_doppler"),
        content_types=("2d_brightness_mode", "1d_pulsed_wave_doppler"),
        frame_counts_by_content_type={"2d_brightness_mode": 6, "1d_pulsed_wave_doppler": 200},
        median_delta_time_by_content_type={},
        array_paths=paths,
    )
    _assert_reused_figure_matches_fresh(_renderer(), record, root=tmp_path, view_mode=view_mode)


def test_strain_recording_reuses_figure(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "rv.zarr", mode="w")
    bmode = np.arange(3, dtype=np.uint8).reshape(3, 1, 1) * np.ones((3, 16, 16), dtype=np.uint8) * 40
    group.create_array("data/2d_brightness_mode", data=bmode)
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    delayed = np.concatenate([np.full((1, 3, 2), np.nan, dtype=np.float32), _contours()], axis=0)
    group.create_array("data/rv_contour", data=delayed)
    group.create_array("timestamps/rv_contour", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    group.create_array("data/rv_curve", data=np.asarray([0.0, -6.0, -12.0], dtype=np.float32))
    group.create_array("timestamps/rv_curve", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    group.attrs["recording_manifest"] = strain_document(("rv",), self_recording_id="rv")
    record = make_record(
        "rv",
        "rv.zarr",
        content_types=("2d_right_ventricular_strain",),
        array_paths=(
            "data/2d_brightness_mode",
            "timestamps/2d_brightness_mode",
            "data/rv_contour",
            "timestamps/rv_contour",
            "data/rv_curve",
            "timestamps/rv_curve",
        ),
    )
    _assert_reused_figure_matches_fresh(
        _renderer(), record, root=tmp_path, view_mode="beamspace", modalities=("strain",), min_frames=2
    )


def test_3d_mosaic_with_moving_mesh_lines_reuses_figure(tmp_path: Path) -> None:
    _write_mesh_only_store(tmp_path, linked=True)
    # Make the volume differ between frames so the pixels move as well as the mesh overlay.
    volume = zarr.open_group(tmp_path / "volume.zarr", mode="r+")
    volume["data/3d_brightness_mode"][1] = 200
    _assert_reused_figure_matches_fresh(
        _renderer(), tmp_path / "mesh.zarr", root=None, view_mode="cartesian", min_frames=2
    )
