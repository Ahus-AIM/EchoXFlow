"""Synthetic native-clock and rejection contracts for anatomical strain plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zarr
from conftest import strain_document
from test_aha_strain import _materialized_store

from echoxflow import plot_recording
from echoxflow.plotting import PlotStyle
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.timeline import select_timeline


def test_anatomical_summary_keeps_all_native_curves_and_rejections(tmp_path: Path) -> None:
    path = tmp_path / "named.zarr"
    _, expected = _materialized_store(path)
    fig = plot_recording(
        path, modalities=("aha_strain",), style=PlotStyle(width_px=1600, height_px=600, show_figure_title=False)
    )
    try:
        assert len(fig.axes) == 4  # no invented ECG or image dependency
        fig.canvas.draw()
        assert all(ax.title.get_window_extent().y1 <= fig.bbox.y1 for ax in fig.axes)
        by_gid = {line.get_gid(): line for ax in fig.axes for line in ax.lines if line.get_gid()}
        assert len(by_gid) == 19
        for segment in expected.segments:
            for curve in segment.contributors:
                line = by_gid[f"aha-{segment.aha_segment_id}-{curve.source_view}"]
                np.testing.assert_array_equal(line.get_xdata(), curve.timestamps - curve.native_ed_time)
                np.testing.assert_array_equal(line.get_ydata(), curve.values)
                assert (line.get_linestyle() == "--") == (segment.accepted is False and segment.aha_segment_id != 17)
        labels = {text.get_gid(): text.get_text() for text in fig.axes[-1].texts if text.get_gid()}
        assert len(labels) == 17
        for segment in expected.segments:
            label = labels[f"aha-value-{segment.aha_segment_id}"]
            assert (label == "X") == (segment.accepted is False)
            if segment.accepted is not False:
                assert label == f"{segment.signed_peak_percent:.0f}"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(("modality", "axes"), [("bullseye", 1), ("aha_curves", 3)])
def test_summary_selection_and_static_playback(tmp_path, modality, axes):
    path = tmp_path / "named.zarr"
    _materialized_store(path)
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=800, height_px=450))
    panels, ecg = renderer._load_specs(
        path, root=None, modalities=(modality,), view_mode="beamspace", show_annotations=True
    )
    assert select_timeline(panels).timestamps.tolist() == [0.0]
    before = plt.get_fignums()
    frames = list(renderer.iter_frames(panels, ecg, np.asarray([0.0, 0.2])))
    np.testing.assert_array_equal(frames[0], frames[1])
    assert plt.get_fignums() == before
    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0, frame_index=0)
    try:
        assert len(figure.axes) == axes
    finally:
        plt.close(figure)


def test_default_anatomical_plot_keeps_cine_clock_and_adds_summary(tmp_path):
    path = tmp_path / "named.zarr"
    group, _ = _materialized_store(path)
    roles = ("A2C", "A3C", "A4C")
    for role in roles:
        source = zarr.open_group(tmp_path / f"{role}.zarr", mode="w")
        source.create_array("data/2d_brightness_mode", data=np.ones((2, 8, 8), dtype=np.uint8))
        source.create_array("timestamps/2d_brightness_mode", data=np.array([0.0, 0.1]))
    document = group.attrs["recording_manifest"]
    document["linked_panels"] = strain_document(roles)["linked_panels"]
    group.attrs["recording_manifest"] = document
    renderer = RecordingPlotRenderer()
    panels, _ = renderer._load_specs(
        path, root=tmp_path, modalities=None, view_mode="beamspace", show_annotations=False
    )
    assert [p.kind for p in panels] == ["image"] * 3 + ["strain_curves"] * 3 + ["bullseye"]
    np.testing.assert_array_equal(select_timeline(panels).timestamps, np.array([0, 0.1], dtype=np.float32))
    # A declared anatomical export must not silently fall back after corruption.
    document["anatomical_strain"]["segments"][0]["segment_name"] = "wrong anatomy"
    group.attrs["recording_manifest"] = document
    with pytest.raises(ValueError):
        renderer._load_specs(path, root=tmp_path, modalities=None, view_mode="beamspace", show_annotations=False)
