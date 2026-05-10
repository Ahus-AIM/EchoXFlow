from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from echoxflow.croissant import RecordingRecord
from echoxflow.plotting.renderer import RecordingPlotRenderer
from echoxflow.plotting.style import PlotStyle
from echoxflow.plotting.timeline import select_timeline

zarr = pytest.importorskip("zarr")


def test_strain_plotter_builds_self_referenced_bmode_with_contour_and_curve(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "rv.zarr", mode="w")
    bmode = np.arange(3, dtype=np.uint8).reshape(3, 1, 1) * np.ones((3, 16, 16), dtype=np.uint8)
    group.create_array("data/2d_brightness_mode", data=bmode)
    group.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    delayed_contours = np.concatenate([np.full((1, 3, 2), np.nan, dtype=np.float32), _contours()], axis=0)
    group.create_array("data/rv_contour", data=delayed_contours)
    group.create_array("timestamps/rv_contour", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    group.create_array("data/rv_curve", data=np.asarray([0.0, -6.0, -12.0], dtype=np.float32))
    group.create_array("timestamps/rv_curve", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float32))
    group.attrs["recording_manifest"] = _strain_document(("rv",), self_recording_id="rv")
    record = _record(
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
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=320, dpi=100))

    panels, ecg = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("strain",),
        view_mode="beamspace",
        show_annotations=True,
    )

    assert [panel.loaded.data_path for panel in panels] == ["data/2d_brightness_mode", "data/rv_curve"]
    assert panels[0].kind == "image"
    assert panels[0].loaded.timestamps is not None
    assert np.allclose(panels[0].loaded.timestamps, [0.1, 0.2])
    assert int(panels[0].loaded.data[0, 0, 0]) == 1
    assert np.isclose(select_timeline(panels).timestamps[0], 0.1)
    assert panels[0].loaded.attrs["annotation_overlays"][0]["target"] == {
        "type": "linked_panel",
        "semantic_id": "rv",
        "field": "contour_points",
    }
    assert panels[0].loaded.attrs["annotation_overlays"][0]["timestamps"].shape == (3,)
    assert panels[1].kind == "line"
    assert panels[1].loaded.attrs["strain_curve"] is True
    assert ecg.signal.shape == (1,)

    figure = renderer.render_figure_from_specs(panels=panels, ecg=ecg, time_s=0.1, frame_index=1, dpi=100)
    try:
        assert len(figure.axes[0].lines) >= 2
        assert any(np.asarray(line.get_xdata()).size >= 3 for line in figure.axes[0].lines)
    finally:
        plt.close(figure)


def test_strain_image_input_chains_reference_load_error() -> None:
    from echoxflow.objects import LinkedArray, RecordingRef, StrainPanel
    from echoxflow.plotting.renderer import _strain_image_input

    original = RuntimeError("broken reference")

    class Store:
        def open_reference(self, recording: RecordingRef) -> object:
            raise original

    panel = StrainPanel(
        role_id="2ch",
        view_code=None,
        sequence_id=None,
        bmode=LinkedArray(
            recording=RecordingRef(recording_id="source", zarr_path="source.zarr"),
            data_path="data/2d_brightness_mode",
        ),
    )

    with pytest.raises(ValueError, match="Could not load linked B-mode") as exc_info:
        _strain_image_input(Store(), panel)

    assert exc_info.value.__cause__ is original


def test_strain_plotter_opens_external_lv_panels_by_default(tmp_path: Path) -> None:
    for index, role in enumerate(("2ch", "3ch", "4ch"), start=1):
        source = zarr.open_group(tmp_path / f"{role}.zarr", mode="w")
        source.create_array("data/2d_brightness_mode", data=np.full((1, 8, 8), index, dtype=np.uint8))
        source.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    strain = zarr.open_group(tmp_path / "strain.zarr", mode="w")
    for role in ("2ch", "3ch", "4ch"):
        strain.create_array(f"data/{role}_contour", data=_contours()[:1])
        strain.create_array(f"timestamps/{role}_contour", data=np.asarray([0.0], dtype=np.float32))
        strain.create_array(f"data/{role}_curve", data=np.asarray([0.0, -10.0], dtype=np.float32))
        strain.create_array(f"timestamps/{role}_curve", data=np.asarray([0.0, 0.1], dtype=np.float32))
    strain.attrs["recording_manifest"] = _strain_document(("2ch", "3ch", "4ch"))
    record = _record(
        "strain",
        "strain.zarr",
        content_types=("2d_left_ventricular_strain",),
        array_paths=tuple(
            item
            for role in ("2ch", "3ch", "4ch")
            for item in (
                f"data/{role}_contour",
                f"timestamps/{role}_contour",
                f"data/{role}_curve",
                f"timestamps/{role}_curve",
            )
        ),
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=520, height_px=360, dpi=100))

    panels, _ = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=None,
        view_mode="beamspace",
        show_annotations=False,
    )

    assert [panel.loaded.attrs["strain_role_id"] for panel in panels[:3]] == ["2ch", "3ch", "4ch"]
    assert [int(panel.loaded.data[0, 0, 0]) for panel in panels[:3]] == [1, 2, 3]
    assert [panel.loaded.data_path for panel in panels[3:]] == ["data/strain_curves"]
    assert all("annotation_overlays" not in panel.loaded.attrs for panel in panels[:3])


def test_strain_linked_panels_start_together_at_qrs(tmp_path: Path) -> None:
    roles = ("2ch", "4ch")
    for role, base_time, first_value in (("2ch", 0.0, 10), ("4ch", 1.0, 20)):
        source = zarr.open_group(tmp_path / f"{role}.zarr", mode="w")
        frames = np.arange(first_value, first_value + 4, dtype=np.uint8).reshape(4, 1, 1)
        source.create_array("data/2d_brightness_mode", data=np.broadcast_to(frames, (4, 8, 8)))
        source.create_array(
            "timestamps/2d_brightness_mode",
            data=np.asarray([base_time, base_time + 0.1, base_time + 0.2, base_time + 0.3], dtype=np.float32),
        )
    strain = zarr.open_group(tmp_path / "strain.zarr", mode="w")
    for role, base_time in (("2ch", 0.0), ("4ch", 1.0)):
        strain.create_array(f"data/{role}_contour", data=np.concatenate([_contours(), _contours()], axis=0))
        strain.create_array(
            f"timestamps/{role}_contour",
            data=np.asarray([base_time, base_time + 0.1, base_time + 0.2, base_time + 0.3], dtype=np.float32),
        )
        strain.create_array(f"data/{role}_curve", data=np.asarray([0.0, -10.0], dtype=np.float32))
        strain.create_array(f"timestamps/{role}_curve", data=np.asarray([0.0, 0.1], dtype=np.float32))
        strain.create_array(f"data/{role}_ecg_qrs", data=np.ones(2, dtype=np.float32))
        strain.create_array(
            f"timestamps/{role}_ecg_qrs",
            data=np.asarray([base_time + 0.1, base_time + 0.3], dtype=np.float32),
        )
    strain.attrs["recording_manifest"] = _strain_document(roles, include_qrs=True)
    record = _record(
        "strain",
        "strain.zarr",
        content_types=("2d_left_ventricular_strain",),
        array_paths=tuple(
            item
            for role in roles
            for item in (
                f"data/{role}_contour",
                f"timestamps/{role}_contour",
                f"data/{role}_curve",
                f"timestamps/{role}_curve",
                f"data/{role}_ecg_qrs",
                f"timestamps/{role}_ecg_qrs",
            )
        ),
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=520, height_px=360, dpi=100))

    panels, _ = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("strain",),
        view_mode="beamspace",
        show_annotations=True,
    )

    image_panels = panels[:2]
    assert [panel.loaded.attrs["strain_role_id"] for panel in image_panels] == ["2ch", "4ch"]
    assert [int(panel.loaded.data[0, 0, 0]) for panel in image_panels] == [11, 21]
    assert all(panel.loaded.timestamps is not None for panel in image_panels)
    assert np.allclose(image_panels[0].loaded.timestamps, [0.0, 0.1, 0.2])
    assert np.allclose(image_panels[1].loaded.timestamps, [0.0, 0.1, 0.2])
    assert np.isclose(select_timeline(panels).timestamps[0], 0.0)
    assert np.isclose(select_timeline(panels).timestamps[-1], 0.2)


def test_strain_plotter_can_select_one_role_from_explicit_modalities(tmp_path: Path) -> None:
    for role in ("2ch", "4ch"):
        source = zarr.open_group(tmp_path / f"{role}.zarr", mode="w")
        source.create_array("data/2d_brightness_mode", data=np.zeros((1, 8, 8), dtype=np.uint8))
        source.create_array("timestamps/2d_brightness_mode", data=np.asarray([0.0], dtype=np.float32))
    strain = zarr.open_group(tmp_path / "strain.zarr", mode="w")
    for role in ("2ch", "4ch"):
        strain.create_array(f"data/{role}_contour", data=_contours()[:1])
        strain.create_array(f"timestamps/{role}_contour", data=np.asarray([0.0], dtype=np.float32))
    strain.attrs["recording_manifest"] = _strain_document(("2ch", "4ch"))
    record = _record(
        "strain",
        "strain.zarr",
        content_types=("2d_left_ventricular_strain",),
        array_paths=(
            "data/2ch_contour",
            "timestamps/2ch_contour",
            "data/4ch_contour",
            "timestamps/4ch_contour",
        ),
    )
    renderer = RecordingPlotRenderer(style=PlotStyle(width_px=420, height_px=320, dpi=100))

    panels, _ = renderer._load_specs(
        record,
        root=tmp_path,
        modalities=("4ch_contour",),
        view_mode="beamspace",
        show_annotations=True,
    )

    assert len(panels) == 1
    assert panels[0].loaded.attrs["strain_role_id"] == "4ch"
    assert panels[0].loaded.attrs["annotation_overlays"][0]["target"] == {
        "type": "linked_panel",
        "semantic_id": "4ch",
        "field": "contour_points",
    }


def _contours() -> np.ndarray:
    return np.asarray(
        [
            [[-0.006, 0.03], [0.0, 0.04], [0.006, 0.03]],
            [[-0.005, 0.032], [0.0, 0.042], [0.005, 0.032]],
        ],
        dtype=np.float32,
    )


def _strain_document(
    roles: tuple[str, ...],
    *,
    self_recording_id: str | None = None,
    include_qrs: bool = False,
) -> dict[str, object]:
    annotations: list[dict[str, object]] = []
    for role in roles:
        annotations.extend(
            [
                {
                    "target": {
                        "type": "linked_panel",
                        "semantic_id": role,
                        "field": "contour_points",
                    },
                    "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                    "time": {"zarr_path": f"timestamps/{role}_contour", "format": "zarr_array"},
                    "value": {"zarr_path": f"data/{role}_contour", "format": "zarr_array"},
                },
                {
                    "target": {
                        "type": "linked_panel",
                        "semantic_id": role,
                        "field": "strain_curve",
                    },
                    "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                    "time": {"zarr_path": f"timestamps/{role}_curve", "format": "zarr_array"},
                    "value": {"zarr_path": f"data/{role}_curve", "format": "zarr_array"},
                },
            ]
        )
        if include_qrs:
            annotations.append(
                {
                    "target": {
                        "type": "linked_panel",
                        "semantic_id": role,
                        "field": "ecg_qrs",
                    },
                    "target_data": {"zarr_path": "data/2d_brightness_mode", "format": "zarr_array"},
                    "time": {"zarr_path": f"timestamps/{role}_ecg_qrs", "format": "zarr_array"},
                    "value": {"zarr_path": f"data/{role}_ecg_qrs", "format": "zarr_array"},
                }
            )
    return {
        "manifest_type": "strain",
        "schema_version": 4,
        "annotation_type": "right_ventricular_strain" if roles == ("rv",) else "left_ventricular_strain",
        "linked_panels": [
            {
                "role_id": role,
                "view_code": role.upper(),
                "geometry": {
                    "depth_start_m": 0.0,
                    "depth_end_m": 0.08,
                    "tilt_rad": 0.0,
                    "width_rad": 0.5,
                },
                "frame_timestamps": {"zarr_path": "timestamps/2d_brightness_mode", "format": "zarr_array"},
                "linked_recording": {
                    "recording_id": self_recording_id or f"{role}_source",
                    "recording_zarr_path": "rv.zarr" if self_recording_id is not None else f"{role}.zarr",
                    "relative_zarr_path": "rv.zarr" if self_recording_id is not None else f"{role}.zarr",
                    "frames_array_path": "data/2d_brightness_mode",
                    "timestamps_array_path": "timestamps/2d_brightness_mode",
                },
            }
            for role in roles
        ],
        "annotations": annotations,
    }


def _record(
    recording_id: str,
    zarr_path: str,
    *,
    content_types: tuple[str, ...],
    array_paths: tuple[str, ...],
) -> RecordingRecord:
    return RecordingRecord(
        exam_id="exam",
        recording_id=recording_id,
        zarr_path=zarr_path,
        modes=("2d", "aps"),
        content_types=content_types,
        frame_counts_by_content_type={content_type: 1 for content_type in content_types},
        median_delta_time_by_content_type={},
        array_paths=array_paths,
    )
