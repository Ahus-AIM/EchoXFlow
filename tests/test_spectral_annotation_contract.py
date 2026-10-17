"""Synthetic annotation ownership and native-row overlay regressions."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from echoxflow.loading import LoadedArray
from echoxflow.objects import recording_object_from_metadata
from echoxflow.plotting.annotations import _annotation_paths_for_loaded, _attach_annotation_overlay
from echoxflow.plotting.panels import _draw_matrix_annotations, _spectral_annotation_y_to_display_y
from echoxflow.plotting.specs import PanelSpec
from echoxflow.plotting.style import PlotStyle
from echoxflow.spectral import SpectralCalibration, SpectralMetadata

PW = "data/1d_pulsed_wave_doppler"
CW = "data/1d_continuous_wave_doppler"


def loaded(path=PW, *, clock=None, metadata=None):
    return LoadedArray(
        path,
        path,
        np.zeros((4, 100)),
        None,
        np.asarray([0.0, 1.0, 2.0, 3.0] if clock is None else clock),
        None,
        {"spectral_metadata": metadata} if metadata else {},
    )


def store(annotations, streams=(PW,)):
    class Group:
        attrs = {"recording_manifest": {"annotations": annotations}}

        def __getitem__(self, path):
            return SimpleNamespace(attrs={"units": "s_mps"})

    return SimpleNamespace(
        group=Group(),
        array_paths=(*streams, *(a["value"]["path"] for a in annotations)),
        load_array=lambda path: np.array([[1.0, 0.25]], dtype=np.float32),
    )


def test_document_identity_and_metadata_only_annotations():
    entry = {"annotation_identity": "measurement:000", "value": {"path": "annotations/root"}}
    nested = {**entry, "value": {"path": "annotations/child"}}
    obj = recording_object_from_metadata(
        None,
        {
            "recording_manifest": {
                "annotations": [entry, {"label": "metadata only"}],
                "merged_documents": [{"annotations": [nested], "merged_documents": [{"annotations": [nested]}]}],
            }
        },
    )
    assert len(obj.annotations) == 4
    assert [a.document_path for a in obj.annotations] == [(), (), (0,), (0, 0)]
    assert obj.annotations[1].value is None
    assert obj.annotations[1].raw["label"] == "metadata only"
    assert obj.annotations[0].annotation_identity == "measurement:000"


def test_explicit_target_beats_filename_and_legacy_requires_unique_owner():
    path = "annotations/1d_continuous_wave_doppler_annotation_00"
    entry = {
        "kind": "measurement",
        "value": {"path": path},
        "target_data": {"path": PW},
        "links": {"coordinate_type": "spectral"},
    }
    current = store([entry], (PW, CW))
    assert _annotation_paths_for_loaded(current, loaded(PW)) == (path,)
    assert _annotation_paths_for_loaded(current, loaded(CW)) == ()
    legacy = {key: value for key, value in entry.items() if key != "target_data"}
    assert _annotation_paths_for_loaded(store([legacy], (PW, CW)), loaded(PW)) == ()
    assert _annotation_paths_for_loaded(store([legacy]), loaded(PW)) == (path,)


def test_opaque_path_and_shared_array_preserve_semantic_records():
    entry = {
        "kind": "spectral_trace",
        "value": {"path": "annotations/opaque"},
        "target_data": {"path": PW},
        "links": {"geometry_kind": "points"},
    }
    attached = _attach_annotation_overlay(
        store([entry, {**entry, "label": "independent"}]), loaded(), show_sampling_gate=False, show_sampling_line=False
    )
    overlays = attached.attrs["annotation_overlays"]
    assert len(overlays) == 2
    assert all(o["kind"] == "spectral_trace" for o in overlays)


def test_native_calibration_steps_missing_outside_and_reset_ownership():
    calibration = SpectralCalibration(np.array([0.5, 0.25, 0.5, 0.5]), np.array([1.0, 2.0, np.nan, 1.0]))
    metadata = SpectralMetadata(PW, calibration=calibration)
    panel = PanelSpec(loaded(), kind="matrix", label="PW")
    panel = PanelSpec(loaded(metadata=metadata), kind="matrix", label="PW")
    overlay = {"y_unit": "m/s"}
    actual = _spectral_annotation_y_to_display_y(
        panel, overlay, np.array([0.5, 0.5, 0.5, 3.0]), y_size=100, times=np.array([0.0, 1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(actual[:2], [75.0, 87.5])
    assert np.isnan(actual[2:]).all()
    reset = PanelSpec(loaded(clock=[0.0, 1.0, 0.0, 1.0], metadata=metadata), kind="matrix", label="PW")
    assert np.isnan(
        _spectral_annotation_y_to_display_y(reset, overlay, np.array([0.5]), y_size=100, times=np.array([0.5]))
    ).all()


def test_geometry_points_polyline_and_independent_segments():
    metadata = SpectralMetadata(PW, row_velocity_mps=np.array([1.0, 0.0, -1.0]))
    for geometry, count, linestyle in [("points", 1, "None"), ("polyline", 4, "-"), ("segments", 4, "-")]:
        points = np.column_stack([np.arange(count), np.zeros(count)])
        data = loaded(metadata=metadata)
        data = LoadedArray(
            data.name,
            data.data_path,
            data.data,
            data.timestamps_path,
            data.timestamps,
            data.sample_rate_hz,
            {
                **data.attrs,
                "annotation_overlays": (
                    {"kind": "spectral_trace", "geometry_kind": geometry, "points": points, "y_unit": "m/s"},
                ),
            },
        )
        fig, ax = plt.subplots()
        try:
            _draw_matrix_annotations(
                ax, PanelSpec(data, kind="matrix", label="PW"), time_s=0.0, y_size=100, style=PlotStyle()
            )
            assert ax.lines[0].get_linestyle() == linestyle
            if geometry == "points":
                assert ax.lines[0].get_marker() == "+"
            if geometry == "segments":
                assert len(ax.lines) == 4
                assert all(len(line.get_xdata()) == 2 for line in ax.lines)
        finally:
            plt.close(fig)


def test_explicit_alias_draws_once_and_independent_identity_stays_visible():
    canonical = {
        "annotation_identity": "measurement:000",
        "kind": "measurement",
        "value": {"path": "annotations/canonical"},
        "target_data": {"path": PW},
        "links": {"coordinate_type": "spectral", "geometry_kind": "points"},
    }
    alias = {
        **canonical,
        "annotation_identity": "trace:000",
        "alias_of": "measurement:000",
        "value": {"path": "annotations/alias"},
    }
    independent = {**canonical, "annotation_identity": "measurement:001"}
    attached = _attach_annotation_overlay(
        store([canonical, alias, independent]), loaded(), show_sampling_gate=False, show_sampling_line=False
    )
    assert [o["annotation_identity"] for o in attached.attrs["annotation_overlays"]] == [
        "measurement:000",
        "measurement:001",
    ]


def test_polyline_breaks_keep_points_across_calibration_step_and_gap():
    from echoxflow.plotting.panels import _spectral_trace_with_breaks

    metadata = SpectralMetadata(PW, calibration=SpectralCalibration(np.array([0.5, 0.5, 0.25, 0.25]), np.ones(4)))
    panel = PanelSpec(loaded(metadata=metadata), kind="matrix", label="PW")
    x, y = _spectral_trace_with_breaks(panel, np.array([1.0, 2.0]), np.array([50.0, 75.0]))
    np.testing.assert_allclose(x, [1.0, np.nan, 2.0], equal_nan=True)
    np.testing.assert_allclose(y, [50.0, np.nan, 75.0], equal_nan=True)
