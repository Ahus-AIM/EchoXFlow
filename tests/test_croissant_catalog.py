from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import croissant_row, write_croissant

from echoxflow import RecordingRecord, find_recordings, load_croissant, load_exam_folds
from echoxflow.schema import consensus_view

BMODE = ["2d_brightness_mode"]


def _view_row(
    recording: str, prime: str, fine: str, *, prime_prob: float = 0.9, fine_prob: float = 0.8, skip: str = ""
):
    return {
        "exam_id": "e",
        "recording_id": recording,
        "skip_reason": skip,
        "echoprime_top1": prime,
        "echoprime_top1_prob": prime_prob,
        "echofine_top1": fine,
        "echofine_top1_prob": fine_prob,
    }


@pytest.fixture
def export(tmp_path: Path) -> Path:
    rows = [
        croissant_row("r_a4c", exam_id="e1", content_types=BMODE, array_paths=["data/2d_brightness_mode"]),
        croissant_row("r_plax", exam_id="e1", content_types=BMODE, array_paths=["data/2d_brightness_mode"]),
        croissant_row("r_mismatch", exam_id="e2", content_types=BMODE, array_paths=["data/2d_brightness_mode"]),
        croissant_row("r_skipped", exam_id="e2", content_types=BMODE, array_paths=["data/2d_brightness_mode"]),
        croissant_row(
            "r_strain",
            exam_id="e3",
            content_types=["2d_left_ventricular_strain"],
            array_paths=["annotations/4ch_contour", "annotations/3d_left_ventricle_mesh/point_values"],
        ),
    ]
    views = [
        _view_row("r_a4c", "A4C", "a4ch-normal"),
        _view_row("r_plax", "Parasternal_Long", "plax-normal", prime_prob=0.5),
        _view_row("r_mismatch", "A2C", "a4ch-normal"),
        _view_row("r_skipped", "A4C", "a4ch-normal", skip="too_short"),
    ]
    write_croissant(tmp_path, rows, splits={"e1": 0, "e2": 4}, views=views)
    return tmp_path


def test_catalog_joins_fold_and_views(export: Path) -> None:
    catalog = load_croissant(root=export)
    by_id = {record.recording_id: record for record in catalog.recordings}

    a4c = by_id["r_a4c"]
    assert (a4c.fold, a4c.view, a4c.view_echoprime, a4c.view_echofine_prob) == (0, "A4C", "A4C", 0.8)
    assert a4c.view_confidence_for() == 0.8
    assert by_id["r_plax"].view == "PLAX"
    assert by_id["r_plax"].view_for("echoprime") == "PLAX"
    assert by_id["r_mismatch"].view is None and by_id["r_mismatch"].view_echofine == "a4ch-normal"
    skipped = by_id["r_skipped"]
    assert (skipped.view, skipped.view_echoprime, skipped.fold) == (None, None, 4)
    assert by_id["r_strain"].fold is None
    assert by_id["r_strain"].annotations == ("lv_mesh", "lv_strain_contour", "strain_contour")


def test_catalog_without_sidecar_files_leaves_fields_empty(tmp_path: Path) -> None:
    write_croissant(tmp_path, [croissant_row("r", content_types=BMODE, array_paths=[])])
    (record,) = load_croissant(root=tmp_path).recordings
    assert (record.fold, record.view, record.view_echoprime_prob, record.annotations) == (None, None, None, ())


def test_explicit_relative_csv_resolves_against_the_manifest(export: Path) -> None:
    (export / "splits.csv").rename(export / "my_folds.csv")
    catalog = load_croissant(export / "croissant.json", splits_csv="my_folds.csv")
    assert {record.fold for record in catalog.recordings} == {0, 4, None}
    with pytest.raises(FileNotFoundError):
        load_croissant(export / "croissant.json", view_csv="nope.csv")


def test_filter_by_view_fold_confidence_and_annotation(export: Path) -> None:
    catalog = load_croissant(root=export)

    def ids(**kwargs: object) -> list[str]:
        return [record.recording_id for record in catalog.filter(**kwargs)]  # type: ignore[arg-type]

    assert ids(views=("a4c", "plax")) == ["r_a4c", "r_plax"]
    assert ids(views=("A4C",), view_source="echofine") == []
    assert ids(views=("a4ch-normal",), view_source="echofine") == ["r_a4c", "r_mismatch"]
    assert ids(views=("PLAX", "A4C"), min_view_confidence=0.6) == ["r_a4c"]
    assert ids(min_view_confidence=0.85, view_source="echoprime") == ["r_a4c", "r_mismatch"]
    assert ids(folds=(0,)) == ["r_a4c", "r_plax"]
    assert ids(exclude_folds=(4,)) == ["r_a4c", "r_plax"]  # recordings of unknown fold never pass
    assert ids(annotated=("lv_mesh", "rv_mesh")) == ["r_strain"]
    assert ids(annotated=("lv_mesh", "rv_mesh"), require_all=True) == []
    assert [r.recording_id for r in find_recordings(croissant=catalog, views=("A4C",), folds=(0,))] == ["r_a4c"]
    with pytest.raises(ValueError):
        ids(views=("A4C",), view_source="nope")


def test_consensus_view_tables() -> None:
    assert consensus_view("mmode-plax-x", "PLAX") == "PLAX"
    assert consensus_view("something-else", "Apical_Doppler") == "Other"
    assert consensus_view("a4ch-x", "A2C") is None
    assert consensus_view("", "A4C") is None


def test_load_exam_folds_rejects_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    path.write_text("exam_id,five_fold_split\nx,1\nx,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        load_exam_folds(path)


def test_record_from_row_matches_exported_row() -> None:
    row = croissant_row("r", content_types=BMODE, array_paths=["data/2d_brightness_mode"], frame_counts={BMODE[0]: 3})
    record = RecordingRecord.from_row(row)
    assert record.frame_count("2d_brightness_mode") == 3 and record.annotations == ()


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"  # the autouse fixture in conftest points ECHOXFLOW_CACHE_DIR here


def _arrays_manifest(root: Path) -> None:
    write_croissant(root, [croissant_row("r", content_types=BMODE, array_paths=["data/2d_brightness_mode"])])
    manifest = json.loads((root / "croissant.json").read_text(encoding="utf-8"))
    manifest["recordSet"].append(
        {
            "@id": "arrays",
            "data": [
                {"arrays/recording_id": "r", "arrays/array_path": "data/2d_brightness_mode", "arrays/shape": [3, 4, 5]}
            ],
        }
    )
    (root / "croissant.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_cache_round_trip_is_lazy_and_equal(export: Path, cache_root: Path) -> None:
    _arrays_manifest(export)
    first = load_croissant(root=export)
    assert len(list(cache_root.glob("catalog-*.pkl"))) == 1
    cached = load_croissant(root=export)

    assert cached.recordings == first.recordings
    assert cached._arrays is None  # nothing but the recordings was read
    assert [array.shape for array in cached.arrays] == [(3, 4, 5)]
    assert cached.arrays_for_recording("r") == first.arrays_for_recording("r")
    assert cached.stream_summaries == ()


def test_cache_is_invalidated_by_manifest_and_csv_changes(export: Path, cache_root: Path) -> None:
    load_croissant(root=export)
    (export / "splits.csv").write_text("exam_id,five_fold_split\ne1,3\ne2,3\n", encoding="utf-8")
    assert {record.fold for record in load_croissant(root=export).recordings} == {3, None}
    assert len(list(cache_root.glob("catalog-*.pkl"))) == 2


def test_cache_false_and_unreadable_cache_fall_back_to_parsing(
    export: Path, cache_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert load_croissant(root=export, cache=False)._arrays == ()
    assert not cache_root.exists()
    load_croissant(root=export)
    (cache_file,) = cache_root.glob("catalog-*.pkl")
    cache_file.write_bytes(b"not a pickle")

    catalog = load_croissant(root=export)

    assert len(catalog.recordings) == 5 and catalog._arrays == ()
    assert "unreadable catalog cache" in caplog.text


def test_unwritable_cache_only_warns(export: Path, cache_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache_root.parent.mkdir(exist_ok=True)
    cache_root.write_text("a file where the directory should be", encoding="utf-8")
    assert len(load_croissant(root=export).recordings) == 5
    assert "could not write catalog cache" in caplog.text


def test_to_dataframe_and_select(export: Path) -> None:
    pd = pytest.importorskip("pandas")
    catalog = load_croissant(root=export)

    frame = catalog.to_dataframe()

    assert isinstance(frame, pd.DataFrame) and len(frame) == 5
    assert frame.view.value_counts().to_dict() == {"A4C": 1, "PLAX": 1}
    assert (
        frame.loc[frame.recording_id == "r_strain", "annotations"].item() == "lv_mesh|lv_strain_contour|strain_contour"
    )
    assert frame.has_lv_mesh.sum() == 1 and frame.has_2d_brightness_mode.sum() == 4
    assert not frame.is_3d.any()
    chosen = frame[(frame.view == "A4C") | frame.has_lv_mesh]
    assert [record.recording_id for record in catalog.select(chosen)] == ["r_a4c", "r_strain"]


def test_sidecar_split_csv_without_folds_is_ignored_unless_requested(tmp_path: Path) -> None:
    from echoxflow.croissant import load_exam_folds

    path = tmp_path / "splits.csv"
    path.write_text("exam_id,camus_split\nexam_a,train\n", encoding="utf-8")
    assert load_exam_folds(path, required=False) == {}
    with pytest.raises(ValueError, match="five_fold_split"):
        load_exam_folds(path)
