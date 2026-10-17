from __future__ import annotations

from pathlib import Path

import pytest
from conftest import croissant_row, write_croissant

from echoxflow import load_croissant
from tasks.video_diffusion.views import build_view_selector

BMODE = ["2d_brightness_mode"]


def _view_row(recording: str, label: str, prob: float, skip: str = "") -> dict[str, object]:
    return {
        "exam_id": "e",
        "recording_id": recording,
        "skip_reason": skip,
        "echoprime_top1": label,
        "echoprime_top1_prob": prob,
        "echofine_top1": "",
        "echofine_top1_prob": "",
    }


def test_csv_selector_keeps_confident_apical_echoprime_labels(tmp_path: Path) -> None:
    rows = [croissant_row(name, content_types=BMODE, array_paths=[]) for name in ("a", "b", "c", "d", "e")]
    views = [
        _view_row("a", "A4C", 0.9),
        _view_row("b", "A2C", 0.85),
        _view_row("c", "A4C", 0.8),
        _view_row("d", "Parasternal_Long", 0.99),
        _view_row("e", "A3C", 0.99, skip="too_short"),
    ]
    write_croissant(tmp_path, rows, views=views)
    settings = {"views": ["A2C", "A3C", "A4C"], "view_source": "csv", "minimum_view_confidence": 0.85}

    selector = build_view_selector(settings, catalog=load_croissant(root=tmp_path))

    assert selector.csv_labels == {("exam", "a"): "A4C", ("exam", "b"): "A2C"}


def test_csv_selector_requires_the_view_table(tmp_path: Path) -> None:
    write_croissant(tmp_path, [croissant_row("a", content_types=BMODE, array_paths=[])])
    with pytest.raises(FileNotFoundError, match="view_classifier_probabilities.csv"):
        build_view_selector({"view_source": "csv"}, catalog=load_croissant(root=tmp_path))
    assert (
        build_view_selector({"view_source": "scanner_view_detector"}, catalog=load_croissant(root=tmp_path)).csv_labels
        is None
    )
