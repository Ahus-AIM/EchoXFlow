"""Apical view selection for the video diffusion task.

Two sources are supported:

``csv`` (default)
    The per-recording view-classifier table written by ``scripts/build_view_classifier_csv.py`` and published next to
    the dataset as ``view_classifier_probabilities.csv``, joined into the catalog as ``record.view_echoprime``. It
    holds the scanner's native view-detector scores and the full EchoPrime and EchoFine probability vectors for every
    B-mode recording. Selection uses EchoPrime (``echoprime_top1`` / ``echoprime_top1_prob``, probabilities averaged
    over eight evenly spaced frames), which matches the EchoPrime selection of the original paper runs for 96.6% of
    recordings. A relative ``data.view_labels_csv`` is resolved against the catalog's directory.

``scanner_view_detector``
    Fallback that needs no extra file: the scanner's own view-detector logits in
    ``recording_manifest.metadata.scanner_view_detector.scores`` (10 values). The class order is not documented;
    comparing each class with EchoPrime and EchoFine on 28,551 recordings gives 0 = A2C, 1 = A4C and 3 = A3C with
    67-87% agreement. Class 2 mixes A5C with A4C, and class 4 is an A4C variant enriched for right-heart work
    (tricuspid CW, TAPSE M-mode), so neither is used. A recording is selected when the softmax probability of its
    arg-max class is at least ``minimum_view_confidence``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from echoxflow import CroissantCatalog

APICAL_VIEWS = ("A2C", "A3C", "A4C")
SCANNER_VIEW_CLASSES: dict[int, str] = {0: "A2C", 1: "A4C", 3: "A3C"}
DEFAULT_VIEW_LABELS_CSV = "view_classifier_probabilities.csv"


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


@dataclass(frozen=True)
class ViewSelector:
    minimum_confidence: float = 0.85
    views: tuple[str, ...] = APICAL_VIEWS
    csv_labels: Mapping[tuple[str, str], str] | None = field(default=None)

    @property
    def uses_csv(self) -> bool:
        return self.csv_labels is not None

    def view_from_csv(self, exam_id: str, recording_id: str) -> str | None:
        if self.csv_labels is None:
            return None
        return self.csv_labels.get((exam_id, recording_id))

    def view_from_manifest(self, manifest: Mapping[str, Any]) -> str | None:
        metadata = manifest.get("metadata")
        detector = metadata.get("scanner_view_detector") if isinstance(metadata, Mapping) else None
        scores = detector.get("scores") if isinstance(detector, Mapping) else None
        if not isinstance(scores, list) or len(scores) != 10:
            return None
        probabilities = softmax([float(value) for value in scores])
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        view = SCANNER_VIEW_CLASSES.get(index)
        if view is None or view not in self.views or probabilities[index] < self.minimum_confidence:
            return None
        return view


def build_view_selector(settings: Mapping[str, Any], *, catalog: CroissantCatalog) -> ViewSelector:
    """Selector for ``data.view_source``; ``catalog`` must be loaded with the settings' ``view_labels_csv``."""
    views = tuple(str(view).upper() for view in settings.get("views", APICAL_VIEWS))
    confidence = float(settings.get("minimum_view_confidence", 0.85))
    source = str(settings.get("view_source", "csv"))
    if source == "scanner_view_detector":
        return ViewSelector(minimum_confidence=confidence, views=views)
    if source == "csv":
        if not any(record.view_echoprime for record in catalog.recordings):
            raise FileNotFoundError(
                "view classifier table not found next to the catalog; download view_classifier_probabilities.csv "
                "next to the dataset, regenerate it with scripts/build_view_classifier_csv.py, or set "
                "data.view_source=scanner_view_detector"
            )
        selected = catalog.filter(views=views, view_source="echoprime", min_view_confidence=confidence)
        labels = {(record.exam_id, record.recording_id): str(record.view_echoprime).upper() for record in selected}
        return ViewSelector(minimum_confidence=confidence, views=views, csv_labels=labels)
    raise ValueError(f"unsupported data.view_source {source!r}; use 'scanner_view_detector' or 'csv'")
