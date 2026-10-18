"""Task 4 comparison tables: per-fold replicates are saved, and the cross-validated table pools them correctly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tasks.utils import stats
from tasks.video_diffusion import cv_table, tables


def _arms(seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    exams = np.repeat([f"exam{index}" for index in range(8)], 3)
    views = np.tile(["A2C", "A4C", "A4C"], 8)
    real = rng.normal(size=(len(exams), 4))
    shared = {
        "real": real,
        "real_exam_ids": exams,
        "generated_exam_ids": exams,
        "generated_views": views,
        "generated_ids": np.arange(len(exams)),
        "kind": np.array("frame"),
    }
    return {
        "beamspace": {**shared, "generated": real + rng.normal(scale=0.3, size=real.shape)},
        "cartesian": {**shared, "generated": real + rng.normal(scale=0.8, size=real.shape) + 0.5},
    }


def test_tables_save_the_replicate_differences_next_to_the_comparison(tmp_path: Path) -> None:
    result = tables.compare_echoxflow(_arms(), kind="frame", replicates=20, seed=17, kid={}, device=torch.device("cpu"))
    deltas = result.pop("replicate_deltas")
    row = {"resolution": 128, "extractor": "inception", "label": "Inception (frame FD)", **result, "p_holm": 0.5}
    tables.write_tables(
        [row],
        tmp_path,
        title="t",
        replicates=20,
        seed=17,
        replicate_deltas={tables.replicate_key(128, "inception"): deltas},
    )

    with np.load(tmp_path / "comparison_replicates.npz") as saved:
        np.testing.assert_array_equal(saved["r0128_inception"], deltas)
    payload = json.loads((tmp_path / "comparison.json").read_text())
    assert "replicate_deltas" not in payload["rows"][0]
    assert payload["rows"][0]["p_two_sided"] == stats.recentered_p_value(deltas, row["delta_cartesian_minus_beamspace"])


def _write_fold(directory: Path, rows: list[dict], deltas: dict[str, np.ndarray], *, replicates: int = 50) -> None:
    directory.mkdir(parents=True)
    (directory / "comparison.json").write_text(json.dumps({"replicates": replicates, "seed": 17, "rows": rows}))
    np.savez(directory / "comparison_replicates.npz", **deltas)


def _row(resolution: int, extractor: str, beam: float, cart: float, p_holm: float) -> dict:
    return {
        "resolution": resolution,
        "extractor": extractor,
        "label": extractor.title(),
        "beamspace": beam,
        "cartesian": cart,
        "p_holm": p_holm,
    }


def _three_folds(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    values = {  # fold: ((beam, cart, per-fold Holm p) for inception, then for echoclip)
        0: ((10.0, 20.0, 0.01), (1.0, 1.1, 0.6)),
        1: ((12.0, 21.0, 0.01), (1.2, 1.0, 0.7)),
        2: ((11.0, 25.0, 0.02), (1.1, 1.2, 0.9)),
    }
    for fold, ((b1, c1, p1), (b2, c2, p2)) in values.items():
        _write_fold(
            tmp_path / f"fold{fold}" / "tables",
            [_row(128, "inception", b1, c1, p1), _row(128, "echoclip", b2, c2, p2)],
            {
                "r0128_inception": (c1 - b1) + rng.normal(scale=1.0, size=50),
                "r0128_echoclip": (c2 - b2) + rng.normal(scale=0.3, size=50),
            },
        )
    return tmp_path


def test_cv_table_pools_folds(tmp_path: Path) -> None:
    root = _three_folds(tmp_path)
    folds = {fold: cv_table.load_fold(root / f"fold{fold}" / "tables") for fold in range(3)}
    rows, replicates, seed = cv_table.pool(folds)

    inception, echoclip = rows
    assert (replicates, seed) == (50, 17)
    assert inception["beamspace_mean"] == pytest.approx(11.0)
    assert inception["beamspace_sd_across_folds"] == pytest.approx(1.0)
    assert inception["cartesian_mean"] == pytest.approx(22.0)
    assert inception["delta_cartesian_minus_beamspace"] == pytest.approx(11.0)
    assert inception["relative_difference_percent"] == pytest.approx(50.0)
    assert inception["folds_beamspace_better"] == 3 and inception["folds_significant_beamspace"] == 3
    assert inception["direction"] == "beamspace" and inception["p_holm"] < 0.05
    assert echoclip["folds_beamspace_better"] == 2 and echoclip["folds_cartesian_better"] == 1
    assert echoclip["folds_significant_beamspace"] == echoclip["folds_significant_cartesian"] == 0
    assert echoclip["direction"] == "n.s."
    assert [row["p_holm"] for row in rows] == stats.holm_adjust([row["p_two_sided"] for row in rows])

    again, _, _ = cv_table.pool(folds)
    assert again == rows  # deterministic


def test_cv_table_pooled_replicates_average_one_permuted_replicate_per_fold(tmp_path: Path) -> None:
    root = _three_folds(tmp_path)
    folds = {fold: cv_table.load_fold(root / f"fold{fold}" / "tables") for fold in range(3)}
    rows, _, _ = cv_table.pool(folds)

    pooled = np.mean(
        [
            folds[fold][1]["r0128_inception"][stats.seeded_permutation(50, seed=17, namespace=f"cv-table:fold{fold}")]
            for fold in range(3)
        ],
        axis=0,
    )
    assert rows[0]["p_two_sided"] == stats.recentered_p_value(pooled, 11.0)
    assert rows[0]["ci95_low"] == float(np.quantile(pooled, 0.025))


def test_camus_cv_preserves_shared_patient_uncertainty(tmp_path: Path) -> None:
    arms = _arms()
    for arm in arms.values():
        arm["real_views"] = arm["generated_views"]
    result = tables.compare_camus(arms, kind="frame", replicates=20, seed=31, kid={}, device=torch.device("cpu"))
    assert result["bootstrap_patient_ids"] == [f"exam{index}" for index in range(8)]
    deltas = result.pop("replicate_deltas")
    row = {"resolution": 128, "extractor": "inception", "label": "Inception", **result, "p_holm": 0.5}
    for fold in range(3):
        tables.write_tables(
            [row],
            tmp_path / f"fold{fold}" / "tables_camus",
            title="CAMUS",
            replicates=20,
            seed=31,
            replicate_deltas={"r0128_inception": deltas},
        )
    folds = {fold: cv_table.load_fold(tmp_path / f"fold{fold}" / "tables_camus") for fold in range(3)}
    pooled, _, _ = cv_table.pool(folds)

    # Repeating the same patient draws across folds must not narrow their uncertainty.
    assert [pooled[0]["ci95_low"], pooled[0]["ci95_high"]] == pytest.approx(result["ci95"])
    assert pooled[0]["p_two_sided"] == result["p_two_sided"]
    assert pooled[0]["beamspace_mean"] == pytest.approx(result["beamspace"])
    assert pooled[0]["cartesian_mean"] == pytest.approx(result["cartesian"])


@pytest.mark.parametrize("tables_name", ["tables_camus", "external"])
def test_camus_cv_reuses_legacy_replicates_and_feature_metadata(tmp_path: Path, tables_name: str) -> None:
    root = _three_folds(tmp_path)
    draws = []
    for fold in range(3):
        directory = root / f"fold{fold}"
        (directory / "tables").rename(directory / tables_name)
        with np.load(directory / tables_name / "comparison_replicates.npz") as data:
            draws.append(data["r0128_inception"])
        for arm in ("beamspace", "cartesian"):
            features = directory / f"{arm}_0128" / "evaluation" / "camus" / "features"
            features.mkdir(parents=True)
            for extractor in ("inception", "echoclip"):
                np.savez(
                    features / f"{extractor}.npz",
                    generated_exam_ids=["patient2", "patient1", "patient1"],
                    real_exam_ids=["patient3", "patient1", "patient2"],
                )
    output = root / "out"
    args = ["--root", str(root), "--folds", "0", "1", "2", "--tables-name", tables_name, "--output-dir", str(output)]
    assert cv_table.main(args + (["--camus"] if tables_name == "external" else [])) == 0

    payload = json.loads((output / "comparison_cv.json").read_text())
    row = payload["rows"][0]
    pooled = np.mean(draws, axis=0)
    assert row["ci95_low"] == float(np.quantile(pooled, 0.025))
    assert row["ci95_high"] == float(np.quantile(pooled, 0.975))
    assert row["p_two_sided"] == stats.recentered_p_value(pooled, 11.0)
    assert row["beamspace_mean"] == 11.0 and row["cartesian_mean"] == 22.0
    assert "patient-cluster bootstrap shared across folds" in payload["protocol"]
    assert (output / "comparison_cv.md").read_text().startswith("# CAMUS generation quality")


@pytest.mark.parametrize("cohort", [None, [], ["patient1"], ["patient2", "patient1"]])
def test_camus_cv_rejects_missing_or_different_patient_order(tmp_path: Path, cohort: list[str] | None) -> None:
    root = _three_folds(tmp_path)
    folds = {fold: cv_table.load_fold(root / f"fold{fold}" / "tables") for fold in range(3)}
    for payload, _ in folds.values():
        for row in payload["rows"]:
            row["bootstrap_patient_ids"] = ["patient1", "patient2"]
    folds[2][0]["rows"][0]["bootstrap_patient_ids"] = cohort
    with pytest.raises(SystemExit, match="missing or different bootstrap patient cohorts"):
        cv_table.pool(folds)


@pytest.mark.parametrize("missing", [True, False])
def test_legacy_camus_cv_requires_matching_feature_cohorts(tmp_path: Path, missing: bool) -> None:
    directory = tmp_path / "fold0" / "tables_camus"
    _write_fold(directory, [_row(128, "inception", 1.0, 2.0, 0.01)], {"r0128_inception": np.zeros(50)})
    for arm in ("beamspace", "cartesian"):
        if arm == "cartesian" and missing:
            continue
        features = directory.parent / f"{arm}_0128" / "evaluation" / "camus" / "features"
        features.mkdir(parents=True)
        np.savez(features / "inception.npz", generated_exam_ids=[arm], real_exam_ids=["patient1"])
    with pytest.raises(SystemExit, match="cannot verify CAMUS patients" if missing else "different bootstrap patient"):
        cv_table.load_fold(directory)


def test_cv_table_refuses_incomplete_or_inconsistent_folds(tmp_path: Path) -> None:
    root = _three_folds(tmp_path)
    _write_fold(root / "fold3" / "tables", [_row(128, "inception", 1.0, 2.0, 0.01)], {"r0128_inception": np.zeros(50)})
    with pytest.raises(SystemExit, match="lacks comparisons"):
        cv_table.main(["--root", str(root), "--folds", "0", "1", "2", "3", "--output-dir", str(tmp_path / "out")])

    _write_fold(
        root / "fold4" / "tables",
        [_row(128, "inception", 1.0, 2.0, 0.01), _row(128, "echoclip", 1.0, 2.0, 0.01)],
        {"r0128_inception": np.zeros(20), "r0128_echoclip": np.zeros(20)},
        replicates=20,
    )
    with pytest.raises(SystemExit, match="different bootstrap settings"):
        cv_table.main(["--root", str(root), "--folds", "0", "4", "--output-dir", str(tmp_path / "out")])

    with pytest.raises(SystemExit, match="missing"):
        cv_table.main(["--root", str(root), "--folds", "0", "9", "--output-dir", str(tmp_path / "out")])


def test_cv_table_writes_every_format(tmp_path: Path) -> None:
    root = _three_folds(tmp_path)
    assert cv_table.main(["--root", str(root), "--folds", "0", "1", "2", "--output-dir", str(tmp_path / "out")]) == 0

    out = tmp_path / "out"
    assert {path.name for path in out.iterdir()} == {f"comparison_cv.{ext}" for ext in ("csv", "json", "md", "tex")}
    tex = (out / "comparison_cv.tex").read_text()
    assert "128 & Inception & \\textbf{11.0} $\\pm$ 1.0 & 22.0 $\\pm$ 2.6 & +50.0\\%" in tex
    assert "3/3 (3) \\\\" in tex


def test_bootstrap_multiplicities_are_unchanged_by_the_shared_seeding() -> None:
    plan = stats.bootstrap_multiplicities(7, replicates=5, seed=17, namespace="exam")
    words = np.frombuffer(hashlib.sha256(b"17:exam").digest()[:16], dtype=np.uint32)
    rng = np.random.default_rng(np.random.SeedSequence((17, *words.tolist())))
    expected = np.stack([np.bincount(row, minlength=7) for row in rng.integers(0, 7, size=(5, 7))])
    np.testing.assert_array_equal(plan, expected)


def test_seeded_permutation_is_a_fixed_permutation() -> None:
    first = stats.seeded_permutation(100, seed=17, namespace="cv-table:fold0")
    assert sorted(first.tolist()) == list(range(100))
    np.testing.assert_array_equal(first, stats.seeded_permutation(100, seed=17, namespace="cv-table:fold0"))
    assert not np.array_equal(first, stats.seeded_permutation(100, seed=17, namespace="cv-table:fold1"))


@pytest.mark.parametrize(("folds", "expected"), [(["0", "1", "2"], True), (["3"], False)])
def test_launcher_pools_the_folds_after_their_tables(tmp_path: Path, folds: list[str], expected: bool) -> None:
    from scripts.reproduce_video_diffusion import Reproduction, _parse_args

    (tmp_path / "croissant.json").write_text("{}")
    (tmp_path / "camus").mkdir(exist_ok=True)
    (tmp_path / "camus" / "croissant.json").write_text("{}")
    args = _parse_args(
        ["--data-root", str(tmp_path), "--camus-root", str(tmp_path / "camus")]
        + ["--output-dir", str(tmp_path / "out"), "--folds", *folds, "--resolutions", "128"]
    )
    jobs = Reproduction(args).launcher.jobs

    cv = [job for job in jobs if job.name == "tables/cv"]
    assert bool(cv) == expected
    if expected:
        assert cv[0].depends_on == ("tables/fold0", "tables/fold1", "tables/fold2")
        assert cv[0].commands(cv[0], "local")[0][-6:] == [
            "--folds",
            "0",
            "1",
            "2",
            "--output-dir",
            str((tmp_path / "out" / "tables").resolve()),
        ]
        assert jobs[-1] is cv[0]
