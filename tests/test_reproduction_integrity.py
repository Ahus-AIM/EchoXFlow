"""Reproduction must reject stale results and report incomplete work as failure."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.reproduce_video_diffusion import Reproduction, _parse_args
from tasks.video_diffusion import evaluate, train
from tasks.video_diffusion.features import MissingExtractorError
from tasks.video_diffusion.provenance import ProvenanceError, bind_output, dataset_identity, file_digest


def _dataset(root: Path) -> dict:
    root.mkdir(exist_ok=True)
    (root / "croissant.json").write_text("{}")
    (root / "splits.csv").write_text("exam_id,five_fold_split\ne,4\n")
    (root / "exams").mkdir(exist_ok=True)
    (root / "exams" / "r.zip").write_bytes(b"payload")
    return {"data": {"root_dir": str(root)}}


def test_provenance_rejects_legacy_and_changed_artifacts(tmp_path):
    output = tmp_path / "evaluation"
    identity = {"checkpoint": "one", "sampling_steps": 32}
    bind_output(output, identity, artifacts=("features",))
    bind_output(output, identity, artifacts=("features",))
    for changed in ({**identity, "checkpoint": "two"}, {**identity, "sampling_steps": 16}):
        with pytest.raises(ProvenanceError, match="identity changed"):
            bind_output(output, changed, artifacts=("features",))
    (output / "provenance.json").unlink()
    (output / "features").mkdir()
    with pytest.raises(ProvenanceError, match="lack provenance"):
        bind_output(output, identity, artifacts=("features",))


@pytest.mark.parametrize("changed", ["splits.csv", "exams/r.zip", "view_classifier_probabilities.csv"])
def test_dataset_identity_detects_changes(tmp_path, changed):
    config = _dataset(tmp_path)
    before = dataset_identity(config)
    (tmp_path / changed).write_text("changed")
    assert dataset_identity(config) != before


def test_completion_markers_do_not_bypass_child_validation_and_single_space_has_no_table(tmp_path):
    _dataset(tmp_path)
    args = _parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--camus-root",
            str(tmp_path),
            "--folds",
            "4",
            "--output-dir",
            str(tmp_path / "out"),
            "--resolutions",
            "128",
            "--spaces",
            "beamspace",
        ]
    )
    output = tmp_path / "out/fold4/beamspace_0128"
    output.mkdir(parents=True)
    (output / "evaluation").mkdir()
    (output / "training_complete.json").write_text("{}")
    (output / "evaluation/metrics.json").write_text("{}")
    reproduction = Reproduction(args)
    assert len(reproduction.launcher.jobs) == 2
    assert all(job.state == "pending" for job in reproduction.launcher.jobs)


def test_missing_extractors_fail_unless_partial_is_explicit(monkeypatch):
    def missing(*args):
        raise MissingExtractorError("missing weights")

    monkeypatch.setattr(evaluate, "build_extractor", missing)
    with pytest.raises(MissingExtractorError):
        evaluate.requested_extractors(["inception"], torch.device("cpu"))
    models, skips = evaluate.requested_extractors(["inception"], torch.device("cpu"), allow_partial=True)
    assert models == {} and skips == {"inception": "missing weights"}


def test_partial_report_is_not_a_completion_marker(tmp_path):
    evaluate.write_metrics(tmp_path, [], skipped={"inception": "missing weights"}, info={})
    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.partial.json").exists()
    assert not evaluate.evaluation_complete(tmp_path, ["inception"])


def test_completed_evaluation_requires_intact_features(tmp_path):
    (tmp_path / "features").mkdir()
    feature = tmp_path / "features/inception.npz"
    np.savez(feature, real=np.ones((2, 3)), generated=np.zeros((2, 3)))
    rows = [{"extractor": "inception", "label": "FD", "metric": "fd", "view": "mean", "value": 1.0}]
    evaluate.write_metrics(tmp_path, rows, skipped={}, info={})
    assert evaluate.evaluation_complete(tmp_path, ["inception"])
    assert not evaluate.evaluation_complete(tmp_path, ["inception", "echoclip"])
    feature.write_bytes(b"corrupt")
    assert not evaluate.evaluation_complete(tmp_path, ["inception"])


def test_actual_extractor_weights_affect_identity():
    extractor = SimpleNamespace(model=torch.nn.Linear(2, 2), input_size=2, kind="frame")
    before = evaluate.extractor_identity(extractor)
    with torch.no_grad():
        extractor.model.weight.add_(1)
    assert evaluate.extractor_identity(extractor) != before


def test_training_completion_requires_config_and_weights(tmp_path, monkeypatch):
    config = train.load_config(None)
    config["data"]["root_dir"] = str(tmp_path)
    _dataset(tmp_path)
    monkeypatch.setattr(train, "implementation_identity", lambda: {"source": "test"})
    run = tmp_path / "run"
    argv = ["--data-root", str(tmp_path), "--output-dir", str(run)]
    resolved = train.apply_overrides(train.load_config(None), train.build_parser().parse_args(argv))
    train.bind_training(run, resolved)
    weights = run / "ema.pt"
    weights.write_bytes(b"weights")
    (run / "training_complete.json").write_text(
        json.dumps({"config_hash": train.config_hash(resolved), "ema_sha256": file_digest(weights)})
    )
    assert train.main(argv) == 0
    assert train.main([*argv, "--epochs", "999"]) == 3
    weights.unlink()
    assert train.main(argv) == 3


@pytest.mark.parametrize(
    "parts,fail",
    [
        ("benchmark", "run_full_benchmark.py"),
        ("benchmark", "build_benchmark_table.py"),
        ("diffusion", "reproduce_video_diffusion.py"),
        ("benchmark", "plot_benchmark_losses.py"),
        ("benchmark", "tasks.segmentation.camus"),
        ("ef", "foundation_model_ef.py"),
        ("diffusion", "none"),
        ("benchmark", "none"),
    ],
)
def test_pipeline_exit_status(tmp_path, parts, fail):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "run_paper_pipeline.sh"
    script.write_text(Path("scripts/run_paper_pipeline.sh").read_text())
    data = tmp_path / "data"
    _dataset(data)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, content in {
        "git": '#!/bin/sh\nif [ "$1" = rev-parse ]; then echo test; fi\n',
        "uv": '#!/bin/sh\ncase "$*" in *"$FAIL_COMMAND"*) exit 7;; esac\nexit 0\n',
    }.items():
        target = binaries / name
        target.write_text(content)
        target.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--data-root",
            str(data),
            "--camus-root",
            str(data),
            "--output-dir",
            str(tmp_path / "out"),
            "--parts",
            parts,
            "--resolutions",
            "128",
            "--folds",
            "4",
        ],
        env={
            **os.environ,
            **{
                key: str(data)
                for key in (
                    "ECHOPRIME_VIEW_CLASSIFIER",
                    "ECHOPRIME_ENCODER",
                    "PANECHO_SOURCE",
                    "PANECHO_WEIGHTS",
                    "ECHOJEPA_SOURCE",
                    "ECHOJEPA_WEIGHTS",
                    "ECHOPRIME_SOURCE",
                    "ECHOPRIME_MODEL_DATA",
                )
            },
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "FAIL_COMMAND": fail,
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (result.returncode == 0) == (fail == "none"), result.stdout + result.stderr


def test_evaluation_rejects_changed_checkpoint_before_reusing_samples(tmp_path, monkeypatch):
    config = _dataset(tmp_path / "data")
    checkpoint = tmp_path / "ema.pt"
    checkpoint.write_bytes(b"first model")
    args = evaluate.build_parser().parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "out")])
    monkeypatch.setattr(evaluate, "implementation_identity", lambda: {"source": "test"})
    evaluate.prepare_evaluation(args, config, {})
    (args.output_dir / "samples.npz").write_bytes(b"old predictions")
    checkpoint.write_bytes(b"new model")
    with pytest.raises(ProvenanceError, match="identity changed"):
        evaluate.prepare_evaluation(args, config, {})
    assert (args.output_dir / "samples.npz").read_bytes() == b"old predictions"


def test_dataset_identity_ignores_permission_changes(tmp_path):
    config = _dataset(tmp_path)
    before = dataset_identity(config)
    payload = tmp_path / "exams/r.zip"
    payload.chmod(payload.stat().st_mode ^ 0o100)
    assert dataset_identity(config) == before


def test_evaluation_requires_a_dataset_before_loading_checkpoint(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("should reject empty evaluation before loading models")

    monkeypatch.setattr(evaluate, "load_checkpoint", unexpected)
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--checkpoint", "unused.pt", "--skip-echoxflow"])
    assert error.value.code == 2
