#!/usr/bin/env python3
"""Compare public EchoCLIP, PanEcho and EchoPrime EF estimates against 3D LV mesh volumes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echoxflow import load_croissant, open_recording  # noqa: E402
from echoxflow.scan import sector_to_cartesian  # noqa: E402
from echoxflow.scan.geometry import sector_geometry_from_mapping  # noqa: E402
from tasks.utils.stats import bootstrap_mean  # noqa: E402
from tasks.video_diffusion.features import (  # noqa: E402
    ECHOPRIME_MEAN,
    ECHOPRIME_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EchoCLIPExtractor,
    EchoPrimeMViTExtractor,
    EchoPrimeViewExtractor,
    PanEchoExtractor,
    _clip_tensor,
    _normalized,
    _to_video,
    _weights_path,
)

MODELS = ("echoclip", "panecho", "echoprime")


class PanEchoEFPredictor(PanEchoExtractor):
    """Official PanEcho EF regression head on the shared video encoder; inputs are full-cine samples at 224 px."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        checkpoint = torch.load(_weights_path("PANECHO_WEIGHTS", "panecho.pt"), map_location="cpu", weights_only=False)
        state = checkpoint["weights"]
        weight, bias = state["EF_head.1.weight"], state["EF_head.1.bias"]
        self.head = nn.Linear(weight.shape[1], 1).eval().requires_grad_(False).to(device)
        self.head.load_state_dict({"weight": weight, "bias": bias})

    def predict(self, videos: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, videos.shape[1] - 1, 16).round().astype(int).tolist()
        frames = _clip_tensor(videos, indices, self.device)
        with torch.inference_mode():
            features = self.model(_normalized(_to_video(frames, len(videos)), IMAGENET_MEAN, IMAGENET_STD))
            return np.asarray(self.head(features).flatten().cpu().numpy(), dtype=np.float32)


class EchoPrimeEFPredictor:
    """EchoPrime's LV-section weighted study retrieval, using the shared encoder and view classifier loaders.

    The official candidate embeddings, study IDs, labels and MIL weights are required in addition to the encoders.
    Only the LV ejection-fraction head is loaded; language models and report generation are unnecessary.
    """

    def __init__(self, device: torch.device) -> None:
        self.encoder = EchoPrimeMViTExtractor(device)
        self.views = EchoPrimeViewExtractor(device)
        self.device = device
        source = _weights_path("ECHOPRIME_SOURCE", "an EchoPrime checkout (assets/MIL_weights.csv)")
        data = _weights_path("ECHOPRIME_MODEL_DATA", "EchoPrime model_data directory with candidates_data/")
        with (source / "assets" / "MIL_weights.csv").open(newline="", encoding="utf-8") as stream:
            row = next(row for row in csv.DictReader(stream) if row["Section"] == "Left Ventricle")
        self.section_weights = np.asarray([float(value) for key, value in row.items() if key != "Section"])
        base = data / "candidates_data"
        with (base / "candidate_studies.csv").open(newline="", encoding="utf-8") as stream:
            studies = [row["Study"] for row in csv.DictReader(stream)]
        with (base / "candidate_labels.pkl").open("rb") as stream:
            labels = pickle.load(stream)
        keys = [key for key in labels if "ejection" in str(key).lower() and "fraction" in str(key).lower()]
        if len(keys) != 1:
            raise ValueError(f"cannot identify EchoPrime EF labels: {keys}")
        # CSV IDs are text; the released label dictionary uses integer study IDs.
        ef = {str(study): value for study, value in labels[keys[0]].items()}
        self.labels = np.asarray([float(ef[study]) if study in ef else np.nan for study in studies])
        self.candidates = torch.cat(
            [
                torch.load(base / f"candidate_embeddings_p{part}.pt", map_location="cpu", weights_only=True)
                for part in (1, 2)
            ]
        ).float()
        if len(self.candidates) != len(studies) or self.candidates.shape[1] != 512:
            raise ValueError("EchoPrime candidate IDs and embeddings differ")

    def _retrieve(self, features: np.ndarray, views: np.ndarray) -> float:
        weights = self.section_weights[views.astype(int)]
        embedding = torch.from_numpy((features * weights[:, None]).sum(axis=0)).float()
        embedding = F.normalize(embedding, dim=0)
        similarities = self.candidates @ embedding
        for neighbors in (50, 100, 200, 500):
            indices = torch.topk(similarities, min(neighbors, len(similarities))).indices.numpy()
            values = self.labels[indices]
            if np.isfinite(values).any():
                return float(np.nanmean(values))
        raise ValueError("EchoPrime nearest studies have no finite EF labels")

    def predict(self, videos: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, videos.shape[1] - 1, 32).round().astype(int)[::2]
        tensor = torch.from_numpy(np.ascontiguousarray(videos[:, indices])).to(self.device, torch.float32)
        tensor = tensor[:, :, None].repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
        with torch.inference_mode():
            features = self.encoder.model(_normalized(tensor, ECHOPRIME_MEAN, ECHOPRIME_STD)).cpu().numpy()
            first = self.views.extract(videos[:, 0], batch_size=2)
            views = self.views.view_head(torch.as_tensor(first, device=self.device)).argmax(dim=1).cpu().numpy()
        # Two single-view predictions and one LV-section weighted two-view study prediction.
        return np.asarray(
            [self._retrieve(features[i : i + 1], views[i : i + 1]) for i in range(len(videos))]
            + [self._retrieve(features, views)]
        )


def closed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    """Closed oriented triangle/quad surface volume in m³; reject malformed or open surfaces."""
    points, faces = np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] not in (3, 4):
        raise ValueError("invalid mesh point/face shape")
    if not np.isfinite(points).all() or not faces.size or np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("invalid mesh vertices or indices")
    triangles = faces if faces.shape[1] == 3 else np.concatenate((faces[:, :3], faces[:, [0, 2, 3]]))
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(unique[:, 0] == unique[:, 1]) or np.any(counts != 2) or np.any(np.bincount(inverse, weights=signs) != 0):
        raise ValueError("mesh is not a closed consistently oriented surface")
    vertices = (points - points.mean(axis=0))[triangles]
    volume = abs(float(np.einsum("ij,ij->i", vertices[:, 0], np.cross(vertices[:, 1], vertices[:, 2])).sum()) / 6)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("mesh volume must be finite and positive")
    return volume


def mesh_references(records: Sequence[Any], root: Path) -> tuple[dict, list[dict]]:
    """Load a bounded number of independent mesh recordings concurrently, reducing references in catalog order."""
    candidates = [record for record in records if "lv_mesh" in record.annotations]

    def reference(record: Any) -> tuple[dict | None, dict | None]:
        try:
            mesh = open_recording(record, root=root).load_packed_mesh_annotation()
            volumes = []
            for index in range(mesh.frame_count):
                frame = mesh.frame(index)
                try:
                    volumes.append((index, closed_volume(frame.points, frame.faces) * 1e6))
                except ValueError:
                    continue
            if len(volumes) < 2:
                raise ValueError("fewer than two valid closed LV mesh volumes")
            ed, edv = max(volumes, key=lambda pair: pair[1])
            es, esv = min(volumes, key=lambda pair: pair[1])
            return {
                "exam_id": record.exam_id,
                "mesh_recording_id": record.recording_id,
                "mesh_frames": mesh.frame_count,
                "valid_mesh_frames": len(volumes),
                "ed_frame": ed,
                "es_frame": es,
                "edv_ml": edv,
                "esv_ml": esv,
                "mesh_ef_percent": 100 * (edv - esv) / edv,
            }, None
        except (ValueError, KeyError, FileNotFoundError) as error:
            return None, {"exam_id": record.exam_id, "recording_id": record.recording_id, "error": str(error)}

    def rank(row: dict) -> tuple:
        return row["valid_mesh_frames"], row["mesh_frames"], row["mesh_recording_id"]

    references, failures = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, (candidate, failure) in enumerate(pool.map(reference, candidates), start=1):
            if failure is not None:
                failures.append(failure)
            if candidate is not None:
                exam = candidate["exam_id"]
                previous = references.get(exam)
                if previous is None or rank(candidate) > rank(previous):
                    references[exam] = candidate
            if number == 1 or number % 25 == 0 or number == len(candidates):
                print(f"Mesh cohort: recording {number}/{len(candidates)}", flush=True)
    return references, failures


def view_choices(records: Sequence[Any]) -> dict:
    """Highest geometric-mean confidence per exam/view, using Prime/Fine consensus and single-plane B-mode."""
    selected: dict = defaultdict(dict)
    for record in records:
        view = record.view
        if view not in ("A2C", "A4C") or "data/2d_brightness_mode" not in record.array_paths:
            continue
        if (record.frame_count("2d_brightness_mode") or 0) < 2:
            continue
        pc, fc = record.view_echoprime_prob, record.view_echofine_prob
        if pc is None or fc is None or not (0 <= pc <= 1 and 0 <= fc <= 1):
            raise ValueError(f"invalid classifier confidence for {record.recording_id}")
        score = math.sqrt(pc * fc)
        current = selected[record.exam_id].get(view)
        if current is None or (score, record.recording_id) > (current[0], current[1].recording_id):
            selected[record.exam_id][view] = (score, record)
    return selected


def prepare_video(record: Any, root: Path, *, count: int, size: int = 224) -> np.ndarray:
    store = open_recording(record, root=root)
    stream = store.load_stream("data/2d_brightness_mode")
    frames = np.asarray(stream.data, dtype=np.float32)
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError(f"{record.recording_id}: expected single-plane multi-frame B-mode")
    geometry = stream.metadata.geometry or sector_geometry_from_mapping(stream.metadata.raw)
    converted = np.stack(
        [sector_to_cartesian(frame, geometry, output_height=size, interpolation="linear").data for frame in frames]
    )
    tensor = torch.from_numpy(np.asarray(converted, dtype=np.float32))[:, None]
    height, width = tensor.shape[-2:]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    tensor = F.pad(tensor, (left, side - width - left, top, side - height - top))
    video = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    finite = video[np.isfinite(video)]
    if not finite.size:
        raise ValueError(f"{record.recording_id}: no finite pixels")
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        raise ValueError(f"{record.recording_id}: no usable intensity range")
    video = np.nan_to_num(np.clip((video - low) / (high - low), 0, 1))
    indices = np.linspace(0, len(video) - 1, count).round().astype(int)
    return np.rint(255 * video[indices]).astype(np.uint8)


class EchoCLIPEF:
    """Official EF prompts: rank prompts per frame, average ranks, median of the top 20% (0..100 EF points)."""

    def __init__(self, device: torch.device, batch_size: int) -> None:
        from open_clip import tokenize

        self.extractor = EchoCLIPExtractor(device)
        self.batch_size = batch_size
        templates = (
            "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
            "LV EJECTION FRACTION IS <#>%. ",
        )
        prompts = [template.replace("<#>", str(value)) for template in templates for value in range(101)]
        with torch.inference_mode():
            self.text = self.extractor.model.encode_text(tokenize(prompts).to(device), normalize=True)
        self.values = np.tile(np.arange(101), len(templates))

    def predict(self, videos: np.ndarray) -> np.ndarray:
        predictions = []
        for video in videos:
            features = self.extractor.extract(video, batch_size=self.batch_size)
            similarity = features @ self.text.float().cpu().numpy().T
            order = np.argsort(-similarity, axis=1)
            ranks = self.values[order].mean(axis=0)
            predictions.append(float(np.median(ranks[: max(1, int(len(ranks) * 0.2))])))
        return np.asarray(predictions)


def summarize(rows: list[dict], models: Sequence[str]) -> list[dict]:
    summaries = []
    target = np.asarray([row["mesh_ef_percent"] for row in rows])
    for model in models:
        for view in ("a2c", "a4c", "two_view"):
            prediction = np.asarray([row[f"{model}_{view}_ef_percent"] for row in rows])
            if not np.isfinite(prediction).all():
                raise ValueError(f"{model}/{view}: non-finite EF predictions")
            errors = prediction - target
            stats = bootstrap_mean(np.abs(errors), namespace="foundation-ef:exam")
            loo_bias = (errors.sum() - errors) / (len(errors) - 1)
            correlation = (
                float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) and np.std(target) else math.nan
            )
            summaries.append(
                {
                    "model": model,
                    "view": view,
                    "n_exams": len(rows),
                    "mae": stats["mean"],
                    "mae_ci95_low": stats["ci95_low"],
                    "mae_ci95_high": stats["ci95_high"],
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "mean_bias": float(errors.mean()),
                    "pearson_r": correlation,
                    "loocv_bias_corrected_mae": float(np.mean(np.abs(errors - loo_bias))),
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--view-csv", type=Path, default=Path("view_classifier_probabilities.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/foundation_model_ef"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--max-exams", type=int, help="Limit paired-view examinations for smoke runs (at least two).")
    parser.add_argument("--batch-size", type=int, default=16, help="EchoCLIP image inference batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--tables-only", action="store_true", help="Rebuild summary.csv and summary.tex from per_exam.csv."
    )
    args = parser.parse_args(argv)
    if args.max_exams is not None and args.max_exams < 2:
        parser.error("--max-exams must be at least two for exam bootstrap")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.tables_only:
        with (args.output_dir / "per_exam.csv").open(newline="", encoding="utf-8") as stream:
            numeric = lambda key, value: float(value) if key.endswith("_percent") else value  # noqa: E731
            rows = [{key: numeric(key, value) for key, value in row.items()} for row in csv.DictReader(stream)]
        write_summaries(args.output_dir, rows, args.models)
        return 0
    root = args.data_root.expanduser().resolve()
    print("Loading Croissant catalog", flush=True)
    catalog = load_croissant(root / "croissant.json", root=root, view_csv=args.view_csv)
    references, failures = mesh_references(catalog.recordings, root)
    choices = view_choices(catalog.recordings)
    cohort = sorted(exam for exam in references if set(choices.get(exam, {})) == {"A2C", "A4C"})
    audit = {
        "mesh_exams": len(references),
        "paired_view_exams": len(cohort),
        "mesh_failures": failures,
        "missing_a2c": sum("A2C" not in choices.get(exam, {}) for exam in references),
        "missing_a4c": sum("A4C" not in choices.get(exam, {}) for exam in references),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 17,
        "croissant": str(root / "croissant.json"),
        "models": args.models,
    }
    selected = cohort[: args.max_exams] if args.max_exams is not None else cohort
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit["evaluated_exams"] = len(selected)
    (args.output_dir / "cohort.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Mesh examinations: {len(references)}; both consensus views: {len(cohort)}; evaluated: {len(selected)}",
        flush=True,
    )
    if len(selected) < 2:
        raise ValueError("at least two mesh examinations with paired consensus views are required")
    rows = [
        {
            **references[exam],
            **{f"{view.lower()}_recording_id": choices[exam][view][1].recording_id for view in ("A2C", "A4C")},
        }
        for exam in selected
    ]
    device = torch.device(args.device)
    for name in args.models:
        predictor = (
            EchoCLIPEF(device, args.batch_size)
            if name == "echoclip"
            else PanEchoEFPredictor(device)
            if name == "panecho"
            else EchoPrimeEFPredictor(device)
        )
        for exam, row in zip(selected, rows, strict=True):
            count = 20 if name == "echoclip" else 32 if name == "echoprime" else 16
            videos = np.stack([prepare_video(choices[exam][view][1], root, count=count) for view in ("A2C", "A4C")])
            values = predictor.predict(videos)
            for view, value in zip(("a2c", "a4c"), values[:2], strict=True):
                row[f"{name}_{view}_ef_percent"] = float(value)
            row[f"{name}_two_view_ef_percent"] = float(values[2] if name == "echoprime" else np.mean(values))
            print(
                f"{name}: {exam} mesh={row['mesh_ef_percent']:.1f}, "
                f"prediction={row[f'{name}_two_view_ef_percent']:.1f}",
                flush=True,
            )
        del predictor
        write_csv(args.output_dir / "per_exam.csv", rows)
    write_summaries(args.output_dir, rows, args.models)
    print(args.output_dir / "summary.csv")
    return 0


def write_summaries(output_dir: Path, rows: list[dict[str, Any]], models: Sequence[str]) -> None:
    """Write ``summary.csv`` and the paper table ``summary.tex`` from the per-examination rows."""
    summaries = summarize(rows, models)
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.tex").write_text(latex_table(summaries, n_exams=len(rows)), encoding="utf-8")


LATEX_MODEL = {"echoclip": "EchoCLIP", "panecho": "PanEcho", "echoprime": "EchoPrime"}
LATEX_VIEW = {"a2c": "A2C", "a4c": "A4C", "two_view": "Two views"}


def latex_table(summaries: list[dict[str, Any]], *, n_exams: int) -> str:
    """Full-width paper table; blue marks it for manual review."""
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \color{blue}",
        r"    \caption{Zero-shot left-ventricular ejection fraction of public echocardiography foundation models "
        f"against the 3D LV mesh volumes of {n_exams} EchoXFlow examinations with both apical views.}}",
        r"    \label{tab:foundation-model-ef}",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll*{5}{r}@{}}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{View} & \textbf{MAE [95\% CI]} & \textbf{RMSE} & \textbf{Bias} & "
        r"\textbf{$r$} & \textbf{MAE$_\text{corr}$} \\",
        r"    \midrule",
    ]
    previous = None
    for row in summaries:
        if row["model"] != previous:
            if previous is not None:
                lines.append(r"    \addlinespace")
            previous = row["model"]
        model = LATEX_MODEL.get(row["model"], row["model"]) if row["view"] == "a2c" else ""
        bias = f"{row['mean_bias']:+.1f}".replace("-", "$-$")
        lines.append(
            f"    {model} & {LATEX_VIEW.get(row['view'], row['view'])} & "
            f"{row['mae']:.1f} [{row['mae_ci95_low']:.1f}, {row['mae_ci95_high']:.1f}] & {row['rmse']:.1f} & "
            f"{bias} & {row['pearson_r']:.2f} & {row['loocv_bias_corrected_mae']:.1f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"    \end{tabular*}",
        r"    \par\vspace{2pt}",
        r"    \begin{minipage}{\linewidth}",
        r"    \footnotesize",
        r"    EF in percentage points. MAE with a 95\% examination-bootstrap confidence interval (2000 replicates); "
        r"bias is prediction minus mesh; $r$ is Pearson's correlation; MAE$_\text{corr}$ is the leave-one-out "
        r"bias-corrected MAE. Two views: EchoCLIP and PanEcho average the view estimates, EchoPrime uses its "
        r"LV-section weighted study retrieval. Reference: largest and smallest closed LV mesh volume per examination.",
        r"    \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
