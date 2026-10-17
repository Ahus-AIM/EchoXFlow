"""Train the ECG-conditioned video diffusion model.

Example::

    uv run python -m tasks.video_diffusion.train --data-root /path/to/EchoXFlow \\
        --coordinate-space beamspace --resolution 128

Crash safety: ``latest.pt`` (model, EMA, optimizer, RNG states, epoch and position within it, running log sums and
``history.jsonl`` length) is written atomically at the end of every epoch and every ``--checkpoint-every-steps``
steps or ``--checkpoint-every-minutes`` minutes. ``--resume <run>/latest.pt`` continues from that step with the same
data order, clip starts and noise draws as an uninterrupted run (up to nondeterministic GPU kernels). ``ema.pt`` is
written at the end of every epoch; on completion ``training_complete.json`` is written and ``latest.pt`` is deleted
unless ``--keep-latest``. A resumed or completed run whose config differs from the requested one is refused.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import signal
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tasks.utils.atomic import (
    atomic_open,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    read_json,
    remove_stale_temporaries,
)
from tasks.utils.optim import AdamMuon
from tasks.video_diffusion.dataset import ResumableEpochSampler, build_datasets
from tasks.video_diffusion.model import ExponentialMovingAverage, build_model
from tasks.video_diffusion.provenance import (
    ProvenanceError,
    bind_output,
    dataset_identity,
    file_digest,
    implementation_identity,
)
from tasks.video_diffusion.task import diffusion_loss_terms, prepare_batch


def bind_training(run_dir: Path, config: Mapping[str, Any]) -> None:
    bind_output(
        run_dir,
        {
            "config_hash": config_hash(config),
            "dataset": dataset_identity(config),
            "implementation": implementation_identity(),
        },
        artifacts=("training_complete.json", "latest.pt", "ema.pt", "config.json"),
    )


DEFAULT_CONFIG = Path(__file__).resolve().parent / "train.yaml"
CHECKPOINT_FORMAT = "echoxflow_video_diffusion"
CHECKPOINT_VERSION = 2
# Settings that change speed or file locations but not the trained model; left out of the config hash.
_OPERATIONAL_KEYS = {
    "model": {"activation_checkpointing"},
    "data": {"num_workers", "index_workers", "root_dir", "split_csv", "index_cache_dir"},
    "trainer": {
        "device",
        "checkpoint_every_steps",
        "checkpoint_every_minutes",
        "keep_ema_epochs",
        "validation_batches",
        "validation_levels",
    },
}


class ConfigMismatchError(RuntimeError):
    """A checkpoint or finished run belongs to a different configuration than the one requested."""


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash of the configuration that determines the trained model (operational settings left out)."""
    raw = copy.deepcopy(dict(config))
    raw.pop("artifacts", None)
    for section, keys in _OPERATIONAL_KEYS.items():
        if isinstance(raw.get(section), dict):
            raw[section] = {key: value for key, value in raw[section].items() if key not in keys}
    encoded = json.dumps(raw, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_config(path: str | Path | None) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected a YAML mapping in {path}")
    return raw


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data, trainer = config.setdefault("data", {}), config.setdefault("trainer", {})
    for key, value in (
        ("root_dir", args.data_root),
        ("split_csv", args.split_csv),
        ("evaluation_fold", args.evaluation_fold),
        ("coordinate_space", args.coordinate_space),
        ("spatial_size", args.resolution),
        ("max_recordings", args.max_recordings),
        ("num_workers", args.num_workers),
    ):
        if value is not None:
            data[key] = value
    for key, value in (
        ("epochs", args.epochs),
        ("max_steps", args.max_steps),
        ("device", args.device),
        ("checkpoint_every_steps", getattr(args, "checkpoint_every_steps", None)),
        ("checkpoint_every_minutes", getattr(args, "checkpoint_every_minutes", None)),
    ):
        if value is not None:
            trainer[key] = value
    if getattr(args, "keep_ema_epochs", None):
        trainer["keep_ema_epochs"] = [int(epoch) for epoch in str(args.keep_ema_epochs).split(",") if epoch.strip()]
    if getattr(args, "logsnr_shift", None) is not None:
        config.setdefault("diffusion", {})["logsnr_shift"] = float(args.logsnr_shift)
    if args.output_dir is not None:
        config.setdefault("artifacts", {})["run_dir"] = str(args.output_dir)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True  # fixed clip shapes: time the convolution kernels once, keep the fastest


def run_directory(config: Mapping[str, Any]) -> Path:
    artifacts = config.get("artifacts", {})
    if artifacts.get("run_dir"):
        return Path(artifacts["run_dir"])
    data = config["data"]
    name = f"{data.get('coordinate_space', 'beamspace')}_{int(data.get('spatial_size', 128)):04d}"
    name += f"_fold{int(data.get('evaluation_fold', 4))}_{time.strftime('%Y%m%d_%H%M%S')}"
    return Path(artifacts.get("root_dir", "outputs/runs/video_diffusion")) / name


@torch.no_grad()
def validate(
    model: torch.nn.Module, loader: DataLoader, config: Mapping[str, Any], *, batches: int, seed: int, levels: int = 8
) -> dict:
    """Diffusion loss on the first ``batches`` held-out clips at a fixed grid of ``levels`` noise levels.

    Every (clip, level) pair uses the same noise in every call, so values are comparable across epochs and runs; the
    mean over the grid and the loss at each level (``t<step>_<term>``) are returned.
    """
    model.eval()
    device = next(model.parameters()).device
    steps = int(config.get("diffusion", {}).get("steps", 1000))
    grid = [int((level + 0.5) * steps / levels) for level in range(levels)]
    totals: dict[str, float] = {}
    count = 0
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        prepared = prepare_batch(batch, device=device, config=config)
        for level, timestep in enumerate(grid):
            generator = torch.Generator(device=device).manual_seed(seed + index * levels + level)
            timesteps = torch.full((len(prepared.first_frame),), timestep, dtype=torch.long, device=device)
            terms = diffusion_loss_terms(model, prepared, config, timesteps=timesteps, generator=generator)
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value)
                key = f"t{timestep:04d}_{name}"
                totals[key] = totals.get(key, 0.0) + float(value) * levels
            count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def checkpoint_payload(
    model: torch.nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: AdamMuon,
    *,
    config: Mapping[str, Any],
    epoch: int,
    step: int,
    step_in_epoch: int = 0,
    running: Mapping[str, float] | None = None,
    elapsed_s: float = 0.0,
    history_lines: int = 0,
) -> dict[str, Any]:
    """Everything needed to continue after ``step``: ``epoch`` completed epochs plus ``step_in_epoch`` batches."""
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": config,
        "config_hash": config_hash(config),
        "epoch": epoch,
        "step": step,
        "step_in_epoch": int(step_in_epoch),
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "running": dict(running or {}),
        "elapsed_s": float(elapsed_s),
        "history_lines": int(history_lines),
    }


def _restore_rng(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _truncate_lines(path: Path, count: int) -> None:
    """Keep the first ``count`` lines of ``path``: rows logged after the checkpoint are rewritten on resume."""
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)[: int(count)]
    with atomic_open(path, "w", encoding="utf-8") as stream:
        stream.write("".join(line if line.endswith("\n") else line + "\n" for line in lines))


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


def train(config: dict[str, Any], *, resume: str | Path | None = None, keep_latest: bool = False) -> Path:
    run_dir = Path(resume).parent if resume is not None else run_directory(config)
    bind_training(run_dir, config)
    trainer, optim = config["trainer"], config["optimizer"]
    seed = int(trainer.get("seed", 0))
    seed_everything(seed)
    training_set, validation_set = build_datasets(config)
    device = torch.device(trainer.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = build_model(config).to(device)
    ema = ExponentialMovingAverage(model, decay=float(optim.get("ema_decay", 0.995)))
    optimizer = AdamMuon(
        model.parameters(),
        lr=float(optim["learning_rate"]),
        muon_momentum=float(optim.get("muon_momentum", 0.95)),
        betas=tuple(float(beta) for beta in optim.get("betas", (0.9, 0.999))),  # type: ignore[arg-type]
        weight_decay=float(optim.get("weight_decay", 0.0)),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_temporaries(run_dir)  # partial checkpoint writes of a killed run
    history_path = run_dir / "history.jsonl"
    start_epoch = step = start_in_epoch = 0
    running: dict[str, float] = {}
    elapsed_before = 0.0
    state: Mapping[str, Any] | None = None
    if resume is not None:
        loaded: dict[str, Any] = torch.load(resume, map_location=device, weights_only=False)
        state = loaded
        if loaded.get("config_hash", config_hash(loaded["config"])) != config_hash(config):
            raise ConfigMismatchError(f"{resume} was trained with another configuration than the one requested")
        model.load_state_dict(loaded["model"])
        ema.model.load_state_dict(loaded["ema_model"])
        ema.updates = int(loaded["ema_updates"])
        optimizer.load_state_dict(loaded["optimizer"])
        start_epoch, step = int(loaded["epoch"]), int(loaded["step"])
        start_in_epoch = int(loaded.get("step_in_epoch", 0))
        running = {str(key): float(value) for key, value in dict(loaded.get("running", {})).items()}
        elapsed_before = float(loaded.get("elapsed_s", 0.0))
        if "history_lines" in loaded:
            _truncate_lines(history_path, int(loaded["history_lines"]))
        print(
            json.dumps(
                {"resumed_from": str(resume), "epoch": start_epoch, "step": step, "step_in_epoch": start_in_epoch}
            ),
            flush=True,
        )
    else:
        history_path.unlink(missing_ok=True)
    atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=False))
    summary = {
        "training_recordings": len(training_set.recordings),
        "validation_recordings": len(validation_set.recordings),
        "training_items_per_epoch": len(training_set),
        "training_view_counts": {view: len(items) for view, items in training_set.grouped.items()},
        "validation_view_counts": {view: len(items) for view, items in validation_set.grouped.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config_hash": config_hash(config),
    }
    atomic_write_json(run_dir / "dataset_summary.json", summary)
    print(json.dumps(summary))
    seed_everything(seed + 10_000)
    if state is not None:
        _restore_rng(state.get("rng"))
    data = config["data"]
    workers = int(data.get("num_workers", 4))
    loader_options: dict[str, Any] = {"num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}
    batch_size = int(data.get("batch_size", 1))
    # The epoch order, clip starts and ECG flips come from the sampler's (seed, epoch) stream, and the loaders own
    # their generators, so the global RNGs (diffusion noise and timesteps) are untouched by data loading and an
    # interrupted epoch resumes exactly.
    sampler = ResumableEpochSampler(len(training_set), seed=seed)
    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2),
        **loader_options,
    )
    max_steps = int(trainer.get("max_steps") or 0)
    epochs = int(trainer.get("epochs", 10))
    log_every = int(trainer.get("log_every_steps", 50))
    every_steps = int(trainer.get("checkpoint_every_steps", 1000) or 0)
    every_seconds = 60.0 * float(trainer.get("checkpoint_every_minutes", 30.0) or 0.0)
    started = time.time() - elapsed_before
    last_checkpoint = time.monotonic()

    def save_latest(*, completed_epochs: int, step_in_epoch: int) -> None:
        payload = checkpoint_payload(
            model,
            ema,
            optimizer,
            config=config,
            epoch=completed_epochs,
            step=step,
            step_in_epoch=step_in_epoch,
            running=running,
            elapsed_s=time.time() - started,
            history_lines=_line_count(history_path),
        )
        atomic_torch_save(payload, run_dir / "latest.pt")

    for epoch in range(start_epoch, epochs):
        in_epoch = start_in_epoch if epoch == start_epoch else 0
        sampler.set_position(epoch, in_epoch * batch_size)
        model.train()
        for batch in training_loader:
            if max_steps and step >= max_steps:
                break
            terms = diffusion_loss_terms(model, batch, config)
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf)
            step += 1
            in_epoch += 1
            optimizer.step()
            ema.update(model)
            for name, value in terms.items():
                running[name] = running.get(name, 0.0) + float(value.detach())
            running["gradient_norm"] = running.get("gradient_norm", 0.0) + float(gradient_norm)
            running["steps"] = running.get("steps", 0.0) + 1.0
            if step % log_every == 0:
                count = running.pop("steps")
                row: dict[str, float] = {"epoch": epoch + 1, "step": step}
                row.update({name: value / count for name, value in running.items()})
                row["elapsed_s"] = round(time.time() - started, 1)
                running = {}
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            due_steps = every_steps > 0 and step % every_steps == 0
            due_time = every_seconds > 0 and time.monotonic() - last_checkpoint >= every_seconds
            if due_steps or due_time:
                save_latest(completed_epochs=epoch, step_in_epoch=in_epoch)
                last_checkpoint = time.monotonic()
                print(json.dumps({"checkpoint": "latest.pt", "epoch": epoch + 1, "step": step}), flush=True)
        metrics = validate(
            ema.model,
            validation_loader,
            config,
            batches=int(trainer.get("validation_batches", 16)),
            levels=int(trainer.get("validation_levels", 8)),
            seed=seed + 200,
        )
        row = {"epoch": epoch + 1, "step": step, **{f"validation_{name}": value for name, value in metrics.items()}}
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        save_latest(completed_epochs=epoch + 1, step_in_epoch=0)
        last_checkpoint = time.monotonic()
        atomic_torch_save(
            {
                "format": "echoxflow_video_diffusion_ema",
                "config": config,
                "config_hash": config_hash(config),
                "step": step,
                "ema_model": ema.model.state_dict(),
            },
            run_dir / "ema.pt",
        )
        if epoch + 1 in {int(kept) for kept in trainer.get("keep_ema_epochs") or ()}:
            shutil.copyfile(run_dir / "ema.pt", run_dir / f"ema_epoch{epoch + 1:03d}.pt")
        if max_steps and step >= max_steps:
            break
    completed = {
        "epochs": epochs,
        "step": step,
        "max_steps": max_steps or None,
        "elapsed_s": round(time.time() - started, 1),
        "config_hash": config_hash(config),
    }
    completed["ema_sha256"] = file_digest(run_dir / "ema.pt")
    atomic_write_json(run_dir / "training_complete.json", completed)
    latest = run_dir / "latest.pt"
    if not keep_latest and latest.is_file():
        size = latest.stat().st_size
        latest.unlink()
        print(json.dumps({"removed": str(latest), "bytes": size, "hint": "--keep-latest keeps it"}), flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, default=None, help="task YAML (default: tasks/video_diffusion/train.yaml)"
    )
    parser.add_argument("--data-root", default=None, help="EchoXFlow root containing croissant.json")
    parser.add_argument(
        "--split-csv", default=None, help="exam_id,five_fold_split CSV (default: <data-root>/splits.csv)"
    )
    parser.add_argument("--evaluation-fold", type=int, default=None, help="held-out fold; the others are trained on")
    parser.add_argument("--coordinate-space", choices=("beamspace", "cartesian"), default=None)
    parser.add_argument("--resolution", type=int, default=None, help="square frame size, e.g. 128, 256 or 320")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    parser.add_argument("--max-recordings", type=int, default=None, help="debug: limit recordings per split")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="run directory (default: timestamped)")
    parser.add_argument("--resume", type=Path, default=None, help="latest.pt of an interrupted run")
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help="also write latest.pt every N steps within an epoch (default 1000; 0 disables)",
    )
    parser.add_argument(
        "--checkpoint-every-minutes",
        type=float,
        default=None,
        help="also write latest.pt after this many minutes since the last one (default 30; 0 disables)",
    )
    parser.add_argument(
        "--keep-latest", action="store_true", help="keep latest.pt after training completes (deleted by default)"
    )
    parser.add_argument(
        "--keep-ema-epochs",
        default=None,
        help="comma-separated epochs whose EMA weights are also kept as ema_epoch<N>.pt (for convergence checks)",
    )
    parser.add_argument(
        "--logsnr-shift",
        type=float,
        default=None,
        help="shift the cosine schedule's log-SNR by this much (negative: more training and sampling at high noise)",
    )
    return parser


def _stop_on_sigterm(signum: int, frame: object) -> None:
    del frame
    print(f"received signal {signum}; stopping (rerun the same command to resume)", flush=True)
    raise SystemExit(128 + signum)  # unwind: in-progress atomic writes remove their temporary files


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    config = apply_overrides(load_config(args.config), args)
    run_dir = Path(args.resume).parent if args.resume is not None else None
    if run_dir is None and args.output_dir is not None:
        run_dir = Path(args.output_dir)
    if run_dir is None:
        run_dir = run_directory(config)
        config.setdefault("artifacts", {})["run_dir"] = str(run_dir)
    try:
        bind_training(run_dir, config)
    except ProvenanceError as error:
        print(error)
        return 3
    if run_dir is not None:
        complete = read_json(run_dir / "training_complete.json")
        if complete is not None:
            if complete.get("config_hash") != config_hash(config):
                print(f"{run_dir} finished training with another configuration; use a new --output-dir")
                return 3
            weights = run_dir / "ema.pt"
            if not weights.is_file() or complete.get("ema_sha256") != file_digest(weights):
                print(f"{run_dir} has missing or changed EMA weights; use a new --output-dir")
                return 3
            print(f"training already complete: {run_dir / 'training_complete.json'}")
            return 0
        if args.resume is None and (run_dir / "latest.pt").is_file():
            print(f"{run_dir / 'latest.pt'} exists: pass --resume {run_dir / 'latest.pt'} or use a new --output-dir")
            return 3
    try:
        run_dir = train(config, resume=args.resume, keep_latest=args.keep_latest)
    except ConfigMismatchError as error:
        print(f"{error}; use a new --output-dir")
        return 3
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
