from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from typing import Any, ContextManager, cast

import torch
from torch import Tensor

from tasks.utils.training.runtime import AmpConfig


def move_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {field.name: move_to_device(getattr(value, field.name), device) for field in fields(value)}
        return replace(value, **updates)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def _resolve_device(trainer_config: Mapping[str, object]) -> torch.device:
    accelerator = str(trainer_config.get("accelerator", "cpu")).lower()
    if accelerator in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return _resolve_cuda_device(trainer_config.get("devices"))
    if accelerator in {"auto", "mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator in {"cuda", "gpu"}:
        raise RuntimeError("CUDA accelerator requested but CUDA is not available")
    if accelerator == "mps":
        raise RuntimeError("MPS accelerator requested but MPS is not available")
    return torch.device("cpu")


def _resolve_cuda_device(devices: object) -> torch.device:
    index = _first_cuda_device_index(devices)
    available = int(torch.cuda.device_count())
    if index < 0:
        raise ValueError(f"CUDA device index must be non-negative, got {index}")
    if index >= available:
        raise RuntimeError(
            f"CUDA device {index} requested by trainer.devices, but only {available} CUDA device(s) exist"
        )
    return torch.device(f"cuda:{index}")


def _first_cuda_device_index(devices: object) -> int:
    if devices is None:
        return 0
    if isinstance(devices, int):
        if devices <= 0:
            raise ValueError(f"trainer.devices must be positive, got {devices}")
        return 0
    if isinstance(devices, str):
        text = devices.strip().lower()
        if not text or text == "auto":
            return 0
        if text.startswith("cuda:"):
            return int(text.removeprefix("cuda:"))
        if "," in text:
            return int(text.split(",", maxsplit=1)[0].strip())
        value = int(text)
        if value <= 0:
            raise ValueError(f"trainer.devices must be positive, got {devices!r}")
        return 0
    if isinstance(devices, Sequence):
        if not devices:
            raise ValueError("trainer.devices list must not be empty")
        return int(cast(Any, devices[0]))
    return 0


def _resolve_amp_config(trainer_config: Mapping[str, object], device: torch.device) -> AmpConfig:
    precision = str(trainer_config.get("precision", "32-true")).strip().lower()
    if precision in {"32", "32-true", "fp32", "float32", "false", "none"}:
        return AmpConfig(precision="32-true", enabled=False, dtype=torch.float32, scaler_enabled=False)
    if precision in {"16", "16-mixed", "fp16", "float16"}:
        if device.type not in {"cuda", "mps"}:
            return AmpConfig(precision="32-true", enabled=False, dtype=torch.float32, scaler_enabled=False)
        return AmpConfig(
            precision="16-mixed",
            enabled=True,
            dtype=torch.float16,
            scaler_enabled=device.type == "cuda",
        )
    if precision in {"bf16", "bf16-mixed", "bfloat16"}:
        return AmpConfig(
            precision="bf16-mixed",
            enabled=device.type in {"cuda", "cpu"},
            dtype=torch.bfloat16,
            scaler_enabled=False,
        )
    raise ValueError(f"Unsupported trainer.precision {precision!r}; use 32-true, 16-mixed, or bf16-mixed")


def _autocast_context(amp: AmpConfig, device: torch.device) -> ContextManager[None]:
    if not amp.enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp.dtype, enabled=True)
