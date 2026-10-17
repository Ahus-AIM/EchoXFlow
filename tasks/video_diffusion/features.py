"""Pretrained feature extractors for generation-quality metrics.

All extractors are optional and imported lazily. Each consumes grayscale ``uint8`` images already scan converted onto
the square physical canvas at ``input_size`` (see :mod:`tasks.video_diffusion.evaluate`).

Frame extractors (Frechet distance)        Clip extractors (paired KID, 31-frame clips)
  inception       torch-fidelity, 299 px      panecho         256 px, centred 16 frames
  echoprime_view  EchoPrime ConvNeXt, 224     echojepa        292 px, 16 uniformly spaced frames
  echofine        EchoView47 Xception, 224    echoprime_mvit  280 px, frames 0, 2, ..., 30
  echoclip        open_clip EchoCLIP, 224

Weights are located through environment variables (see the task README for download instructions):

  ECHOPRIME_VIEW_CLASSIFIER  EchoPrime ``model_data/weights/view_classifier.pt``
  ECHOPRIME_ENCODER          EchoPrime ``model_data/weights/echo_prime_encoder.pt``
  ECHOFINE_WEIGHTS           EchoView47 ``TTE47_Classifier_Model.h5`` (or its converted ``.pt`` state dict);
                             downloaded from Hugging Face ``thrive-2025/EchoView47`` when unset
  PANECHO_SOURCE / PANECHO_WEIGHTS     PanEcho git checkout and ``panecho.pt`` release asset
  ECHOJEPA_SOURCE / ECHOJEPA_WEIGHTS   EchoJEPA git checkout and ``vitl-vmix22m-pt220-c55.pt``
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ECHOPRIME_MEAN = (29.110628, 28.076836, 29.096405)
ECHOPRIME_STD = (47.989223, 46.456997, 47.20083)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FRAME_EXTRACTORS = ("echoprime_view", "echofine", "echoclip", "inception")
CLIP_EXTRACTORS = ("panecho", "echojepa", "echoprime_mvit")
EXTRACTOR_LABELS = {
    "echoprime_view": "EchoPrime view (frame FD)",
    "echofine": "EchoFine (frame FD)",
    "echoclip": "EchoCLIP (frame FD)",
    "inception": "Inception (frame FD)",
    "panecho": "PanEcho (clip KID)",
    "echojepa": "EchoJEPA (clip KID)",
    "echoprime_mvit": "EchoPrime MViT (clip KID)",
}


class MissingExtractorError(RuntimeError):
    """Raised when an optional dependency or weight file is unavailable."""


class Extractor(Protocol):
    name: str
    kind: str  # "frame" or "clip"
    input_size: int

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray: ...


def _require(module: str, install: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingExtractorError(f"{module} is required for this extractor; install it with `{install}`") from error


def _weights_path(variable: str, description: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise MissingExtractorError(f"set {variable} to {description} (see tasks/video_diffusion/README.md)")
    path = Path(value).expanduser()
    if not path.exists():
        raise MissingExtractorError(f"{variable}={path} does not exist")
    return path


def _state_dict(path: Path) -> Mapping[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _batched(values: np.ndarray, batch_size: int, function: Callable[[np.ndarray], Tensor]) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            outputs.append(function(values[start : start + batch_size]).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _rgb(images: np.ndarray, device: torch.device) -> Tensor:
    """``[N, H, W]`` uint8 -> float ``[N, 3, H, W]`` in 0..255."""
    return torch.from_numpy(np.ascontiguousarray(images)).to(device, torch.float32)[:, None].repeat(1, 3, 1, 1)


def _normalized(tensor: Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> Tensor:
    shape = (1, 3) + (1,) * (tensor.ndim - 2)
    mean_tensor = torch.tensor(mean, device=tensor.device).view(shape)
    std_tensor = torch.tensor(std, device=tensor.device).view(shape)
    return (tensor - mean_tensor) / std_tensor


# ---------------------------------------------------------------------------------------------------------------------
# Frame extractors


class InceptionExtractor:
    """torch-fidelity Inception-v3 (TF-compatible FID weights), 2048-d pool3."""

    name, kind, input_size = "inception", "frame", 299

    def __init__(self, device: torch.device) -> None:
        _require("torch_fidelity", "uv pip install torch-fidelity")
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"], feature_extractor_internal_dtype="float32")
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        def run(batch: np.ndarray) -> Tensor:
            tensor = torch.from_numpy(np.ascontiguousarray(batch))[:, None].repeat(1, 3, 1, 1).to(self.device)
            return cast(Tensor, self.model(tensor)[0])

        return _batched(values, batch_size, run)


class EchoPrimeViewExtractor:
    """EchoPrime view-classifier ConvNeXt-B, 1024-d penultimate features."""

    name, kind, input_size = "echoprime_view", "frame", 224

    def __init__(self, device: torch.device) -> None:
        torchvision = _require("torchvision", "uv pip install torchvision")
        path = _weights_path("ECHOPRIME_VIEW_CLASSIFIER", "EchoPrime model_data/weights/view_classifier.pt")
        model = torchvision.models.convnext_base(weights=None)
        model.classifier[-1] = nn.Linear(int(model.classifier[-1].in_features), 11)
        model.load_state_dict(_state_dict(path))
        self.view_head = model.classifier[-1].eval().requires_grad_(False).to(device)
        model.classifier[-1] = nn.Identity()
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        return _batched(
            values,
            batch_size,
            lambda batch: self.model(_normalized(_rgb(batch, self.device), ECHOPRIME_MEAN, ECHOPRIME_STD)),
        )


class _KerasSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.pointwise(self.depthwise(inputs)))


def _batch_norm(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=0.001, momentum=0.01)


class EchoFineXception(nn.Module):
    """Torch port of the EchoView47 Keras Xception (layer names follow the Keras model)."""

    def __init__(self, num_classes: int = 47) -> None:
        super().__init__()
        self.block1_conv1 = nn.Conv2d(3, 32, 3, stride=2, bias=False)
        self.block1_conv1_bn = _batch_norm(32)
        self.block1_conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.block1_conv2_bn = _batch_norm(64)
        entry: tuple[tuple[int, int, str, str], ...]
        entry = ((64, 128, "conv2d", "batch_normalization"), (128, 256, "conv2d_1", "batch_normalization_1"))
        entry += ((256, 728, "conv2d_2", "batch_normalization_2"),)
        for block, (inputs, outputs, conv, norm) in enumerate(entry, start=2):
            setattr(self, conv, nn.Conv2d(inputs, outputs, 1, stride=2, bias=False))
            setattr(self, norm, _batch_norm(outputs))
            setattr(self, f"block{block}_sepconv1", _KerasSeparableConv2d(inputs, outputs))
            setattr(self, f"block{block}_sepconv1_bn", _batch_norm(outputs))
            setattr(self, f"block{block}_sepconv2", _KerasSeparableConv2d(outputs, outputs))
            setattr(self, f"block{block}_sepconv2_bn", _batch_norm(outputs))
        for block in range(5, 13):
            for index in (1, 2, 3):
                setattr(self, f"block{block}_sepconv{index}", _KerasSeparableConv2d(728, 728))
                setattr(self, f"block{block}_sepconv{index}_bn", _batch_norm(728))
        self.conv2d_3 = nn.Conv2d(728, 1024, 1, stride=2, bias=False)
        self.batch_normalization_3 = _batch_norm(1024)
        self.block13_sepconv1, self.block13_sepconv1_bn = _KerasSeparableConv2d(728, 728), _batch_norm(728)
        self.block13_sepconv2, self.block13_sepconv2_bn = _KerasSeparableConv2d(728, 1024), _batch_norm(1024)
        self.block14_sepconv1, self.block14_sepconv1_bn = _KerasSeparableConv2d(1024, 1536), _batch_norm(1536)
        self.block14_sepconv2, self.block14_sepconv2_bn = _KerasSeparableConv2d(1536, 2048), _batch_norm(2048)
        self.dense = nn.Linear(2048, num_classes)

    def _layer(self, name: str, inputs: Tensor) -> Tensor:
        return cast(Tensor, getattr(self, f"{name}_bn")(getattr(self, name)(inputs)))

    @staticmethod
    def _same_max_pool(inputs: Tensor) -> Tensor:
        height, width = inputs.shape[-2:]
        pad_h = max((math.ceil(height / 2) - 1) * 2 + 3 - height, 0)
        pad_w = max((math.ceil(width / 2) - 1) * 2 + 3 - width, 0)
        padded = F.pad(inputs, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=float("-inf"))
        return F.max_pool2d(padded, kernel_size=3, stride=2)

    def forward_features(self, inputs: Tensor) -> Tensor:
        x = torch.relu(self._layer("block1_conv1", inputs / 255.0))
        x = torch.relu(self._layer("block1_conv2", x))
        for block, (conv, norm) in zip(
            (2, 3, 4),
            (
                ("conv2d", "batch_normalization"),
                ("conv2d_1", "batch_normalization_1"),
                ("conv2d_2", "batch_normalization_2"),
            ),
        ):
            residual = getattr(self, norm)(getattr(self, conv)(x))
            x = x if block == 2 else torch.relu(x)
            x = self._layer(f"block{block}_sepconv2", torch.relu(self._layer(f"block{block}_sepconv1", x)))
            x = self._same_max_pool(x) + residual
        for block in range(5, 13):
            residual = x
            for index in (1, 2, 3):
                x = self._layer(f"block{block}_sepconv{index}", torch.relu(x))
            x = x + residual
        residual = self.batch_normalization_3(self.conv2d_3(x))
        x = self._layer("block13_sepconv2", torch.relu(self._layer("block13_sepconv1", torch.relu(x))))
        x = self._same_max_pool(x) + residual
        x = torch.relu(self._layer("block14_sepconv1", x))
        x = torch.relu(self._layer("block14_sepconv2", x))
        return x.mean(dim=(2, 3))


def convert_echofine_h5(path: Path) -> dict[str, Tensor]:
    """Convert the EchoView47 Keras ``.h5`` weights into an :class:`EchoFineXception` state dict."""
    h5py = _require("h5py", "uv pip install h5py")
    state = EchoFineXception().state_dict()

    def weight(group: Any, layer: str, name: str) -> np.ndarray:
        found: list[Any] = []
        group[layer].visititems(lambda key, item: found.append(item) if key.endswith(name) else None)
        if len(found) != 1:
            raise ValueError(f"expected one weight {layer}/{name} in {path}")
        return np.asarray(found[0], dtype=np.float32)

    def tensor(values: np.ndarray, *axes: int) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(values, axes) if axes else values))

    with h5py.File(path, "r") as file:
        layers = file["model_weights"]["sequential_1"]
        convolutions = ["block1_conv1", "block1_conv2", "conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
        norms = ["block1_conv1_bn", "block1_conv2_bn"] + [f"batch_normalization{s}" for s in ("", "_1", "_2", "_3")]
        for layer in convolutions:
            state[f"{layer}.weight"] = tensor(weight(layers, layer, "kernel:0"), 3, 2, 0, 1)
        for block in range(2, 15):
            for index in range(1, (2 if block in {2, 3, 4, 13, 14} else 3) + 1):
                layer = f"block{block}_sepconv{index}"
                state[f"{layer}.depthwise.weight"] = tensor(weight(layers, layer, "depthwise_kernel:0"), 2, 3, 0, 1)
                state[f"{layer}.pointwise.weight"] = tensor(weight(layers, layer, "pointwise_kernel:0"), 3, 2, 0, 1)
                norms.append(f"{layer}_bn")
        for layer in norms:
            for target, source in (
                ("weight", "gamma"),
                ("bias", "beta"),
                ("running_mean", "moving_mean"),
                ("running_var", "moving_variance"),
            ):
                state[f"{layer}.{target}"] = tensor(weight(layers, layer, f"{source}:0"))
        state["dense.weight"] = tensor(weight(file["model_weights"]["dense"], "dense", "kernel:0"), 1, 0)
        state["dense.bias"] = tensor(weight(file["model_weights"]["dense"], "dense", "bias:0"))
    return state


class EchoFineExtractor:
    """EchoFine / EchoView47 Xception, 2048-d pooled features (aspect-preserving 224 px input)."""

    name, kind, input_size = "echofine", "frame", 224

    def __init__(self, device: torch.device) -> None:
        value = os.environ.get("ECHOFINE_WEIGHTS", "").strip()
        if value:
            path = Path(value).expanduser()
        else:
            hub = _require("huggingface_hub", "uv pip install huggingface_hub h5py")
            path = Path(hub.hf_hub_download(repo_id="thrive-2025/EchoView47", filename="TTE47_Classifier_Model.h5"))
        state = convert_echofine_h5(path) if path.suffix in {".h5", ".keras"} else _state_dict(path)
        model = EchoFineXception()
        model.load_state_dict(state)
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        return _batched(values, batch_size, lambda batch: self.model.forward_features(_rgb(batch, self.device)))


class EchoCLIPExtractor:
    """EchoCLIP (``hf-hub:mkaichristensen/echo-clip``) L2-normalized 512-d image embedding."""

    name, kind, input_size = "echoclip", "frame", 224

    def __init__(self, device: torch.device) -> None:
        open_clip = _require("open_clip", "uv pip install open_clip_torch")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("hf-hub:mkaichristensen/echo-clip")
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        from PIL import Image

        def run(batch: np.ndarray) -> Tensor:
            images = [self.preprocess(Image.fromarray(image, mode="L").convert("RGB")) for image in batch]
            return cast(Tensor, self.model.encode_image(torch.stack(images).to(self.device), normalize=True))

        return _batched(values, batch_size, run)


# ---------------------------------------------------------------------------------------------------------------------
# Clip extractors (inputs ``[N, 31, H, W]`` uint8: the generated future frames)


def _clip_tensor(clips: np.ndarray, indices: list[int], device: torch.device) -> Tensor:
    """``[N, T, H, W]`` uint8 -> float ``[N*len(indices), 1, H, W]`` in 0..1."""
    selected = np.ascontiguousarray(clips[:, indices])
    return torch.from_numpy(selected).to(device, torch.float32).flatten(0, 1)[:, None] / 255.0


def _to_video(frames: Tensor, batch: int) -> Tensor:
    """``[N*T, 1, H, W]`` -> ``[N, 3, T, H, W]``."""
    return frames.reshape(batch, -1, 1, *frames.shape[-2:]).repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)


def _import_file(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MissingExtractorError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PanEchoExtractor:
    """PanEcho video backbone (ConvNeXt-T frame encoder + temporal transformer), 768-d.

    Uses the centred 16 of the 31 frames, bilinear resize to 256 and a 224 centre crop.
    """

    name, kind, input_size = "panecho", "clip", 256

    def __init__(self, device: torch.device) -> None:
        _require("timm", "uv pip install timm torchvision")
        torchvision = _require("torchvision", "uv pip install torchvision")
        source = _weights_path("PANECHO_SOURCE", "a checkout of https://github.com/CarDS-Yale/PanEcho")
        weights = _weights_path("PANECHO_WEIGHTS", "panecho.pt from the PanEcho v1.0 GitHub release")
        module = _import_file(source / "src" / "models.py", "_panecho_models")
        original = torchvision.models.convnext_tiny
        torchvision.models.convnext_tiny = lambda *args, **kwargs: original(*args, **{**kwargs, "weights": None})
        try:
            model = module.FrameTransformer("convnext_tiny", 8, 4, 0.0, "mean", 16)
        finally:
            torchvision.models.convnext_tiny = original
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)["weights"]
        state = {key.removeprefix("encoder."): value for key, value in checkpoint.items() if key.startswith("encoder.")}
        model.load_state_dict(state)
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        start = (values.shape[1] - 16) // 2

        def run(batch: np.ndarray) -> Tensor:
            frames = F.interpolate(
                _clip_tensor(batch, list(range(start, start + 16)), self.device), size=(256, 256), mode="bilinear"
            )
            return cast(
                Tensor,
                self.model(
                    _normalized(_to_video(frames[..., 16:240, 16:240], len(batch)), IMAGENET_MEAN, IMAGENET_STD)
                ),
            )

        return _batched(values, batch_size, run)


class EchoJEPAExtractor:
    """EchoJEPA ViT-L (``vitl-vmix22m-pt220-c55``), mean of final tokens, 1024-d.

    Uses 16 uniformly spaced frames (0, 2, ..., 30), resize to 292 and a 256 centre crop.
    """

    name, kind, input_size = "echojepa", "clip", 292

    def __init__(self, device: torch.device) -> None:
        _require("einops", "uv pip install einops timm")
        source = _weights_path("ECHOJEPA_SOURCE", "a checkout of https://github.com/bowang-lab/EchoJEPA")
        weights = _weights_path("ECHOJEPA_WEIGHTS", "the EchoJEPA vitl-vmix22m-pt220-c55.pt checkpoint")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        vision_transformer = importlib.import_module("src.models.vision_transformer")
        model = vision_transformer.vit_large(
            img_size=(256, 256),
            num_frames=16,
            patch_size=16,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=True,
            use_rope=True,
            use_activation_checkpointing=False,
        )
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)["target_encoder"]
        state = {key.removeprefix("module.").removeprefix("backbone."): value for key, value in checkpoint.items()}
        model.load_state_dict(state)
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        indices = np.linspace(0, values.shape[1] - 1, 16).astype(np.int64).tolist()

        def run(batch: np.ndarray) -> Tensor:
            frames = F.interpolate(_clip_tensor(batch, indices, self.device), size=(292, 292), mode="bilinear")
            video = _normalized(_to_video(frames[..., 18:274, 18:274], len(batch)), IMAGENET_MEAN, IMAGENET_STD)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                tokens = self.model(video)
            return cast(Tensor, tokens.float().mean(dim=1))

        return _batched(values, batch_size, run)


class EchoPrimeMViTExtractor:
    """EchoPrime video encoder (torchvision MViT-v2-S with 512-d head).

    Uses frames 0, 2, ..., 30 of the clip materialized at 280 px, cropped to the central 224 px (EchoPrime's 10% zoom).
    """

    name, kind, input_size = "echoprime_mvit", "clip", 280

    def __init__(self, device: torch.device) -> None:
        torchvision = _require("torchvision", "uv pip install torchvision")
        path = _weights_path("ECHOPRIME_ENCODER", "EchoPrime model_data/weights/echo_prime_encoder.pt")
        model = torchvision.models.video.mvit_v2_s(weights=None)
        model.head[-1] = nn.Linear(model.head[-1].in_features, 512)
        model.load_state_dict(_state_dict(path))
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    def extract(self, values: np.ndarray, *, batch_size: int) -> np.ndarray:
        def run(batch: np.ndarray) -> Tensor:
            frames = torch.from_numpy(np.ascontiguousarray(batch[:, 0:31:2, 28:252, 28:252])).to(
                self.device, torch.float32
            )
            video = frames[:, :, None].repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
            return cast(Tensor, self.model(_normalized(video, ECHOPRIME_MEAN, ECHOPRIME_STD)))

        return _batched(values, batch_size, run)


EXTRACTORS: dict[str, Callable[[torch.device], Extractor]] = {
    "inception": InceptionExtractor,
    "echoprime_view": EchoPrimeViewExtractor,
    "echofine": EchoFineExtractor,
    "echoclip": EchoCLIPExtractor,
    "panecho": PanEchoExtractor,
    "echojepa": EchoJEPAExtractor,
    "echoprime_mvit": EchoPrimeMViTExtractor,
}


def build_extractor(name: str, device: torch.device) -> Extractor:
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name](device)
