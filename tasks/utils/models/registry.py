"""Registry of the dense backbones used by the EchoXFlow task benchmark.

Each entry in :data:`METHODS` names one architecture. ``build_model(backbone=<name>, ...)`` looks the name up here,
so adding a model is a matter of writing a :class:`~tasks.utils.models.backbone.DenseVideoBackbone` and registering it.

Every builder is called as ``builder(in_channels=..., out_channels=..., **options)`` and must return a module mapping
``[B,C,T,H,W]`` to ``[B,out_channels,T,H,W]``. ``options`` are written into the resolved ``model`` config section by
:func:`method_config`, so a run's ``config.yaml`` records exactly what was built. Builders are referenced by import
path and imported only when a model is built, keeping optional dependencies (timm, segmentation-models-pytorch,
torchvision) lazy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

from tasks.utils.models.backbone import DenseVideoBackbone

#: U-Net-only ``model`` keys removed when a config switches to a registered backbone.
UNET_CONFIG_KEYS = frozenset(
    {
        "channels",
        "framewise",
        "kernel_sizes",
        "num_res_units",
        "strides",
        "transpose_post_conv_kernel_size",
        "up_kernel_sizes",
    }
)


@dataclass(frozen=True)
class MethodSpec:
    """One benchmark architecture.

    ``builder`` is a ``"module:attribute"`` path to a backbone class or factory, or ``None`` for the built-in U-Net
    configured by the task's ``model`` section. ``config_transform`` optionally rewrites the full raw config first.
    """

    builder: str | None
    options: Mapping[str, object] = field(default_factory=dict)
    config_transform: str | None = None
    framewise: bool = False


_FRAMEWISE = "tasks.utils.models.framewise"
_SPATIOTEMPORAL = "tasks.utils.models.spatiotemporal"

METHODS: Mapping[str, MethodSpec] = {
    # Controls: the task U-Net as a spatiotemporal (3D) and as a framewise (2D) network.
    "temporal_unet": MethodSpec(builder=None),
    "frame_unet": MethodSpec(
        builder=None,
        config_transform="tasks.utils.models.unet:framewise_temporal_config",
        framewise=True,
    ),
    # Framewise 2D networks.
    "convnext_v2_atto": MethodSpec(
        builder=f"{_FRAMEWISE}:ConvNeXtV2AttoBackbone",
        options={"decoder_pyramid_channels": 64, "decoder_segmentation_channels": 32, "pretrained": False},
        framewise=True,
    ),
    "segformer_b0": MethodSpec(
        builder=f"{_FRAMEWISE}:SegFormerB0Backbone",
        options={"decoder_channels": 256, "pretrained": False},
        framewise=True,
    ),
    "dinov3_vit_s16": MethodSpec(
        builder=f"{_FRAMEWISE}:TimmFeaturePyramidBackbone",
        options={"model_name": "vit_small_patch16_dinov3", "decoder_channels": 64, "pretrained": False},
        framewise=True,
    ),
    "mambaout_femto": MethodSpec(
        builder=f"{_FRAMEWISE}:TimmFeaturePyramidBackbone",
        options={"model_name": "mambaout_femto", "decoder_channels": 48, "pretrained": False},
        framewise=True,
    ),
    # Spatiotemporal 3D networks.
    "video_swin_t": MethodSpec(
        builder=f"{_SPATIOTEMPORAL}:VideoSwinTinyBackbone",
        options={"decoder_channels": 32, "pretrained": False},
    ),
    "mvit_v2_s": MethodSpec(
        builder=f"{_SPATIOTEMPORAL}:MViTV2SmallBackbone",
        # Activation checkpointing only trades speed for memory here: the batch-1 activations fit at 256 px.
        options={"decoder_channels": 32, "gradient_checkpointing": False, "pretrained": False},
    ),
    # UniFormer keeps checkpointing although it would fit without: the recomputed forward pass updates the
    # BatchNorm running statistics of its local blocks a second time, so turning it off changes the trained model.
    "uniformer_s_32f": MethodSpec(
        builder=f"{_SPATIOTEMPORAL}:UniFormerSmallBackbone",
        options={
            "embed_dims": [64, 128, 320, 512],
            "depths": [3, 4, 8, 3],
            "head_dim": 64,
            "mlp_ratio": 4.0,
            "decoder_channels": 32,
            "dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
            "drop_path_rate": 0.1,
            "gradient_checkpointing": True,
            "pretrained": False,
        },
    ),
    "u_mamba_bot": MethodSpec(
        builder=f"{_SPATIOTEMPORAL}:UMambaBotBackbone",
        options={
            "backbone_channels": [16, 32, 64, 128, 192, 256],
            "backbone_kernel_sizes": [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        },
    ),
}


def method_spec(name: str) -> MethodSpec:
    try:
        return METHODS[str(name)]
    except KeyError:
        raise ValueError(f"Unknown model method {name!r}; expected one of {sorted(METHODS)}") from None


def method_config(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return ``raw`` with its ``model`` section configured for method ``name``."""
    spec = method_spec(name)
    config = deepcopy(dict(raw))
    if spec.config_transform is not None:
        config = _load_attribute(spec.config_transform)(config)
        if not isinstance(config, dict):
            raise TypeError(f"config_transform for {name!r} must return a dict")
    if spec.builder is None:
        return config
    model = config.get("model")
    if not isinstance(model, dict):
        raise TypeError(f"method {name!r} requires a mapping `model` config section")
    for key in UNET_CONFIG_KEYS:
        model.pop(key, None)
    model["backbone"] = str(name)
    model.update(deepcopy(dict(spec.options)))
    return config


def build_backbone(name: str, *, in_channels: int, out_channels: int, **options: object) -> DenseVideoBackbone:
    """Instantiate the registered backbone ``name`` with ``options`` (normally from the ``model`` config)."""
    spec = method_spec(name)
    if spec.builder is None:
        raise ValueError(f"method {name!r} uses the built-in U-Net; build it with backbone='unet'")
    builder = cast(Callable[..., DenseVideoBackbone], _load_attribute(spec.builder))
    backbone = builder(in_channels=int(in_channels), out_channels=int(out_channels), **options)
    if not isinstance(backbone, DenseVideoBackbone):
        raise TypeError(f"builder for {name!r} must return a DenseVideoBackbone, got {type(backbone).__name__}")
    return backbone


def _load_attribute(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"Invalid builder path {path!r}; expected 'module:attribute'")
    return getattr(import_module(module_name), attribute)


__all__ = ["METHODS", "UNET_CONFIG_KEYS", "MethodSpec", "build_backbone", "method_config", "method_spec"]
