from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import overload

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(gradient: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    if gradient.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization expects a 2D tensor")
    a, b, c = (3.4445, -4.7750, 2.0315)
    update = gradient.bfloat16()
    update /= update.norm() + eps
    transposed = gradient.size(0) > gradient.size(1)
    if transposed:
        update = update.T
    for _ in range(int(steps)):
        gram = update @ update.T
        correction = b * gram + c * gram @ gram
        update = a * update + correction @ update
    return update.T if transposed else update


class Muon(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, momentum: float, weight_decay: float) -> None:
        defaults = {"lr": float(lr), "momentum": float(momentum)}
        super().__init__(params, defaults)
        self.weight_decay = float(weight_decay)

    @overload
    def step(self, closure: None = ...) -> None: ...  # noqa: E704

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...  # noqa: E704

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                gradient = gradient.add(buffer, alpha=momentum)
                update = zeropower_via_newtonschulz5(gradient.flatten(1)).reshape_as(gradient)
                if self.weight_decay != 0.0:
                    parameter.data.mul_(1.0 - lr * self.weight_decay)
                parameter.data.add_(update, alpha=-lr)
        return loss


class AdamMuon(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | list[dict[str, object]],
        lr: float = 1e-3,
        muon_momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ) -> None:
        muon_params, adam_params = _split_muon_adam_params(params)
        self.muon = Muon(muon_params, lr=lr, momentum=muon_momentum, weight_decay=weight_decay) if muon_params else None
        self.adam = AdamW(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) if adam_params else None

        param_groups = []
        if self.muon is not None:
            for group in self.muon.param_groups:
                group["optimizer"] = "muon"
                param_groups.append(group)
        if self.adam is not None:
            for group in self.adam.param_groups:
                group["optimizer"] = "adam"
                param_groups.append(group)

        defaults = {"lr": lr, "momentum": muon_momentum, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(param_groups, defaults)
        if self.muon is not None:
            self.muon.state = self.state
        if self.adam is not None:
            self.adam.state = self.state

    @overload
    def step(self, closure: None = ...) -> None: ...  # noqa: E704

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...  # noqa: E704

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        if self.muon is not None:
            self.muon.step()
        if self.adam is not None:
            self.adam.step()
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam is not None:
            self.adam.zero_grad(set_to_none=set_to_none)


def _split_muon_adam_params(params: Iterable[Tensor] | list[dict[str, object]]) -> tuple[list[Tensor], list[Tensor]]:
    muon_params: list[Tensor] = []
    adam_params: list[Tensor] = []
    parameter_groups = list(params)
    if parameter_groups and isinstance(parameter_groups[0], dict):
        for group in parameter_groups:
            if not isinstance(group, Mapping):
                raise TypeError("Optimizer parameter groups must all be dictionaries")
            group_params = group.get("params")
            if not isinstance(group_params, Iterable):
                raise TypeError("Optimizer parameter group `params` must be iterable")
            for parameter in group_params:
                _append_by_rank(parameter, muon_params=muon_params, adam_params=adam_params)
        return muon_params, adam_params
    for parameter in parameter_groups:
        _append_by_rank(parameter, muon_params=muon_params, adam_params=adam_params)
    return muon_params, adam_params


def _append_by_rank(parameter: object, *, muon_params: list[Tensor], adam_params: list[Tensor]) -> None:
    if not isinstance(parameter, Tensor):
        raise TypeError(f"Expected torch.Tensor optimizer parameter, got {type(parameter).__name__}")
    if parameter.ndim > 1:
        muon_params.append(parameter)
    else:
        adam_params.append(parameter)


__all__ = ["AdamMuon", "Muon", "zeropower_via_newtonschulz5"]
