"""Distribution distances (Frechet distance, kernel inception distance) with optional integer row weights.

Integer weights implement cluster-bootstrap replicates without materializing duplicated rows: a row with weight ``w``
behaves exactly like ``w`` copies of that row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from torch import Tensor


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary values (independent of Python's hash randomization)."""
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _tensor(values: np.ndarray | Tensor, device: torch.device) -> Tensor:
    return torch.as_tensor(values, device=device, dtype=torch.float64)


def _moment_factor(features: Tensor, weights: Tensor | None) -> tuple[Tensor, Tensor]:
    """Return the mean and a factor ``F`` with ``F.T @ F`` equal to the unbiased (weighted) covariance."""
    if weights is None:
        weights = torch.ones(len(features), dtype=features.dtype, device=features.device)
    keep = weights > 0
    features, weights = features[keep], weights[keep]
    total = weights.sum()
    mean = (weights[:, None] * features).sum(0) / total
    factor = (features - mean) * weights.sqrt()[:, None] / torch.sqrt(total - 1.0)
    return mean, factor


def _trace_sqrt_product(real_factor: Tensor, generated_factor: Tensor) -> Tensor:
    """``tr sqrt(S_r^1/2 S_g S_r^1/2)`` for ``S = F.T @ F``, via the smaller of the Gram or covariance forms."""
    dimension = real_factor.shape[1]
    if min(len(real_factor), len(generated_factor)) <= dimension:
        cross = generated_factor @ real_factor.T
        gram = cross @ cross.T if len(cross) <= cross.shape[1] else cross.T @ cross
        return cast(Tensor, torch.linalg.eigvalsh(gram).clamp_min(0.0).sqrt().sum())
    real_covariance = real_factor.T @ real_factor
    values, vectors = torch.linalg.eigh(real_covariance)
    root = (vectors * values.clamp_min(0.0).sqrt()) @ vectors.T
    product = root @ (generated_factor.T @ generated_factor) @ root
    return cast(Tensor, torch.linalg.eigvalsh((product + product.T) / 2).clamp_min(0.0).sqrt().sum())


def frechet_distance(
    real: np.ndarray | Tensor,
    generated: np.ndarray | Tensor,
    *,
    real_weights: np.ndarray | Tensor | None = None,
    generated_weights: np.ndarray | Tensor | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """``||mu_r - mu_g||^2 + tr S_r + tr S_g - 2 tr sqrt(S_r^1/2 S_g S_r^1/2)`` with unbiased covariances."""
    device = torch.device(device)
    real_mean, real_factor = _moment_factor(
        _tensor(real, device), None if real_weights is None else _tensor(real_weights, device)
    )
    generated_mean, generated_factor = _moment_factor(
        _tensor(generated, device), None if generated_weights is None else _tensor(generated_weights, device)
    )
    value = (
        (real_mean - generated_mean).square().sum()
        + real_factor.square().sum()
        + generated_factor.square().sum()
        - 2.0 * _trace_sqrt_product(real_factor, generated_factor)
    )
    return max(float(value), 0.0)


def polynomial_kernel(left: Tensor, right: Tensor) -> Tensor:
    """KID kernel ``(x.y / d + 1)^3``."""
    return (left @ right.T / left.shape[1] + 1.0) ** 3


def kid_subsets(count: int, *, subsets: int, subset_size: int, seed: int, namespace: str) -> list[np.ndarray]:
    """Fixed random subsets (without replacement) of ``range(count)``, one RNG stream per subset."""
    size = min(subset_size, count)
    return [
        np.sort(np.random.default_rng(stable_seed(seed, namespace, repeat)).choice(count, size=size, replace=False))
        for repeat in range(subsets)
    ]


def _weighted_offdiagonal_mean(kernel: Tensor, weights: Tensor, diagonal: Tensor) -> Tensor | None:
    total = weights.sum()
    if total < 2:
        return None
    return (weights @ kernel @ weights - (weights * diagonal).sum()) / (total * (total - 1.0))


def paired_kid(
    real: np.ndarray | Tensor,
    generated: np.ndarray | Tensor,
    subsets: Sequence[np.ndarray],
    *,
    weights: np.ndarray | Tensor | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Unbiased KID where row ``i`` of ``generated`` is paired with row ``i`` of ``real``.

    The cross term excludes the matched pairs (the diagonal), mirroring the within-set terms. Averaged over subsets.
    """
    device = torch.device(device)
    real_tensor, generated_tensor = _tensor(real, device), _tensor(generated, device)
    weight_tensor = torch.ones(len(real_tensor), device=device, dtype=torch.float64)
    if weights is not None:
        weight_tensor = _tensor(weights, device)
    values = []
    for subset in subsets:
        index = torch.as_tensor(subset, device=device)
        r, g, w = real_tensor[index], generated_tensor[index], weight_tensor[index]
        terms = []
        for kernel in (polynomial_kernel(r, r), polynomial_kernel(g, g), polynomial_kernel(r, g)):
            terms.append(_weighted_offdiagonal_mean(kernel, w, torch.diagonal(kernel)))
        real_term, generated_term, cross_term = terms
        if real_term is None or generated_term is None or cross_term is None:
            continue
        values.append(float(real_term + generated_term - 2.0 * cross_term))
    return float(np.mean(values)) if values else float("nan")


def unpaired_kid(
    real: np.ndarray | Tensor,
    generated: np.ndarray | Tensor,
    *,
    subsets: int,
    subset_size: int,
    seed: int,
    namespace: str,
    real_weights: np.ndarray | None = None,
    generated_weights: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Unbiased KID between independent sets, averaged over random subsets drawn separately from each set."""
    device = torch.device(device)
    real_tensor, generated_tensor = _tensor(real, device), _tensor(generated, device)
    real_w = torch.ones(len(real_tensor), device=device, dtype=torch.float64)
    generated_w = torch.ones(len(generated_tensor), device=device, dtype=torch.float64)
    if real_weights is not None:
        real_w = _tensor(real_weights, device)
    if generated_weights is not None:
        generated_w = _tensor(generated_weights, device)
    real_subsets = kid_subsets(
        len(real_tensor), subsets=subsets, subset_size=subset_size, seed=seed, namespace=f"{namespace}:real"
    )
    generated_subsets = kid_subsets(
        len(generated_tensor), subsets=subsets, subset_size=subset_size, seed=seed, namespace=f"{namespace}:generated"
    )
    values = []
    for real_subset, generated_subset in zip(real_subsets, generated_subsets, strict=True):
        r, rw = real_tensor[real_subset], real_w[real_subset]
        g, gw = generated_tensor[generated_subset], generated_w[generated_subset]
        kernel_rr, kernel_gg = polynomial_kernel(r, r), polynomial_kernel(g, g)
        rr = _weighted_offdiagonal_mean(kernel_rr, rw, torch.diagonal(kernel_rr))
        gg = _weighted_offdiagonal_mean(kernel_gg, gw, torch.diagonal(kernel_gg))
        if rr is None or gg is None:
            continue
        cross = (rw @ polynomial_kernel(r, g) @ gw) / (rw.sum() * gw.sum())
        values.append(float(rr + gg - 2.0 * cross))
    return float(np.mean(values)) if values else float("nan")
