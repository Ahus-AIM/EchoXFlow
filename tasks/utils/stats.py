"""Shared cluster-bootstrap sampling and two-sided tests with Holm correction."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def _rng(seed: int, namespace: str) -> np.random.Generator:
    words = np.frombuffer(hashlib.sha256(f"{seed}:{namespace}".encode()).digest()[:16], dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence((seed, *words.tolist())))


def bootstrap_multiplicities(count: int, *, replicates: int, seed: int, namespace: str) -> np.ndarray:
    """``[replicates, count]`` cluster multiplicities from resampling ``count`` clusters with replacement."""
    draws = _rng(seed, namespace).integers(0, count, size=(replicates, count))
    return np.stack([np.bincount(row, minlength=count) for row in draws]).astype(np.float64)


def seeded_permutation(count: int, *, seed: int, namespace: str) -> np.ndarray:
    """A permutation of ``range(count)`` fixed by ``seed`` and ``namespace``."""
    return _rng(seed, namespace).permutation(count)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family = all supplied values)."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    count = len(values)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(values[order] * (count - np.arange(count))))
    adjusted = np.empty(count)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def recentered_p_value(replicates: np.ndarray, observed: float) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0 from the recentered replicate distribution."""
    extreme = np.count_nonzero(np.abs(replicates - observed) >= abs(observed))
    return float((extreme + 1.0) / (len(replicates) + 1.0))


def bootstrap_mean(
    values: Sequence[float] | np.ndarray, *, replicates: int = 2000, seed: int = 17, namespace: str = "exam"
) -> dict[str, float]:
    """Equal-cluster mean, percentile CI and recentered two-sided p-value; pass one value per cluster."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all() or replicates < 1:
        raise ValueError("bootstrap needs at least two finite cluster values and positive replicates")
    observed = float(array.mean())
    draws = (
        bootstrap_multiplicities(len(array), replicates=replicates, seed=seed, namespace=namespace) @ array / len(array)
    )
    return {
        "mean": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "p_two_sided": recentered_p_value(draws, observed),
    }
