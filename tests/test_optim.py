from __future__ import annotations

import io

import torch
from torch import nn

from tasks.utils.optim import AdamMuon


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))


def _step(model: nn.Module, optimizer: AdamMuon, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(5, 6, generator=generator)
    targets = torch.randn(5, 3, generator=generator)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(inputs), targets).backward()
    optimizer.step()


def test_adam_muon_resume_matches_uninterrupted_training() -> None:
    reference = _model()
    reference_optimizer = AdamMuon(reference.parameters(), lr=1e-2)
    for seed in range(3):
        _step(reference, reference_optimizer, seed)

    interrupted = _model()
    interrupted_optimizer = AdamMuon(interrupted.parameters(), lr=1e-2)
    for seed in range(2):
        _step(interrupted, interrupted_optimizer, seed)
    buffer = io.BytesIO()
    torch.save({"model": interrupted.state_dict(), "optimizer": interrupted_optimizer.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)

    resumed = _model()
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = AdamMuon(resumed.parameters(), lr=1e-2)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    assert resumed_optimizer.muon is not None and resumed_optimizer.adam is not None
    assert resumed_optimizer.muon.state is resumed_optimizer.state
    assert resumed_optimizer.adam.state is resumed_optimizer.state
    _step(resumed, resumed_optimizer, 2)

    for expected, actual in zip(reference.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_adam_muon_learning_rate_changes_reach_delegates_after_load() -> None:
    model = _model()
    optimizer = AdamMuon(model.parameters(), lr=1e-2)
    _step(model, optimizer, 0)
    optimizer.load_state_dict(optimizer.state_dict())
    for group in optimizer.param_groups:
        group["lr"] = 0.5
    assert optimizer.muon is not None and optimizer.adam is not None
    assert all(group["lr"] == 0.5 for group in optimizer.muon.param_groups + optimizer.adam.param_groups)
