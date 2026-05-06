from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from echoxflow.scan import SectorGeometry
from tasks.tissue_doppler.roi import sampling_gate_roi_mask, tissue_doppler_sampling_gate_metadata


def test_sampling_gate_roi_mask_rasterizes_sample_volume_box() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=0.08,
        tilt_rad=0.0,
        width_rad=1.0,
    )
    mask = sampling_gate_roi_mask(
        {
            "gate_center_depth_m": 0.04,
            "gate_tilt_rad": 0.0,
            "gate_sample_volume_m": 0.02,
        },
        geometry=geometry,
        target_shape=(9, 9),
    )

    assert mask is not None
    assert mask.shape == (9, 9)
    assert mask[4, 4]
    assert not mask[0, 4]
    assert not mask[4, 0]
    assert np.count_nonzero(mask) > 1


def test_sampling_gate_roi_mask_clamps_to_sector_bounds() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=0.08,
        tilt_rad=0.0,
        width_rad=0.5,
    )
    mask = sampling_gate_roi_mask(
        {
            "gate_center_depth_m": 0.005,
            "gate_tilt_rad": -0.25,
            "gate_sample_volume_m": 0.02,
        },
        geometry=geometry,
        target_shape=(9, 9),
    )

    assert mask is not None
    assert mask[0, 0]
    assert not mask[-1, -1]


def test_sampling_gate_roi_mask_rejects_invalid_metadata() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=0.08,
        tilt_rad=0.0,
        width_rad=0.5,
    )

    assert sampling_gate_roi_mask({}, geometry=geometry, target_shape=(9, 9)) is None
    assert (
        sampling_gate_roi_mask(
            {
                "gate_center_depth_m": 0.04,
                "gate_tilt_rad": 0.0,
                "gate_sample_volume_m": 0.0,
            },
            geometry=geometry,
            target_shape=(9, 9),
        )
        is None
    )


def test_tissue_doppler_sampling_gate_metadata_reads_sector_gate() -> None:
    gate = {
        "kind": "tissue_doppler_gate",
        "gate_center_depth_m": 0.04,
        "gate_tilt_rad": 0.0,
        "gate_sample_volume_m": 0.01,
    }
    store = SimpleNamespace(
        group=SimpleNamespace(
            attrs={
                "recording_manifest": {
                    "sectors": [
                        {"semantic_id": "bmode"},
                        {"semantic_id": "tissue_doppler", "sampling_gate_metadata": gate},
                    ]
                }
            }
        )
    )

    assert tissue_doppler_sampling_gate_metadata(cast(Any, store)) == gate


def test_roi_valid_mask_limits_velocity_band_l1_loss_to_selected_pixels() -> None:
    pytest.importorskip("monai")

    from tasks.utils.training.loss import masked_velocity_band_l1_loss

    prediction = torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    target[..., 1, 1] = 2.0
    limit = torch.tensor([2.0], dtype=torch.float32)
    roi_mask = torch.zeros((1, 1, 1, 3, 3), dtype=torch.float32)
    roi_mask[..., 1, 1] = 1.0

    full_loss = masked_velocity_band_l1_loss(prediction, target, limit)
    roi_loss = masked_velocity_band_l1_loss(prediction, target, limit, valid_mask=roi_mask)

    assert torch.isclose(full_loss, torch.tensor(1.0 / 9.0))
    assert torch.isclose(roi_loss, torch.tensor(1.0))
