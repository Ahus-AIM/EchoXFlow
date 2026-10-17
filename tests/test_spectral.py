from __future__ import annotations

import pytest

from echoxflow.spectral import spectral_metadata_from_attrs


def test_spectral_metadata_preserves_shifted_baseline_and_nyquist_limits() -> None:
    metadata = spectral_metadata_from_attrs(
        "data/1d_pulsed_wave_doppler",
        {
            "recording_manifest": {
                "tracks": [
                    {
                        "semantic_id": "1d_pulsed_wave_doppler",
                        "nyquist_mps": 2.0,
                        "spectral_row_baseline_frac": 0.25,
                    }
                ]
            }
        },
        row_count=5,
    )

    assert metadata.nyquist_limit_mps == pytest.approx(2.0)
    assert metadata.baseline_row == pytest.approx(1.0)
    assert metadata.row_velocity_mps is not None
    assert metadata.row_velocity_mps.tolist() == pytest.approx([2.0, 0.0, -2.0 / 3.0, -4.0 / 3.0, -2.0])
