"""ECG windows for video diffusion stay valid on a reset ECG clock."""

from __future__ import annotations

import numpy as np

from tasks.video_diffusion.dataset import ecg_windows


def test_ecg_windows_match_the_monotonic_case_and_survive_a_reset() -> None:
    times = np.arange(0.0, 2.0, 1.0 / 500.0)
    signal = np.sin(times * 7.0)
    frames = np.asarray([0.5, 1.0, 1.5])
    options = {"sample_rate_hz": 600.0, "history_seconds": 0.2, "future_seconds": 0.05, "maximum_gap_samples": 4}

    windows = ecg_windows(times, signal, frames, **options)
    queries = (frames[:, None] + np.arange(-120, 31)[None, :] / 600.0).reshape(-1)
    np.testing.assert_allclose(windows[:, 0].reshape(-1), np.interp(queries, times, signal), atol=1e-6)
    assert windows[:, 1].all()

    reset = times.copy()
    reset[500:] -= 0.004  # the second acquisition chunk restarts 4 ms early, overlapping the first
    windows = ecg_windows(reset, signal, frames, **options)
    assert np.isfinite(windows).all()
    assert 0 < int((windows[:, 1] == 0).sum()) < windows[:, 1].size
