# Task 2: Estimation of Blood Velocity

<p align="center">
  <img src="../../assets/task2_color_doppler.webp" width="540">
</p>

Predict color Doppler velocity, power, and local variation from B-mode clips. The task learns blood-flow appearance from the brightness-mode sequence for the same cardiac window.

The benchmark metrics are `velocity_l1` (m/s, reported in cm/s), `power_l1`, and `variation_l1`. They are computed per
clip and averaged over examinations (see [Evaluation Metrics](../README.md#evaluation-metrics)).

```bash
uv run python -m tasks.train color_doppler --data-root /path/to/EchoXFlow
```
