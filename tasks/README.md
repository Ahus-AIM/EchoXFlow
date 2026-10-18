# Benchmark Tasks

Tasks 1-3 (tissue velocity, blood flow, LV segmentation) share one training and evaluation pipeline. Task 4 (video
diffusion) has its own, see [video_diffusion/README.md](video_diffusion/README.md).

## Running

Install the task dependencies with `uv sync --extra tasks`, then train one task on the five-fold `splits.csv` of the
data root (fold 4 held out by default):

```bash
uv run python -m tasks.train <task> --data-root /path/to/EchoXFlow
uv run python -m tasks.train <task> --data-root /path/to/EchoXFlow --max-train-steps 1 --max-val-steps 0  # smoke
uv run python scripts/run_full_benchmark.py --data-root /path/to/EchoXFlow --output-dir /path/with/space --smoke
```

`tasks.bench` runs one benchmark cell (task, method, domain, fold) and `--protocol-profile echoxflow_paper_v1` applies
the article's settings from [configs/protocols/echoxflow_paper_v1.yaml](../configs/protocols/echoxflow_paper_v1.yaml).
The benchmark methods are defined in [configs/benchmark_methods.yaml](../configs/benchmark_methods.yaml) and the table
cells in [configs/benchmark_table.yaml](../configs/benchmark_table.yaml). The launcher and the paper protocol are
described in [docs/REPRODUCE.md](../docs/REPRODUCE.md#tasks-1-3).

Training recordings and discovery results are cached under `outputs/cache/`. Both caches are safe to delete.

## Making a New Task

A task is a folder with three files:

```text
tasks/my_task/
  task.py       # TASK_SPEC = TaskSpec(name="my_task", loss_fn=loss_terms)
  dataset.py    # build_dataloaders(*, config, data_root=None) -> (train, val, selection)
  train.yaml    # model, data, trainer, optimizer, loss, artifacts (and metrics, if needed)
```

`loss_terms(model, sample, config)` returns a dict with at least `"loss"`. Optional `TaskSpec` fields add validation
metrics (`val_metrics_fn`, `metric_keys`), epoch previews (`preview`), the temporal-mean baseline (`baseline`) and a
custom model head (`model_factory`). See [registry.py](registry.py). The shared training loop, models and dataset
helpers live in [utils/](utils/).

## Model Selection

Training never sees the reported validation fold. A fixed 12.5% of the training-fold examinations (chosen by a seeded
hash, the same for every method) form a selection split. After every epoch the selection loss is evaluated as an
examination-equal mean of per-clip losses, the best epoch's `weights.pt` is kept, and training stops after 3 epochs
without improvement (at most 25 epochs). `early_stopping.json` records the split and the stopping state.

## Evaluation Metrics

Every metric is defined per clip and averaged within each examination, then over examinations (`val_metrics.json`,
per-clip values in `val_clip_metrics.csv`). L1 metrics are masked means over a clip's valid pixels. Dice is computed per
clip and channel and averaged over the structures present. The segmentation headline `dice_mean` is the thresholded
(argmax) Dice. Clips without valid pixels are left out of a metric, not scored 0.

Clips longer than the training clip are predicted in sliding windows of the training length, starting every half
window, and overlapping predictions are averaged, so every method sees the same temporal context. Tissue Doppler is
scored on the Cartesian metric grid inside the scan sector in both domains. Its headline `velocity_l1` is the absolute
error in m/s after wrapping into one Nyquist period.

## Models

| Type | Method | Name in configs |
| --- | --- | --- |
| 2D control | Framewise U-Net | `frame_unet` |
| 2D | ConvNeXt V2 Atto + FPN | `convnext_v2_atto` |
| 2D | SegFormer-B0 | `segformer_b0` |
| 2D | DINOv3 ViT-S/16 + dense decoder | `dinov3_vit_s16` |
| 2D | MambaOut-Femto + FPN | `mambaout_femto` |
| 3D control | Spatiotemporal U-Net | `temporal_unet` |
| 3D | Video Swin-T | `video_swin_t` |
| 3D | MViT-v2-S | `mvit_v2_s` |
| 3D | UniFormer-S (32 frames) | `uniformer_s_32f` |
| 3D | U-Mamba-Bot-style | `u_mamba_bot` |

All models are trained from random initialisation with an effective batch of four clips. 2D models process each frame
independently. 3D models see the whole clip.

## Adding a Model

1. Implement a `DenseVideoBackbone` ([utils/models/backbone.py](utils/models/backbone.py)) mapping `[B,C,T,H,W]` to
   `[B,out_channels,T,H,W]`; for a 2D network, subclass `FramewiseDenseVideoBackbone` and implement `forward_frames`.
2. Register it in `METHODS` ([utils/models/registry.py](utils/models/registry.py)), for example
   `"my_net": MethodSpec(builder="tasks.utils.models.my_module:MyNetBackbone", options={"width": 64})`.
3. Add `my_net: {kind: trainable}` to [configs/benchmark_methods.yaml](../configs/benchmark_methods.yaml).

`tasks.bench --method my_net` then trains it, and saved weights reload from the run directory alone.
