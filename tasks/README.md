# Making a New Task

Tasks are discovered from local `tasks/<name>/task.py` files. A new CLI-visible task needs this minimum structure:

```text
tasks/my_task/
  task.py
  dataset.py
  train.yaml
```

`task.py` defines the task spec:

```python
from tasks.registry import TaskSpec


def loss_terms(model, sample, config):
    prediction = model(sample.frames)
    loss = ...
    return {"loss": loss}


TASK_SPEC = TaskSpec(name="my_task", loss_fn=loss_terms)
```

`dataset.py` provides dataloaders:

```python
def build_dataloaders(*, config, data_root=None):
    train_loader = ...
    val_loader = ...
    return train_loader, val_loader
```

`train.yaml` must include the sections consumed by the shared training loop: `model`, `data`, `trainer`,
`optimizer`, `loss`, and `artifacts`. Add `metrics` when validation metrics need task-specific options.

## Setup and Running

Install the library, task dependencies, and development tools from an active `uv` environment:

```bash
uv pip install --editable ".[tasks]" --requirement requirements-dev.txt
```

Run one task by name:

```bash
uv run python -m tasks.train <task> --data-root /path/to/EchoXFlow
```

For a quick smoke-style training step, limit train and validation work:

```bash
uv run python -m tasks.train <task> --data-root /path/to/EchoXFlow --max-train-steps 1 --max-val-steps 0
```

Run the benchmark matrix in smoke mode:

```bash
scripts/run_full_benchmark.sh --data-root /path/to/EchoXFlow --smoke
```

## Optional Hooks

`TaskSpec` defaults `dataset_module` to `tasks.<name>.dataset`, `config_path` to the task folder's `train.yaml`, and
`batch_size_fn` to `sample.frames.shape[0]`.

Use optional fields only when the task needs them:

- `val_metrics_fn` and `metric_keys` for evaluation metrics and benchmark output names.
- `result_factory` for task-specific result metadata.
- `preview=PreviewSpec(prediction_fn=..., writer=...)` for epoch previews. Tasks without this skip previews cleanly.
- `baseline=BaselineSpec(...)` for temporal-mean benchmark support. Tasks without it report an unsupported baseline.
- `model_factory` or a dotted model head path in config, for example `kind: my_package.my_heads.MyHead`, to avoid editing
  the built-in head registry.

Package exports such as `tasks.segmentation.run_training` require the task package to call
`bind_task_api("segmentation", globals())` in `__init__.py`. This is useful for imports, but it is not required for CLI
discovery.

Shared training behavior lives outside task folders:

- `tasks/registry.py` discovers task specs and exposes stable helpers used by `tasks/train.py`, `tasks/bench.py`, and
  package imports.
- `tasks/utils/models/` contains model implementations, currently the shared U-Net heads in `unet.py` and temporal-mean
  benchmark model in `temporal_mean.py`.
- `tasks/utils/training/` contains the common training loop, evaluation, loss helpers, logging, checkpoints, metrics, and
  epoch preview machinery.
- `tasks/utils/` contains reusable dataset, optimizer, seed, and type helpers.

## Dataset Caches

Use `data.recording_cache_dir` when repeated window sampling would otherwise duplicate the same source frames on disk.
This raw recording-array cache stores each complete pre-resize training array once under `arrays/`, then task datasets
slice their windows from the cached arrays. By default the task datasets cache only their training-relevant raw arrays:

- segmentation: `data/2d_brightness_mode*` and `data/*_contour`
- color Doppler: `data/2d_brightness_mode`, `data/2d_color_doppler_velocity`, and `data/2d_color_doppler_power`
- tissue Doppler: `data/2d_brightness_mode` and `data/tissue_doppler`

Set `data.recording_cache: true` in a task config to use the default `outputs/cache/recordings` directory. Use
`data.recording_cache_read_only: true` to read existing cached arrays without writing misses, and use
`data.recording_cache_include` or `data.recording_cache_exclude` for shell-style path patterns when a task needs a
tighter or broader set.

The benchmark launcher exposes the same recording cache controls as CLI flags: `--recording-cache-dir`,
`--recording-cache-read-only`, `--recording-cache-include`, and `--recording-cache-exclude`. These flags are forwarded to
`tasks.bench` for each benchmark cell.

The older `data.sample_cache_dir` and `--sample-cache-dir` store completed `dataset[index]` samples. They can still be
useful for deterministic preprocessed samples, but sliding-window tasks may duplicate overlapping frames across many
cached samples.

Benchmark methods and table rows are config-driven:

- `configs/benchmark_methods.yaml` defines benchmark methods such as temporal mean, frame U-Net, and temporal U-Net.
  Frame U-Net is implemented as a config transform that forces temporal kernel sizes to `1`.
- `configs/benchmark_table.yaml` defines the task/method/domain cells that `scripts/run_full_benchmark.sh` launches.
  Add rows or columns there instead of editing the launcher.

Keep reusable logic in `tasks/utils/` or `src/echoxflow/`. Add task-local `types.py`, model heads, preview writers,
benchmarks, package exports, or a README only when they are useful for that task.
