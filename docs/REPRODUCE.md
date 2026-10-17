# Reproducing the EchoXFlow results

The [README](../README.md#reproduce-our-results) gives the pipeline and dataset-statistics commands. This page lists the individual commands behind them, the evaluation protocol and the disk footprint.

## Commands per result

Install the task dependencies first (see [Installation](../README.md#installation)).

Set `DATA` to the dataset, `CAMUS` to the [CAMUS EchoXFlow package](https://huggingface.co/datasets/Ahus-AIM/CAMUS-beamspace)
(downloaded separately), and `BENCH` and `DIFF` to output directories with enough space. Run the architecture and
domain commands once per resolution (`R` = 128, 256, 320).

| Result | Reproduction command | Output |
|---|---|---|
| Every trained or evaluated model below at 128/256/320 with 5-fold CV, one resolution at a time | `scripts/run_paper_pipeline.sh --data-root "$DATA" --camus-root "$CAMUS" --output-dir "$OUT"` | `$OUT/{benchmark,video_diffusion,foundation_model_ef}`, commit in `$OUT/pipeline_runs.jsonl` |
| Architecture tables | `uv run python scripts/run_full_benchmark.py --data-root "$DATA" --output-dir "$BENCH" --run-name paper --paper --cv --resolution "$R" --gpus 0 1` | `$BENCH/tables/paper_r$R.{tex,csv}` |
| Domain comparison | `uv run python scripts/build_benchmark_table.py --root "$BENCH" --run-name "paper_r$R" --resolution "$R" --domain-comparison --output-dir "$BENCH/tables"` | `$BENCH/tables/paper_r$R.domain_comparison.r$R.{csv,tex}` |
| Task 4, EchoXFlow and CAMUS | `uv run python scripts/reproduce_video_diffusion.py --data-root "$DATA" --camus-root "$CAMUS" --output-dir "$DIFF"` | `$DIFF/tables{,_camus}/comparison_cv.{tex,csv,md,json}` (5-fold), `$DIFF/fold<k>/tables{,_camus}/comparison.{csv,md,json}` |
| Task 3, CAMUS | `uv run python -m tasks.segmentation.camus --package-root "$CAMUS" --benchmark-root "$BENCH" --output-dir "$BENCH/camus"` | `$BENCH/camus/per_recording_r<R>_<domain>.csv` |
| Foundation-model EF | `uv run --extra diffusion-eval python scripts/foundation_model_ef.py --data-root "$DATA" --output-dir outputs/foundation_model_ef` | `outputs/foundation_model_ef/{per_exam.csv,summary.csv,summary.tex,cohort.json}` |
| Dataset statistics and views per recording | `uv run python scripts/croissant_summary_table.py "$DATA/croissant.json" --view-csv "$DATA/view_classifier_probabilities.csv"` | `outputs/tables/croissant_summary_table{.tex,_report.json,_views.tex,_views.csv}` |
| Loss curves | `uv run python scripts/plot_benchmark_losses.py --benchmark-root "$BENCH" --run-name paper --resolution "$R"` | `outputs/figures/*_r$R.{png,csv}` |

Domain tables exclude the temporal-mean baseline, count wins over method × available fold pairs, and pool out-of-fold
clips into exam-equal values. Differences are beamspace minus scan-converted in each metric's display units. The paired
exam bootstrap uses 2,000 replicates and Holm correction across methods within each resolution and task metric.
Partial smoke tables may use `--allow-missing`. Their counts reflect only the available comparisons.

Generate the dataset statistics table (`tab:modality-stats`) and the views-per-recording table (`tab:view-stats`)
with [scripts/croissant_summary_table.py](../scripts/croissant_summary_table.py):

```bash
uv run python scripts/croissant_summary_table.py "$DATA/croissant.json" \
  --view-csv "$DATA/view_classifier_probabilities.csv"
```

The script reads the Croissant record sets, plus the `.zattrs` of annotated recordings and a few small timestamp
arrays for rows the catalog cannot resolve. Bytes per item are uncompressed array sizes. Views are the classes on which
EchoPrime and EchoFine agree. 2D annotation panels without a view of their own take it from their AHA view timelines,
then from the view classifier. `croissant_summary_table_report.json` records how panels were resolved and which were
left out.

## Tasks 1-3

[scripts/run_full_benchmark.py](../scripts/run_full_benchmark.py) trains every row of
[configs/benchmark_table.yaml](../configs/benchmark_table.yaml) on every task, domain and validation fold, skips finished
cells, and builds the table. `--paper` applies the article protocol. Run it once per resolution:

```bash
uv run python scripts/run_full_benchmark.py --data-root /path/to/EchoXFlow --output-dir /path/with/space \
  --paper --cv --resolution 256 --gpus 0 1 2
uv run python scripts/run_full_benchmark.py --data-root /path/to/EchoXFlow --output-dir /path/with/space \
  --paper --cv --resolution 128 --gpus 0 1 2
```

The runs take days. Rerunning the same command continues where it stopped. `OUTPUT_ROOT/status/RUN_NAME.txt` shows
every cell. Disk: about 19 GB per resolution.

The paper protocol, applied identically to every method:

- Five-fold cross-validation over examinations. Each fold is evaluated once, with weights chosen without looking at it.
- Checkpoints are selected by early stopping (patience 3, at most 25 epochs) on an inner split of 12.5% of the
  training-fold examinations, the same for every method.
- Metrics are computed per clip on unpadded clips, averaged within each examination and then over examinations, with
  the same temporal evaluation window for all models (see [tasks/README.md](../tasks/README.md#evaluation-metrics)).
- Tables report the mean over folds ± SD across folds, plus paired per-fold differences to the 2D U-Net.
- Each run writes a manifest with the code revision, dataset version and configuration hash.

Use `--smoke` for a quick end-to-end check on 10% of the data.

## Task 4

[scripts/reproduce_video_diffusion.py](../scripts/reproduce_video_diffusion.py) trains beamspace and Cartesian
models at 128, 256 and 320 pixels for each of the five held-out folds, evaluates them on the held-out fold, builds the
bootstrap comparison table per fold and pools the folds into the cross-validated table, for EchoXFlow and for CAMUS
(`--camus-root`, or `ECHOXFLOW_CAMUS_ROOT`). The feature extractor weights are listed in
[tasks/video_diffusion/PROTOCOL.md](../tasks/video_diffusion/PROTOCOL.md#evaluation).

```bash
uv run python scripts/reproduce_video_diffusion.py --data-root /path/to/EchoXFlow --output-dir /path/with/space
```

`OUTPUT_ROOT/status/video_diffusion.txt` shows the state of every stage. Disk: about 15 GB for the thirty paper runs.

## Foundation-model EF

Compare EchoCLIP, PanEcho and EchoPrime against the largest and smallest closed LV
mesh volumes per recording. Install `uv sync --extra tasks --extra diffusion-eval` and follow the
[feature-extractor weight instructions](../tasks/video_diffusion/PROTOCOL.md#evaluation). EchoPrime also
needs its official checkout (`ECHOPRIME_SOURCE`, containing `assets/MIL_weights.csv`) and unpacked release
`model_data/` (`ECHOPRIME_MODEL_DATA`, containing `candidates_data/`). The shared encoder environment variables
`ECHOPRIME_ENCODER`, `ECHOPRIME_VIEW_CLASSIFIER`, `PANECHO_SOURCE`, and `PANECHO_WEIGHTS` apply here too.

```bash
uv run python scripts/foundation_model_ef.py --data-root /path/to/EchoXFlow \
  --view-csv view_classifier_probabilities.csv --output-dir outputs/foundation_model_ef
```

Add `--max-exams 10` for a smoke, and `--tables-only` to rebuild `summary.csv` and the paper table `summary.tex`
from an existing `per_exam.csv`. Relative classifier CSV paths resolve against the data root. Both classifiers must
agree on A2C/A4C. Select the recording with the highest geometric-mean confidence for each view. The command reports
actual mesh and paired-view cohort counts in `cohort.json`, predictions and references in `per_exam.csv`, and metrics
in `summary.csv` and `summary.tex`: MAE with an exam-bootstrap 95% CI (2,000 replicates), RMSE, prediction-minus-
mesh bias, Pearson r, and LOOCV bias-corrected MAE. All EF values are percentage points. Summaries cover each view and
the two-view estimate. PanEcho and EchoCLIP average the view estimates. EchoPrime uses its LV-section weighted study
retrieval. No labels from the evaluation cohort train these models.

## Reuse and completion checks

Task 4 validates training and evaluation provenance on every launch, including
existing completion markers. Evaluation reuse checks the checkpoint contents,
configuration, metadata and split contents, implementation, and actual loaded
extractor weights. Missing requested extractors fail by default. Explicit
`--allow-partial` evaluation writes `metrics.partial.json` and exits nonzero;
it is not a completed paper evaluation.

Existing Task 4 artifacts without `provenance.json`, or artifacts whose identity
has changed, require a new output directory. They are not silently adopted or
overwritten. Completed training also verifies the EMA checkpoint digest;
completed evaluation verifies required feature files and their digests.
Reruns load the extractors to verify their weights even when evaluation is complete.
Single-space runs train and evaluate without planning a two-space comparison.
The paper pipeline exits nonzero if any phase or table generation fails.

Dataset exports must remain immutable during a run. Provenance hashes catalog,
split and view metadata, and records the payload file inventory (sizes and
modification times, plus content hashes for files up to 1 MiB). This detects
ordinary local edits without rereading the entire corpus; it is not a full
cryptographic verification of large payloads. Verify publisher checksums when
acquiring or transferring data. Checkpoint and extractor weights are hashed in full.

Permission changes alone do not invalidate the payload inventory. Preserve file
modification times when copying an export; a different dataset root is still a
different run identity. Source identity deliberately includes reader, preprocessing
and shared task code, not just the model: changes there can change results.
Plotting-only edits may therefore require a fresh output directory too.

To evaluate a legacy checkpoint without modifying its original results, run
`tasks.video_diffusion.evaluate` with `--checkpoint` and a new `--output-dir`.
The new evaluation records current provenance; it does not certify the original
training run or automatically reproduce historical paper tables. Never fabricate
`provenance.json` for old artifacts.

Provenance checks scan the full export on each launch, including completed runs.
They may take minutes and several GiB of memory on large exports, before training
or evaluation starts. Small payload files are read in full; plan for that I/O even
when reusing results. Do not run many launch-time checks concurrently on shared storage.
