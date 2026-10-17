# Video diffusion technical reference

Task overview and examples: [README.md](README.md). Defaults: [train.yaml](train.yaml).

## Data

Recordings are read through `echoxflow`: B-mode from `data/2d_brightness_mode`, ECG from `data/ecg`. A recording is
used when it is an apical A2C/A3C/A4C view, its sector starts at the probe (`depth_start <= 1e-5 m`), its frame rate is
at least 25 Hz (above 50 Hz every second frame is used) and it holds one 32-frame clip inside the ECG time span.

Frames are scaled to `[0, 1]` and resized bilinearly to `S x S` in beamspace. Cartesian frames are then scan converted
onto a square canvas spanning `[0, depth]` axially and `[-depth/2, depth/2]` laterally. The ECG is normalized per
recording (median, 1.4826 x MAD, clipped at 5, scaled to `[-1, 1]`) and resampled at 600 Hz from 1.2 s before to 0.2 s
after each frame. Each epoch draws one clip per training recording with a random start and a random ECG polarity flip
(p = 0.5).

Views come from `view_classifier_probabilities.csv` next to the dataset (EchoPrime top-1 A2C/A3C/A4C with probability
>= 0.85, regenerate with [scripts/build_view_classifier_csv.py](../../scripts/build_view_classifier_csv.py)).
`data.view_source: scanner_view_detector` falls back to the scanner's detector (classes `0 -> A2C`, `1 -> A4C`,
`3 -> A3C`), which selects a different set of recordings.

## Model

A factorized spatiotemporal 3D U-Net ([model.py](model.py), 43.0 M parameters):

- 2x2 pixel rearrangement: `1 x S x S` frames become `4 x S/2 x S/2`.
- Three encoder, two bottleneck and three decoder residual blocks, widths 128/256/512, factorized `1x3x3` spatial and
  `3x1x1` temporal convolutions, spatial skip connections.
- Diffusion time and per-frame ECG embeddings (512-d, added in every block). The ECG embedding is a residual 1D CNN
  (channels 16/32/64, kernel 15, 8 pooled samples). The clean first frame is concatenated to every future frame.
- Cosine schedule, 1000 steps, v-prediction MSE. AdamMuon, learning rate 3.7e-3, no weight decay, batch 1, 10 epochs,
  bf16. EMA decay 0.995. Sampling uses the EMA weights with 32 DDIM steps.

## Training

```bash
uv run python -m tasks.video_diffusion.train --data-root /path/to/EchoXFlow \
  --coordinate-space beamspace --resolution 128 --evaluation-fold 4
```

`--coordinate-space cartesian` trains the scan-converted arm; `--resolution` is 128, 256 or 320. A run writes
`config.yaml`, `history.jsonl`, `latest.pt`, `ema.pt` and `training_complete.json` to
`outputs/runs/video_diffusion/<space>_<size>_fold<k>_<time>/` or `--output-dir`; `--resume <run>/latest.pt` continues
a run, and evaluation uses `ema.pt`. `--max-recordings 6 --max-steps 300` is a quick check. One step takes about
0.25 s at 256 px on an RTX 5090 (15 GB). `model.activation_checkpointing: true` halves the memory.

## Evaluation

```bash
uv run python -m tasks.video_diffusion.evaluate --checkpoint <run>/ema.pt --data-root /path/to/EchoXFlow
```

68 held-out clips per view (one per recording, deterministic) are generated with 32 steps from per-clip fixed noise.
Generated and real frames are mapped onto each extractor's canvas (beamspace scan converted, Cartesian resized, real
frames re-read from the native data). FD is computed per view over all 2108 frames, paired KID over clips (100 subsets
of 32, kernel `(x.y/d + 1)^3`); both are averaged over the three views. Outputs under `<run>/evaluation/`:
`features/<extractor>.npz`, `metrics.csv`, `metrics.json` (written last) and, with `--keep-samples`, `samples.npz`.

Extractors whose package or weights are missing are skipped. `--extractors` selects a subset.

| Extractor | Input | Package | Weights |
|---|---|---|---|
| `inception` | 299 px frames | `torch-fidelity` | downloaded by torch-fidelity into `$TORCH_HOME` |
| `echoprime_view` | 224 px frames | `torchvision` | `ECHOPRIME_VIEW_CLASSIFIER=model_data/weights/view_classifier.pt` from the [EchoPrime v1.0.0 release](https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip) |
| `echofine` | 224 px frames | `h5py`, `huggingface_hub` | [`thrive-2025/EchoView47`](https://huggingface.co/thrive-2025/EchoView47) `TTE47_Classifier_Model.h5`, downloaded automatically, or `ECHOFINE_WEIGHTS` |
| `echoclip` | 224 px frames | `open_clip_torch` | [`mkaichristensen/echo-clip`](https://huggingface.co/mkaichristensen/echo-clip), downloaded automatically |
| `panecho` | 256 px clips | `timm`, `torchvision` | `PANECHO_SOURCE` = clone of [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho) (`836c46ef`); `PANECHO_WEIGHTS` = [`panecho.pt`](https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt) |
| `echojepa` | 292 px clips | `einops`, `timm` | `ECHOJEPA_SOURCE` = clone of [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA) (`77208e1a`); `ECHOJEPA_WEIGHTS` = `vitl-vmix22m-pt220-c55.pt` from its README |
| `echoprime_mvit` | 280 px clips | `torchvision` | `ECHOPRIME_ENCODER=model_data/weights/echo_prime_encoder.pt` from the EchoPrime release |

```bash
uv pip install torch-fidelity open_clip_torch h5py huggingface_hub einops timm torchvision
```

### Tables

```bash
uv run python -m tasks.video_diffusion.tables \
  --pair 128 <beamspace-128-run>/evaluation <cartesian-128-run>/evaluation \
  --pair 256 <beamspace-256-run>/evaluation <cartesian-256-run>/evaluation \
  --pair 320 <beamspace-320-run>/evaluation <cartesian-320-run>/evaluation \
  --output-dir outputs/video_diffusion/tables
uv run python -m tasks.video_diffusion.cv_table --root outputs/video_diffusion --folds 0 1 2 3 4 \
  --output-dir outputs/video_diffusion/tables
```

The per-fold table reports `100 x (D_cartesian - D_beamspace) / D_cartesian` (positive favours beamspace) with a
two-sided paired exam-cluster bootstrap (2000 replicates, the same exam multiplicities for real, beamspace and
Cartesian rows, views averaged equally), recentered p-values and Holm adjustment across all feature-space x resolution
comparisons. Output: `comparison.{md,csv,json}` and `comparison_replicates.npz`.

The cross-validated table pools the five folds: mean ± SD across folds per arm, the difference averaged over folds
relative to the mean Cartesian distance, and pooled replicates that average one independent replicate per fold.
Output: `comparison_cv.{csv,json,md,tex}`.

## Reproducing the paper grid

```bash
uv run python scripts/reproduce_video_diffusion.py --data-root /path/to/EchoXFlow --camus-root /path/to/CAMUS-beamspace
```

For each held-out fold, this trains both arms at 128, 256 and 320 px, evaluates them on EchoXFlow and CAMUS, builds
`outputs/video_diffusion/fold<k>/tables{,_camus}/` and pools the folds into
`outputs/video_diffusion/tables{,_camus}/comparison_cv.*`. `--camus-root` defaults to `ECHOXFLOW_CAMUS_ROOT`.
Finished steps are skipped on rerun. `--resolutions`, `--folds` and `--extractors` select a subset.

## External evaluation on CAMUS

The [CAMUS EchoXFlow package](https://huggingface.co/datasets/Ahus-AIM/CAMUS-beamspace) stores every
[CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) half-sequence as an EchoXFlow recording (B-mode on the fitted
fan in beamspace, with the ED and ES frame indices). Every patient (500) is conditioned in both views (A2C, A4C) on its
end-diastolic frame paired with the ECG and timing of an evaluation-fold EchoXFlow clip of the same view. The reference
is the patients' own non-ED frames and ED-to-ES clips. FD and KID are unpaired, and the bootstrap resamples patients.
Cite Leclerc et al., IEEE TMI 38(9), 2019.

```bash
uv run python -m tasks.video_diffusion.evaluate --checkpoint <run>/ema.pt --camus-root /path/to/CAMUS-beamspace
uv run python -m tasks.video_diffusion.tables --camus --pair 128 <beam-128>/evaluation <cart-128>/evaluation ... \
  --output-dir outputs/video_diffusion/tables_camus
```

CAMUS results are written to `<run>/evaluation/camus/`.

The cross-validated CAMUS table uses the same patients in every fold, so it averages matching patient-bootstrap
replicate indices across folds after verifying the ordered patient cohorts, replicate counts and seeds match.
Rebuild this table from existing per-fold `comparison.json` and `comparison_replicates.npz` files with:

```bash
uv run python -m tasks.video_diffusion.cv_table --root outputs/video_diffusion --folds 0 1 2 3 4 \
  --tables-name tables_camus --output-dir outputs/video_diffusion/tables_camus
```

This only recalculates pooled uncertainty and significance; mean distances remain unchanged. It does not rerun
training, generation, feature extraction or per-fold bootstrap distances. Older per-fold tables lack patient IDs;
the command reads them from the saved `fold<k>/<space>_<size>/evaluation/camus/features/*.npz` metadata instead.
For a custom per-fold table directory, also pass `--camus`.

## Media

[scripts/render_task4_media.py](../../scripts/render_task4_media.py) renders the README examples from saved
evaluation outputs (`--pick VIEW=EXAM/RECORDING` selects a clip, `--trim-to-beat A3C` cuts a clip to one heartbeat).
[scripts/render_camus_examples.py](../../scripts/render_camus_examples.py) renders the CAMUS animation:

```bash
uv run python -m scripts.render_task4_media --beamspace <beamspace-run>/ema_epoch010.pt \
  --cartesian <cartesian-run>/ema_epoch010.pt --saved-evaluation evaluation_epoch010 --device cpu
uv run python -m scripts.render_camus_examples --checkpoint <beamspace-256-run>/ema.pt --camus-root /path/to/CAMUS-beamspace
```
