#!/usr/bin/env bash
# Run every result of the paper that trains or evaluates a model, from one commit, one resolution at a time: Tasks 1-3
# (full benchmark matrix, 5-fold CV) and Task 4 (video diffusion, 5-fold CV) at the first resolution, then both at the
# next, and so on. Afterwards: the benchmark loss curves and the Task 3 CAMUS evaluation of every checkpoint
# (OUT/benchmark/{figures,camus}), the diffusion comparison tables (EchoXFlow and CAMUS) per fold and pooled over the
# folds (OUT/video_diffusion/tables{,_camus}/comparison_cv.*), and the foundation-model EF comparison
# (OUT/foundation_model_ef). The dataset tables need no model: scripts/croissant_summary_table.py.
#
#   scripts/run_paper_pipeline.sh --data-root DATA --output-dir OUT --camus-root CAMUS
#                                 [--resolutions 128 256 320] [--folds 0 1 2 3 4] [--parts benchmark diffusion ef]
#
# CAMUS is the CAMUS EchoXFlow package (https://huggingface.co/datasets/Ahus-AIM/CAMUS-beamspace; also read from
# ECHOXFLOW_CAMUS_ROOT). Within a phase the benchmark launcher and one diffusion launcher per fold share the machine's
# free GPUs (--gpus auto). Rerun the same command after an interruption. The working tree must be clean (the commit is
# recorded in OUT/pipeline_runs.jsonl) unless --allow-dirty is given. The diffusion and ef parts refuse to start unless
# the weights of every feature extractor and foundation model are configured (tasks/video_diffusion/PROTOCOL.md,
# docs/REPRODUCE.md).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root=""
output_root=""
camus_root="${ECHOXFLOW_CAMUS_ROOT:-}"
resolutions="128 256 320"
folds="0 1 2 3 4"
parts="benchmark diffusion ef"
allow_dirty=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --output-dir) output_root="$2"; shift 2 ;;
    --camus-root) camus_root="$2"; shift 2 ;;
    --resolutions|--folds|--parts)
      flag="$1"; shift; values=()
      while [[ $# -gt 0 && "$1" != --* ]]; do values+=("$1"); shift; done
      case "$flag" in
        --resolutions) resolutions="${values[*]}" ;;
        --folds) folds="${values[*]}" ;;
        --parts) parts="${values[*]}" ;;
      esac ;;
    --allow-dirty) allow_dirty=1; shift ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$data_root" && -n "$output_root" ]] || { echo "--data-root and --output-dir are required" >&2; exit 2; }
[[ -f "$camus_root/croissant.json" ]] || {
  echo "--camus-root (or ECHOXFLOW_CAMUS_ROOT) must point to the CAMUS EchoXFlow package containing croissant.json" >&2
  exit 2
}
run_benchmark=0; run_diffusion=0; run_ef=0
for part in $parts; do
  case "$part" in
    benchmark) run_benchmark=1 ;;
    diffusion) run_diffusion=1 ;;
    ef) run_ef=1 ;;
    *) echo "--parts takes benchmark, diffusion and/or ef, got: $part" >&2; exit 2 ;;
  esac
done
required=()
# The paper reports all seven feature spaces; the evaluation would otherwise skip the missing ones silently.
[[ $run_diffusion -eq 1 ]] && required+=(ECHOPRIME_VIEW_CLASSIFIER ECHOPRIME_ENCODER PANECHO_SOURCE PANECHO_WEIGHTS \
  ECHOJEPA_SOURCE ECHOJEPA_WEIGHTS)
[[ $run_ef -eq 1 ]] && required+=(ECHOPRIME_VIEW_CLASSIFIER ECHOPRIME_ENCODER PANECHO_SOURCE PANECHO_WEIGHTS \
  ECHOPRIME_SOURCE ECHOPRIME_MODEL_DATA)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" && -e "${!variable}" ]] || {
    echo "$variable is unset or missing; see tasks/video_diffusion/PROTOCOL.md and docs/REPRODUCE.md" >&2
    exit 2
  }
done
split_csv="$data_root/splits.csv"
bench_root="$output_root/benchmark"
diffusion_root="$output_root/video_diffusion"
logs="$output_root/pipeline_logs"
mkdir -p "$logs"

cd "$repo"
commit="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  if [[ $allow_dirty -eq 0 ]]; then
    echo "Refusing to run from a working tree with uncommitted changes (commit them, or pass --allow-dirty)." >&2
    exit 2
  fi
  commit="$commit-dirty"
fi
python3 - "$output_root/pipeline_runs.jsonl" "$commit" "$data_root" "$resolutions" "$folds" "$parts" <<'PY'
import datetime, hashlib, json, pathlib, socket, sys
path, commit, data_root, resolutions, folds, parts = sys.argv[1:]
digest = lambda name: hashlib.sha256((pathlib.Path(data_root) / name).read_bytes()).hexdigest()
record = {
    "started": datetime.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(), "commit": commit,
    "data_root": data_root, "croissant_sha256": digest("croissant.json"), "splits_sha256": digest("splits.csv"),
    "resolutions": resolutions.split(), "folds": folds.split(), "parts": parts.split(),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
print(json.dumps(record, indent=2))
PY

children=()
stop() {
  echo "$(date '+%F %T') stopping: forwarding SIGTERM to the launchers" | tee -a "$logs/pipeline.log"
  for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait || true
  exit 143
}
trap stop INT TERM

say() { echo "$(date '+%F %T') $*" | tee -a "$logs/pipeline.log"; }
total_failed=0
for resolution in $resolutions; do
  say "phase ${resolution}px ($parts, folds $folds)"
  children=()
  if [[ $run_benchmark -eq 1 ]]; then
    uv run python scripts/run_full_benchmark.py --data-root "$data_root" --split-csv "$split_csv" --output-dir "$bench_root" \
      --run-name paper --paper --cv --folds $folds --resolution "$resolution" --gpus auto \
      >> "$logs/benchmark_r${resolution}.log" 2>&1 &
    children+=("$!")
  fi
  if [[ $run_diffusion -eq 1 ]]; then
    for fold in $folds; do
      uv run python scripts/reproduce_video_diffusion.py --data-root "$data_root" --split-csv "$split_csv" \
        --camus-root "$camus_root" --output-dir "$diffusion_root" --resolutions "$resolution" --folds "$fold" --skip-tables \
        --status-name "video_diffusion_r${resolution}_fold${fold}" --gpus auto \
        >> "$logs/diffusion_r${resolution}_fold${fold}.log" 2>&1 &
      children+=("$!")
    done
  fi
  failed=0
  for pid in "${children[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  total_failed=$((total_failed + failed))
  say "phase ${resolution}px finished; launchers with failures: $failed (see the status files)"
  if [[ $run_benchmark -eq 1 ]]; then
    uv run python scripts/build_benchmark_table.py --root "$bench_root" --run-name "paper_r${resolution}" \
      --resolution "$resolution" --domain-comparison --output-dir "$bench_root/tables" \
      >> "$logs/benchmark_r${resolution}.log" 2>&1 || { total_failed=$((total_failed + 1)); say "domain comparison table for ${resolution}px failed"; }
  fi
done

if [[ $run_ef -eq 1 ]]; then
  say "foundation-model EF comparison"
  uv run python scripts/foundation_model_ef.py --data-root "$data_root" --output-dir "$output_root/foundation_model_ef" \
    >> "$logs/foundation_model_ef.log" 2>&1 || { total_failed=$((total_failed + 1)); say "foundation-model EF comparison failed (see $logs/foundation_model_ef.log)"; }
fi
if [[ $run_benchmark -eq 1 ]]; then
  for resolution in $resolutions; do
    uv run python scripts/plot_benchmark_losses.py --benchmark-root "$bench_root" --run-name paper \
      --resolution "$resolution" --output-dir "$bench_root/figures" >> "$logs/benchmark_figures.log" 2>&1 \
      || { total_failed=$((total_failed + 1)); say "loss curves for ${resolution}px failed (see $logs/benchmark_figures.log)"; }
  done
  say "Task 3 CAMUS evaluation of every benchmark checkpoint"
  uv run python -m tasks.segmentation.camus --package-root "$camus_root" --benchmark-root "$bench_root" \
    --resolutions $resolutions --folds $folds --output-dir "$bench_root/camus" >> "$logs/benchmark_camus.log" 2>&1 \
    || { total_failed=$((total_failed + 1)); say "Task 3 CAMUS evaluation failed (see $logs/benchmark_camus.log)"; }
fi
[[ $run_diffusion -eq 1 ]] || { say "pipeline finished (no diffusion part); failures: $total_failed"; [[ $total_failed -eq 0 ]]; exit $?; }
say "diffusion comparison tables over resolutions $resolutions"
children=()
for fold in $folds; do
  uv run python scripts/reproduce_video_diffusion.py --data-root "$data_root" --split-csv "$split_csv" \
    --camus-root "$camus_root" --output-dir "$diffusion_root" --resolutions $resolutions --folds "$fold" \
    --status-name "video_diffusion_tables_fold${fold}" --gpus auto \
    >> "$logs/diffusion_tables_fold${fold}.log" 2>&1 &
  children+=("$!")
done
failed=0
for pid in "${children[@]}"; do wait "$pid" || failed=$((failed + 1)); done
say "diffusion table launchers with failures: $failed"
if [[ $failed -eq 0 && $(wc -w <<< "$folds") -gt 1 ]]; then
  uv run python -m tasks.video_diffusion.cv_table --root "$diffusion_root" --folds $folds \
    --output-dir "$diffusion_root/tables" >> "$logs/diffusion_tables_cv.log" 2>&1 \
    || { say "cross-validated diffusion table failed (see $logs/diffusion_tables_cv.log)"; failed=1; }
  uv run python -m tasks.video_diffusion.cv_table --root "$diffusion_root" --folds $folds --tables-name tables_camus \
    --output-dir "$diffusion_root/tables_camus" >> "$logs/diffusion_tables_cv.log" 2>&1 \
    || { say "cross-validated CAMUS table failed (see $logs/diffusion_tables_cv.log)"; failed=1; }
fi
say "pipeline finished; diffusion table failures: $failed"

total_failed=$((total_failed + failed))
[[ $total_failed -eq 0 ]]
