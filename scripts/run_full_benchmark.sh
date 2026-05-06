#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the EchoXFlow benchmark matrix and build the table.

Usage:
  scripts/run_full_benchmark.sh [options]

Options:
  --data-root PATH     Dataset root containing croissant.json. Defaults to DATA_ROOT, then config/ECHOXFLOW_DATA_ROOT.
  --output-root PATH   Benchmark artifact root. Default: outputs/bench
  --seed N             Random seed. Default: 0
  --cv                 Run CSV-defined 5-fold CV with validation folds 0,1,2,3,4.
  --folds LIST         Run comma-separated validation folds, e.g. 0,1,2. Implies --cv.
  --fold-split-path PATH
                       CSV fold split path. Defaults to auto-discovery beside croissant.json.
  --sample-cache-dir PATH
                       Enable generic preprocessed sample cache under PATH.
  --recording-cache-dir PATH
                       Enable raw recording-array cache under PATH.
  --recording-cache-read-only
                       Read recording cache without writing misses.
  --recording-cache-include PATTERN
                       Only cache matching recording array paths. May be repeated; supports shell-style globs.
  --recording-cache-exclude PATTERN
                       Do not cache matching recording array paths. May be repeated; supports shell-style globs.
  --gpus LIST          Run cells concurrently across comma-separated GPUs, e.g. 0,1,2.
  --baseline-jobs N    CPU slots for temporal_mean cells when --gpus is set. Default: 10.
  --run-name NAME      Run folder name. Default: RUN_NAME, then current datetime.
  --force-rerun        Re-run cells even when val_metrics.json already exists.
  --smoke              Run a smoke benchmark with tiny model configs for 1 epoch on 10% data.
  --python CMD         Python command. Default: "uv run python" if uv exists, otherwise "python".
  -h, --help           Show this help.

Environment:
  DATA_ROOT            Optional dataset root.
  OUTPUT_ROOT          Optional artifact root.
  SEED                 Optional seed.
  CV                   Set to 1/true/yes/on to run 5-fold CV.
  FOLDS                Optional comma-separated validation folds.
  FOLD_SPLIT_PATH      Optional CSV fold split path.
  SAMPLE_CACHE_DIR     Optional preprocessed sample cache directory.
  RECORDING_CACHE_DIR  Optional raw recording-array cache directory.
  RECORDING_CACHE_READ_ONLY
                       Set to 1/true/yes/on to read recording cache without writing misses.
  RECORDING_CACHE_INCLUDE
                       Optional comma-separated recording cache include patterns.
  RECORDING_CACHE_EXCLUDE
                       Optional comma-separated recording cache exclude patterns.
  GPUS                 Optional comma-separated GPU list.
  BASELINE_JOBS        Optional CPU slots for temporal_mean cells.
  RUN_NAME             Optional run folder name. Default: current datetime.
  PYTHON_CMD           Optional Python command.

Examples:
  scripts/run_full_benchmark.sh --data-root /data/EchoXFlow
  scripts/run_full_benchmark.sh --data-root /data/EchoXFlow --cv
  scripts/run_full_benchmark.sh --data-root /data/EchoXFlow --smoke
  scripts/run_full_benchmark.sh --gpus 0,1,2
EOF
}

data_root="${DATA_ROOT:-}"
output_root="${OUTPUT_ROOT:-outputs/bench}"
seed="${SEED:-0}"
cv="${CV:-0}"
folds_csv="${FOLDS:-}"
fold_split_path="${FOLD_SPLIT_PATH:-}"
sample_cache_dir="${SAMPLE_CACHE_DIR:-}"
recording_cache_dir="${RECORDING_CACHE_DIR:-}"
recording_cache_read_only="${RECORDING_CACHE_READ_ONLY:-0}"
recording_cache_include_csv="${RECORDING_CACHE_INCLUDE:-}"
recording_cache_exclude_csv="${RECORDING_CACHE_EXCLUDE:-}"
gpus_csv="${GPUS:-}"
baseline_jobs="${BASELINE_JOBS:-10}"
smoke=0
python_cmd="${PYTHON_CMD:-}"
run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
force_rerun=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --cv)
      cv=1
      shift
      ;;
    --folds)
      folds_csv="$2"
      cv=1
      shift 2
      ;;
    --fold-split-path)
      fold_split_path="$2"
      shift 2
      ;;
    --sample-cache-dir)
      sample_cache_dir="$2"
      shift 2
      ;;
    --recording-cache-dir)
      recording_cache_dir="$2"
      shift 2
      ;;
    --recording-cache-read-only)
      recording_cache_read_only=1
      shift
      ;;
    --recording-cache-include)
      if [[ -n "$recording_cache_include_csv" ]]; then
        recording_cache_include_csv+=","
      fi
      recording_cache_include_csv+="$2"
      shift 2
      ;;
    --recording-cache-exclude)
      if [[ -n "$recording_cache_exclude_csv" ]]; then
        recording_cache_exclude_csv+=","
      fi
      recording_cache_exclude_csv+="$2"
      shift 2
      ;;
    --gpus)
      gpus_csv="$2"
      shift 2
      ;;
    --baseline-jobs)
      baseline_jobs="$2"
      shift 2
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --force-rerun)
      force_rerun=1
      shift
      ;;
    --smoke)
      smoke=1
      shift
      ;;
    --python)
      python_cmd="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$python_cmd" ]]; then
  if command -v uv >/dev/null 2>&1; then
    python_cmd="uv run python"
  else
    python_cmd="python"
  fi
fi

common_args=(--seed "$seed" --output-root "$output_root" --run-name "$run_name")
if [[ -n "$data_root" ]]; then
  common_args+=(--data-root "$data_root")
fi
if [[ -n "$fold_split_path" ]]; then
  common_args+=(--fold-split-path "$fold_split_path")
fi
if [[ -n "$sample_cache_dir" ]]; then
  common_args+=(--sample-cache-dir "$sample_cache_dir")
fi
case "${recording_cache_read_only,,}" in
  1|true|yes|on)
    recording_cache_read_only=1
    ;;
  0|false|no|off|"")
    recording_cache_read_only=0
    ;;
  *)
    echo "RECORDING_CACHE_READ_ONLY must be one of 1/0/true/false/yes/no/on/off, got: $recording_cache_read_only" >&2
    exit 2
    ;;
esac
if [[ -n "$recording_cache_dir" ]]; then
  common_args+=(--recording-cache-dir "$recording_cache_dir")
  if [[ "$recording_cache_read_only" -eq 1 ]]; then
    common_args+=(--recording-cache-read-only)
  fi
fi
if [[ -n "$recording_cache_include_csv" ]]; then
  IFS=',' read -r -a recording_cache_includes <<< "$recording_cache_include_csv"
  for pattern in "${recording_cache_includes[@]}"; do
    pattern="${pattern#"${pattern%%[![:space:]]*}"}"
    pattern="${pattern%"${pattern##*[![:space:]]}"}"
    if [[ -n "$pattern" ]]; then
      common_args+=(--recording-cache-include "$pattern")
    fi
  done
fi
if [[ -n "$recording_cache_exclude_csv" ]]; then
  IFS=',' read -r -a recording_cache_excludes <<< "$recording_cache_exclude_csv"
  for pattern in "${recording_cache_excludes[@]}"; do
    pattern="${pattern#"${pattern%%[![:space:]]*}"}"
    pattern="${pattern%"${pattern##*[![:space:]]}"}"
    if [[ -n "$pattern" ]]; then
      common_args+=(--recording-cache-exclude "$pattern")
    fi
  done
fi

limit_args=()
if [[ "$smoke" -eq 1 ]]; then
  limit_args+=(
    --smoke-config
    --data-fraction 0.1
    --max-epochs 1
  )
fi
train_args=(
  --max-val-steps 0
)

gpus=()
if [[ -n "$gpus_csv" ]]; then
  IFS=',' read -r -a raw_gpus <<< "$gpus_csv"
  for raw_gpu in "${raw_gpus[@]}"; do
    gpu="${raw_gpu//[[:space:]]/}"
    if [[ -n "$gpu" ]]; then
      gpus+=("$gpu")
    fi
  done
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "--gpus must contain at least one GPU id" >&2
    exit 2
  fi
  if (( BASH_VERSINFO[0] < 5 )); then
    echo "--gpus requires Bash 5 or newer for wait -n support" >&2
    exit 2
  fi
fi
if [[ "$baseline_jobs" == "auto" ]]; then
  baseline_jobs="${#gpus[@]}"
fi
if ! [[ "$baseline_jobs" =~ ^[0-9]+$ ]]; then
  echo "--baseline-jobs must be a non-negative integer, got: $baseline_jobs" >&2
  exit 2
fi

folds=()
case "${cv,,}" in
  1|true|yes|on)
    cv=1
    ;;
  0|false|no|off|"")
    cv=0
    ;;
  *)
    echo "CV must be one of 1/0/true/false/yes/no/on/off, got: $cv" >&2
    exit 2
    ;;
esac
if [[ -n "$folds_csv" ]]; then
  cv=1
  IFS=',' read -r -a raw_folds <<< "$folds_csv"
  for raw_fold in "${raw_folds[@]}"; do
    fold="${raw_fold//[[:space:]]/}"
    if [[ -z "$fold" ]]; then
      continue
    fi
    if ! [[ "$fold" =~ ^[0-9]+$ ]]; then
      echo "--folds must contain non-negative integer fold ids, got: $fold" >&2
      exit 2
    fi
    folds+=("$fold")
  done
elif [[ "$cv" -eq 1 ]]; then
  folds=(0 1 2 3 4)
fi
if [[ "$cv" -eq 1 && "${#folds[@]}" -eq 0 ]]; then
  echo "CV mode requires at least one fold" >&2
  exit 2
fi

mapfile -t benchmark_cells < <($python_cmd - <<'PY'
import yaml

with open("configs/benchmark_table.yaml", encoding="utf-8") as stream:
    spec = yaml.safe_load(stream)
tasks = list(dict.fromkeys(str(column["task"]) for column in spec["columns"]))
for task in tasks:
    for row in spec["rows"]:
        print(f'{task}\t{row["method"]}\t{row["domain"]}')
PY
)

benchmark_runs=()
if [[ "$cv" -eq 1 ]]; then
  for fold in "${folds[@]}"; do
    for cell in "${benchmark_cells[@]}"; do
      benchmark_runs+=("$cell"$'\t'"$fold")
    done
  done
else
  for cell in "${benchmark_cells[@]}"; do
    benchmark_runs+=("$cell"$'\t')
  done
fi

all_run_count="${#benchmark_runs[@]}"
skipped_runs=()
if [[ "$force_rerun" -eq 0 ]]; then
  pending_runs=()
  for run in "${benchmark_runs[@]}"; do
    IFS=$'\t' read -r task method domain fold <<< "$run"
    metrics_path="$output_root/$task/$run_name/$method/$domain"
    if [[ -n "$fold" ]]; then
      metrics_path+="/fold=$fold"
    fi
    metrics_path+="/seed=$seed/val_metrics.json"
    if [[ -f "$metrics_path" ]]; then
      skipped_runs+=("$run")
    else
      pending_runs+=("$run")
    fi
  done
  benchmark_runs=("${pending_runs[@]}")
fi

echo "Benchmark output: $output_root"
echo "Run name: $run_name"
echo "Seed: $seed"
if [[ -n "$data_root" ]]; then
  echo "Data root: $data_root"
else
  echo "Data root: from config or ECHOXFLOW_DATA_ROOT"
fi
if [[ "$cv" -eq 1 ]]; then
  echo "Validation folds: ${folds[*]}"
  if [[ -n "$fold_split_path" ]]; then
    echo "Fold split CSV: $fold_split_path"
  else
    echo "Fold split CSV: auto-discovered from data root"
  fi
fi
if [[ "$smoke" -eq 1 ]]; then
  echo "Smoke mode: tiny model configs, 10% data, 1 epoch"
fi
echo "Training validation: disabled (--max-val-steps 0); final benchmark evaluation still runs"
if [[ -n "$sample_cache_dir" ]]; then
  echo "Sample cache: $sample_cache_dir"
fi
if [[ -n "$recording_cache_dir" ]]; then
  echo "Recording cache: $recording_cache_dir"
  if [[ "$recording_cache_read_only" -eq 1 ]]; then
    echo "Recording cache mode: read-only"
  fi
fi
if [[ "${#gpus[@]}" -gt 0 ]]; then
  echo "Parallel GPUs: ${gpus[*]}"
  echo "Temporal mean CPU jobs: $baseline_jobs"
fi
if [[ "$force_rerun" -eq 1 ]]; then
  echo "Completed-cell skipping: disabled (--force-rerun)"
else
  echo "Completed-cell skipping: ${#skipped_runs[@]} skipped, ${#benchmark_runs[@]} pending, $all_run_count total"
fi

run_cell() {
  local task="$1"
  local method="$2"
  local domain="$3"
  local fold="${4:-}"
  local fold_args=()
  if [[ -n "$fold" ]]; then
    fold_args+=(--validation-fold "$fold")
  fi
  $python_cmd -m tasks.bench \
    --task "$task" \
    --method "$method" \
    --domain "$domain" \
    --stage train_and_evaluate \
    "${common_args[@]}" \
    "${fold_args[@]}" \
    "${train_args[@]}" \
    "${limit_args[@]}"
}

build_table() {
  local label="${1:-}"
  if [[ -n "$label" ]]; then
    echo "Building benchmark table ($label)"
  else
    echo "Building benchmark table"
  fi
  $python_cmd scripts/build_benchmark_table.py \
    --root "$output_root" \
    --run-name "$run_name" \
    --out "$output_root/tables/$run_name.tex"
}

run_sequential_batch() {
  local batch_name="$1"
  shift
  local batch_runs=("$@")
  local run task method domain fold label
  if [[ "${#batch_runs[@]}" -eq 0 ]]; then
    echo "No pending cells for $batch_name"
    return 0
  fi
  for run in "${batch_runs[@]}"; do
    IFS=$'\t' read -r task method domain fold <<< "$run"
    label="$task / $method / $domain"
    if [[ -n "$fold" ]]; then
      label="$label / fold=$fold"
    fi
    echo "Running $label"
    run_cell "$task" "$method" "$domain" "$fold"
  done
}

if [[ "${#gpus[@]}" -eq 0 ]]; then
  if [[ "$cv" -eq 1 ]]; then
    for fold in "${folds[@]}"; do
      fold_runs=()
      for run in "${benchmark_runs[@]}"; do
        IFS=$'\t' read -r _task _method _domain run_fold <<< "$run"
        if [[ "$run_fold" == "$fold" ]]; then
          fold_runs+=("$run")
        fi
      done
      echo "Running validation fold $fold"
      run_sequential_batch "fold=$fold" "${fold_runs[@]}"
      build_table "after fold=$fold"
    done
  else
    run_sequential_batch "benchmark matrix" "${benchmark_runs[@]}"
    build_table
  fi
else
  log_dir="$output_root/logs/$run_name"
  mkdir -p "$log_dir"
  echo "Cell logs: $log_dir"

  active_pids=()
  active_resources=()
  active_labels=()
  active_logs=()
  free_gpus=()
  active_baseline_jobs=0

  stop_active_jobs() {
    local pid
    for pid in "${active_pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  }
  trap 'echo "Stopping active benchmark jobs" >&2; stop_active_jobs; exit 130' INT TERM

  forget_active_job() {
    local index="$1"
    unset 'active_pids[index]'
    unset 'active_resources[index]'
    unset 'active_labels[index]'
    unset 'active_logs[index]'
    active_pids=("${active_pids[@]}")
    active_resources=("${active_resources[@]}")
    active_labels=("${active_labels[@]}")
    active_logs=("${active_logs[@]}")
  }

  wait_for_one_job() {
    local finished_pid status index resource label log_file found
    set +e
    wait -n -p finished_pid "${active_pids[@]}"
    status="$?"
    set -e
    found=0
    for index in "${!active_pids[@]}"; do
      if [[ "${active_pids[$index]}" == "$finished_pid" ]]; then
        resource="${active_resources[$index]}"
        label="${active_labels[$index]}"
        log_file="${active_logs[$index]}"
        if [[ "$resource" == gpu:* ]]; then
          free_gpus+=("${resource#gpu:}")
        elif [[ "$resource" == "cpu-baseline" ]]; then
          active_baseline_jobs=$((active_baseline_jobs - 1))
        fi
        forget_active_job "$index"
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "A benchmark job exited with status $status" >&2
      return "$status"
    fi
    if [[ "$status" -ne 0 ]]; then
      echo "Failed $label on $resource; log: $log_file" >&2
      tail -n 60 "$log_file" >&2 || true
      stop_active_jobs
      exit "$status"
    fi
    echo "Completed $label on $resource"
  }

  start_gpu_cell() {
    local cell="$1"
    local task method domain fold gpu label log_file fold_slug
    IFS=$'\t' read -r task method domain fold <<< "$cell"
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    label="$task / $method / $domain"
    fold_slug=""
    if [[ -n "$fold" ]]; then
      label="$label / fold=$fold"
      fold_slug="_fold-${fold}"
    fi
    log_file="$log_dir/${task}_${method}_${domain}${fold_slug}_seed-${seed}.log"
    echo "Starting $label on GPU $gpu; log: $log_file"
    (export CUDA_VISIBLE_DEVICES="$gpu"; run_cell "$task" "$method" "$domain" "$fold") > "$log_file" 2>&1 &
    active_pids+=("$!")
    active_resources+=("gpu:$gpu")
    active_labels+=("$label")
    active_logs+=("$log_file")
  }

  start_baseline_cell() {
    local cell="$1"
    local task method domain fold label log_file fold_slug
    IFS=$'\t' read -r task method domain fold <<< "$cell"
    label="$task / $method / $domain"
    fold_slug=""
    if [[ -n "$fold" ]]; then
      label="$label / fold=$fold"
      fold_slug="_fold-${fold}"
    fi
    log_file="$log_dir/${task}_${method}_${domain}${fold_slug}_seed-${seed}.log"
    echo "Starting $label on CPU baseline slot; log: $log_file"
    (export CUDA_VISIBLE_DEVICES=""; run_cell "$task" "$method" "$domain" "$fold") > "$log_file" 2>&1 &
    active_pids+=("$!")
    active_resources+=("cpu-baseline")
    active_labels+=("$label")
    active_logs+=("$log_file")
    active_baseline_jobs=$((active_baseline_jobs + 1))
  }

  run_parallel_batch() {
    local batch_name="$1"
    shift
    local batch_runs=("$@")
    local run _task method _domain _fold
    if [[ "${#batch_runs[@]}" -eq 0 ]]; then
      echo "No pending cells for $batch_name"
      return 0
    fi

    gpu_cells=()
    baseline_cells=()
    for run in "${batch_runs[@]}"; do
      IFS=$'\t' read -r _task method _domain _fold <<< "$run"
      if [[ "$method" == "temporal_mean" && "$baseline_jobs" -gt 0 ]]; then
        baseline_cells+=("$run")
      else
        gpu_cells+=("$run")
      fi
    done

    active_pids=()
    active_resources=()
    active_labels=()
    active_logs=()
    free_gpus=("${gpus[@]}")
    active_baseline_jobs=0

    local gpu_index=0
    local baseline_index=0
    while [[
      "$gpu_index" -lt "${#gpu_cells[@]}"
      || "$baseline_index" -lt "${#baseline_cells[@]}"
      || "${#active_pids[@]}" -gt 0
    ]]; do
      while [[ "${#free_gpus[@]}" -gt 0 && "$gpu_index" -lt "${#gpu_cells[@]}" ]]; do
        start_gpu_cell "${gpu_cells[$gpu_index]}"
        gpu_index=$((gpu_index + 1))
      done
      while [[ "$active_baseline_jobs" -lt "$baseline_jobs" && "$baseline_index" -lt "${#baseline_cells[@]}" ]]; do
        start_baseline_cell "${baseline_cells[$baseline_index]}"
        baseline_index=$((baseline_index + 1))
      done
      if [[ "${#active_pids[@]}" -gt 0 ]]; then
        wait_for_one_job
      fi
    done
  }

  if [[ "$cv" -eq 1 ]]; then
    for fold in "${folds[@]}"; do
      fold_runs=()
      for run in "${benchmark_runs[@]}"; do
        IFS=$'\t' read -r _task _method _domain run_fold <<< "$run"
        if [[ "$run_fold" == "$fold" ]]; then
          fold_runs+=("$run")
        fi
      done
      echo "Running validation fold $fold"
      run_parallel_batch "fold=$fold" "${fold_runs[@]}"
      build_table "after fold=$fold"
    done
  else
    run_parallel_batch "benchmark matrix" "${benchmark_runs[@]}"
    build_table
  fi
fi
