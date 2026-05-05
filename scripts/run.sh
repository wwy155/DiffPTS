#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run.sh --model <D3U|DiffPTS|NsDiffPE|...> --datasets <d1> <d2> ... [options] [-- extra_args...]

Examples:
  # Minimal: just choose model/datasets (auto GPU)
  bash scripts/run.sh --model DiffPTS --datasets ETTm1 ETTm2

  # Auto pick a GPU, run D3U on multiple datasets
  bash scripts/run.sh --model D3U --datasets ETTm1 ExchangeRate Traffic \
    --batch_size 32 --horizon 1 --pred_len 192 --windows 168 \
    --seeds "[1,2,3]"

  # Force GPU 1, run DiffPTS with the same args + wandb config
  bash scripts/run.sh --model DiffPTS --datasets ETTm1 ETTm2 --gpu 1 \
    --batch_size 32 --horizon 1 --pred_len 192 --windows 168 \
    --with_wandb --seeds "[1,2,3]"

Notes:
  - By default, runs datasets sequentially on one selected GPU.
  - You can pass additional experiment args after `--`.
  - If you omit --pred_len / --windows: ILI uses pred_len=32 (see scripts/DiffPTS/ILI.sh);
    other datasets use pred_len=192; windows default to 168 for all.
EOF
}

pick_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return 0
  fi
  local line
  line="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -nr | head -n1 || true)"
  if [[ -z "${line}" ]]; then
    echo "0"
    return 0
  fi
  echo "${line%%,*}" | xargs
}

MODEL=""
GPU="auto"
WITH_WANDB="0"
SEEDS="[1,2,3]"

COMMON_ARGS=()
DATASETS=()

has_flag() {
  local flag="$1"; shift
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" ]] || [[ "${arg}" == ${flag}=* ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0;;
    --model)
      MODEL="$2"; shift 2;;
    --datasets)
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        DATASETS+=("$1"); shift
      done;;
    --gpu)
      GPU="$2"; shift 2;;
    --device)
      # allow explicit device string, e.g. cuda:1
      GPU="${2#cuda:}"; shift 2;;
    --with_wandb)
      WITH_WANDB="1"; shift;;
    --seeds)
      SEEDS="$2"; shift 2;;
    --no_seeds)
      SEEDS=""; shift;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        COMMON_ARGS+=("$1"); shift
      done
      ;;
    *)
      # passthrough args (shared across datasets)
      COMMON_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "ERROR: missing --model"
  usage
  exit 2
fi
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "ERROR: missing --datasets"
  usage
  exit 2
fi

if [[ "${GPU}" == "auto" ]]; then
  GPU="$(pick_gpu)"
fi
DEVICE="cuda:${GPU}"

EXP_PY="${ROOT_DIR}/src/experiments/${MODEL}.py"
if [[ ! -f "${EXP_PY}" ]]; then
  echo "ERROR: cannot find experiment file: ${EXP_PY}"
  exit 2
fi

echo "Model: ${MODEL}"
echo "Device: ${DEVICE}"
echo "Datasets: ${DATASETS[*]}"
echo "Shared args (before per-dataset defaults): ${COMMON_ARGS[*]}"
if [[ -n "${SEEDS}" ]]; then
  echo "Seeds: ${SEEDS}"
else
  echo "Seeds: (not set)"
fi
echo

for ds in "${DATASETS[@]}"; do
  echo "===== Running ${MODEL} on dataset=${ds} (${DEVICE}) ====="
  RUN_ARGS=("${COMMON_ARGS[@]}")
  # Per-dataset defaults (only if not provided); ILI matches scripts/DiffPTS/ILI.sh
  if ! has_flag --pred_len "${RUN_ARGS[@]}"; then
    if [[ "${ds}" == "ILI" ]]; then
      RUN_ARGS+=(--pred_len 32)
    else
      RUN_ARGS+=(--pred_len 192)
    fi
  fi
  if ! has_flag --windows "${RUN_ARGS[@]}"; then
    RUN_ARGS+=(--windows 168)
  fi
  echo "Effective args for ${ds}: ${RUN_ARGS[*]}"
  if [[ "${WITH_WANDB}" == "1" ]]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
      python3 "${EXP_PY}" \
        --dataset_type="${ds}" \
        --device="${DEVICE}" \
        "${RUN_ARGS[@]}" \
        config_wandb ProbForecastBase \
        runs ${SEEDS:+--seeds="${SEEDS}"}
  else
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
      python3 "${EXP_PY}" \
        --dataset_type="${ds}" \
        --device="${DEVICE}" \
        "${RUN_ARGS[@]}" \
        runs ${SEEDS:+--seeds="${SEEDS}"}
  fi
done

