#!/usr/bin/env bash
# Cluster entry point of corsika_sparse_v2 (used by the jobs/*.sub files except
# chain_page, chain_steps, active_vs_grid and v1_samples).
#
#   run_pipeline.sh selftest                     synthetic end-to-end check (configs/smoke.yaml, CPU)
#   run_pipeline.sh notebook <nb> [options]      execute a notebook headless (scripts/run_notebook.py)
#   run_pipeline.sh generate [options]           new showers from trained_model/ (python -m sparseshower.generate)
#   run_pipeline.sh <cli command> [cli options]  anything else goes to  python -m sparseshower.cli
#
# Examples (what the jobs/*.sub files pass as `arguments`):
#   run_pipeline.sh memtest --kinds ae dit struct attr
#   run_pipeline.sh train --kind ae --case v2a
#   run_pipeline.sh ae-eval --case v2a --n-samples 256 --name ae_best
#   run_pipeline.sh encode --case v2a
#   run_pipeline.sh sample --case v2a --chain sa_recon --run sa_recon --n-samples 128 --shard 0 --num-shards 4
#   run_pipeline.sh evaluate --case v2a --runs sa_truth sa_recon full --name main
#
# Environment: SPARSE_PROJECT (code directory), SPARSE_ENV (activate script),
# SPARSE_CONFIG (default configs/v2.yaml; not used by selftest / notebook).
set -euo pipefail

mode="${1:-selftest}"
if [[ $# -gt 0 ]]; then shift; fi
project_dir="${SPARSE_PROJECT:-/data/user/qwang/corsika_sparse_v2}"
env_activate="${SPARSE_ENV:-/data/user/qwang/env/env2025/bin/activate}"
config_path="${SPARSE_CONFIG:-configs/v2.yaml}"

if [[ ! -d "$project_dir" ]]; then echo "Project directory not found: $project_dir" >&2; exit 2; fi
if [[ -f "$env_activate" ]]; then
  # shellcheck disable=SC1090
  source "$env_activate"
else
  echo "WARNING: environment file not found ($env_activate); using the current python" >&2
fi
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

child_pid=""
on_terminate() {
  if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  exit 143
}
trap on_terminate TERM INT
run() {
  "$@" &
  child_pid=$!
  local code=0
  wait "$child_pid" || code=$?
  child_pid=""
  return "$code"
}

echo "sparseshower v2: mode=$mode args=[$*] host=$(hostname) date=$(date -Is) config=$config_path"
case "$mode" in
  selftest)
    SPARSE_CONFIG=configs/smoke.yaml bash scripts/check_environment.sh
    run python -m sparseshower.cli selftest --config configs/smoke.yaml "$@"
    ;;
  notebook)
    SPARSE_CONFIG="$config_path" bash scripts/check_environment.sh
    run python scripts/run_notebook.py "$@"
    ;;
  generate)
    SPARSE_CONFIG="$config_path" bash scripts/check_environment.sh
    run python -m sparseshower.generate "$@"
    ;;
  *)
    SPARSE_CONFIG="$config_path" bash scripts/check_environment.sh
    run python -m sparseshower.cli "$mode" --config "$config_path" "$@"
    ;;
esac
echo "pipeline completed: $mode"
