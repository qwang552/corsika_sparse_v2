#!/usr/bin/env bash
# Runs sparseshower.chain_page (interactive 3D comparison page) with the cluster environment.
#   bash scripts/chain_page.sh --case v2a --runs sa_truth sa_recon sdedit_1 --full-run full
# Environment: SPARSE_PROJECT (code directory), SPARSE_ENV (activate script), SPARSE_CONFIG (default configs/v2.yaml).
set -euo pipefail
project_dir="${SPARSE_PROJECT:-/data/user/qwang/corsika_sparse_v2}"
env_activate="${SPARSE_ENV:-/data/user/qwang/env/env2025/bin/activate}"
config_path="${SPARSE_CONFIG:-configs/v2.yaml}"
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
echo "chain_page: args=[$*] host=$(hostname) date=$(date -Is) config=$config_path"
python -m sparseshower.chain_page --config "$config_path" "$@"
