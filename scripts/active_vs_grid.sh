#!/bin/bash
# Wrapper for jobs/active_vs_grid.sub: activate env, run from the project root.
set -euo pipefail
source "$SPARSE_ENV"
cd "$SPARSE_PROJECT"
python scripts/active_vs_grid.py --config "$SPARSE_CONFIG" "$@"
