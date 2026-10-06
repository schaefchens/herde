#!/bin/bash
# Starts the ARCHE voice worker (or `./run-arche-worker.sh check` / `./run-arche-worker.sh once`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo "Run ./setup-arche-worker.sh first." >&2
    exit 1
fi
exec .venv/bin/python -m arche_worker "${@:-run}"
