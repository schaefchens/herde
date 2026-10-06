#!/bin/bash
# Starts the worker (or `./run.sh check` / `./run.sh once`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo "Run ./setup.sh first." >&2
    exit 1
fi
exec .venv/bin/python -m arche_worker "${@:-run}"
