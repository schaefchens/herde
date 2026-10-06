#!/bin/bash
# The worker's tests: no model, no GPU, no network — pytest and httpx in a venv of their own.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python="$(command -v python3.13 || command -v python3)"
if [[ ! -x .venv-test/bin/python ]]; then
    "$python" -m venv .venv-test
fi
.venv-test/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-dev.txt
exec .venv-test/bin/python -m pytest -q tests "$@"
