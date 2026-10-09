#!/bin/bash
# herde's tests: no model, no GPU, no network — pytest, jsonschema and ruff in a venv of their own.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python="$(command -v python3.13 || command -v python3.12 || command -v python3)"
if [[ ! -x .venv-test/bin/python ]]; then
    "$python" -m venv .venv-test
fi
.venv-test/bin/python -m pip install --quiet --disable-pip-version-check -e ".[dev]"
.venv-test/bin/ruff check .
exec .venv-test/bin/python -m pytest "$@"
