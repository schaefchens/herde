#!/bin/bash
# Installs the worker on an Apple Silicon Mac: Python 3.13, its own venv, mlx-audio.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "The worker runs Qwen3-TTS on Apple's GPU: it needs an Apple Silicon Mac." >&2
    exit 1
fi
if ! command -v python3.13 >/dev/null; then
    echo "Install Python 3.13 first: brew install python@3.13" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    python3.13 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python -m pip check
if ! command -v ffmpeg >/dev/null && [[ ! -x /opt/homebrew/bin/ffmpeg ]]; then
    echo "ffmpeg is missing (it turns speech into MP3): brew install ffmpeg" >&2
fi
config="${ARCHE_WORKER_CONFIG:-$HOME/.config/arche-worker/config.toml}"
if [[ ! -f "$config" ]]; then
    mkdir -p "$(dirname "$config")"
    cp config.example.toml "$config"
    chmod 600 "$config"
    echo "Wrote $config: paste the key from the station's /mod there."
fi
echo "Ready. Next: ./run.sh check, then ./run.sh"
