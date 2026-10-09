#!/bin/bash
# Installs herde on an Apple Silicon Mac: a venv with Qwen3-TTS through mlx-audio (Docker cannot reach a
# Mac's GPU, so herde runs natively here), a config, and a launchd job file.
#
# It starts nothing and stops nothing: an arche-worker may still be serving its stations. At the end it
# prints the commands that switch over.
#
#   macos/install.sh                         asks whether to take over ~/.config/arche-worker
#   macos/install.sh --import-arche-worker   takes it over without asking
#   macos/install.sh --no-import             starts with an empty config
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
cd "$repo"
import="ask"
for arg in "$@"; do
    case "$arg" in
        --import-arche-worker) import="yes" ;;
        --no-import) import="no" ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This installs Qwen3-TTS on Apple's GPU: it needs an Apple Silicon Mac. Elsewhere use docker/compose.yaml." >&2
    exit 1
fi
if ! command -v python3.13 >/dev/null; then
    echo "Install Python 3.13 first: brew install python@3.13" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    python3.13 -m venv .venv
fi
# The tested set of packages first (the qwen3-tts-ui demo's on an M1 Max), then herde itself on top.
.venv/bin/python -m pip install --disable-pip-version-check -r macos/requirements-mlx.lock
.venv/bin/python -m pip install --disable-pip-version-check --no-deps -e .
.venv/bin/python -m pip check
if ! command -v ffmpeg >/dev/null && [[ ! -x /opt/homebrew/bin/ffmpeg ]]; then
    echo "ffmpeg is missing (it turns speech into MP3): brew install ffmpeg" >&2
fi

config="${HERDE_CONFIG:-$HOME/.config/herde/config.toml}"
old="$HOME/.config/arche-worker/config.toml"
if [[ ! -f "$config" && -f "$old" ]]; then
    if [[ "$import" == "ask" && -t 0 ]]; then
        read -r -p "Take over $old (its stations' keys, voices and speech settings)? [Y/n] " answer
        [[ "${answer:-y}" =~ ^[Yy] ]] && import="yes"
    fi
    if [[ "$import" == "yes" ]]; then
        .venv/bin/herde --config "$config" import-arche-worker "$old"
    fi
fi
if [[ ! -f "$config" ]]; then
    .venv/bin/python - "$config" <<'PY'
import sys
from pathlib import Path
from herde.cli import starter_config, write_secret
import socket
write_secret(Path(sys.argv[1]), starter_config(socket.gethostname(), tts_backend="mlx"))
PY
    echo "Wrote $config. Join a project with the invite its admin made: .venv/bin/herde join <url> <code>"
fi

label="de.schaefchens.herde"
plist="$HOME/Library/LaunchAgents/$label.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s|__HERDE__|$repo/.venv/bin/herde|g" -e "s|__REPO__|$repo|g" -e "s|__CONFIG__|$config|g" -e "s|__LOG__|$HOME/Library/Logs/herde.log|g" \
    macos/de.schaefchens.herde.plist.template > "$plist.new"
mv "$plist.new" "$plist"
echo
echo "Installed. Next:"
echo "  .venv/bin/herde check                      the config, ffmpeg, Ollama and each project"
if launchctl print "gui/$(id -u)/de.schaefchens.arche-worker" >/dev/null 2>&1; then
    echo "  launchctl bootout gui/$(id -u)/de.schaefchens.arche-worker     stop the old worker (its stations must speak protocol 2 first)"
fi
echo "  launchctl bootstrap gui/$(id -u) $plist   start herde at login, and now"
echo "  open http://127.0.0.1:8737                 its status page; the log is ~/Library/Logs/herde.log"
