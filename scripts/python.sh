#!/usr/bin/env bash
set -euo pipefail
PTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PTS_REPOS="$(dirname -- "$PTS_ROOT")"
PTS_MODEL_ROOT="${PTS_OPENPI_ROOT:-$PTS_REPOS/openpi}"
PTS_CONTROL_ROOT="${PTS_FORCEBENCH_ROOT:-$PTS_REPOS/ForceBenchWeb}"
PTS_INTERPRETER="${PTS_PYTHON:-$PTS_MODEL_ROOT/.venv/bin/python}"
export PYTHONPATH="$PTS_ROOT/src:$PTS_CONTROL_ROOT:$PTS_MODEL_ROOT/src:$PTS_MODEL_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
PTS_ASSET_CACHE="$(dirname -- "$PTS_REPOS")/models/openpi"
if [[ -z "${OPENPI_DATA_HOME:-}" && -f "$PTS_ASSET_CACHE/big_vision/paligemma_tokenizer.model" ]]; then
    export OPENPI_DATA_HOME="$PTS_ASSET_CACHE"
fi
if [[ ! -x "$PTS_INTERPRETER" ]]; then
    echo "Set PTS_PYTHON to the Python executable in your existing inference/control environment." >&2
    exit 1
fi
exec "$PTS_INTERPRETER" "$@"
