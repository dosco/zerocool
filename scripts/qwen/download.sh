#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_PYTHON="${ZEROCOOL_PYTHON:-$TASK_ROOT/.cache/qwen-download-venv/bin/python}"
if [ ! -x "$TASK_PYTHON" ]; then
    python3 -m venv "$TASK_ROOT/.cache/qwen-download-venv"
fi
"$TASK_PYTHON" -c 'import huggingface_hub' 2>/dev/null || "$TASK_PYTHON" -m pip install 'huggingface_hub==1.30.0'
"$TASK_PYTHON" "$TASK_ROOT/scripts/qwen/download.py" "$@"
