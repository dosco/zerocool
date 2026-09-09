#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_ENV="$TASK_ROOT/.cache/qwen-reference-venv"
TASK_PYTHON="${FREELLM_PYTHON:-python3}"
"$TASK_PYTHON" -m venv "$TASK_ENV"
"$TASK_ENV/bin/python" -m pip install -r "$TASK_ROOT/scripts/qwen/requirements-reference.txt"
printf 'Reference Python: %s\n' "$TASK_ENV/bin/python"
