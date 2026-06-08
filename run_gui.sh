#!/usr/bin/env bash
# Launch the Thesis Pipeline graphical interface.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pick_python() {
  local candidates=(
    "${HOME}/thesis-sim/venv/bin/python"
    /usr/bin/python3.12
    /usr/bin/python3.11
    /usr/bin/python3
  )
  local py
  for py in "${candidates[@]}"; do
    [[ -x "$py" ]] || continue
    if "$py" -c "import tkinter" 2>/dev/null; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "Tkinter is not available for any Python on this system."
  echo "Run setup first:  bash setup.sh"
  echo "Or install:       sudo apt install python3-tk"
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/thesis_gui.py" "$@"
