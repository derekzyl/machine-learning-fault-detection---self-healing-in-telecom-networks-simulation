#!/usr/bin/env bash
# Run NS-3's ./ns3 via thesis venv Python 3.11.
# System python3 may be 3.14+, which breaks NS-3 3.38's argparse usage.
set -euo pipefail
NS3_DIR="${NS3_DIR:-$HOME/ns-3.38}"
VENV_PY="${VENV_PY:-$HOME/thesis-sim/venv/bin/python}"
cd "$NS3_DIR"
exec "$VENV_PY" ./ns3 "$@"
