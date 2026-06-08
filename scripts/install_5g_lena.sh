#!/usr/bin/env bash
# Install CTTC 5G-LENA (nr module) for NS-3 3.38 — thesis 5G NR support.
set -euo pipefail

NS3_DIR="${NS3_DIR:-$HOME/ns-3.38}"
NS3_WRAPPER="${NS3_WRAPPER:-$HOME/thesis-sim/bin/ns3}"
NR_DIR="$NS3_DIR/contrib/nr"
NR_BRANCH="5g-lena-v2.4.y"

echo "══ 5G-LENA install (NS-3 3.38 + nr $NR_BRANCH) ══"

if ! command -v git >/dev/null; then
  echo "ERROR: git required"
  exit 1
fi

if ! dpkg -s libsqlite3-dev >/dev/null 2>&1; then
  echo "Install sqlite dev headers: sudo apt-get install -y libsqlite3-dev"
  exit 1
fi

mkdir -p "$NS3_DIR/contrib"
if [ ! -d "$NR_DIR/.git" ]; then
  echo "[1/3] Cloning https://gitlab.com/cttc-lena/nr.git ($NR_BRANCH)..."
  git clone --depth 1 --branch "$NR_BRANCH" https://gitlab.com/cttc-lena/nr.git "$NR_DIR"
else
  echo "[1/3] NR module present at $NR_DIR"
fi

MODULES="core,network,internet,applications,mobility,spectrum,propagation,antenna,lte,nr,energy,flow-monitor,point-to-point,stats"

echo "[2/3] Configuring NS-3 with modules: $MODULES"
cd "$NS3_DIR"
"$NS3_WRAPPER" configure --build-profile=optimized --enable-modules="$MODULES"

echo "[3/3] Building (10–40 min)..."
"$NS3_WRAPPER" build

echo ""
echo "Done. Verify:"
echo "  cd $NS3_DIR && $NS3_WRAPPER run cttc-nr-demo --help 2>/dev/null | head -3 || $NS3_WRAPPER run nr-demo --help | head -3"
echo "  Next: thesis-fault-sim-nr.cc (Phase 3) — not yet wired to run_all_trials.py"
