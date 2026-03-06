#!/bin/bash
# =============================================================================
# THESIS NS-3 INSTALLATION SCRIPT (Debian 12 / Ubuntu — uv + locked venv)

# Run: bash install_ns3_thesis.sh
# =============================================================================

set -e
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH

THESIS_DIR="$HOME/thesis-sim"
NS3_DIR="$HOME/ns-3.38"
VENV_DIR="$THESIS_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"

echo ""
echo "============================================================"
echo "  THESIS NS-3 SIMULATION — INSTALLATION"
echo "============================================================"
echo ""

# STEP 1 — System packages
echo "[1/6] System build packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake ninja-build ccache g++ gcc \
    git wget curl unzip \
    libboost-all-dev libssl-dev libxml2-dev libsqlite3-dev \
    gsl-bin libgsl-dev libgtk-3-dev pkg-config \
    python3 python3-dev ca-certificates
echo "  Done."

# STEP 2 — uv
echo ""
echo "[2/6] Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

# STEP 3 — Python environment
echo ""
echo "[3/6] Python environment..."
if "$VENV_PY" -c "import tensorflow as tf; assert tf.__version__ == '2.15.0'" 2>/dev/null; then
    echo "  Venv already has correct packages — skipping reinstall."
elif [ -f "$THESIS_DIR/requirements.txt" ]; then
    echo "  Installing from locked requirements.txt..."
    [ -d "$VENV_DIR" ] || uv venv "$VENV_DIR" --python 3.11 --seed
    uv pip install --python "$VENV_PY" -r "$THESIS_DIR/requirements.txt"
    echo "  Done."
else
    echo "  No requirements.txt found."
    echo "  Run: bash lock_and_repair.sh  first."
    exit 1
fi

# STEP 4 — Download NS-3.38
echo ""
echo "[4/6] NS-3.38..."
if [ -d "$NS3_DIR" ]; then
    echo "  Already at $NS3_DIR — skipping."
else
    cd "$HOME"
    wget -q --show-progress \
        https://www.nsnam.org/releases/ns-allinone-3.38.tar.bz2
    tar xjf ns-allinone-3.38.tar.bz2
    mv ns-allinone-3.38/ns-3.38 "$NS3_DIR"
    rm -rf ns-allinone-3.38 ns-allinone-3.38.tar.bz2
    echo "  Ready at $NS3_DIR"
fi

# STEP 5 — Configure and build NS-3
echo ""
echo "[5/6] Building NS-3 (10-30 minutes)..."
if [ -f "$NS3_DIR/build/lib/libns3.38-lte-optimized.so" ] || \
   [ -f "$NS3_DIR/build/lib/libns3-dev-lte-default.so" ] || \
   ls "$NS3_DIR"/build/lib/libns3*lte* 2>/dev/null | head -1 | grep -q lte; then
    echo "  NS-3 already built — skipping."
else
    cd "$NS3_DIR"
    export PYTHON="$VENV_PY"
    ./ns3 configure \
        --build-profile=optimized \
        --enable-modules=lte,network,internet,applications,mobility,energy,flow-monitor,point-to-point
    ./ns3 build
    echo "  NS-3 build complete."
fi

# STEP 6 — Thesis simulation script
echo ""
echo "[6/6] Thesis simulation script..."
if [ -f "$THESIS_DIR/scripts/thesis-fault-sim.cc" ]; then
    cp "$THESIS_DIR/scripts/thesis-fault-sim.cc" "$NS3_DIR/scratch/"
    cd "$NS3_DIR"
    ./ns3 build thesis-fault-sim
    echo "  Compiled."
    ./ns3 run "thesis-fault-sim --trial=0 --fault=power --outputDir=$THESIS_DIR/output/raw" 2>&1 | tail -3
else
    echo "  Place thesis-fault-sim.cc in $THESIS_DIR/scripts/ then re-run."
fi

echo ""
echo "============================================================"
echo "  DONE"
echo ""
echo "  source ~/thesis-sim/activate_thesis.sh"
echo "  python3 check_environment.py"
echo "  python3 run_all_trials.py --workers 3"
echo "============================================================"
