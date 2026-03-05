#!/bin/bash
# =============================================================================
# THESIS SIM — PYTHON ENVIRONMENT FIX SCRIPT
# Peters PG/2415890
#
# This script fixes the numpy conflict where the system numpy at
# /usr/lib/python3/dist-packages/numpy/ bleeds into the venv.
#
# Run: bash fix_python_env.sh
# =============================================================================

set -e

THESIS_DIR="$HOME/thesis-sim"
VENV_DIR="$THESIS_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"

echo ""
echo "============================================================"
echo "  PYTHON ENVIRONMENT FIX"
echo "  Fixing numpy/TensorFlow conflict"
echo "============================================================"
echo ""

# Make sure uv is on PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[!] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
echo "[OK] uv: $(uv --version)"

# =============================================================================
# STEP 1 — Nuke the old venv completely
# =============================================================================
echo ""
echo "[1/5] Removing old virtual environment..."
rm -rf "$VENV_DIR"
echo "  Removed: $VENV_DIR"

# =============================================================================
# STEP 2 — Create a brand new isolated venv
# The key flag is --system-site-packages is NOT passed,
# which means the venv is 100% isolated from system packages.
# =============================================================================
echo ""
echo "[2/5] Creating fresh isolated virtual environment..."
uv venv "$VENV_DIR" \
    --python 3.11 \
    --seed

echo "  Created: $VENV_DIR"

# Confirm the venv Python is isolated from system site-packages
echo "  Checking isolation..."
"$VENV_PY" -c "
import sys
# Make sure system dist-packages is NOT in sys.path
bad_paths = [p for p in sys.path if 'dist-packages' in p or '/usr/lib/python3' in p]
if bad_paths:
    print('  WARNING: System paths detected in venv:', bad_paths)
    print('  Proceeding anyway - uv install will override.')
else:
    print('  Venv is fully isolated from system packages.')
print(f'  sys.executable = {sys.executable}')
"

# =============================================================================
# STEP 3 — Install numpy 1.26.4 first (required by TF 2.15)
# =============================================================================
echo ""
echo "[3/5] Installing numpy 1.26.4 (TensorFlow-compatible version)..."
uv pip install \
    --python "$VENV_PY" \
    "numpy==1.26.4"

# Verify the venv has the RIGHT numpy — not the system one
"$VENV_PY" -c "
import numpy
print(f'  numpy version:    {numpy.__version__}')
print(f'  numpy location:   {numpy.__file__}')
if 'thesis-sim/venv' not in numpy.__file__:
    print('  ERROR: Wrong numpy loaded! Not from venv.')
    raise SystemExit(1)
print('  numpy OK — loaded from venv.')
print(f'  np.dtypes test:   {hasattr(numpy, \"dtypes\")}')
"

# =============================================================================
# STEP 4 — Install TensorFlow + all other packages
# =============================================================================
echo ""
echo "[4/5] Installing TensorFlow and ML packages..."

uv pip install \
    --python "$VENV_PY" \
    "tensorflow-cpu==2.15.0"

echo "  TensorFlow installed. Verifying..."
"$VENV_PY" -c "
import tensorflow as tf
print(f'  TensorFlow:       {tf.__version__}')
print('  TensorFlow OK.')
"

echo ""
echo "  Installing remaining ML packages..."
uv pip install \
    --python "$VENV_PY" \
    pandas \
    scipy \
    "scikit-learn>=1.3" \
    imbalanced-learn \
    matplotlib \
    seaborn \
    joblib \
    shap \
    jupyter \
    notebook

echo ""
echo "  Final verification..."
"$VENV_PY" -c "
import sys
print(f'  Python:           {sys.version.split()[0]}')
print(f'  Executable:       {sys.executable}')

import numpy;      print(f'  numpy:            {numpy.__version__}  from {numpy.__file__[:50]}...')
import tensorflow as tf; print(f'  tensorflow:       {tf.__version__}')
import sklearn;    print(f'  scikit-learn:     {sklearn.__version__}')
import imblearn;   print(f'  imbalanced-learn: {imblearn.__version__}')
import pandas;     print(f'  pandas:           {pandas.__version__}')
import scipy;      print(f'  scipy:            {scipy.__version__}')
import matplotlib; print(f'  matplotlib:       {matplotlib.__version__}')
import seaborn;    print(f'  seaborn:          {seaborn.__version__}')
import joblib;     print(f'  joblib:           {joblib.__version__}')
print()
print('  ALL PACKAGES OK.')
"

# =============================================================================
# STEP 5 — Rewrite activate_thesis.sh (was missing before)
# =============================================================================
echo ""
echo "[5/5] Writing activate_thesis.sh..."

mkdir -p "$THESIS_DIR"

cat > "$THESIS_DIR/activate_thesis.sh" << 'ACTIVATE'
#!/bin/bash
# SOURCE this at the start of every new terminal session:
#   source ~/thesis-sim/activate_thesis.sh

VENV_DIR="$HOME/thesis-sim/venv"
VENV_PY="$VENV_DIR/bin/python"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
source "$VENV_DIR/bin/activate"

# Confirm we have the right numpy (must come from venv, not system)
NUMPY_LOC=$("$VENV_PY" -c "import numpy; print(numpy.__file__)" 2>/dev/null || echo "NOT FOUND")
NUMPY_VER=$("$VENV_PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "?")
TF_VER=$("$VENV_PY"    -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "?")

echo ""
echo "  Thesis venv activated"
echo "  Python:     $(which python3)"
echo "  NumPy:      $NUMPY_VER  (from: $NUMPY_LOC)"
echo "  TensorFlow: $TF_VER"
echo ""
echo "  Commands:"
echo "    python3 check_environment.py"
echo "    python3 run_all_trials.py --workers 3"
echo "    python3 preprocess_and_train.py"
echo "    python3 mapek_loop.py --model all"
echo ""
ACTIVATE

chmod +x "$THESIS_DIR/activate_thesis.sh"
echo "  Written: $THESIS_DIR/activate_thesis.sh"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "============================================================"
echo "  FIX COMPLETE"
echo "============================================================"
echo ""
echo "  To use your environment, run:"
echo ""
echo "    source ~/thesis-sim/activate_thesis.sh"
echo ""
echo "  Then verify:"
echo ""
echo "    cd ~/thesis-sim"
echo "    python3 check_environment.py"
echo ""
echo "============================================================"
