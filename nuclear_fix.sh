#!/bin/bash
# =============================================================================
# THESIS SIM — NUCLEAR NUMPY FIX
# Peters PG/2415890
#
# Debian 12 injects /usr/lib/python3/dist-packages into every venv via
# sitecustomize.py, which causes the system numpy (1.24) to override the
# venv numpy (1.26.4). This script blocks that injection permanently.
#
# Run: bash nuclear_fix.sh
# =============================================================================

set -e

VENV_DIR="$HOME/thesis-sim/venv"
VENV_PY="$VENV_DIR/bin/python"
THESIS_DIR="$HOME/thesis-sim"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo ""
echo "============================================================"
echo "  NUCLEAR NUMPY FIX"
echo "  Blocking Debian system path injection into venv"
echo "============================================================"
echo ""

# =============================================================================
# STEP 1 — Nuke old venv
# =============================================================================
echo "[1/6] Removing old venv..."
rm -rf "$VENV_DIR"
echo "  Removed."

# =============================================================================
# STEP 2 — Create fresh venv
# =============================================================================
echo ""
echo "[2/6] Creating new venv..."
uv venv "$VENV_DIR" --python 3.11 --seed
echo "  Created: $VENV_DIR"

# =============================================================================
# STEP 3 — THE KEY FIX: Write a sitecustomize.py inside the venv that
# explicitly removes all Debian system paths from sys.path BEFORE
# any packages are imported.
# =============================================================================
echo ""
echo "[3/6] Blocking Debian system path injection..."

SITE_PACKAGES=$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
echo "  Venv site-packages: $SITE_PACKAGES"

cat > "$SITE_PACKAGES/sitecustomize.py" << 'SITECUSTOMIZE'
"""
sitecustomize.py — placed inside the venv site-packages.
Runs automatically on Python startup. Removes Debian system dist-packages
from sys.path so the venv packages are always found first.
"""
import sys

SYSTEM_PATHS = [
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.11/dist-packages',
    '/usr/local/lib/python3.11/dist-packages',
    '/usr/lib/python3.11',
]

removed = []
for p in SYSTEM_PATHS:
    if p in sys.path:
        sys.path.remove(p)
        removed.append(p)

# Optional: uncomment to debug
# if removed:
#     import os
#     if os.environ.get('THESIS_DEBUG'):
#         print(f"[sitecustomize] Removed system paths: {removed}", file=sys.stderr)
SITECUSTOMIZE

echo "  Written: $SITE_PACKAGES/sitecustomize.py"

# Verify it worked — sys.path should be clean now
"$VENV_PY" -c "
import sys
bad = [p for p in sys.path if 'dist-packages' in p]
if bad:
    print('  STILL DIRTY — remaining system paths:', bad)
    raise SystemExit(1)
else:
    print('  sys.path is clean — no system dist-packages.')
    print('  sys.path:', sys.path[:4])
"

# =============================================================================
# STEP 4 — Install numpy 1.26.4 and verify it lands in the venv
# =============================================================================
echo ""
echo "[4/6] Installing numpy 1.26.4..."
uv pip install --python "$VENV_PY" "numpy==1.26.4"

"$VENV_PY" -c "
import numpy
print(f'  numpy version:  {numpy.__version__}')
print(f'  numpy location: {numpy.__file__}')
assert 'thesis-sim/venv' in numpy.__file__, f'WRONG: {numpy.__file__}'
assert numpy.__version__ == '1.26.4', f'WRONG VERSION: {numpy.__version__}'
print('  numpy OK — correct version from venv.')
assert hasattr(numpy, 'dtypes'), 'numpy.dtypes missing — too old!'
print('  numpy.dtypes present — TensorFlow will be happy.')
"

# =============================================================================
# STEP 5 — Install TensorFlow + all ML packages
# =============================================================================
echo ""
echo "[5/6] Installing TensorFlow and ML packages..."

uv pip install --python "$VENV_PY" "tensorflow-cpu==2.15.0"

"$VENV_PY" -c "
import tensorflow as tf
print(f'  TensorFlow: {tf.__version__}')
print('  TensorFlow OK.')
"

uv pip install --python "$VENV_PY" \
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
echo "  Full verification..."
"$VENV_PY" -c "
import sys
print(f'  Python:           {sys.version.split()[0]}')
print(f'  Executable:       {sys.executable}')
import numpy;          print(f'  numpy:            {numpy.__version__}  [{numpy.__file__[:55]}]')
import tensorflow as tf; print(f'  tensorflow:       {tf.__version__}')
import sklearn;        print(f'  scikit-learn:     {sklearn.__version__}')
import imblearn;       print(f'  imbalanced-learn: {imblearn.__version__}')
import pandas;         print(f'  pandas:           {pandas.__version__}')
import scipy;          print(f'  scipy:            {scipy.__version__}')
import matplotlib;     print(f'  matplotlib:       {matplotlib.__version__}')
import seaborn;        print(f'  seaborn:          {seaborn.__version__}')
import joblib;         print(f'  joblib:           {joblib.__version__}')
print()
print('  ALL PACKAGES OK.')
"

# =============================================================================
# STEP 6 — Write activate_thesis.sh
# =============================================================================
echo ""
echo "[6/6] Writing activate_thesis.sh..."

cat > "$THESIS_DIR/activate_thesis.sh" << 'ACTIVATE'
#!/bin/bash
# SOURCE at the start of every new terminal session:
#   source ~/thesis-sim/activate_thesis.sh

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
source "$HOME/thesis-sim/venv/bin/activate"

VENV_PY="$HOME/thesis-sim/venv/bin/python"
NP_VER=$("$VENV_PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "?")
NP_LOC=$("$VENV_PY" -c "import numpy; print(numpy.__file__)" 2>/dev/null || echo "?")
TF_VER=$("$VENV_PY" -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "?")

echo ""
echo "  Thesis venv activated."
echo "  Python:     $(which python3)"
echo "  NumPy:      $NP_VER"
echo "  numpy from: $NP_LOC"
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

echo ""
echo "============================================================"
echo "  DONE. Run:"
echo ""
echo "    source ~/thesis-sim/activate_thesis.sh"
echo "    python3 check_environment.py"
echo "============================================================"
