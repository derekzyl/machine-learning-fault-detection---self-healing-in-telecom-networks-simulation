#!/bin/bash
# =============================================================================
# THESIS SIM — FINAL CLEAN INSTALL

#
# Fixes: numpy being upgraded to 2.x by shap/numba dependencies.
# Solution: install ALL packages in ONE uv call with numpy<2 constraint,
# so the resolver never picks numpy 2.x. Also drops shap (pulls numba
# which forces numpy>=2) — we do not need shap for the thesis pipeline.
#
# Run: bash final_clean_install.sh
# =============================================================================

set -e
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH

VENV_DIR="$HOME/thesis-sim/venv"
THESIS_DIR="$HOME/thesis-sim"

echo ""
echo "============================================================"
echo "  FINAL CLEAN INSTALL"
echo "  numpy<2 constraint enforced across all packages"
echo "============================================================"
echo ""

# ── 1. Nuke old venv ──────────────────────────────────────────────────────────
echo "[1/4] Removing old venv..."
rm -rf "$VENV_DIR"

# ── 2. Fresh venv ─────────────────────────────────────────────────────────────
echo "[2/4] Creating fresh venv..."
uv venv "$VENV_DIR" --python 3.11 --seed
echo "  Created: $VENV_DIR"

# ── 3. Install everything in ONE call with numpy pinned <2 ────────────────────
# Key points:
#   - "numpy>=1.23.5,<2.0"  is a hard constraint uv will honour globally
#   - tensorflow 2.15 + numpy 1.26.4 is a known-good combination
#   - shap removed (pulls numba which requires numpy>=2)
#   - numba removed (same reason)
#   - We use torch-free, gpu-free tensorflow-cpu to avoid heavy CUDA deps

echo ""
echo "[3/4] Installing all packages with numpy<2 constraint..."
echo "  (This takes 3-5 minutes — uv resolves all versions together)"
echo ""

uv pip install \
    --python "$VENV_DIR/bin/python" \
    "numpy>=1.23.5,<2.0" \
    "tensorflow-cpu==2.15.0" \
    "pandas>=1.5,<3" \
    "scipy>=1.9" \
    "scikit-learn>=1.3,<1.6" \
    "imbalanced-learn>=0.11" \
    "matplotlib>=3.6" \
    "seaborn>=0.12" \
    "joblib>=1.2" \
    "jupyter" \
    "notebook"

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying all packages..."
"$VENV_DIR/bin/python" -c "
import sys
print(f'Python:     {sys.version.split()[0]}  |  {sys.executable}')
print()

import numpy
loc = numpy.__file__
assert 'thesis-sim/venv' in loc, f'WRONG numpy location: {loc}'
assert numpy.__version__.startswith('1.'), f'WRONG numpy version: {numpy.__version__}'
assert hasattr(numpy, 'dtypes'), 'numpy.dtypes missing'
print(f'numpy:            {numpy.__version__}  [venv OK]')

import tensorflow as tf
print(f'tensorflow:       {tf.__version__}')

import sklearn
print(f'scikit-learn:     {sklearn.__version__}')

import imblearn
print(f'imbalanced-learn: {imblearn.__version__}')

import pandas
print(f'pandas:           {pandas.__version__}')

import scipy
print(f'scipy:            {scipy.__version__}')

import matplotlib
print(f'matplotlib:       {matplotlib.__version__}')

import seaborn
print(f'seaborn:          {seaborn.__version__}')

import joblib
print(f'joblib:           {joblib.__version__}')

print()
print('ALL PACKAGES OK.')
"

# ── Write activate_thesis.sh ───────────────────────────────────────────────────
cat > "$THESIS_DIR/activate_thesis.sh" << 'ACTIVATE'
#!/bin/bash
# SOURCE this at the start of every terminal session:
#   source ~/thesis-sim/activate_thesis.sh

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
source "$HOME/thesis-sim/venv/bin/activate"

VENV_PY="$HOME/thesis-sim/venv/bin/python"
NP=$("$VENV_PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "ERROR")
TF=$("$VENV_PY" -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "ERROR")

echo ""
echo "  Thesis venv activated  (PYTHONPATH cleared)"
echo "  Python:     $(which python3)"
echo "  NumPy:      $NP  (must be 1.x)"
echo "  TensorFlow: $TF"
echo ""
echo "  Commands:"
echo "    python3 check_environment.py"
echo "    python3 run_all_trials.py --workers 3"
echo "    python3 preprocess_and_train.py"
echo "    python3 mapek_loop.py --model all"
echo ""
ACTIVATE
chmod +x "$THESIS_DIR/activate_thesis.sh"

echo ""
echo "============================================================"
echo "  DONE."
echo ""
echo "  Run:"
echo "    source ~/thesis-sim/activate_thesis.sh"
echo "    python3 check_environment.py"
echo "============================================================"
