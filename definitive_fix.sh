#!/bin/bash
# =============================================================================
# THESIS SIM — DEFINITIVE FIX (diagnose then fix)
# Peters PG/2415890
# Run: bash definitive_fix.sh
# =============================================================================

set -e
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

VENV_DIR="$HOME/thesis-sim/venv"
THESIS_DIR="$HOME/thesis-sim"

echo ""
echo "============================================================"
echo "  STEP 0 — DIAGNOSING HOW /usr/lib/python3/dist-packages"
echo "  IS GETTING INTO THE VENV"
echo "============================================================"
echo ""

# Find every .pth file and sitecustomize in the venv
echo "--- .pth files in venv ---"
find "$VENV_DIR" -name "*.pth" 2>/dev/null && echo "(none found)" || true

echo ""
echo "--- sitecustomize files on the whole system ---"
find /usr/lib/python3.11 /usr/lib/python3 -name "sitecustomize.py" 2>/dev/null | head -10

echo ""
echo "--- usercustomize ---"
find "$HOME" -name "usercustomize.py" 2>/dev/null | head -5 && echo "(none in HOME)" || true

echo ""
echo "--- PYTHONPATH env var ---"
echo "${PYTHONPATH:-<not set>}"

echo ""
echo "--- Full sys.path inside venv ---"
"$VENV_DIR/bin/python" -c "import sys; [print('  ', p) for p in sys.path]"

echo ""
echo "--- Where dist-packages is being added from ---"
"$VENV_DIR/bin/python" -c "
import sys, site
print('site.ENABLE_USER_SITE:', site.ENABLE_USER_SITE)
print('site.getusersitepackages():', site.getusersitepackages())

# Trace exactly which file adds dist-packages
import importlib.util
for name in ['sitecustomize', 'usercustomize']:
    spec = importlib.util.find_spec(name)
    if spec:
        print(f'{name}.py found at: {spec.origin}')
    else:
        print(f'{name}.py: not found')
"

echo ""
echo "============================================================"
echo "  APPLYING DEFINITIVE FIX"
echo "============================================================"
echo ""

# =============================================================================
# THE FIX: Use PYTHONNOUSERSITE + a .pth file that removes system paths
# We ALSO set PYTHONPATH="" to block env leakage
# AND we patch the venv's python wrapper to always set these env vars
# =============================================================================

VENV_PY="$VENV_DIR/bin/python"
VENV_BIN="$VENV_DIR/bin"
SITE_PKG=$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")

echo "[A] Writing sitecustomize.py with aggressive path cleaning..."
cat > "$SITE_PKG/sitecustomize.py" << 'EOF'
import sys

# Remove ALL system dist-packages paths — must happen before any import
_BAD = set()
for _p in list(sys.path):
    if any(x in _p for x in [
        '/usr/lib/python3/dist-packages',
        '/usr/lib/python3.11/dist-packages',
        '/usr/local/lib/python3.11/dist-packages',
        '/usr/lib/python3.11/lib-dynload',   # keep this actually
    ]):
        if 'lib-dynload' not in _p:          # lib-dynload is needed
            sys.path.remove(_p)
            _BAD.add(_p)

# Also pre-empt usercustomize from re-adding them
import site as _site
_orig_addsitepackages = _site.addsitepackages
def _patched_addsitepackages(paths, prefixes=None):
    if prefixes is None:
        prefixes = [sys.prefix, sys.exec_prefix]
    result = _orig_addsitepackages(paths, prefixes)
    for _p in list(sys.path):
        if '/usr/lib/python3/dist-packages' in _p:
            try:
                sys.path.remove(_p)
            except ValueError:
                pass
    return result
_site.addsitepackages = _patched_addsitepackages
EOF

echo "  Written."

echo ""
echo "[B] Verifying clean sys.path..."
CLEAN=$("$VENV_PY" -c "
import sys
bad = [p for p in sys.path if '/usr/lib/python3/dist-packages' in p]
print('DIRTY' if bad else 'CLEAN')
if bad: print('remaining:', bad)
")
echo "  Result: $CLEAN"

if echo "$CLEAN" | grep -q "DIRTY"; then
    echo ""
    echo "[C] sitecustomize still not working — using PYTHONNOUSERSITE wrapper..."

    # Rename the real python binary and wrap it with env vars set
    if [ ! -f "$VENV_BIN/python3.11.real" ]; then
        mv "$VENV_BIN/python3.11" "$VENV_BIN/python3.11.real"
    fi

    cat > "$VENV_BIN/python3.11" << WRAPPER
#!/bin/bash
# Wrapper that forces clean Python environment
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
exec "$VENV_BIN/python3.11.real" "\$@"
WRAPPER
    chmod +x "$VENV_BIN/python3.11"

    # Update the python and python3 symlinks to point through wrapper
    ln -sf "$VENV_BIN/python3.11" "$VENV_BIN/python3"
    ln -sf "$VENV_BIN/python3.11" "$VENV_BIN/python"

    echo "  Wrapper installed."

    echo ""
    echo "[D] Re-verifying..."
    CLEAN2=$("$VENV_BIN/python3.11" -c "
import sys
bad = [p for p in sys.path if '/usr/lib/python3/dist-packages' in p]
print('DIRTY' if bad else 'CLEAN')
if bad: print('remaining:', bad)
else: print('sys.path:', [p for p in sys.path if p][:5])
")
    echo "  Result: $CLEAN2"
fi

# =============================================================================
# Now install packages into the now-clean venv
# =============================================================================
echo ""
echo "[1] Installing numpy 1.26.4..."
uv pip install --python "$VENV_DIR/bin/python" "numpy==1.26.4"

echo "  Verifying numpy..."
"$VENV_DIR/bin/python" -c "
import numpy
print(f'  version:  {numpy.__version__}')
print(f'  location: {numpy.__file__}')
assert 'thesis-sim/venv' in numpy.__file__, f'WRONG LOCATION: {numpy.__file__}'
assert numpy.__version__ == '1.26.4', f'WRONG VERSION: {numpy.__version__}'
assert hasattr(numpy, 'dtypes'), 'numpy.dtypes missing'
print('  numpy VERIFIED.')
"

echo ""
echo "[2] Installing TensorFlow CPU..."
uv pip install --python "$VENV_DIR/bin/python" "tensorflow-cpu==2.15.0"

"$VENV_DIR/bin/python" -c "
import tensorflow as tf
print(f'  TensorFlow: {tf.__version__}')
print('  TensorFlow VERIFIED.')
"

echo ""
echo "[3] Installing remaining ML packages..."
uv pip install --python "$VENV_DIR/bin/python" \
    pandas scipy "scikit-learn>=1.3" imbalanced-learn \
    matplotlib seaborn joblib shap jupyter notebook

echo ""
echo "[4] Final full verification..."
"$VENV_DIR/bin/python" -c "
import sys
print(f'Python: {sys.version.split()[0]}  |  {sys.executable}')
print()
import numpy;          print(f'numpy:            {numpy.__version__:10s}  {numpy.__file__[:60]}')
import tensorflow as tf; print(f'tensorflow:       {tf.__version__}')
import sklearn;        print(f'scikit-learn:     {sklearn.__version__}')
import imblearn;       print(f'imbalanced-learn: {imblearn.__version__}')
import pandas;         print(f'pandas:           {pandas.__version__}')
import scipy;          print(f'scipy:            {scipy.__version__}')
import matplotlib;     print(f'matplotlib:       {matplotlib.__version__}')
import seaborn;        print(f'seaborn:          {seaborn.__version__}')
import joblib;         print(f'joblib:           {joblib.__version__}')
print()
print('ALL PACKAGES VERIFIED.')
"

# =============================================================================
# Write activate_thesis.sh
# =============================================================================
echo ""
echo "[5] Writing activate_thesis.sh..."
cat > "$THESIS_DIR/activate_thesis.sh" << 'ACTIVATE'
#!/bin/bash
# SOURCE this at the start of every terminal session:
#   source ~/thesis-sim/activate_thesis.sh

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
source "$HOME/thesis-sim/venv/bin/activate"

VENV_PY="$HOME/thesis-sim/venv/bin/python"
NP_VER=$("$VENV_PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "ERROR")
NP_LOC=$("$VENV_PY" -c "import numpy; print(numpy.__file__)" 2>/dev/null || echo "ERROR")
TF_VER=$("$VENV_PY" -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "ERROR")

echo ""
echo "  Thesis venv activated."
echo "  Python:     $(which python3)"
echo "  NumPy:      $NP_VER  ($NP_LOC)"
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
echo "  Written."

echo ""
echo "============================================================"
echo "  DONE. Now run:"
echo ""
echo "    source ~/thesis-sim/activate_thesis.sh"
echo "    python3 check_environment.py"
echo "============================================================"
