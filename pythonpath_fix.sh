#!/bin/bash
# =============================================================================
# THESIS SIM — PYTHONPATH FIX
# Peters PG/2415890
#
# Root cause: PYTHONPATH=/usr/lib/python3/dist-packages is set in your shell,
# which overrides all venv isolation. This script clears it permanently.
# Run: bash pythonpath_fix.sh
# =============================================================================

set -e
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

VENV_DIR="$HOME/thesis-sim/venv"
THESIS_DIR="$HOME/thesis-sim"

echo ""
echo "============================================================"
echo "  ROOT CAUSE: PYTHONPATH is contaminating the venv"
echo "  Current PYTHONPATH: ${PYTHONPATH:-<empty>}"
echo "============================================================"
echo ""

# =============================================================================
# STEP 1 — Find where PYTHONPATH is being set and remove it
# =============================================================================
echo "[1/5] Locating PYTHONPATH in shell config files..."

for f in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" \
          "$HOME/.zshrc"  "$HOME/.zprofile" "$HOME/.zshenv" \
          "/etc/environment" "/etc/profile"; do
    if [ -f "$f" ] && grep -q "PYTHONPATH" "$f" 2>/dev/null; then
        echo "  FOUND in: $f"
        grep -n "PYTHONPATH" "$f"
    fi
done

echo ""
echo "[2/5] Removing PYTHONPATH from ~/.bashrc and ~/.profile..."

# Remove any PYTHONPATH= line from .bashrc
if [ -f "$HOME/.bashrc" ]; then
    # Back up first
    cp "$HOME/.bashrc" "$HOME/.bashrc.thesis_backup"
    # Remove lines containing PYTHONPATH
    grep -v "PYTHONPATH" "$HOME/.bashrc.thesis_backup" > "$HOME/.bashrc"
    echo "  Cleaned ~/.bashrc  (backup: ~/.bashrc.thesis_backup)"
fi

if [ -f "$HOME/.profile" ]; then
    cp "$HOME/.profile" "$HOME/.profile.thesis_backup"
    grep -v "PYTHONPATH" "$HOME/.profile.thesis_backup" > "$HOME/.profile"
    echo "  Cleaned ~/.profile (backup: ~/.profile.thesis_backup)"
fi

if [ -f "$HOME/.bash_profile" ]; then
    cp "$HOME/.bash_profile" "$HOME/.bash_profile.thesis_backup"
    grep -v "PYTHONPATH" "$HOME/.bash_profile.thesis_backup" > "$HOME/.bash_profile"
    echo "  Cleaned ~/.bash_profile"
fi

# Clear it in the current session immediately
unset PYTHONPATH
echo "  Cleared PYTHONPATH in current session."
echo "  PYTHONPATH is now: '${PYTHONPATH:-<empty — correct>}'"

# =============================================================================
# STEP 2 — Rebuild a fresh clean venv (now that PYTHONPATH is gone)
# =============================================================================
echo ""
echo "[3/5] Rebuilding venv from scratch (PYTHONPATH is now clear)..."
rm -rf "$VENV_DIR"
uv venv "$VENV_DIR" --python 3.11 --seed
echo "  Created: $VENV_DIR"

# Verify clean sys.path
"$VENV_DIR/bin/python" -c "
import sys
bad = [p for p in sys.path if 'dist-packages' in p]
if bad:
    print('STILL DIRTY:', bad)
    raise SystemExit(1)
print('  sys.path is CLEAN.')
print('  sys.path:', [p for p in sys.path if p][:4])
"

# =============================================================================
# STEP 3 — Install all packages
# =============================================================================
echo ""
echo "[4/5] Installing all ML packages..."

uv pip install --python "$VENV_DIR/bin/python" "numpy==1.26.4"

"$VENV_DIR/bin/python" -c "
import numpy
assert 'thesis-sim/venv' in numpy.__file__, f'Wrong numpy: {numpy.__file__}'
assert numpy.__version__ == '1.26.4'
assert hasattr(numpy, 'dtypes')
print(f'  numpy {numpy.__version__} — OK from venv')
"

uv pip install --python "$VENV_DIR/bin/python" "tensorflow-cpu==2.15.0"

"$VENV_DIR/bin/python" -c "
import tensorflow as tf
print(f'  TensorFlow {tf.__version__} — OK')
"

uv pip install --python "$VENV_DIR/bin/python" \
    pandas scipy "scikit-learn>=1.3" imbalanced-learn \
    matplotlib seaborn joblib shap jupyter notebook

echo ""
echo "  Full verification..."
"$VENV_DIR/bin/python" -c "
import sys, numpy, tensorflow as tf, sklearn, imblearn, pandas, scipy, matplotlib, seaborn, joblib
print(f'Python:           {sys.version.split()[0]}')
print(f'numpy:            {numpy.__version__}  [{numpy.__file__[:55]}]')
print(f'tensorflow:       {tf.__version__}')
print(f'scikit-learn:     {sklearn.__version__}')
print(f'imbalanced-learn: {imblearn.__version__}')
print(f'pandas:           {pandas.__version__}')
print(f'scipy:            {scipy.__version__}')
print(f'matplotlib:       {matplotlib.__version__}')
print(f'seaborn:          {seaborn.__version__}')
print(f'joblib:           {joblib.__version__}')
print()
print('ALL PACKAGES OK.')
"

# =============================================================================
# STEP 4 — Write activate_thesis.sh that always clears PYTHONPATH on entry
# =============================================================================
echo ""
echo "[5/5] Writing activate_thesis.sh..."
cat > "$THESIS_DIR/activate_thesis.sh" << 'ACTIVATE'
#!/bin/bash
# SOURCE at the start of every terminal session:
#   source ~/thesis-sim/activate_thesis.sh

# Always clear PYTHONPATH — this was the root cause of the numpy conflict
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

source "$HOME/thesis-sim/venv/bin/activate"

VENV_PY="$HOME/thesis-sim/venv/bin/python"
NP_VER=$("$VENV_PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "ERROR")
TF_VER=$("$VENV_PY" -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "ERROR")

echo ""
echo "  Thesis venv activated.  PYTHONPATH cleared."
echo "  Python:     $(which python3)"
echo "  NumPy:      $NP_VER"
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

echo ""
echo "============================================================"
echo "  DONE."
echo ""
echo "  PYTHONPATH has been removed from your shell config."
echo "  activate_thesis.sh will also clear it on every session."
echo ""
echo "  Now run:"
echo "    source ~/thesis-sim/activate_thesis.sh"
echo "    python3 check_environment.py"
echo "============================================================"
