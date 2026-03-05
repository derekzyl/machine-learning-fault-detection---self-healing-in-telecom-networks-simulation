#!/bin/bash
# source ~/thesis-sim/activate_thesis.sh
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
echo "  NumPy:      $NP  (must be 1.26.4)"
echo "  TensorFlow: $TF  (must be 2.15.0)"
echo ""
echo "  Commands:"
echo "    python3 check_environment.py"
echo "    python3 run_all_trials.py --workers 3"
echo "    python3 preprocess_and_train.py"
echo "    python3 mapek_loop.py --model all"
echo ""
