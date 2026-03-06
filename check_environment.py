#!/usr/bin/env python3
"""
THESIS SIMULATION — ENVIRONMENT CHECK
Peters

Run this ONLY after activating the venv:
    source ~/thesis-sim/activate_thesis.sh
    python3 check_environment.py

Or run it directly using the venv python without activating:
    ~/thesis-sim/venv/bin/python check_environment.py
"""

import os
import subprocess
import sys

# ── Determine the venv python path ──────────────────────────────────────────
VENV_PY = os.path.expanduser("~/thesis-sim/venv/bin/python")


def run(cmd):
    """Run a shell command, return (ok, stdout, stderr)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip(), r.stderr.strip()


def check_cmd(cmd, label):
    ok, out, err = run(cmd)
    tag = "[OK]     " if ok else "[MISSING]"
    detail = out.split("\n")[0] if ok and out else ""
    print(f"  {tag} {label}" + (f"  ({detail})" if detail else ""))
    if not ok and err:
        print(f"           {err[:100]}")
    return ok


def check_pymod(module, label, py=None):
    """Check a Python module using a specific python binary."""
    python = py or VENV_PY
    code = f"import importlib; m=importlib.import_module('{module}'); print(getattr(m,'__version__','OK'))"
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        ver = r.stdout.strip()
        print(f"  [OK]     {label:32s} v{ver}")
        return True
    else:
        print(f"  [MISSING] {label}")
        err_line = r.stderr.strip().split("\n")[-1][:100] if r.stderr else ""
        if err_line:
            print(f"           {err_line}")
        return False


def check_numpy_source(py=None):
    """Verify numpy is coming from the venv, not the system."""
    python = py or VENV_PY
    code = "import numpy; print(numpy.__file__)"
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print("  [ERROR]   Cannot import numpy at all")
        return False
    loc = r.stdout.strip()
    if "thesis-sim/venv" in loc:
        print(f"  [OK]     numpy location: {loc[:70]}")
        return True
    else:
        print(f"  [ERROR]  numpy loaded from WRONG location: {loc}")
        print("           It must come from ~/thesis-sim/venv/")
        print("           Fix: bash fix_python_env.sh")
        return False


# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print("  THESIS ENVIRONMENT CHECK  |  Cybergenii")
print("=" * 62)

# ── [1] Virtual environment ─────────────────────────────────────────────────
print("\n[1] Virtual Environment")

# Check if THIS process is running inside the venv
in_venv = sys.prefix != sys.base_prefix
if in_venv and "thesis-sim/venv" in sys.executable:
    print(f"  [OK]     Venv active: {sys.prefix}")
    print("  [OK]     This script is running inside the venv")
    PY = sys.executable  # use current python
elif os.path.isfile(VENV_PY):
    print("  [WARN]   Venv exists but not currently activated")
    print("           Run: source ~/thesis-sim/activate_thesis.sh")
    print("           Checking venv packages directly anyway...")
    PY = VENV_PY
else:
    print("  [ERROR]  Venv not found at ~/thesis-sim/venv/")
    print("           Run: bash fix_python_env.sh")
    PY = sys.executable

print(f"  [INFO]   Checking with: {PY}")

# ── [2] uv ─────────────────────────────────────────────────────────────────
print("\n[2] uv Package Manager")
check_cmd("uv --version", "uv")

# ── [3] System build tools ──────────────────────────────────────────────────
print("\n[3] System Build Tools")
check_cmd("cmake --version | head -1", "CMake")
check_cmd("g++ --version | head -1", "g++ compiler")
check_cmd("ninja --version", "Ninja build system")

# ── [4] NS-3 ────────────────────────────────────────────────────────────────
print("\n[4] NS-3 Simulator")
ns3 = os.path.expanduser("~/ns-3.38")
check_cmd(f"test -d {ns3}", "NS-3 directory  (~/ns-3.38)")
check_cmd(f"test -f {ns3}/ns3", "NS-3 executable")
check_cmd(f"test -d {ns3}/src/lte", "LTE module")
check_cmd(
    f"test -f {ns3}/scratch/thesis-fault-sim.cc", "thesis-fault-sim.cc in scratch/"
)

# ── [5] Python packages (via venv python) ───────────────────────────────────
print("\n[5] Python ML Packages (checked via venv python)")

# First check numpy source — most critical
print("  --- numpy source check (critical) ---")
numpy_ok = check_numpy_source(PY)

print("  --- package versions ---")
packages = [
    ("numpy", "NumPy"),
    ("tensorflow", "TensorFlow"),
    ("pandas", "Pandas"),
    ("scipy", "SciPy"),
    ("sklearn", "scikit-learn"),
    ("imblearn", "imbalanced-learn (SMOTE)"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("joblib", "Joblib"),
    ("shap", "SHAP (XAI)"),
]
results = [check_pymod(mod, lbl, PY) for mod, lbl in packages]

# ── [6] Workspace folders ────────────────────────────────────────────────────
print("\n[6] Thesis Workspace Folders")
base = os.path.expanduser("~/thesis-sim")
for folder in ["output/raw", "models", "reports", "scripts"]:
    path = os.path.join(base, folder)
    ok = os.path.isdir(path)
    print(f"  {'[OK]    ' if ok else '[MISSING]'} {path}")

# ── [7] GPU (optional) ──────────────────────────────────────────────────────
print("\n[7] GPU Acceleration (optional)")
code = "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(gpus[0].name if gpus else 'NONE')"
r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
if r.returncode == 0:
    gpu_info = r.stdout.strip()
    if gpu_info == "NONE":
        print("  [INFO]   No GPU — CPU training will be used (fine, just slower)")
    else:
        print(f"  [OK]     GPU: {gpu_info}")
else:
    print("  [INFO]   Could not check GPU")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
all_ok = numpy_ok and all(results)
if all_ok:
    print("  ALL CHECKS PASSED — ready to run simulations.")
    print("")
    print("  Next:")
    print("    source ~/thesis-sim/activate_thesis.sh")
    print("    python3 run_all_trials.py --workers 3")
else:
    print("  SOME CHECKS FAILED.")
    print("")
    if not numpy_ok:
        print("  numpy is loading from the WRONG location.")
        print("  Fix with:")
        print("    bash fix_python_env.sh")
    else:
        print("  Fix missing packages:")
        print("    uv pip install --python ~/thesis-sim/venv/bin/python \\")
        print("        numpy==1.26.4 tensorflow-cpu scikit-learn imbalanced-learn \\")
        print("        pandas scipy matplotlib seaborn joblib shap")
print("=" * 62 + "\n")
