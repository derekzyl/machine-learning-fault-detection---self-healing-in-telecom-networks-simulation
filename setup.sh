#!/usr/bin/env bash
# =============================================================================
# THESIS-SIM — UNIVERSAL SETUP SCRIPT
# Machine Learning Fault Detection & Self-Healing in Telecom Networks
#
# Supports: Linux (Ubuntu/Debian/Fedora/Arch) | macOS (Intel + Apple Silicon)
# Windows:  Must run inside WSL2 — this script guides you there automatically
#
# Usage (run from the directory containing your thesis scripts):
#   bash setup.sh
#
# What it does:
#   1.  Detect OS / architecture / WSL status
#   2.  Install system build dependencies
#   3.  Install uv (fast Python package manager)
#   4.  Create an isolated venv with numpy<2 constraint (fixes system-numpy bleed)
#   5.  Install NS-3 3.38 and compile thesis-fault-sim.cc
#   6.  Wire up all project files (copy & verify placement)
#   7.  Generate all 5 thesis figures
#   8.  Run environment check (check_environment.py)
#   9.  Ask y/n before running the long training pipeline
# =============================================================================

set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()   { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()     { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()  { echo -e "${RED}[ERROR]${NC} $*"; }
header() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }
die()    { error "$*"; exit 1; }

# ── paths ─────────────────────────────────────────────────────────────────────
THESIS_DIR="$HOME/thesis-sim"
VENV_DIR="$THESIS_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
NS3_DIR="$HOME/ns-3.38"
NS3_ARCHIVE="ns-allinone-3.38.tar.bz2"
NS3_URL="https://www.nsnam.org/releases/${NS3_ARCHIVE}"

# ── log file (always written regardless of errors) ────────────────────────────
LOG="$HOME/thesis_setup.log"
exec > >(tee -a "$LOG") 2>&1
info "Full log will be written to: $LOG"

# =============================================================================
# ERROR TRAP — auto repair on failure
# =============================================================================
_LAST_STEP="(unknown)"

on_error() {
  local exit_code=$?
  local line=$1
  echo ""
  echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${RED}║  SETUP FAILED                                                ║${NC}"
  echo -e "${RED}║  Step: $_LAST_STEP                                           ${NC}"
  echo -e "${RED}║  Line: $line  |  Exit code: $exit_code                       ${NC}"
  echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo "  Full log saved to: $LOG"
  echo ""

  # ── Ask if user wants auto-repair ──────────────────────────────────────────
  echo -e "${YELLOW}  The most common failure is the Python venv / numpy isolation issue.${NC}"
  echo -e "${YELLOW}  Would you like to run the auto-repair (nuke venv + reinstall)?${NC}"
  echo ""
  read -rp "  Run repair now? [y/N]: " REPAIR_ANSWER
  if [[ "${REPAIR_ANSWER,,}" == "y" ]]; then
    echo ""
    info "Running venv repair (numpy<2 constraint enforced)..."
    rm -rf "$VENV_DIR"
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1

    if command -v uv &>/dev/null; then
      uv venv "$VENV_DIR" --python 3.11 --seed 2>/dev/null \
        || uv venv "$VENV_DIR" --python 3.10 --seed

      uv pip install \
        --python "$VENV_PY" \
        "numpy>=1.23.5,<2.0" \
        "tensorflow-cpu==2.15.0" \
        "pandas>=1.5,<3" \
        "scipy>=1.9" \
        "scikit-learn>=1.3,<1.6" \
        "imbalanced-learn>=0.11" \
        "matplotlib>=3.6" \
        "seaborn>=0.12" \
        "joblib>=1.2" \
        "jupyter" "notebook"

      "$VENV_PY" -c "
import numpy as np
assert np.__version__.startswith('1.'), f'Still wrong: {np.__version__}'
print(f'  Repair OK — numpy {np.__version__} ({np.__file__})')
" && ok "Repair successful! Please re-run: bash setup.sh" \
    || error "Repair failed — check $LOG and open an issue."
    else
      error "uv not found — cannot auto-repair. Install uv first:"
      echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
      echo "    source ~/.bashrc  (or open a new terminal)"
      echo "    bash setup.sh"
    fi
  else
    echo ""
    echo "  Manual repair commands:"
    echo "    rm -rf $VENV_DIR"
    echo "    bash setup.sh"
    echo ""
    echo "  Or to only reinstall Python packages:"
    echo "    bash final_clean_install.sh"
  fi
  exit $exit_code
}

trap 'on_error $LINENO' ERR

# ── helper: y/n prompt ────────────────────────────────────────────────────────
ask_yn() {
  # ask_yn "Question text" [default=y|n]
  local prompt="$1"
  local default="${2:-n}"
  local hint
  if [[ "${default,,}" == "y" ]]; then hint="[Y/n]"; else hint="[y/N]"; fi

  while true; do
    echo ""
    read -rp "  ${prompt} ${hint}: " ANSWER
    ANSWER="${ANSWER:-$default}"
    case "${ANSWER,,}" in
      y|yes) return 0 ;;
      n|no)  return 1 ;;
      *)     echo "  Please enter y or n." ;;
    esac
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — DETECT PLATFORM
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Platform detection"
header "STEP 0 — Detecting platform"

OS="$(uname -s)"
ARCH="$(uname -m)"
IS_WSL=false
DISTRO=""

case "$OS" in
  Linux)
    if grep -qi microsoft /proc/version 2>/dev/null; then
      IS_WSL=true
      info "Running inside WSL2 on Windows"
    fi
    if [ -f /etc/os-release ]; then
      . /etc/os-release
      DISTRO="${ID:-unknown}"
    fi
    info "OS: Linux ($DISTRO) | Arch: $ARCH | WSL: $IS_WSL"
    ;;

  Darwin)
    DISTRO="macos"
    info "OS: macOS | Arch: $ARCH"
    if [ "$ARCH" = "arm64" ]; then
      info "Apple Silicon (M-series) detected — native arm64 builds"
    fi
    ;;

  MINGW*|MSYS*|CYGWIN*)
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  WINDOWS DETECTED — WSL2 REQUIRED                           ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  NS-3 does not run on native Windows. You must use WSL2."
    echo ""
    echo "  ── One-time WSL2 setup ──────────────────────────────────────────"
    echo "  1. Open PowerShell as Administrator and run:"
    echo "       wsl --install -d Ubuntu-22.04"
    echo ""
    echo "  2. Reboot, open the Ubuntu app from the Start Menu."
    echo ""
    echo "  3. Copy this setup.sh into your Ubuntu home:"
    echo "       cp /mnt/c/Users/<YOU>/Downloads/setup.sh ~/"
    echo "       bash setup.sh"
    echo ""
    echo "  ── WSL2 memory tip ──────────────────────────────────────────────"
    echo "  Create C:\\Users\\<YOU>\\.wslconfig:"
    echo "    [wsl2]"
    echo "    memory=8GB"
    echo "    processors=4"
    echo "  Then run: wsl --shutdown"
    echo ""
    exit 0
    ;;

  *)
    die "Unsupported OS: $OS"
    ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SYSTEM DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="System dependencies"
header "STEP 1 — Installing system dependencies"

if [ "$DISTRO" = "macos" ]; then
  if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ "$ARCH" = "arm64" ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
      grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null \
        || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
    fi
  fi
  brew install cmake ninja git wget python3 boost openssl@3 libxml2 gsl sqlite || true
  ok "Homebrew packages installed"

elif command -v apt-get &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y \
    build-essential cmake ninja-build g++ git wget curl \
    python3 python3-dev python3-pip \
    libboost-all-dev libssl-dev libxml2-dev \
    gsl-bin libgsl-dev libsqlite3-dev \
    tar bzip2 2>/dev/null || true
  ok "apt packages installed"

elif command -v dnf &>/dev/null; then
  sudo dnf install -y \
    gcc gcc-c++ cmake ninja-build git wget curl \
    python3 python3-devel \
    boost-devel openssl-devel libxml2-devel gsl-devel sqlite-devel \
    tar bzip2
  ok "dnf packages installed"

elif command -v pacman &>/dev/null; then
  sudo pacman -Sy --noconfirm \
    base-devel cmake ninja git wget curl python \
    boost openssl libxml2 gsl sqlite
  ok "pacman packages installed"

else
  warn "Unknown package manager — please install: cmake g++ ninja boost openssl libxml2 gsl"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INSTALL UV
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="uv installation"
header "STEP 2 — Installing uv (fast Python package manager)"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
  info "Downloading uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
ok "uv: $(uv --version)"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ISOLATED VENV  (numpy < 2 fix)
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Python venv creation + numpy isolation"
header "STEP 3 — Isolated Python venv (numpy<2 constraint)"

# Always nuke old venv for a guaranteed clean state
rm -rf "$VENV_DIR"

# Install Python 3.11 via uv if not available
uv python install 3.11 2>/dev/null || true
uv venv "$VENV_DIR" --python 3.11 --seed 2>/dev/null \
  || uv venv "$VENV_DIR" --python 3.10 --seed

ok "Venv created: $VENV_DIR"

# Critical: block system packages from bleeding in at install and at runtime
unset PYTHONPATH
export PYTHONNOUSERSITE=1

info "Installing all ML packages (single uv call, numpy<2 enforced)..."
info "This takes 3–8 minutes on first run..."

if [ "$DISTRO" = "macos" ] && [ "$ARCH" = "arm64" ]; then
  # Apple Silicon: tensorflow-macos + tensorflow-metal
  uv pip install \
    --python "$VENV_PY" \
    "numpy>=1.23.5,<2.0" \
    "tensorflow-macos" \
    "tensorflow-metal" \
    "pandas>=1.5,<3" \
    "scipy>=1.9" \
    "scikit-learn>=1.3,<1.6" \
    "imbalanced-learn>=0.11" \
    "matplotlib>=3.6" \
    "seaborn>=0.12" \
    "joblib>=1.2" \
    "jupyter" "notebook"
else
  # Linux x86_64/aarch64 + macOS Intel — tensorflow-cpu (no CUDA deps)
  uv pip install \
    --python "$VENV_PY" \
    "numpy>=1.23.5,<2.0" \
    "tensorflow-cpu==2.15.0" \
    "pandas>=1.5,<3" \
    "scipy>=1.9" \
    "scikit-learn>=1.3,<1.6" \
    "imbalanced-learn>=0.11" \
    "matplotlib>=3.6" \
    "seaborn>=0.12" \
    "joblib>=1.2" \
    "jupyter" "notebook"
fi

# Verify numpy is from THE VENV and is 1.x — fail fast here if not
"$VENV_PY" -c "
import numpy, sys
loc  = numpy.__file__
ver  = numpy.__version__
assert 'thesis-sim/venv' in loc, f'SYSTEM NUMPY BLEED — got: {loc}'
assert ver.startswith('1.'),     f'WRONG numpy version {ver} (must be 1.x)'
print(f'  numpy {ver} — venv isolated OK ({loc})')
"
ok "Python packages installed and isolated"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PROJECT DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Project files"
header "STEP 4 — Project directory + file placement"

mkdir -p "$THESIS_DIR"/{output/raw,models,reports,scripts}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
info "Source dir: $SCRIPT_DIR"

copy_if_exists() {
  local src="$SCRIPT_DIR/$1"
  local dst="$THESIS_DIR/$2"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    ok "Copied $1"
  else
    warn "$1 not found in source dir — skipping"
  fi
}

copy_if_exists "run_all_trials.py"            "run_all_trials.py"
copy_if_exists "preprocess_and_train.py"      "preprocess_and_train.py"
copy_if_exists "mapek_loop.py"                "mapek_loop.py"
copy_if_exists "check_environment.py"         "check_environment.py"
copy_if_exists "thesis-fault-sim.cc"          "scripts/thesis-fault-sim.cc"
copy_if_exists "scripts/generate_figures.py"  "scripts/generate_figures.py"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — NS-3 3.38
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="NS-3 download + build"
header "STEP 5 — NS-3 3.38"

if [ -d "$NS3_DIR" ] && [ -f "$NS3_DIR/ns3" ]; then
  ok "NS-3 already at $NS3_DIR — skipping download"
else
  info "Downloading NS-3 3.38 (~120 MB)..."
  cd "$HOME"
  [ -f "$NS3_ARCHIVE" ] || wget -q --show-progress "$NS3_URL" -O "$NS3_ARCHIVE"
  info "Extracting..."
  tar xjf "$NS3_ARCHIVE"
  mv -f ns-allinone-3.38/ns-3.38 "$NS3_DIR" 2>/dev/null || true
  rm -rf ns-allinone-3.38
  ok "NS-3 extracted to $NS3_DIR"
fi

NS3_BUILT=false
if [ -d "$NS3_DIR/cmake_cache" ] || find "$NS3_DIR/build" -name "libns3*" 2>/dev/null | grep -q .; then
  ok "NS-3 already built — skipping configure/build"
  NS3_BUILT=true
else
  info "Configuring NS-3 (optimised build)..."
  cd "$NS3_DIR"
  if [ "$DISTRO" = "macos" ]; then
    export PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig:$(brew --prefix libxml2)/lib/pkgconfig"
    export CPPFLAGS="-I$(brew --prefix openssl@3)/include"
  fi
  ./ns3 configure \
    --build-profile=optimized \
    --enable-modules=core,network,internet,applications,mobility 2>&1 | tail -20

  info "Building NS-3 — this takes 10–30 min ..."
  ./ns3 build 2>&1 | tail -5
  ok "NS-3 built"
  NS3_BUILT=true
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — COMPILE SIMULATION SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Compiling thesis-fault-sim.cc"
header "STEP 6 — Compiling thesis-fault-sim.cc"

SIM_CC="$THESIS_DIR/scripts/thesis-fault-sim.cc"
SCRATCH_CC="$NS3_DIR/scratch/thesis-fault-sim.cc"

if [ -f "$SIM_CC" ]; then
  cp "$SIM_CC" "$SCRATCH_CC"
  cd "$NS3_DIR"
  ./ns3 build thesis-fault-sim 2>&1 | tail -5
  ok "thesis-fault-sim compiled"
else
  warn "thesis-fault-sim.cc not found — skipping compile"
  warn "Place it in $THESIS_DIR/scripts/ and re-run to compile"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ACTIVATE SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Writing activate_thesis.sh"
header "STEP 7 — Writing activate_thesis.sh"

cat > "$THESIS_DIR/activate_thesis.sh" << 'ACTIVATE'
#!/usr/bin/env bash
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
echo "  ✓ Thesis venv activated  (PYTHONPATH cleared)"
echo "  Python:     $(which python3)"
echo "  NumPy:      $NP  ← must be 1.x"
echo "  TensorFlow: $TF"
echo ""
echo "  Commands:"
echo "    python3 check_environment.py"
echo "    python3 run_all_trials.py --workers 3"
echo "    python3 preprocess_and_train.py"
echo "    python3 mapek_loop.py --model all"
echo "    python3 scripts/generate_figures.py"
echo ""
ACTIVATE
chmod +x "$THESIS_DIR/activate_thesis.sh"
ok "activate_thesis.sh written"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — GENERATE THESIS FIGURES
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Figure generation"
header "STEP 8 — Generating thesis figures"

FIG_SCRIPT="$THESIS_DIR/scripts/generate_figures.py"
if [ -f "$FIG_SCRIPT" ]; then
  info "Running generate_figures.py ..."
  PYTHONNOUSERSITE=1 PYTHONPATH="" "$VENV_PY" "$FIG_SCRIPT" \
    && ok "All figures saved to $THESIS_DIR/reports/" \
    || warn "Figure generation failed (non-fatal) — check $LOG"
else
  warn "generate_figures.py not found — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — ENVIRONMENT CHECK
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Environment check"
header "STEP 9 — Running check_environment.py"

CHK="$THESIS_DIR/check_environment.py"
if [ -f "$CHK" ]; then
  echo ""
  PYTHONNOUSERSITE=1 PYTHONPATH="" "$VENV_PY" "$CHK" \
    && ok "Environment check passed" \
    || warn "Some environment checks failed — see output above"
else
  warn "check_environment.py not found — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — OPTIONAL: RUN FULL PIPELINE (y/n prompts)
# ─────────────────────────────────────────────────────────────────────────────
_LAST_STEP="Pipeline prompts"
header "STEP 10 — Run the full training pipeline?"

echo ""
echo -e "${BOLD}  The following steps are OPTIONAL and take several hours.${NC}"
echo -e "  You can run them now or manually later with:"
echo -e "    source ~/thesis-sim/activate_thesis.sh"
echo ""

# ── 10a: NS-3 simulation trials ──────────────────────────────────────────────
RUN_TRIALS=false
if [ -f "$THESIS_DIR/run_all_trials.py" ]; then
  echo -e "  ${BOLD}Step A — NS-3 simulation trials${NC}"
  echo    "  Runs 200 simulation trials (50 × 4 fault types)."
  echo    "  Estimated time: 4–8 hours. Needs NS-3 compiled."
  echo    "  Output: ~/thesis-sim/output/kpi_master_dataset.csv"
  if ask_yn "Run all 50 simulation trials now?" "n"; then
    RUN_TRIALS=true
    echo ""
    read -rp "  How many CPU workers? (press Enter for 2): " N_WORKERS
    N_WORKERS="${N_WORKERS:-2}"
    echo ""
    info "Starting simulation trials with $N_WORKERS workers..."
    cd "$THESIS_DIR"
    PYTHONNOUSERSITE=1 PYTHONPATH="" "$VENV_PY" run_all_trials.py --workers "$N_WORKERS" \
      && ok "All trials complete. Output: output/kpi_master_dataset.csv" \
      || { warn "Trials failed — check $LOG. You can re-run manually later."; RUN_TRIALS=false; }
  else
    echo "  Skipped. Run manually:"
    echo "    cd ~/thesis-sim && source activate_thesis.sh"
    echo "    python3 run_all_trials.py --workers 3"
  fi
fi

# ── 10b: ML training ─────────────────────────────────────────────────────────
RUN_TRAIN=false
if [ -f "$THESIS_DIR/preprocess_and_train.py" ]; then
  echo ""
  echo -e "  ${BOLD}Step B — ML model training (RF + LSTM + SVM)${NC}"
  echo    "  Requires: kpi_master_dataset.csv from Step A."
  echo    "  Estimated time: 1–2 hours."
  echo    "  Output: ~/thesis-sim/models/ (random_forest.pkl, lstm.h5, svm.pkl)"

  DATASET="$THESIS_DIR/output/kpi_master_dataset.csv"
  if [ ! -f "$DATASET" ] && [ "$RUN_TRIALS" = false ]; then
    warn "  kpi_master_dataset.csv not found — skipping training prompt"
    warn "  Run Step A first, then: python3 preprocess_and_train.py"
  else
    if ask_yn "Run ML training now (RF + LSTM + SVM)?" "n"; then
      RUN_TRAIN=true
      echo ""
      info "Starting ML training pipeline..."
      info "(TIP: If SVM is very slow, Ctrl-C and re-run with --skip_svm)"
      echo ""
      cd "$THESIS_DIR"
      PYTHONNOUSERSITE=1 PYTHONPATH="" "$VENV_PY" preprocess_and_train.py \
        && ok "Training complete — models saved to models/" \
        || { warn "Training failed — check $LOG. Re-run manually later."; RUN_TRAIN=false; }
    else
      echo "  Skipped. Run manually:"
      echo "    python3 preprocess_and_train.py"
    fi
  fi
fi

# ── 10c: MAPE-K evaluation ────────────────────────────────────────────────────
if [ -f "$THESIS_DIR/mapek_loop.py" ]; then
  echo ""
  echo -e "  ${BOLD}Step C — MAPE-K self-healing evaluation${NC}"
  echo    "  Requires: trained models from Step B."
  echo    "  Estimated time: 15–30 minutes."
  echo    "  Output: ~/thesis-sim/reports/mapek_summary.json"

  MODELS_EXIST=false
  [ -f "$THESIS_DIR/models/random_forest.pkl" ] && MODELS_EXIST=true

  if [ "$MODELS_EXIST" = false ] && [ "$RUN_TRAIN" = false ]; then
    warn "  No trained models found — skipping MAPE-K prompt"
    warn "  Run Step B first, then: python3 mapek_loop.py --model all"
  else
    if ask_yn "Run MAPE-K evaluation now?" "n"; then
      echo ""
      info "Running MAPE-K evaluation (all models + reactive baseline)..."
      cd "$THESIS_DIR"
      PYTHONNOUSERSITE=1 PYTHONPATH="" "$VENV_PY" mapek_loop.py --model all \
        && ok "MAPE-K evaluation complete — results in reports/mapek_summary.json" \
        || warn "MAPE-K failed — check $LOG. Re-run: python3 mapek_loop.py --model all"
    else
      echo "  Skipped. Run manually:"
      echo "    python3 mapek_loop.py --model all"
    fi
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
header "SETUP COMPLETE"

echo ""
"$VENV_PY" -c "
import sys, importlib

def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', 'ok')
    except ImportError:
        return 'MISSING'

import numpy as np
print(f'  Python:           {sys.version.split()[0]}')
print(f'  numpy:            {np.__version__}  (1.x — isolated)')
print(f'  tensorflow:       {ver(\"tensorflow\")}')
print(f'  scikit-learn:     {ver(\"sklearn\")}')
print(f'  imbalanced-learn: {ver(\"imblearn\")}')
print(f'  pandas:           {ver(\"pandas\")}')
print(f'  matplotlib:       {ver(\"matplotlib\")}')
"

echo ""
info "Generated figures (in reports/):"
for f in fig3_1_topology fig3_2_pipeline fig3_3_mapek fig3_4_timeline fig3_5_lstm_arch; do
  fp="$THESIS_DIR/reports/${f}.png"
  if [ -f "$fp" ]; then
    size=$(du -h "$fp" | cut -f1)
    ok "  ${f}.png  (${size})"
  else
    warn "  ${f}.png — not generated"
  fi
done

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗"
echo    "║  ALL DONE — your thesis environment is ready.                ║"
echo    "╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Activate in any new terminal:"
echo -e "    ${BOLD}source ~/thesis-sim/activate_thesis.sh${NC}"
echo ""
echo "  Full execution order (run these after activating):"
echo "    1. python3 run_all_trials.py --workers 3    # 4–8 hours"
echo "    2. python3 preprocess_and_train.py          # 1–2 hours"
echo "    3. python3 mapek_loop.py --model all        # 15–30 min"
echo "    4. python3 scripts/generate_figures.py      # 30 sec"
echo ""
echo "  Full log: $LOG"
echo ""
