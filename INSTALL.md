# 🔧 NS-3 Simulation — Installation & Execution Guide

> **Machine Learning Fault Detection & Self-Healing in Telecom Networks**  
> Complete setup, code placement, and step-by-step execution path.

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project File Structure](#project-file-structure)
- [The NS-3 Simulation Script](#the-ns-3-simulation-script)
- [Step-by-Step Execution](#step-by-step-execution)
- [Troubleshooting](#troubleshooting)
- [Manual Dataset Merge](#manual-dataset-merge)
- [Time-Saving Tips](#time-saving-tips)
- [Quick Command Reference](#quick-command-reference)

---

## Overview

The full pipeline has four sequential stages:

| Stage | Script / Tool | Output | Est. Time |
|-------|--------------|--------|-----------|
| 1 | `install_ns3_thesis.sh` | Working NS-3 environment | 30–60 min |
| 2 | `run_all_trials.py` | `kpi_master_dataset.csv` (~51k rows) | 4–8 hours |
| 3 | `preprocess_and_train.py` | RF, LSTM, SVM model files | 1–2 hours |
| 4 | `mapek_loop.py` | MTTR & availability results | 15–30 min |

> **All code files are provided separately.** This document explains what each one does, where to place it, and the exact commands to run.

---

## System Requirements

### Operating System

NS-3 runs natively on Linux. **Ubuntu 22.04 LTS** is strongly recommended.

- **Windows users** — use WSL2 or a VirtualBox VM running Ubuntu
- **macOS users** — Homebrew installation is possible but Ubuntu is preferred

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04 LTS (or WSL2) | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Disk | 20 GB free | 50 GB free |
| CPU | 4-core x86-64 | 8-core+ |
| GPU | Not required | NVIDIA + CUDA (5–10× LSTM speedup) |
| Python | 3.8+ | 3.10 |

### WSL2 Setup (Windows only)

```powershell
# In PowerShell (as Administrator):
wsl --install -d Ubuntu-22.04
# After reboot, open Ubuntu from Start Menu and continue this guide inside the Ubuntu terminal.
```

---

## Installation

### Step 1 — Create the workspace and place files

```bash
mkdir -p ~/thesis-sim/scripts
mkdir -p ~/thesis-sim/output/raw
mkdir -p ~/thesis-sim/models
mkdir -p ~/thesis-sim/reports

cp run_all_trials.py        ~/thesis-sim/
cp preprocess_and_train.py  ~/thesis-sim/
cp mapek_loop.py            ~/thesis-sim/
cp check_environment.py     ~/thesis-sim/
cp thesis-fault-sim.cc      ~/thesis-sim/scripts/
```

### Step 2 — Run the installation script

```bash
chmod +x install_ns3_thesis.sh
bash install_ns3_thesis.sh
```

The script will:
1. Install build tools (`cmake`, `g++`, `ninja`)
2. Install Python ML packages (TensorFlow, scikit-learn, etc.)
3. Download NS-3 3.38 (~120 MB)
4. Configure and build NS-3 (~10–30 min)
5. Copy the simulation script to the NS-3 scratch folder
6. Run a 30-second test simulation

### Manual Installation (if the script fails)

**System packages:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build g++ \
    python3 python3-pip python3-dev git wget \
    libboost-all-dev libssl-dev libxml2-dev gsl-bin libgsl-dev
```

**Python packages:**
```bash
pip3 install numpy pandas scipy scikit-learn imbalanced-learn \
             tensorflow matplotlib seaborn joblib shap jupyter
```

**Download and build NS-3:**
```bash
cd ~
wget https://www.nsnam.org/releases/ns-allinone-3.38.tar.bz2
tar xjf ns-allinone-3.38.tar.bz2
mv ns-allinone-3.38/ns-3.38 ~/ns-3.38

cd ~/ns-3.38
./ns3 configure --build-profile=optimized \
    --enable-modules=lte,network,internet,applications,mobility,energy,flow-monitor
./ns3 build
```

**Place the simulation script:**
```bash
cp ~/thesis-sim/scripts/thesis-fault-sim.cc ~/ns-3.38/scratch/
cd ~/ns-3.38 && ./ns3 build thesis-fault-sim
```

### Step 3 — Verify installation

```bash
cd ~/thesis-sim
python3 check_environment.py
```

Expected output:
```
[OK] Python 3
[OK] CMake
[OK] g++ compiler
[OK] NS-3 directory at ~/ns-3.38
[OK] NS-3 executable
[OK] LTE module
[OK] TensorFlow 2.x
[OK] Scikit-learn
[OK] imbalanced-learn (SMOTE)
```

---

## Project File Structure

```
~/ns-3.38/
└── scratch/
    └── thesis-fault-sim.cc         ← NS-3 C++ simulation script (PLACE HERE)

~/thesis-sim/
├── install_ns3_thesis.sh           ← One-time installation script
├── check_environment.py            ← Environment verification
├── run_all_trials.py               ← Step 2: runs all 50 trials
├── preprocess_and_train.py         ← Step 3: ML training pipeline
├── mapek_loop.py                   ← Step 4: MAPE-K evaluation
│
├── scripts/
│   └── thesis-fault-sim.cc         ← Backup copy of simulation script
│
├── output/
│   ├── raw/                        ← NS-3 CSV output (auto-created)
│   │   ├── kpi_trial0_power.csv
│   │   ├── kpi_trial0_congestion.csv
│   │   └── ... (200 CSV files: 4 fault types × 50 trials)
│   └── kpi_master_dataset.csv      ← Merged dataset (auto-created)
│
├── models/                         ← Trained model files (auto-created)
│   ├── random_forest.pkl
│   ├── lstm_model.h5
│   ├── svm_baseline.pkl
│   ├── scaler_lstm.pkl / scaler_tab.pkl / pca.pkl
│   └── metadata.json
│
└── reports/                        ← Plots and evaluation results (auto-created)
    ├── lstm_training_history.png
    └── mapek_summary.json
```

---

## The NS-3 Simulation Script

`thesis-fault-sim.cc` creates the entire 5G network, injects faults, and writes KPI data to CSV. It is provided — you do not need to write it. This section documents what it produces.

### Network Topology

| Parameter | Value |
|-----------|-------|
| Macro gNBs | 7 (hexagonal layout) |
| Small cells | 21 (3 per macro) |
| UEs | 500 (Poisson distributed) |
| Inter-site distance | 500 m |
| Carrier frequency | 3.5 GHz (Band n78) |
| Channel bandwidth | 100 MHz |
| UE speed | 0.83–8.33 m/s |
| Simulation time | 300 s per trial |
| Random seeds | 50 independent seeds |

### Fault Types

| Fault | Label | KPI Signature |
|-------|-------|---------------|
| **Normal** | 0 | RSRP −70 to −80 dBm, PRB 60–70%, nominal throughput, loss <2% |
| **Power Fault** | 1 | RSRP → −115 dBm, Throughput → ~0, HO success → ~5%, Loss → 95% |
| **Congestion** | 2 | PRB >90%, Latency → 450 ms+, Throughput −65% |
| **gNB HW Failure** | 3 | Cell collapse + neighbour load increase |

### Output CSV Format

One row per gNB per second → **~2,100 rows per trial per fault type** → **~51,000 rows total**.

| Column | Type | Description |
|--------|------|-------------|
| `trial` | int | Trial index (0–49) |
| `time` | float | Simulation time (s) |
| `gnb_id` | int | gNB identifier (0–6) |
| `rsrp_avg_dbm` | float | Avg. RSRP (dBm) |
| `sinr_avg_db` | float | Avg. SINR (dB) |
| `prb_utilisation` | float | PRB utilisation (0–1) |
| `dl_throughput_mbps` | float | Downlink throughput (Mbps) |
| `ul_throughput_mbps` | float | Uplink throughput (Mbps) |
| `packet_loss_rate` | float | Packet loss fraction (0–1) |
| `handover_success_rate` | float | HO success fraction (0–1) |
| `latency_avg_ms` | float | Avg. UE downlink latency (ms) |
| `fault_label` | int | Ground truth class (0–3) |

---

## Step-by-Step Execution

### Step 1 — Test a single trial first

```bash
cd ~/ns-3.38
./ns3 run "thesis-fault-sim \
    --trial=0 \
    --fault=power \
    --outputDir=$HOME/thesis-sim/output/raw"

# Verify output:
head -3 ~/thesis-sim/output/raw/kpi_trial0_power.csv
```

> If you see `Build failed`, ensure `thesis-fault-sim.cc` is in `~/ns-3.38/scratch/` then run `./ns3 build thesis-fault-sim`.

### Step 2 — Run all 50 trials

```bash
cd ~/thesis-sim

# Full run (4–8 hours)
python3 run_all_trials.py --workers 3

# Quick test (5 trials only)
python3 run_all_trials.py --trials 5 --workers 2

# Single fault type
python3 run_all_trials.py --fault power --workers 2
```

> Set `--workers` to **CPU cores − 1**. Do not exceed your core count.

### Step 3 — Verify the master dataset

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('output/kpi_master_dataset.csv')
print('Shape:', df.shape)
labels = {0:'Normal',1:'Power Fault',2:'Congestion',3:'HW Failure'}
for k,v in df['fault_label'].value_counts().sort_index().items():
    print(f'  {labels[k]}: {v:,}  ({100*v/len(df):.1f}%)')
print('Missing values:', df.isnull().sum().sum())
"
```

Expected:
```
Shape: (51340, 12)
  Normal:        43820  (85.4%)
  Power Fault:    2876  ( 5.6%)
  Congestion:     2541  ( 4.9%)
  HW Failure:     2103  ( 4.1%)
Missing values: 0
```

### Step 4 — Train the ML models

```bash
cd ~/thesis-sim

python3 preprocess_and_train.py          # full pipeline (RF + LSTM + SVM)
python3 preprocess_and_train.py --skip_svm   # skip SVM if time-constrained
```

### Step 5 — Run the MAPE-K evaluation

```bash
cd ~/thesis-sim

python3 mapek_loop.py --model all    # evaluates all models + reactive baseline
python3 mapek_loop.py --model lstm   # LSTM only
```

Expected summary:
```
Condition                   MTTR (min)   Availability   MTTR Reduction
Reactive Baseline              312.4        94.17%          Reference
LSTM                           101.6        98.96%           67.5%
RF                             118.7        98.11%           62.0%
SVM                            187.3        96.73%           40.1%
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ns3: command not found` | NS-3 not built | `cd ~/ns-3.38 && ./ns3 build` |
| `thesis-fault-sim not found` | `.cc` not in `scratch/` | `cp ~/thesis-sim/scripts/thesis-fault-sim.cc ~/ns-3.38/scratch/` then `./ns3 build thesis-fault-sim` |
| Build fails with LTE error | LTE module not included | `./ns3 configure --enable-modules=lte,network,internet,...` then `./ns3 build` |
| `No module named tensorflow` | Not installed | `pip3 install tensorflow` |
| `No module named imblearn` | Not installed | `pip3 install imbalanced-learn` |
| Output CSV empty / header only | Simulation crashed | `NS_LOG=ThesisFaultSim=info ./ns3 run "thesis-fault-sim..."` |
| LSTM accuracy < 60% | SMOTE not applied | Check for `Post-SMOTE training size` in output |
| `File not found: kpi_master_dataset.csv` | Merge step failed | See [Manual Dataset Merge](#manual-dataset-merge) |
| Very slow (>10 min/trial) | Too many workers | Try `--workers 1`; reduce `N_UE` in `.cc` for testing |
| WSL2 out of memory | RAM limit too low | Create `C:\Users\YOU\.wslconfig`: `[wsl2]` / `memory=8GB` / `processors=4` then `wsl --shutdown` |

---

## Manual Dataset Merge

If `run_all_trials.py` completed some trials but the merge step failed, merge manually:

```python
import pandas as pd, os, glob

raw_dir = os.path.expanduser('~/thesis-sim/output/raw')
out_csv = os.path.expanduser('~/thesis-sim/output/kpi_master_dataset.csv')

all_files = glob.glob(os.path.join(raw_dir, 'kpi_trial*.csv'))
print(f'Found {len(all_files)} CSV files')

dfs = [pd.read_csv(f) for f in sorted(all_files)]
master = pd.concat(dfs, ignore_index=True)
master.to_csv(out_csv, index=False)
print(f'Merged {len(dfs)} files → {len(master):,} rows')
print(master['fault_label'].value_counts().sort_index())
```

---

## Time-Saving Tips

### Run overnight with `nohup`

```bash
cd ~/thesis-sim
nohup python3 run_all_trials.py --workers 3 > sim_log.txt 2>&1 &
echo "PID: $!"

tail -f sim_log.txt        # monitor progress
ps aux | grep run_all_trials   # check if still running
```

### Reduce scope for quick testing

```bash
python3 run_all_trials.py --trials 5 --workers 2
python3 preprocess_and_train.py   # verify pipeline works
# then commit to full run:
python3 run_all_trials.py --trials 50 --workers 3
```

### GPU acceleration for LSTM

```bash
# Check if GPU is detected:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install CUDA-enabled TensorFlow:
pip3 install tensorflow[and-cuda]
```

---

## Quick Command Reference

```bash
# ── ONE-TIME SETUP ─────────────────────────────────────────────────────────
bash install_ns3_thesis.sh
cp thesis-fault-sim.cc ~/ns-3.38/scratch/
cd ~/ns-3.38 && ./ns3 build thesis-fault-sim

# ── STEP 1: Test single trial ───────────────────────────────────────────────
cd ~/ns-3.38
./ns3 run "thesis-fault-sim --trial=0 --fault=power --outputDir=$HOME/thesis-sim/output/raw"

# ── STEP 2: Run all 50 trials ───────────────────────────────────────────────
cd ~/thesis-sim
python3 run_all_trials.py --workers 3

# ── STEP 3: Train all ML models ─────────────────────────────────────────────
python3 preprocess_and_train.py

# ── STEP 4: Run MAPE-K evaluation ───────────────────────────────────────────
python3 mapek_loop.py --model all

# ── CHECK RESULTS ────────────────────────────────────────────────────────────
cat reports/mapek_summary.json
ls -lh models/
ls -lh output/raw/ | wc -l    # should show 200 CSV files
```
