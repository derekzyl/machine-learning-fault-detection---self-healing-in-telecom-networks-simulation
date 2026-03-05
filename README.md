# 🛡️ ML Fault Detection & Self-Healing in Telecom Networks

> **MSc Thesis Simulation** · 
> A full-stack research pipeline for proactive fault detection and autonomous self-healing in 5G NR radio access networks, combining NS-3 network simulation with machine-learning classifiers guided by the MAPE-K autonomic control loop.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Research Objectives](#research-objectives)
- [Project Structure](#project-structure)
- [KPI Features](#kpi-features)
- [Fault Classes](#fault-classes)
- [Pipeline Stages](#pipeline-stages)
- [Quickstart](#quickstart)
- [Detailed Usage](#detailed-usage)
- [Models & Hyperparameters](#models--hyperparameters)
- [MAPE-K Self-Healing Loop](#mape-k-self-healing-loop)
- [Output & Reports](#output--reports)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a four-stage end-to-end simulation and ML pipeline designed to answer the core thesis question:

> *Can machine-learning-based proactive fault detection significantly reduce Mean Time to Recovery (MTTR) and improve network availability compared to conventional threshold-based reactive monitoring in 5G NR networks?*

The simulation models a **7-cell hexagonal macro-gNB layout** over **50 randomised Monte Carlo trials**, injects three distinct fault types with stochastic onset times, trains three ML classifiers on the resulting KPI time-series, and benchmarks each against a reactive baseline using the MAPE-K autonomic loop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SIMULATION PIPELINE                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  NS-3 3.38   │    │  ML Training │    │  MAPE-K Loop     │  │
│  │  Fault Sim   │───▶│  Pipeline    │───▶│  Self-Healing    │  │
│  │  (C++)       │    │  (Python)    │    │  Evaluation      │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                     │            │
│   50 trials ×          RF + LSTM + SVM      MTTR / Avail.      │
│   4 fault types        + PCA + SMOTE        vs. Baseline        │
└─────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────  MAPE-K LOOP  ───────────────────────────┐
│                                                                 │
│   Monitor ──▶ Analyse ──▶ Plan ──▶ Execute                     │
│      │           │          │         │                         │
│   KPI window  ML model  Remediation  Action log                 │
│   pre-filter  inference   policy     + recovery time           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Research Objectives

| # | Objective |
|---|-----------|
| 1 | Simulate realistic 5G NR KPI degradation for three fault types using NS-3 |
| 2 | Train and compare Random Forest, LSTM, and SVM classifiers for 4-class fault detection |
| 3 | Implement the MAPE-K autonomic loop for proactive self-healing |
| 4 | Quantify MTTR reduction and network availability improvement vs. reactive baseline |

---

## Project Structure

```
thesis-sim/
│
├── thesis-fault-sim.cc        # NS-3 C++ simulation script (→ ~/ns-3.38/scratch/)
├── run_all_trials.py          # Stage 2: run 50 × 4 NS-3 trials in parallel
├── preprocess_and_train.py    # Stage 3: ML preprocessing + model training
├── mapek_loop.py              # Stage 4: MAPE-K self-healing evaluation
│
├── requirements.txt           # Python dependencies (pip freeze)
├── activate_thesis.sh         # Activate Python venv + set PYTHONPATH
├── check_environment.py       # Verify all dependencies are installed
│
├── output/
│   ├── raw/                   # Per-trial CSVs: kpi_trial{N}_{fault}.csv
│   └── kpi_master_dataset.csv # Merged dataset (~420,000 rows)
│
├── models/                    # Saved models (generated after training)
│   ├── random_forest.pkl
│   ├── lstm_model.h5
│   ├── svm_baseline.pkl
│   ├── scaler_lstm.pkl
│   ├── scaler_tab.pkl
│   ├── pca.pkl
│   └── metadata.json
│
└── reports/                   # Generated plots and evaluation reports
    ├── lstm_training_history.png
    └── mapek_summary.json
```

---

## KPI Features

Eight per-second KPIs are collected from each of the 7 simulated gNBs:

| Column | Unit | Normal Range | Description |
|--------|------|-------------|-------------|
| `rsrp_avg_dbm` | dBm | −70 to −85 | Reference Signal Received Power |
| `sinr_avg_db` | dB | 15 to 22 | Signal-to-Interference-plus-Noise Ratio |
| `prb_utilisation` | 0–1 | 0.55 to 0.70 | Physical Resource Block utilisation |
| `dl_throughput_mbps` | Mbps | 80 to 150 | Downlink throughput |
| `ul_throughput_mbps` | Mbps | 20 to 40 | Uplink throughput |
| `packet_loss_rate` | 0–1 | 0.001 to 0.01 | Packet loss ratio |
| `handover_success_rate` | 0–1 | 0.95 to 1.0 | Handover success ratio |
| `latency_avg_ms` | ms | 10 to 25 | Average round-trip latency |

---

## Fault Classes

| Label | Fault Type | Key Signatures |
|-------|-----------|---------------|
| `0` | **Normal** | All KPIs within nominal range |
| `1` | **Power Fault** | RSRP → −118 dBm, SINR → 1.5 dB, throughput → 0, latency → 2400 ms |
| `2` | **Congestion** | PRB > 93%, latency → 420 ms, throughput drops ~65%, RSRP stable |
| `3` | **gNB HW Failure** | RSRP → −114 dBm, partial throughput collapse, latency → 2100 ms |

Each fault is injected at a stochastic onset time (30–250 s) with a random duration (15–45 s) on a randomly selected gNB, across all 50 Monte Carlo trials.

---

## Pipeline Stages

### Stage 1 — NS-3 Simulation (`thesis-fault-sim.cc`)
- 7 macro-gNBs, 300 s simulation time, 1 s KPI collection interval
- 4 fault types × 50 trials = 200 simulation runs
- Outputs per-trial CSVs with ~2,100 rows each (~2,100 × 200 = 420,000 total)

### Stage 2 — Trial Runner (`run_all_trials.py`)
- Parallel execution using `ProcessPoolExecutor` (configurable workers)
- Automatic build, sanity-check, and CSV merge into `kpi_master_dataset.csv`

### Stage 3 — Preprocessing & Training (`preprocess_and_train.py`)
1. **Sliding window extraction** — 10 s windows, 1 s stride, per gNB group
2. **Feature engineering** — 6 statistical features × 8 KPIs = 48 tabular features
3. **Train/val/test split** — 70 / 15 / 15%, stratified
4. **Normalisation** — `StandardScaler` (separate for LSTM and tabular inputs)
5. **PCA** — 95% explained variance retention on tabular features
6. **SMOTE** — minority class oversampling on training partition only
7. **Model training** — Random Forest, LSTM, SVM
8. **Evaluation** — Accuracy, Macro F1, AUC-ROC, confusion matrix per model

### Stage 4 — MAPE-K Evaluation (`mapek_loop.py`)
- Simulates Monitor → Analyse → Plan → Execute loop over the test partition
- Computes **Mean Time to Recovery (MTTR)** and **Network Availability %**
- Benchmarks all three ML models against the reactive threshold-only baseline
- Exports results to `reports/mapek_summary.json`

---

## Quickstart

### Prerequisites

- **NS-3 3.38** installed at `~/ns-3.38` (with core and network modules)
- **Python 3.10+**
- **Ubuntu 22.04** or compatible Linux

### 1 · Set up the Python environment

```bash
cd ~/thesis-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or use the convenience script:

```bash
source activate_thesis.sh
```

### 2 · Place the NS-3 simulation script

```bash
cp ~/thesis-sim/thesis-fault-sim.cc ~/ns-3.38/scratch/
cd ~/ns-3.38 && ./ns3 build thesis-fault-sim
```

### 3 · Run all simulation trials

```bash
cd ~/thesis-sim
python3 run_all_trials.py --workers 4
```

### 4 · Train the ML models

```bash
python3 preprocess_and_train.py
```

### 5 · Run the MAPE-K self-healing evaluation

```bash
python3 mapek_loop.py --model all
```

---

## Detailed Usage

### `run_all_trials.py`

```
usage: run_all_trials.py [-h] [--trials N] [--workers N] [--fault TYPE] [--debug]

  --trials N      Number of Monte Carlo trials (default: 50)
  --workers N     Parallel worker processes (default: 2)
  --fault TYPE    Limit to one fault type: none | power | congestion | hardware
  --debug         Run one trial in foreground (full NS-3 output) for diagnosis
```

**Examples:**
```bash
# Quick test — 1 trial, 1 worker, verbose
python3 run_all_trials.py --trials 1 --workers 1 --debug

# Full 50-trial run with 4 parallel workers
python3 run_all_trials.py --trials 50 --workers 4

# Power faults only
python3 run_all_trials.py --fault power
```

---

### `preprocess_and_train.py`

```
usage: preprocess_and_train.py [-h] [--data PATH] [--skip_svm]

  --data PATH     Path to master KPI CSV (default: ~/thesis-sim/output/kpi_master_dataset.csv)
  --skip_svm      Skip SVM training (SVM can be slow on large datasets)
```

**Examples:**
```bash
# Full pipeline (RF + LSTM + SVM)
python3 preprocess_and_train.py

# Custom dataset path, skip SVM
python3 preprocess_and_train.py --data /path/to/dataset.csv --skip_svm
```

---

### `mapek_loop.py`

```
usage: mapek_loop.py [-h] [--model MODEL]

  --model MODEL   Model to evaluate: lstm | rf | svm | all (default: all)
```

**Examples:**
```bash
# Evaluate all models + reactive baseline
python3 mapek_loop.py --model all

# LSTM only
python3 mapek_loop.py --model lstm
```

---

## Models & Hyperparameters

### Random Forest
| Parameter | Value |
|-----------|-------|
| Trees | 200 |
| Max depth | 20 |
| Min samples leaf | 2 |
| Max features | `sqrt` |
| Class weight | `balanced` |
| Bootstrap | Yes |

### LSTM Network
| Layer | Config |
|-------|--------|
| LSTM 1 | 128 units, `return_sequences=True` |
| Dropout 1 | 0.3 |
| LSTM 2 | 128 units |
| Dropout 2 | 0.3 |
| Dense | 64 units, ReLU |
| Output | 4 units, Softmax |
| Optimiser | Adam (lr=0.001) |
| Epochs | 100 (early stopping, patience=10) |
| Batch size | 64 |
| Input shape | (10 timesteps × 8 KPIs) |

### SVM Baseline
| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C | 10 |
| Gamma | 0.01 |
| Class weight | `balanced` |
| Training cap | 20,000 samples (stratified subsample) |

---

## MAPE-K Self-Healing Loop

### Remediation Policies

| Fault | Action | Expected Recovery |
|-------|--------|-------------------|
| **Power Fault** | Backup power restore + neighbour tilt/power boost | 45 s |
| **Congestion** | Handover offset adjust + PRB reallocation | 30 s |
| **HW Failure** | Neighbour cell UL/DL power boost compensation | 50 s |
| **Uncertain** | Conservative: alert NOC + monitor | 90 s |

### Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAPEK_CYCLE` | 5 s | Inference-action cycle interval |
| `CONFIDENCE_THRESHOLD` | 0.70 | Minimum posterior to confirm a fault |
| `ESCALATION_TIMEOUT` | 30 s | Fallback retry if KPIs don't recover |
| `WINDOW_SIZE` | 10 s | KPI sliding window size |

### Reactive Baseline Thresholds
| KPI | Threshold | Condition |
|-----|-----------|-----------|
| PRB utilisation | > 95% | Overload trigger |
| RSRP | < −110 dBm | Signal loss trigger |
| Packet loss | > 30% | Quality trigger |
| DL throughput | = 0 Mbps | Outage trigger |

---

## Output & Reports

After a complete run, the following artefacts are generated:

```
output/
  kpi_master_dataset.csv        # ~420,000 rows of labelled KPI data

models/
  random_forest.pkl             # Trained RF model
  lstm_model.h5                 # Trained LSTM network (Keras/h5)
  svm_baseline.pkl              # Trained SVM model
  scaler_lstm.pkl               # StandardScaler for LSTM inputs
  scaler_tab.pkl                # StandardScaler for tabular inputs
  pca.pkl                       # PCA transformer (95% variance)
  metadata.json                 # Window size, KPI columns, class names

reports/
  lstm_training_history.png     # Training & validation loss/accuracy curves
  mapek_summary.json            # MTTR and availability per model vs. baseline
```

### Expected MAPE-K Summary Format

```json
{
  "Reactive Baseline": [<mttr_s>, <availability_%>],
  "LSTM":             [<mttr_s>, <availability_%>],
  "RF":               [<mttr_s>, <availability_%>],
  "SVM":              [<mttr_s>, <availability_%>]
}
```

---

## Requirements

### System
- **OS**: Ubuntu 22.04 LTS (or compatible Debian-based Linux)
- **NS-3**: Version 3.38 with `core-module` and `network-module`
- **Python**: 3.10 or higher
- **RAM**: ≥ 8 GB recommended (LSTM training + SMOTE on ~300K windows)
- **Disk**: ≥ 2 GB free (raw CSVs + models)

### Key Python Packages
| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow-cpu` | 2.15.0 | LSTM training |
| `scikit-learn` | 1.5.2 | RF, SVM, preprocessing |
| `imbalanced-learn` | 0.12.4 | SMOTE oversampling |
| `pandas` | 2.3.3 | Data handling |
| `numpy` | 1.26.4 | Numerical operations |
| `scipy` | 1.17.1 | Statistical features |
| `matplotlib` / `seaborn` | 3.10 / 0.13 | Visualisation |
| `joblib` | 1.5.3 | Model serialisation |

Full pinned dependencies: [`requirements.txt`](requirements.txt)

---

## Troubleshooting

### NS-3 simulation fails to build
```bash
# Check that the script is in the correct directory
cp thesis-fault-sim.cc ~/ns-3.38/scratch/
cd ~/ns-3.38 && ./ns3 build thesis-fault-sim

# Verify NS-3 modules
ls ~/ns-3.38/src/ | grep -E "core|network"
```

### Simulation trials crash / produce empty CSVs
```bash
# Run in debug mode to see full NS-3 output
python3 run_all_trials.py --debug
```

### Python environment issues
```bash
# Verify environment
source activate_thesis.sh
python3 check_environment.py
```

### LSTM training is very slow
- Ensure `tensorflow-cpu` is installed (not GPU version) to avoid driver conflicts
- Reduce epochs or enable early stopping (already configured, patience=10)
- Use `--skip_svm` if SVM is the bottleneck

### `FileNotFoundError` in `mapek_loop.py`
- Ensure `preprocess_and_train.py` has been run first to populate `models/`
- Check that `kpi_master_dataset.csv` exists in `output/`

---

## Citation / Academic Context

This simulation was developed as part of an MSc research thesis investigating autonomous network management in 5G environments. All simulation code, ML models, and evaluation methodology are original work.

 
**Simulation Engine**: NS-3 Network Simulator v3.38  
**ML Framework**: TensorFlow/Keras 2.15, scikit-learn 1.5  

---

*Built with [NS-3](https://www.nsnam.org/) · [TensorFlow](https://www.tensorflow.org/) · [scikit-learn](https://scikit-learn.org/)*
