# Installation & Execution Guide

> **ML Fault Detection & Self-Healing in Telecom Networks**  
> Complete setup, simulation execution, and pipeline walkthrough.

**Companion docs:** [README.md](README.md) (chapter map & integrity) · [docs/Project_Overview.pdf](docs/Project_Overview.pdf) · [docs/Project_Overview.pptx](docs/Project_Overview.pptx) (technical overview).

---

## Table of Contents

- [Real-World Context (MTN / Airtel)](#real-world-context-mtn--airtel)
- [Autopilot Install (`setup.sh`)](#autopilot-install-setupsh)
- [System Requirements](#system-requirements)
- [Three Simulation Backends](#three-simulation-backends)
- [Manual Installation](#manual-installation)
- [Project File Structure](#project-file-structure)
- [Network Topology & Fault Model](#network-topology--fault-model)
- [CSV Output Format](#csv-output-format)
- [Step-by-Step Execution](#step-by-step-execution)
- [Chapter 4 Pipeline](#chapter-4-pipeline)
- [Troubleshooting](#troubleshooting)
- [Manual Dataset Merge](#manual-dataset-merge)
- [Time-Saving Tips](#time-saving-tips)
- [Quick Command Reference](#quick-command-reference)

---

## Real-World Context (MTN / Airtel)

This is a **simulation study** — not a live integration with MTN Nigeria, Airtel Nigeria, or any operator OSS. However, the entire pipeline is designed to **imitate real operational scenarios** that field engineers and NOC teams encounter in Nigerian mobile networks.

### Why simulation instead of live network data?

| Constraint | Implication |
|------------|-------------|
| Operator KPI data is proprietary | Cannot publish labelled fault datasets from live MTN/Airtel OSS |
| Ground-truth fault labels are rare in production | Supervised ML needs known fault onset/time/class |
| MAPE-K actions cannot be executed on live RAN safely | Recovery policies are **modelled** with operational time estimates |
| Reproducibility for thesis examination | NS-3 gives controlled, repeatable Monte Carlo trials |

**Thesis positioning:** *"A simulation-based digital twin of dense urban HetNet behaviour, calibrated to Nigerian operational remediation practices and NCC availability expectations."*

### Operator scenario → simulation mapping

| Real-world scenario (MTN / Airtel context) | Typical field cause | Simulation proxy | KPI signature in CSV |
|--------------------------------------------|---------------------|--------------------|----------------------|
| **Cell outage / no service** | Grid failure, diesel genset fault, rectifier trip, vandalism | **Power fault** — gNB TX power collapse | RSRP → −115 dBm, throughput → ~0, HO failure, loss ↑ |
| **Severe congestion / slow data** | Peak-hour data surge, stadium/event load, backhaul saturation | **Congestion** — 3× UDP traffic on affected cell | PRB > 90%, latency ↑, throughput −65% |
| **Hardware / site failure** | RRU/BBU fault, fibre cut to site, cooling failure | **HW failure** — PHY deactivation (TX → 0) | Partial collapse, neighbour load shift, high loss |
| **Normal operations** | Typical urban mobility + mixed traffic | **None** campaign (no active fault window) | RSRP −70 to −85 dBm, PRB 55–70%, stable HO |

### Nigerian operational assumptions (Table 4.6)

These values in `thesis_constants.py` reflect **reactive OSS workflows** (manual dispatch, field visit, spare parts) common in Nigerian RAN operations:

| Fault class | Reactive MTTR (manual) | MAPE-K automated target (LSTM) |
|-------------|------------------------|--------------------------------|
| Power fault | ~240 min (4 h) | ~35 min |
| Congestion | ~300 min (5 h) | ~48 min |
| HW failure | ~290 min | ~45 min |

The **reactive baseline** in `mapek_loop.py` waits for **severe** thresholds (PRB > 90%, RSRP < −110 dBm, loss > 30%) — imitating threshold-only OSS alarms that fire late. The **ML + MAPE-K path** detects degradation earlier from KPI windows, imitating proactive SON/analytics platforms.

### Regulatory benchmark

Chapter 4 availability plots include the **NCC 99%** reference line (`generate_chapter4_figures.py`) — the Nigerian Communications Commission quality-of-service expectation for network availability.

### What to tell examiners

> *We do not claim this ran on MTN or Airtel live cores. We claim the fault types, KPI signatures, HetNet topology, and remediation latencies are **operationally grounded** in Nigerian MNO practice, and the comparative benefit of ML+MAPE-K over reactive monitoring is quantified in a controlled, reproducible NS-3 environment.*

---

## Autopilot Install (`setup.sh`)

**Start here on a fresh machine.**

```bash
bash setup.sh
```

### What `setup.sh` does (10 steps)

| Step | Action |
|------|--------|
| **0** | Detect OS, architecture, WSL2 |
| **1** | Install build tools (apt / dnf / pacman / Homebrew) |
| **2** | Install `uv` package manager |
| **3** | Create Python venv with `numpy<2` isolation |
| **4** | Copy all thesis scripts to `~/thesis-sim/` |
| **5** | Download/build NS-3 3.38; clone **5G-LENA** (`contrib/nr`) if missing |
| **6** | Compile `thesis-fault-sim`, `thesis-fault-sim-lte`, `thesis-fault-sim-nr` (if NR built) |
| **7** | Write `activate_thesis.sh` |
| **8** | Generate Chapter 3 figures |
| **9** | Run `check_environment.py` |
| **10** | Prompt y/n for trials → training → MAPE-K |

### Step 10 prompts

```
  Step A — NS-3 simulation trials
  Run all 50 simulation trials now? [y/N]:
    • Fast KPI generator: ~2–15 min (200 trials)
    • LTE (--lte): hours to days
    • NR (--nr): very slow; use --workers 1

  Step B — ML training (RF + LSTM + SVM)
  Run ML training now? [y/N]:

  Step C — MAPE-K evaluation
  Run MAPE-K evaluation now? [y/N]:
```

### After setup — every new terminal

```bash
source ~/thesis-sim/activate_thesis.sh
```

### Platform support

| Platform | Status |
|----------|--------|
| Ubuntu / Debian | Full support |
| Fedora / RHEL | Full support |
| Arch Linux | Full support |
| macOS | Full support (Homebrew) |
| WSL2 on Windows | Detected automatically |
| Native Windows | WSL2 install guide only |

### Windows WSL2 (one-time)

```powershell
wsl --install -d Ubuntu-22.04
```

Optional `C:\Users\<YOU>\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=4
```

Then `wsl --shutdown` and reopen Ubuntu.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04 / WSL2 | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ (LTE/NR sims) |
| Disk | 20 GB free | **50 GB+** (NS-3 + NR build) |
| CPU | 4-core x86-64 | 8-core+ |
| Python | 3.10+ | 3.11 (NS-3 wrapper) |
| GPU | Not required | Optional for LSTM speedup |

### NS-3 modules required

```
core, network, internet, applications, mobility, spectrum,
propagation, antenna, lte, nr, energy, flow-monitor, point-to-point, stats
```

`nr` requires 5G-LENA: `bash scripts/install_5g_lena.sh`

---

## Three Simulation Backends

Choose based on **speed vs. RAN fidelity**. All produce the **same CSV schema**.

| Script | Stack | Speed | Realism | Command |
|--------|-------|-------|---------|---------|
| `thesis-fault-sim` | KPI event generator | Fastest (~1 min/trial) | Simplified physics; correct labels | `python3 run_all_trials.py` |
| `thesis-fault-sim-lte` | LENA LTE + EPC | Slow (~30–120 min/trial) | Real PHY traces + FlowMonitor | `python3 run_all_trials.py --lte` |
| `thesis-fault-sim-nr` | 5G-LENA NR + EPC | Slowest (hours/trial) | True NR n78 (3.5 GHz) | `python3 run_all_trials.py --nr --workers 1` |

### Recommended workflow

1. **Develop ML pipeline** — fast KPI generator (200 trials, ~1.68M rows).
2. **Validate RAN traces** — LTE batch (`--lte --sim-time 120 --num-ues 280`).
3. **5G NR extension** — NR smoke test then selective trials (`--nr`).

### Build all sims manually

```bash
cp ~/thesis-sim/scripts/thesis-fault-sim*.cc ~/ns-3.38/scratch/
cd ~/ns-3.38
~/thesis-sim/bin/ns3 build thesis-fault-sim
~/thesis-sim/bin/ns3 build thesis-fault-sim-lte
~/thesis-sim/bin/ns3 build thesis-fault-sim-nr   # requires contrib/nr
```

---

## Manual Installation

Use if `setup.sh` fails or you already have NS-3.

### 1. Workspace

```bash
mkdir -p ~/thesis-sim/{scripts,output/raw,models,reports,bin}
# Copy all repo files into ~/thesis-sim/ (or run setup.sh Step 4)
```

### 2. System packages (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build g++ \
    python3 python3-pip python3-dev python3-venv git wget \
    libboost-all-dev libssl-dev libxml2-dev gsl-bin libgsl-dev
```

### 3. Python environment

```bash
cd ~/thesis-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. NS-3 3.38

```bash
cd ~
wget https://www.nsnam.org/releases/ns-allinone-3.38.tar.bz2
tar xjf ns-allinone-3.38.tar.bz2
mv ns-allinone-3.38/ns-3.38 ~/ns-3.38

cd ~/ns-3.38
~/thesis-sim/bin/ns3 configure --build-profile=optimized \
    --enable-modules=core,network,internet,applications,mobility,spectrum,propagation,antenna,lte,nr,energy,flow-monitor,point-to-point,stats
~/thesis-sim/bin/ns3 build
```

### 5. 5G-LENA (for `--nr` only)

```bash
bash ~/thesis-sim/scripts/install_5g_lena.sh
# Reconfigure + rebuild NS-3 with nr module
```

### 6. Verify

```bash
source ~/thesis-sim/activate_thesis.sh
python3 check_environment.py
```

---

## Project File Structure

```
~/ns-3.38/scratch/
├── thesis-fault-sim.cc       # Fast KPI generator
├── thesis-fault-sim-lte.cc   # LENA LTE/EPC
└── thesis-fault-sim-nr.cc    # 5G-LENA NR/EPC

~/thesis-sim/
├── setup.sh                  # Autopilot installer
├── activate_thesis.sh        # Source in every terminal
├── thesis_constants.py       # Ch. 3/4 approved parameters
├── run_all_trials.py         # Stage 2: trial orchestration
├── preprocess_and_train.py   # Stage 3: ML training
├── mapek_loop.py             # Stage 4: MAPE-K evaluation
├── run_chapter4_pipeline.sh  # Train → MAPE-K → Ch. 4 figures
├── thesis_eval.py            # MAPE-K metric helpers
│
├── scripts/
│   ├── thesis-fault-sim*.cc  # Backup copies
│   ├── generate_figures.py           # Ch. 3 figures
│   ├── generate_chapter4_figures.py    # Ch. 4 figures
│   ├── install_5g_lena.sh
│   └── ns3_thesis.sh         # NS-3 wrapper (venv Python 3.11)
│
├── output/
│   ├── raw/                  # kpi_trial{N}_{fault}.csv (200 files)
│   └── kpi_master_dataset.csv
│
├── models/                   # RF, LSTM, SVM, scalers, PCA
└── reports/                  # JSON metrics + PNG figures
```

---

## Network Topology & Fault Model

Aligned with **Table 3.1** (Ch. 3):

| Parameter | Value |
|-----------|-------|
| Macro gNBs / eNBs | 7 (hexagonal, ISD 500 m) |
| Small cells per macro | 3 |
| **Total cells** | **28** |
| UEs (thesis target) | 500 |
| UEs (stable default) | 280 (`--num-ues 280`) |
| Simulation time | 300 s (C++ default); 120 s typical in batch |
| KPI interval | 1 s |
| Trials | 50 × 4 fault campaigns = **200 runs** |
| UE mobility | Random walk 0.8–8.3 m/s (urban) |

### Fault injection (stochastic per trial)

- Random fault cell (0–27)
- Random onset between 10 s and ~65% of sim time
- Duration 5–30 s (scaled to sim length)
- Ground truth written to `fault_label` column

### Fault classes

| Label | Name | Real-world analogue | Sim mechanism |
|-------|------|---------------------|---------------|
| 0 | Normal | Healthy cell | No active fault window |
| 1 | Power fault | Site power outage | gNB TX power → 0 |
| 2 | Congestion | Peak load / event surge | UDP interval 80 ms → 8 ms on cell UEs |
| 3 | HW failure | RRU/BBU / transport failure | PHY TX deactivation |

---

## CSV Output Format

**One row per cell per second.**

| Metric | Full run (300 s, 200 files) |
|--------|----------------------------|
| Rows per trial file | 28 × 300 = **8,400** |
| Merged master dataset | **~1,680,000 rows** |

| Column | Description |
|--------|-------------|
| `trial` | Monte Carlo index (0–49) |
| `fault_type` | Campaign: `none`, `power`, `congestion`, `hardware` |
| `time` | Simulation time (s) |
| `gnb_id` | Cell index **0–27** |
| `macro_id` | Parent macro 0–6 |
| `cell_type` | `macro` or `small` |
| `rsrp_avg_dbm` | Reference signal power |
| `sinr_avg_db` | SINR |
| `prb_utilisation` | PRB load (0–1) |
| `dl_throughput_mbps` / `ul_throughput_mbps` | Throughput |
| `packet_loss_rate` | Loss ratio |
| `handover_success_rate` | HO success |
| `latency_avg_ms` | Mean latency |
| `fault_start_s` / `fault_end_s` | Injected window (9999 if none) |
| `fault_label` | **Ground truth** 0–3 for ML |

> After windowing + Table 4.1 subsampling, ML sees ~51k windows — not 1.68M raw rows. Both numbers are correct at different pipeline stages.

---

## Step-by-Step Execution

### Step 1 — Test a single trial

```bash
source ~/thesis-sim/activate_thesis.sh
mkdir -p ~/thesis-sim/output/raw

# Fast KPI generator (~30 s)
~/thesis-sim/bin/ns3 run "thesis-fault-sim --trial=0 --fault=power \
    --outputDir=$HOME/thesis-sim/output/raw"

# LTE (real RAN — slow)
~/thesis-sim/bin/ns3 run "thesis-fault-sim-lte --trial=0 --fault=none \
    --outputDir=$HOME/thesis-sim/output/raw --simTime=60 --numUes=56"

# NR (5G — very slow)
~/thesis-sim/bin/ns3 run "thesis-fault-sim-nr --trial=0 --fault=none \
    --outputDir=$HOME/thesis-sim/output/raw --simTime=60 --numUes=56"

head -3 ~/thesis-sim/output/raw/kpi_trial0_power.csv
```

### Step 2 — Run all trials

```bash
cd ~/thesis-sim

# A) Fast dataset for ML pipeline (~minutes)
python3 run_all_trials.py --workers 4

# B) Real LTE HetNet (hours–days)
python3 run_all_trials.py --lte --sim-time 120 --num-ues 280 --workers 2

# C) 5G NR (days; single worker)
python3 run_all_trials.py --nr --sim-time 120 --num-ues 280 --workers 1

# Debug any backend
python3 run_all_trials.py --lte --debug
python3 run_all_trials.py --nr --debug --sim-time 60 --num-ues 56
```

Set `--workers` to **CPU cores − 1** for KPI/LTE; use **`--workers 1`** for NR.

### Step 3 — Verify master dataset

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('output/kpi_master_dataset.csv')
print('Shape:', df.shape)
print('Cells:', df['gnb_id'].nunique(), '(expect 28)')
labels = {0:'Normal',1:'Power',2:'Congestion',3:'HW Failure'}
for k,v in df['fault_label'].value_counts().sort_index().items():
    print(f'  {labels[k]}: {v:,}  ({100*v/len(df):.1f}%)')
print('Missing:', df.isnull().sum().sum())
"
```

### Step 4 — Train ML models

```bash
python3 preprocess_and_train.py
# Optional: python3 preprocess_and_train.py --skip_svm
```

Results → `reports/ml_metrics.json` (**observed** accuracy/F1/AUC by default).

### Step 5 — MAPE-K evaluation

```bash
python3 mapek_loop.py --model all
```

Results → `reports/mapek_summary.json`, `reports/chapter4_results.json`

### Step 6 — Generate figures

```bash
python3 scripts/generate_figures.py           # Ch. 3 (Figs 3.1–3.5)
python3 scripts/generate_chapter4_figures.py  # Ch. 4 (Figs 4.2b, 4.4, 4.5)
```

---

## Chapter 4 Pipeline

One command after `kpi_master_dataset.csv` exists:

```bash
bash run_chapter4_pipeline.sh
```

Runs: train (if models missing) → MAPE-K → Chapter 4 figures.

### Metric honesty

`thesis_constants.py` sets `METRIC_BLEND_WEIGHT = 0` — results are **observed-only** by default. Do not use `--blend-thesis-metrics` in final thesis results without disclosure.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ns3: command not found` | NS-3 not built | `cd ~/ns-3.38 && ~/thesis-sim/bin/ns3 build` |
| `thesis-fault-sim-lte not found` | `.cc` not in scratch | `cp ~/thesis-sim/scripts/thesis-fault-sim-lte.cc ~/ns-3.38/scratch/` then build |
| LTE HARQ assert at 500 UEs | NS-3 stability limit | `--num-ues 280` |
| NR build missing | 5G-LENA not cloned | `bash scripts/install_5g_lena.sh` |
| NR sim extremely slow | Expected | `--workers 1`, shorter `--sim-time` for smoke |
| Empty CSV | Sim crashed | `python3 run_all_trials.py --debug` (add `--lte`/`--nr`) |
| Python 3.14 breaks `./ns3` | Wrong system Python | Use `~/thesis-sim/bin/ns3` wrapper |
| `numpy` version conflict | System numpy 2.x bleed | Re-run `setup.sh` repair or recreate venv |
| `No module named tensorflow` | Venv not active | `source activate_thesis.sh` |
| MAPE-K missing models | Skipped training | Run `preprocess_and_train.py` first |
| WSL2 OOM | Low memory cap | Increase `.wslconfig` memory to 8 GB+ |

Full log from setup: `~/thesis_setup.log`

---

## Manual Dataset Merge

If trials completed but merge failed:

```python
import pandas as pd, os, glob

raw_dir = os.path.expanduser("~/thesis-sim/output/raw")
out_csv = os.path.expanduser("~/thesis-sim/output/kpi_master_dataset.csv")

files = [f for f in glob.glob(os.path.join(raw_dir, "kpi_trial*.csv"))
         if os.path.getsize(f) > 100]
print(f"Found {len(files)} valid CSV files")

dfs = [pd.read_csv(f) for f in sorted(files)]
master = pd.concat(dfs, ignore_index=True)
master.to_csv(out_csv, index=False)
print(f"Merged → {len(master):,} rows")
print(master["fault_label"].value_counts().sort_index())
```

---

## Time-Saving Tips

### Overnight run

```bash
cd ~/thesis-sim
nohup python3 run_all_trials.py --lte --sim-time 120 --num-ues 280 --workers 2 \
    > sim_lte_log.txt 2>&1 &
tail -f sim_lte_log.txt
```

### Quick pipeline validation

```bash
python3 run_all_trials.py --trials 2 --workers 2
python3 preprocess_and_train.py --skip_svm
python3 mapek_loop.py --model rf
```

### GPU for LSTM (optional)

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Quick Command Reference

```bash
# ── Autopilot ─────────────────────────────────────────────────────────────
bash setup.sh
source ~/thesis-sim/activate_thesis.sh

# ── Simulations ───────────────────────────────────────────────────────────
python3 run_all_trials.py                              # fast KPI generator
python3 run_all_trials.py --lte --sim-time 120 --num-ues 280 --workers 2
python3 run_all_trials.py --nr  --sim-time 120 --num-ues 280 --workers 1
python3 run_all_trials.py --debug                      # diagnose failures

# ── ML + MAPE-K + Figures ─────────────────────────────────────────────────
python3 preprocess_and_train.py
python3 mapek_loop.py --model all
bash run_chapter4_pipeline.sh
python3 scripts/generate_figures.py
python3 scripts/generate_chapter4_figures.py

# ── Verify ────────────────────────────────────────────────────────────────
python3 check_environment.py
ls ~/thesis-sim/output/raw/*.csv | wc -l    # up to 200
cat ~/thesis-sim/reports/mapek_summary.json
cat ~/thesis-sim/reports/ml_metrics.json
```

---

## Pipeline time estimates

| Stage | KPI generator | LTE (`--lte`) | NR (`--nr`) |
|-------|---------------|---------------|-------------|
| 200 trials | 5–30 min | 1–5 days | 1–2+ weeks |
| ML training | 1–2 h | 1–2 h | 1–2 h |
| MAPE-K | 15–30 min | 15–30 min | 15–30 min |

---

*For thesis chapter mapping, viva defense points, and figure documentation, see [README.md](README.md).*
