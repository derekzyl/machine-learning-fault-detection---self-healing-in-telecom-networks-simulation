# ML Fault Detection & Self-Healing in Telecom Networks

> **MSc Thesis — Simulation & ML Pipeline**  
> End-to-end research framework for proactive RAN fault detection and MAPE-K-guided self-healing, built on NS-3 3.38 with Python ML and evaluation tooling.

**Related docs:** [INSTALL.md](INSTALL.md) (install + MTN/Airtel mapping) · [docs/Project_Overview.pdf](docs/Project_Overview.pdf) (PDF) · [docs/Project_Overview.pptx](docs/Project_Overview.pptx) (PowerPoint) · [thesis_constants.py](thesis_constants.py) (Ch. 3/4 parameters)

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [Research Question & Contribution](#research-question--contribution)
- [Thesis Chapter Map](#thesis-chapter-map)
- [Three RAN Simulation Backends](#three-ran-simulation-backends)
- [Network Topology](#network-topology)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [CSV Dataset Schema](#csv-dataset-schema)
- [Figures & Thesis Paperwork](#figures--thesis-paperwork)
- [Academic Integrity & Defense Guide](#academic-integrity--defense-guide)
- [Quickstart](#quickstart)
- [Detailed Usage](#detailed-usage)
- [Models & MAPE-K](#models--mape-k)
- [Output Artefacts](#output-artefacts)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

---

## What This Project Is

This repository is the **reproducible implementation** behind the thesis chapters on methodology (Ch. 3) and results (Ch. 4). It is **not** a live MTN/Airtel network integration — but fault types, KPI signatures, HetNet layout, and Nigerian remediation times are designed to **imitate real MNO operational scenarios**. See [INSTALL.md § Real-World Context](INSTALL.md#real-world-context-mtn--airtel).

| Layer | Role |
|-------|------|
| **NS-3 simulation (C++)** | Generates labelled per-cell KPI time-series under controlled faults |
| **Python ML pipeline** | Trains RF, LSTM, and SVM on sliding windows; exports Ch. 4 metrics |
| **MAPE-K evaluator** | Simulates proactive self-healing vs. reactive OSS-style monitoring |
| **Figure generators** | Produces Ch. 3 methodology diagrams and Ch. 4 results plots |

Everything flows from simulation ground truth (`fault_label` in CSV) → supervised ML → operational impact (MTTR, availability).

---

## Research Question & Contribution

**Core question (Ch. 4):**

> *Can machine-learning-based proactive fault detection significantly reduce Mean Time to Recovery (MTTR) and improve network availability compared to conventional threshold-based reactive monitoring in dense HetNet RAN deployments?*

**What the thesis contributes:**

1. A **28-cell HetNet** fault-injection framework in NS-3 with stochastic trial seeds (Monte Carlo reproducibility).
2. A **labelled KPI corpus** (8 features × per-second sampling × per-cell rows) for 4-class fault detection.
3. A comparative study of **Random Forest, LSTM, and SVM** on multivariate time-series windows.
4. A **MAPE-K closed loop** (Monitor → Analyse → Plan → Execute) with Nigerian operational remediation assumptions (Table 4.6).
5. A **reactive baseline** that only acts at severe thresholds — quantifying the detection delay gap.

**What it does not claim:** field trials on a live MNO network, vendor-specific OSS integration, or guaranteed NR production-scale runs without stating runtime constraints.

---

## Thesis Chapter Map

Use this table when writing or defending the dissertation — every thesis section maps to code.

| Thesis section | Content | Code / output |
|----------------|---------|---------------|
| **Table 3.1** | 28-cell HetNet, 500 UEs, 300 s, KPI interval 1 s | `thesis_constants.py`, `thesis-fault-sim*.cc` |
| **§3.2.3** | Fault injection (power, congestion, hardware) | `thesis-fault-sim*.cc` — `Trigger*Fault()` |
| **§3.4.3** | Sliding windows, SMOTE, PCA, RF/LSTM/SVM | `preprocess_and_train.py` |
| **§3.4.4** | MAPE-K loop, monitor thresholds, remediation | `mapek_loop.py`, `thesis_eval.py` |
| **Fig. 3.1–3.5** | Methodology diagrams | `scripts/generate_figures.py` |
| **Table 4.1** | Class distribution | `TABLE_41_COUNTS` in `thesis_constants.py`; subsampling in `preprocess_and_train.py` |
| **§4.3–4.5** | ML accuracy, F1, AUC | `reports/ml_metrics.json` (observed values) |
| **Table 4.6 / §4.7** | MTTR, availability vs. reactive | `reports/mapek_summary.json`, `reports/chapter4_results.json` |
| **Fig. 4.2b, 4.4, 4.5** | Results plots | `scripts/generate_chapter4_figures.py` |

**One-command Ch. 4 pipeline** (after trials + dataset exist):

```bash
bash run_chapter4_pipeline.sh
# train (if needed) → MAPE-K → Chapter 4 figures
```

---

## Three RAN Simulation Backends

The thesis supports **three** NS-3 programs. Choose based on speed vs. RAN fidelity. All write the **same CSV schema** so the Python pipeline is unchanged.

```
                    ┌─────────────────────┐
                    │ kpi_master_dataset  │
                    │       .csv          │
                    └──────────▲──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
  thesis-fault-sim    thesis-fault-sim-lte   thesis-fault-sim-nr
  (KPI generator)     (LENA LTE/EPC)         (5G-LENA NR/EPC)
  FAST                MEDIUM                 SLOWEST
  simplified physics  real PHY traces        true NR n78 band
```

| Script | Stack | When to use | Typical command |
|--------|-------|-------------|-----------------|
| `thesis-fault-sim` | NS-3 event-based KPI generator | Large-scale ML dataset (200 trials in ~minutes) | `python3 run_all_trials.py` |
| `thesis-fault-sim-lte` | LENA + PointToPoint EPC | **Primary RAN validation** — real LTE HetNet traces | `python3 run_all_trials.py --lte --workers 2` |
| `thesis-fault-sim-nr` | 5G-LENA v2.4 + NR EPC | **5G NR extension** — n78 (3.5 GHz, 100 MHz) | `python3 run_all_trials.py --nr --workers 1` |

### Recommended thesis wording (RAN)

> *Network behaviour is modelled in NS-3 using a 28-cell HetNet. LTE-A LENA/EPC (`thesis-fault-sim-lte`) serves as the primary validated RAN stack for KPI generation from PHY traces and FlowMonitor. A 5G-LENA NR implementation (`thesis-fault-sim-nr`) is provided for NR-specific validation. The fast KPI generator (`thesis-fault-sim`) supports large-scale Monte Carlo dataset construction with identical labelling semantics.*

Install 5G-LENA if missing: `bash scripts/install_5g_lena.sh`

---

## Network Topology

Aligned with **Table 3.1** (Ch. 3):

| Parameter | Value |
|-----------|-------|
| Macro gNBs | 7 (hexagonal layout, ISD 500 m) |
| Small cells per macro | 3 (offset ISD/3) |
| **Total cells** | **28** |
| UEs (thesis target) | 500 |
| UEs (stable LENA/NR default) | 280 (`--num-ues 280` — avoids NS-3 HARQ assert on typical hardware) |
| Simulation time | 300 s (default in C++); 120 s common in batch runs (`--sim-time 120`) |
| KPI sample interval | 1 s |
| Trials | 50 per fault campaign |
| Fault campaigns | `none`, `power`, `congestion`, `hardware` → **200 runs** total |

**Rows per trial file:** `28 cells × sim_time_seconds`  
**Full merged dataset (300 s, 200 files):** ~**1,680,000 rows**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                              │
│                                                                          │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────────────┐ │
│  │ NS-3 3.38   │   │ preprocess_and_  │   │ mapek_loop.py           │ │
│  │ fault sim   │──▶│ train.py         │──▶│ (MAPE-K evaluation)     │ │
│  │ (C++)       │   │ RF + LSTM + SVM  │   │ MTTR + availability     │ │
│  └─────────────┘   └──────────────────┘   └─────────────────────────┘ │
│        │                    │                         │                  │
│  run_all_trials.py     models/*.pkl/.h5         reports/*.json/png      │
│  → kpi_master_dataset.csv                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌────────────────────────  MAPE-K LOOP (§3.4.4)  ────────────────────────┐
│                                                                          │
│   Monitor ──▶ Analyse ──▶ Plan ──▶ Execute ──▶ Knowledge (update)       │
│      │           │          │         │                                  │
│   KPI window  ML model   Remediation  Log action + recovery time         │
│   pre-filter  (conf≥0.7)  policy      (Table 4.6 minutes)                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 0 — Environment (`setup.sh`)

Full autopilot install: system deps, Python venv, NS-3 3.38, optional 5G-LENA, compile all sim scripts. See [INSTALL.md](INSTALL.md).

### Stage 1 — NS-3 simulation

- **Input:** trial index, fault type, output directory, optional `simTime` / `numUes`
- **Process:** Deploy UEs, staggered attach, UDP downlink via EPC remote host, per-second KPI collection
- **Faults:** random cell, random onset/duration window; power/HW → TX collapse; congestion → traffic surge
- **Output:** `output/raw/kpi_trial{N}_{fault}.csv`

### Stage 2 — Trial runner (`run_all_trials.py`)

- Builds active sim script, sanity-checks trial 0, runs parallel workers, merges CSVs
- Flags: `--lte`, `--nr`, `--sim-time`, `--num-ues`, `--debug`, `--fault`, `--trials`, `--workers`

### Stage 3 — ML (`preprocess_and_train.py`)

1. Load `kpi_master_dataset.csv`
2. Subsample windows toward **Table 4.1** class counts (`TABLE_41_COUNTS`)
3. **10 s windows**, stride 1, per `gnb_id` group
4. **48 tabular features** (6 stats × 8 KPIs) + raw sequences for LSTM
5. Train/val/test **70 / 15 / 15%**, stratified, `RANDOM_STATE=42`
6. `StandardScaler`, PCA (95% variance on tabular), **SMOTE** on train only
7. Train **RF, LSTM, SVM**; export `reports/ml_metrics.json`

**Metric reporting:** `METRIC_BLEND_WEIGHT = 0` in `thesis_constants.py` → **observed-only** by default. Approved Ch. 4 headline numbers are validation targets, not forced outputs. Optional `--blend-thesis-metrics` exists for draft alignment only.

### Stage 4 — MAPE-K (`mapek_loop.py`)

- Replays test partition through Monitor → Analyse → Plan → Execute
- **Reactive baseline:** severe thresholds only (PRB > 90%, RSRP < −110 dBm, etc.)
- **ML path:** earlier detection bonus + shorter remediation (minutes, Table 4.6)
- Exports `reports/mapek_summary.json`, `reports/chapter4_results.json`
- Same observed-only default for MTTR/availability (`MAPEK_MTTR_BLEND = 0`)

### Stage 5 — Figures

```bash
python3 scripts/generate_figures.py          # Ch. 3 (Figs 3.1–3.5)
python3 scripts/generate_chapter4_figures.py  # Ch. 4 (Figs 4.2b, 4.4, 4.5)
```

---

## CSV Dataset Schema

Every simulation backend produces identical columns:

| Column | Description |
|--------|-------------|
| `trial` | Monte Carlo trial index (0–49) |
| `fault_type` | Campaign label: `none`, `power`, `congestion`, `hardware` |
| `time` | Simulation time (seconds) |
| `gnb_id` | Cell index 0–27 |
| `macro_id` | Parent macro 0–6 |
| `cell_type` | `macro` or `small` |
| `rsrp_avg_dbm` | Reference signal power (dBm) |
| `sinr_avg_db` | Signal-to-interference ratio (dB) |
| `prb_utilisation` | PRB load (0–1) |
| `dl_throughput_mbps` | Downlink throughput |
| `ul_throughput_mbps` | Uplink throughput |
| `packet_loss_rate` | Loss ratio |
| `handover_success_rate` | HO success ratio |
| `latency_avg_ms` | Mean latency |
| `fault_start_s` | Injected fault start (9999 if none) |
| `fault_end_s` | Injected fault end |
| `fault_label` | **Ground truth:** 0=Normal, 1=Power, 2=Congestion, 3=HW |

`fault_label` is the supervised learning target — derived from simulation truth, not from ML.

---

## Figures & Thesis Paperwork

### Chapter 3 figures (`scripts/generate_figures.py`)

| File | Figure | Type | Notes |
|------|--------|------|-------|
| `fig3_1_topology.png` | 3.1 | Schematic | Hexagonal macro layout (methodology) |
| `fig3_2_pipeline.png` | 3.2 | Schematic | End-to-end ML pipeline |
| `fig3_3_mapek.png` | 3.3 | Schematic | MAPE-K control loop |
| `fig3_4_timeline.png` | 3.4 | **Data-driven** when `output/raw/` exists | Fault injection windows from CSVs |
| `fig3_5_lstm_arch.png` | 3.5 | Schematic | LSTM architecture |

### Chapter 4 figures (`scripts/generate_chapter4_figures.py`)

| File | Figure | Source |
|------|--------|--------|
| `fig4_2b_availability.png` | 4.2b | `mapek_summary.json` / `chapter4_results.json` |
| `fig4_4_normal_fraction.png` | 4.4 | `kpi_master_dataset.csv` |
| `fig4_5_mttr.png` | 4.5 | MAPE-K results JSON |

### Tables → JSON artefacts

| Thesis table | Generated file | Key fields |
|--------------|----------------|------------|
| ML metrics (§4.3–4.5) | `reports/ml_metrics.json` | `observed_accuracy`, `observed_macro_f1`, `observed_auc_roc` |
| MTTR / availability (§4.7) | `reports/mapek_summary.json` | Per-model MTTR (min) and availability (%) |
| Full Ch. 4 bundle | `reports/chapter4_results.json` | Structured export for tables |

**For the dissertation:** cite `observed_*` fields in results text. Reference `THESIS_*` constants in `thesis_constants.py` only as *expected* or *approved draft* values when comparing.

---

## Academic Integrity & Defense Guide

### One-sentence pitch for viva / supervisor meetings

> *We built a reproducible NS-3 HetNet pipeline that generates labelled RAN KPIs under controlled faults, trains ML classifiers for early detection, and quantifies self-healing benefit via a MAPE-K loop versus reactive threshold monitoring.*

### What is defensible

| Claim | Evidence |
|-------|----------|
| 28-cell HetNet with 4 fault classes | `thesis-fault-sim*.cc`, raw CSVs |
| Supervised ML on 8 KPIs | `preprocess_and_train.py`, `models/` |
| MAPE-K reduces detection delay vs. reactive | `mapek_loop.py`, `thesis_eval.py` |
| Reproducibility | Fixed seeds, `setup.sh`, documented commands |
| Honest metrics | `METRIC_BLEND_WEIGHT = 0` — no artificial tuning toward tables |

### Common examiner questions

**Why LTE if the title says 5G?**  
LTE-A LENA is the mature NS-3 RAN stack; it captures HetNet density, handover, PRB stress, and EPC behaviour relevant to fault KPIs. NR (`thesis-fault-sim-nr`) validates the same methodology on 5G-LENA. This is standard practice in simulation studies.

**Are Chapter 4 numbers fabricated?**  
No — the pipeline reports **observed** simulation and model outputs. Constants like `THESIS_ML_METRICS` are comparison targets. Never use `--blend-thesis-metrics` in final results without explicit disclosure.

**500 UEs or 280?**  
Table 3.1 target is 500. 280 is the stability-validated default for LENA/NR on this NS-3 build — state it as a **platform limitation**, not a hidden downgrade.

**Is this a live network test?**  
No. It is a **simulation study** with operationally grounded remediation times (Nigerian OSS context, Table 4.6).

### Limitations to state in Ch. 5

1. Simulation ≠ commercial vendor RAN (simplified PHY/schedulers).
2. LTE primary / NR extension — clarify which dataset each result uses.
3. UE scaling and runtime (NR trials can take hours per short run).
4. MAPE-K Execute phase is **modelled** recovery time, not real OSS API integration.
5. Some Ch. 3 figures are schematic by design.

### Red lines — do not claim

- Live MNO network deployment or field KPI capture
- All results from NR unless NR batch actually completed
- Exact match to Table 4.x without showing observed JSON values

---

## Quickstart

### Option A — Full autopilot (recommended)

```bash
bash setup.sh
# Follow prompts for trials (Step A), training, MAPE-K
```

### Option B — Manual (existing NS-3 install)

```bash
source ~/thesis-sim/activate_thesis.sh

# 1. Fast KPI dataset (minutes)
python3 run_all_trials.py --workers 4

# OR real LTE RAN (hours)
python3 run_all_trials.py --lte --sim-time 120 --num-ues 280 --workers 2

# OR 5G NR (very slow — use 1 worker)
python3 run_all_trials.py --nr --sim-time 120 --num-ues 280 --workers 1

# 2. Train models
python3 preprocess_and_train.py

# 3. MAPE-K + Ch. 4 figures
bash run_chapter4_pipeline.sh
```

### Debug a failing sim

```bash
python3 run_all_trials.py --debug          # KPI generator
python3 run_all_trials.py --lte --debug    # LTE
python3 run_all_trials.py --nr --debug --sim-time 60 --num-ues 56
```

---

## Detailed Usage

### `run_all_trials.py`

```
usage: run_all_trials.py [-h] [--trials N] [--workers N] [--fault TYPE]
                         [--debug] [--lte] [--nr]
                         [--sim-time SEC] [--num-ues N]

  --trials N       Monte Carlo trials (default: 50)
  --workers N      Parallel processes (default: 2; use 1 for --nr)
  --fault TYPE     One fault only: none | power | congestion | hardware
  --lte            Use thesis-fault-sim-lte (LENA/EPC)
  --nr             Use thesis-fault-sim-nr (5G-LENA)
  --sim-time SEC   RAN duration (default: 120 with --lte/--nr)
  --num-ues N      UE count (default: 280 with --lte/--nr)
  --debug          One foreground trial with full NS-3 output
```

### `preprocess_and_train.py`

```
  --data PATH              Master CSV path
  --skip_svm               Skip slow SVM training
  --blend-thesis-metrics   Optional draft alignment (not for final defense)
```

### `mapek_loop.py`

```
  --model MODEL            lstm | rf | svm | all
  --blend-thesis-metrics   Optional draft alignment (not for final defense)
```

---

## Models & MAPE-K

### Fault classes

| Label | Class | Injection proxy |
|-------|-------|-----------------|
| 0 | Normal | No active fault window |
| 1 | Power fault | gNB TX power collapse |
| 2 | Congestion | 3× UDP traffic surge on affected cell |
| 3 | gNB HW failure | PHY deactivation (TX → 0) |

### ML models (summary)

| Model | Role | Key config |
|-------|------|------------|
| **Random Forest** | Strong tabular baseline | 200 trees, max depth 20, balanced weights |
| **LSTM** | Temporal KPI patterns | 2×128 LSTM, 10×8 input, early stopping |
| **SVM** | Classical baseline | RBF kernel, capped training samples |

### MAPE-K parameters (`thesis_constants.py`)

| Parameter | Value |
|-----------|-------|
| Monitor cycle | 5 s |
| Confidence threshold | 0.70 |
| Window size | 10 s |
| Reactive PRB trigger | > 90% |
| Reactive RSRP trigger | < −110 dBm |

Remediation times are in **minutes** (Table 4.6) — LSTM typically achieves lowest MTTR in the approved narrative because it detects earliest.

---

## Output Artefacts

```
~/thesis-sim/
├── output/
│   ├── raw/kpi_trial{N}_{fault}.csv   # Per-run KPI logs
│   └── kpi_master_dataset.csv         # Merged labelled dataset
├── models/
│   ├── random_forest.pkl
│   ├── lstm_model.h5
│   ├── svm_baseline.pkl
│   ├── scaler_*.pkl, pca.pkl
│   └── metadata.json
└── reports/
    ├── ml_metrics.json                # Ch. 4 ML tables (observed + optional blend)
    ├── mapek_summary.json             # MTTR & availability per model
    ├── chapter4_results.json          # Structured Ch. 4 export
    ├── fig3_*.png                     # Chapter 3 figures
    └── fig4_*.png                     # Chapter 4 figures
```

---

## Project Structure

```
machine-learning-fault-detection---self-healing-in-telecom-networks-simulation/
│
├── thesis-fault-sim.cc           # Fast 28-cell KPI generator
├── thesis-fault-sim-lte.cc       # LENA LTE/EPC HetNet
├── thesis-fault-sim-nr.cc        # 5G-LENA NR/EPC HetNet
├── thesis_constants.py           # Approved Ch. 3/4 parameters
├── thesis_eval.py                # MAPE-K metric helpers
├── run_all_trials.py             # Stage 2: trial orchestration
├── preprocess_and_train.py       # Stage 3: ML training
├── mapek_loop.py                 # Stage 4: MAPE-K evaluation
├── run_chapter4_pipeline.sh      # Train → MAPE-K → Ch. 4 figures
├── setup.sh                      # Full environment installer
├── INSTALL.md                    # Detailed install guide
├── scripts/
│   ├── generate_figures.py       # Chapter 3 figures
│   ├── generate_chapter4_figures.py
│   ├── install_5g_lena.sh
│   └── ns3_thesis.sh             # NS-3 wrapper (venv Python 3.11)
└── requirements.txt
```

Runtime workspace (created by `setup.sh`): `~/thesis-sim/`  
NS-3 install: `~/ns-3.38/` (scratch copies of `thesis-fault-sim*.cc`)

---

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| OS | Ubuntu 22.04 / Debian | Linux natively or WSL2 |
| Python | 3.10+ (venv uses 3.11 for NS-3 wrapper) | 3.11 |
| NS-3 | 3.38 + lte, flow-monitor, point-to-point | + nr (5G-LENA) for `--nr` |
| RAM | 8 GB | 16 GB for LENA/NR |
| Disk | 5 GB free | 30 GB+ for NS-3 build + NR |
| CPU | 4 cores | 8+ cores for parallel trials |

Key Python packages: `tensorflow-cpu`, `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy` — see [requirements.txt](requirements.txt).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| NS-3 build fails | `cp thesis-fault-sim*.cc ~/ns-3.38/scratch/` then `~/thesis-sim/bin/ns3 build <name>` |
| Empty CSV / crash | `python3 run_all_trials.py --debug` (or `--lte` / `--nr`) |
| LTE HARQ assert at 500 UEs | Use `--num-ues 280` |
| NR extremely slow | Expected; use `--workers 1`, shorter `--sim-time` for smoke tests |
| NR module missing | `bash scripts/install_5g_lena.sh` |
| Python/numpy conflicts | `source activate_thesis.sh` then `python3 check_environment.py` |
| MAPE-K missing models | Run `preprocess_and_train.py` first |

---

## Citation & Acknowledgement

Developed as part of an MSc thesis on autonomous network management in dense HetNet RAN environments.

| Component | Technology |
|-----------|------------|
| Network simulation | [NS-3](https://www.nsnam.org/) 3.38 |
| LTE RAN | ns-3 LENA module |
| 5G NR RAN | [5G-LENA](https://gitlab.com/cttc-lena/nr) v2.4.y |
| ML | TensorFlow/Keras 2.15, scikit-learn 1.5 |

---

*For step-by-step installation, Windows/WSL2 notes, and time estimates, see [INSTALL.md](INSTALL.md).*
