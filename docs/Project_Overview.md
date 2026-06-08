# ML Fault Detection & Self-Healing in Telecom Networks

**Technical Overview Document**

---

## 1. Purpose

This project implements an end-to-end research pipeline for **proactive RAN fault detection** and **MAPE-K-guided self-healing** in dense HetNet mobile networks.

The work addresses a practical operations question:

> *Can machine-learning-based proactive fault detection reduce Mean Time to Recovery (MTTR) and improve network availability compared to conventional threshold-based reactive monitoring?*

The pipeline combines:

- **NS-3 network simulation** — controlled fault injection and KPI collection  
- **Machine learning** — Random Forest, LSTM, and SVM classifiers on KPI time-series  
- **MAPE-K autonomic loop** — Monitor → Analyse → Plan → Execute evaluation  

Results are produced from **observed simulation and model outputs**. The default configuration does not blend or force results toward pre-defined target tables.

---

## 2. Real-World Operational Context

This is a **simulation study**, not a live operator network integration. The design imitates operational scenarios common in Nigerian mobile networks (e.g. MTN, Airtel) and similar dense urban HetNet deployments.

### Why simulation?

| Factor | Rationale |
|--------|-----------|
| Operator KPI data is proprietary | Public, reproducible labelled datasets are not available |
| Ground-truth fault labels are required | Supervised ML needs known fault class, onset, and duration |
| Safe evaluation of self-healing | Recovery actions are modelled, not executed on live RAN |
| Reproducibility | Fixed seeds and documented commands enable repeat experiments |

### Operator scenario → simulation mapping

| Operational scenario | Typical field cause | Simulation proxy | KPI signature |
|---------------------|---------------------|------------------|---------------|
| Cell outage / no service | Grid failure, generator fault, rectifier trip | **Power fault** — gNB TX collapse | RSRP ↓, throughput → 0, HO failure |
| Severe congestion / slow data | Peak-hour surge, event traffic | **Congestion** — traffic multiplier on cell UEs | PRB > 90%, latency ↑, throughput ↓ |
| Hardware / site failure | RRU/BBU fault, fibre cut, cooling failure | **HW failure** — PHY deactivation | Partial collapse, high packet loss |
| Normal operations | Typical urban mobility and traffic | **None** campaign | Stable RSRP, PRB, handover metrics |

**Reactive monitoring model:** Severe KPI thresholds only (PRB > 90%, RSRP < −110 dBm) with manual remediation times of **240–300 minutes**.

**MAPE-K model:** Earlier detection from ML on KPI windows with automated remediation policies of **35–55 minutes** (Table 4.6 operational assumptions).

**Regulatory reference:** Nigerian Communications Commission (NCC) **99% availability** benchmark appears in Chapter 4 result figures.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    END-TO-END PIPELINE                          │
│                                                                 │
│  NS-3 HetNet Simulation  →  KPI CSV Dataset  →  ML Training     │
│         │                                              │        │
│         └──────────────────────────────────→  MAPE-K Eval       │
│                                                    │            │
│                                            MTTR + Availability  │
└─────────────────────────────────────────────────────────────────┘
```

### MAPE-K control loop

```
Monitor  →  Analyse  →  Plan  →  Execute
   │           │          │         │
 KPI       ML model   Remediation  Recovery
 window    (conf≥0.7)  policy      time log
```

---

## 4. Network Topology (Table 3.1)

| Parameter | Value |
|-----------|-------|
| Macro cells | 7 (hexagonal layout, ISD 500 m) |
| Small cells per macro | 3 |
| **Total cells** | **28** |
| UEs (design target) | 500 |
| UEs (stable default) | 280 |
| Simulation duration | 300 s (120 s common in batch runs) |
| KPI sample interval | 1 s |
| Monte Carlo trials | 50 per fault campaign |
| Fault campaigns | none, power, congestion, hardware (**200 runs**) |

**Dataset scale (300 s, full merge):** ~**1,680,000** raw CSV rows (28 cells × 300 s × 200 files).

After sliding-window extraction and Table 4.1 subsampling, the ML pipeline uses approximately **51,000** labelled windows.

---

## 5. Three Simulation Backends

All backends write the **same CSV schema** so the Python ML pipeline is unchanged.

| Backend | Script | Stack | Speed | Use case |
|---------|--------|-------|-------|----------|
| **KPI generator** | `thesis-fault-sim` | Event-based KPI synthesis | Fastest | Large-scale ML dataset |
| **LTE HetNet** | `thesis-fault-sim-lte` | LENA LTE + EPC | Slow | Real PHY traces + FlowMonitor |
| **5G NR HetNet** | `thesis-fault-sim-nr` | 5G-LENA NR + EPC (n78) | Slowest | True NR validation |

### Recommended usage

1. **KPI generator** — Full 200-trial dataset for ML and MAPE-K evaluation.  
2. **LTE** — RAN trace validation on identical topology and labelling.  
3. **NR** — Extension for 5G-specific claims when runtime permits.

### Commands

```bash
python3 run_all_trials.py                                    # KPI generator
python3 run_all_trials.py --lte --sim-time 120 --num-ues 280 # LTE
python3 run_all_trials.py --nr  --sim-time 120 --num-ues 280 --workers 1  # NR
```

---

## 6. Fault Classes

| Label | Class | Simulation mechanism |
|-------|-------|---------------------|
| 0 | Normal | No active fault window |
| 1 | Power fault | gNB transmit power collapse |
| 2 | Congestion | UDP traffic surge on affected cell |
| 3 | Hardware failure | gNB PHY deactivation |

Faults are injected per trial with random cell selection, stochastic onset time, and bounded duration. Ground truth is stored in the `fault_label` column.

---

## 7. KPI Features (8 per cell per second)

| Feature | Unit | Description |
|---------|------|-------------|
| `rsrp_avg_dbm` | dBm | Reference Signal Received Power |
| `sinr_avg_db` | dB | Signal-to-Interference-plus-Noise Ratio |
| `prb_utilisation` | 0–1 | Physical Resource Block utilisation |
| `dl_throughput_mbps` | Mbps | Downlink throughput |
| `ul_throughput_mbps` | Mbps | Uplink throughput |
| `packet_loss_rate` | 0–1 | Packet loss ratio |
| `handover_success_rate` | 0–1 | Handover success ratio |
| `latency_avg_ms` | ms | Average latency |

---

## 8. Machine Learning Pipeline

### Preprocessing (`preprocess_and_train.py`)

1. Load `kpi_master_dataset.csv`  
2. Subsample toward Table 4.1 class distribution  
3. Extract **10-second sliding windows** (stride 1) per cell  
4. Engineer 48 tabular features (6 statistics × 8 KPIs)  
5. Train / validation / test split: **70 / 15 / 15%** (stratified)  
6. Normalise with `StandardScaler`; PCA at 95% variance on tabular features  
7. Apply **SMOTE on training data only**  
8. Train **Random Forest**, **LSTM**, **SVM**  

### Models

| Model | Role |
|-------|------|
| Random Forest | Strong tabular baseline (200 trees) |
| LSTM | Temporal KPI pattern detection (2×128 units) |
| SVM | Classical baseline (RBF kernel) |

### Outputs

- `models/*.pkl`, `models/lstm_model.h5`  
- `reports/ml_metrics.json` — observed accuracy, macro-F1, AUC-ROC  

---

## 9. MAPE-K Evaluation

`mapek_loop.py` replays the test partition through the autonomic loop:

| Phase | Function |
|-------|----------|
| **Monitor** | KPI window pre-filter (PRB, RSRP, loss, throughput drop) |
| **Analyse** | ML inference with confidence threshold 0.70 |
| **Plan** | Fault-specific remediation policy |
| **Execute** | Modelled recovery duration (minutes, Table 4.6) |

**Reactive baseline** uses severe thresholds only and longer manual remediation times.

### Outputs

- `reports/mapek_summary.json` — MTTR and availability per model  
- `reports/chapter4_results.json` — structured Chapter 4 export  

### Reference results (validation targets)

| Approach | MTTR (min) | Availability (%) |
|----------|------------|------------------|
| Reactive baseline | ~312 | ~94.2 |
| LSTM + MAPE-K | ~102 | ~99.0 |
| RF + MAPE-K | ~119 | ~98.1 |
| SVM + MAPE-K | ~187 | ~96.7 |

*Report observed values from JSON artefacts. Targets above are for comparison only.*

---

## 10. Dataset Tracks Explained

When presenting or documenting results, state which RAN backend produced the data.

### Track A — KPI generator only

- **Strength:** Full Monte Carlo scale (~1.68M rows), fast, reproducible.  
- **Limitation:** Simplified RAN physics; designed fault signatures.  
- **Statement:** *ML and MAPE-K evaluation on NS-3 KPI event generator with operationally calibrated fault labels.*

### Track B — LTE only

- **Strength:** Real LENA LTE/EPC PHY traces and FlowMonitor KPIs.  
- **Limitation:** LTE-A surrogate for 5G; 280 UE stability default.  
- **Statement:** *RAN KPIs from NS-3 LENA/EPC 28-cell HetNet with identical labelling schema.*

### Track C — KPI + LTE (combined)

- **Strength:** Scale from KPI generator; RAN credibility from LTE trials.  
- **Statement:** *Classifier and MAPE-K trends on full KPI dataset; LTE trials cross-validate RAN trace behaviour.*

---

## 11. Figures & Reports

### Chapter 3 (`scripts/generate_figures.py`)

| Figure | Content | Type |
|--------|---------|------|
| 3.1 | HetNet topology | Schematic |
| 3.2 | ML pipeline flowchart | Schematic |
| 3.3 | MAPE-K loop | Schematic |
| 3.4 | Fault injection timeline | Data-driven when raw CSVs exist |
| 3.5 | LSTM architecture | Schematic |

### Chapter 4 (`scripts/generate_chapter4_figures.py`)

| Figure | Content | Source |
|--------|---------|--------|
| 4.2b | Network availability | `mapek_summary.json` |
| 4.4 | Normal-cell fraction over time | `kpi_master_dataset.csv` |
| 4.5 | MTTR comparison | MAPE-K results |

---

## 12. Execution Summary

```bash
# Environment
bash setup.sh
source ~/thesis-sim/activate_thesis.sh

# Simulation
python3 run_all_trials.py --workers 4

# ML + MAPE-K + figures
python3 preprocess_and_train.py
python3 mapek_loop.py --model all
bash run_chapter4_pipeline.sh
```

---

## 13. Limitations

1. Simulation is not a live commercial RAN deployment.  
2. MAPE-K Execute phase models recovery time; it does not call real OSS APIs.  
3. LTE is the primary validated RAN stack; NR extension requires longer runtime.  
4. 280 UEs is the stability-validated default (500 is the design target).  
5. Some Chapter 3 figures are schematic methodology diagrams.  
6. Metric blending toward draft tables is disabled by default (`METRIC_BLEND_WEIGHT = 0`).

---

## 14. Technology Stack

| Component | Technology |
|-----------|------------|
| Network simulator | NS-3 3.38 |
| LTE RAN | ns-3 LENA module |
| 5G NR RAN | 5G-LENA v2.4.y (`contrib/nr`) |
| ML | TensorFlow/Keras 2.15, scikit-learn 1.5 |
| Oversampling | imbalanced-learn (SMOTE) |

---

## 15. Key Files

| File | Role |
|------|------|
| `thesis-fault-sim.cc` | Fast KPI generator |
| `thesis-fault-sim-lte.cc` | LTE/EPC HetNet |
| `thesis-fault-sim-nr.cc` | 5G NR HetNet |
| `run_all_trials.py` | Trial orchestration |
| `preprocess_and_train.py` | ML training |
| `mapek_loop.py` | MAPE-K evaluation |
| `thesis_constants.py` | Approved parameters |
| `output/kpi_master_dataset.csv` | Merged labelled dataset |
| `reports/ml_metrics.json` | ML results |
| `reports/mapek_summary.json` | MTTR and availability |

---

*Document version: aligned with NS-3 3.38 pipeline — KPI, LTE, and NR simulation backends.*
