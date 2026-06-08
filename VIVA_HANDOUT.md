# Viva Handout

> **ML Fault Detection & Self-Healing in Telecom Networks**  
> One document for oral defense, supervisor meetings, and PG client briefing.  
> Choose your **dataset track** (KPI / LTE / Both) — wording below is marked for each.

**Print tip:** Pages 1–2 = core handout. Pages 3–4 = Q&A. Page 5 = slide outline.

---

## 1. Opening Statement (30 seconds)

Use this verbatim or adapt:

> *This thesis presents a **reproducible simulation framework** for proactive RAN fault detection and MAPE-K-guided self-healing in a **28-cell urban HetNet**. We inject operationally realistic faults — power outage, congestion, and hardware failure — modelled after scenarios common in Nigerian mobile networks such as **MTN and Airtel** field operations. Labelled KPI time-series from NS-3 feed three ML classifiers (RF, LSTM, SVM). A MAPE-K control loop is compared against a **reactive OSS-style baseline** using Nigerian remediation time assumptions. Results are reported from **observed simulation outputs**, not fabricated targets.*

**Do not say:** “We tested on MTN/Airtel live networks.”

---

## 2. Research Question & Hypothesis

| Item | Content |
|------|---------|
| **Research question** | Can ML-based proactive fault detection reduce MTTR and improve availability vs. threshold-only reactive monitoring? |
| **Null hypothesis** | ML + MAPE-K offers no meaningful MTTR or availability gain over reactive monitoring. |
| **Alternative hypothesis** | Earlier KPI-pattern detection enables faster remediation → lower MTTR, higher availability. |
| **Evaluation** | 4-class fault detection (Normal, Power, Congestion, HW) + per-trial MTTR and availability (Eq. 3.8–3.9). |

---

## 3. Real-World Context (MTN / Airtel) — Say This in Defense

This is a **simulation imitating real MNO operations**, not a live network trial.

| Real operator scenario | Field cause (Nigeria) | What we simulate | KPI effect |
|------------------------|----------------------|------------------|------------|
| **Cell outage** | Grid failure, genset fault, rectifier trip, vandalism | Power fault — TX collapse | RSRP ↓, throughput → 0, HO fails |
| **Slow / unusable data** | Peak hour, event surge, backhaul stress | Congestion — traffic surge | PRB > 90%, latency ↑, throughput ↓ |
| **Site / HW down** | RRU/BBU fault, fibre cut, cooling failure | HW fault — PHY off | Partial collapse, high loss |
| **Normal ops** | Typical urban mobility | No fault window | Stable RSRP, PRB, HO |

**Reactive baseline** = late OSS alarms (PRB > 90%, RSRP < −110 dBm) → manual dispatch **240–300 min** (Table 4.6).  
**MAPE-K + ML** = early window detection → automated remediation policy **35–55 min** (modelled).

**NCC benchmark:** 99% availability reference line in Ch. 4 figures.

---

## 4. System Architecture (One Diagram to Draw on Board)

```
NS-3 HetNet (28 cells, faults)  →  kpi_master_dataset.csv
         ↓
preprocess_and_train.py  →  RF / LSTM / SVM  (+ SMOTE, PCA, windows)
         ↓
mapek_loop.py  →  MTTR + Availability  vs  Reactive Baseline
         ↓
reports/*.json + Chapter 3/4 figures
```

**MAPE-K loop:** Monitor (KPI window) → Analyse (ML, conf ≥ 0.70) → Plan (fault policy) → Execute (recovery time).

---

## 5. Three Dataset Tracks — Choose Yours

Your defense must **state which RAN backend produced the results** you present.

### Track A — KPI Generator (`thesis-fault-sim`) ONLY

| Aspect | What to say |
|--------|-------------|
| **Purpose** | Large-scale Monte Carlo dataset for ML pipeline validation and class-balance studies |
| **Strength** | Fast, reproducible, 200 trials → ~1.68M rows; identical CSV schema |
| **Limitation** | Simplified RAN physics; KPIs follow designed fault signatures |
| **Thesis wording** | *“Primary ML and MAPE-K evaluation used the NS-3 KPI event generator with operationally calibrated fault labels. This enables statistically robust classifier comparison under controlled ground truth.”* |
| **Command used** | `python3 run_all_trials.py --workers 4` |
| **Evidence** | `output/kpi_master_dataset.csv`, `reports/ml_metrics.json`, `reports/mapek_summary.json` |

**When Track A alone is acceptable:** Examiners focus on ML methodology, MAPE-K framework, and comparative MTTR logic — not vendor PHY fidelity.

---

### Track B — LTE (`thesis-fault-sim-lte`) ONLY

| Aspect | What to say |
|--------|-------------|
| **Purpose** | RAN validation with **real LENA LTE/EPC** — PHY traces + FlowMonitor |
| **Strength** | Genuine attach, handover, UDP traffic, per-cell KPIs from simulation stack |
| **Limitation** | LTE-A not NR; 280 UEs default (500 caused HARQ instability); slow runtime |
| **Thesis wording** | *“RAN KPIs were generated from NS-3 LENA/EPC simulating a 28-cell LTE-A HetNet as a **3GPP-aligned surrogate** for dense small-cell deployments. Fault injection proxies field power, congestion, and hardware outages.”* |
| **Command used** | `python3 run_all_trials.py --lte --sim-time 120 --num-ues 280 --workers 2` |
| **Evidence** | Raw CSVs with non-default KPI variation; build log for `thesis-fault-sim-lte` |

**When Track B alone is acceptable:** Examiners demand “real simulation stack” evidence; you accept LTE-as-surrogate for 5G HetNet claims.

---

### Track C — BOTH (Recommended for Strongest Defense)

| Aspect | What to say |
|--------|-------------|
| **Purpose** | KPI generator for **scale**; LTE for **RAN trace validation** |
| **Strength** | Shows ML pipeline is backend-agnostic; LTE confirms KPI patterns are physically plausible |
| **Limitation** | Must report which results come from which backend — do not mix without disclosure |
| **Thesis wording** | *“ML classifier ranking and MAPE-K comparative trends were established on the full Monte Carlo KPI dataset. Key RAN metrics were cross-validated on the LENA LTE/EPC implementation with identical topology and labelling schema.”* |
| **Structure** | Ch. 4 tables from KPI data; Ch. 3 §3.3 or appendix cites LTE validation trials |
| **NR extension** | If `thesis-fault-sim-nr` trials exist, add: *“5G-LENA NR implementation provided for NR-specific extension.”* |

**When Track C is best:** Full thesis defense where one examiner asks “Is this real simulation?” and another asks “Is the ML sound?”

---

### Track comparison (quick reference)

| | KPI generator | LTE | NR (optional) |
|--|---------------|-----|---------------|
| Speed | Minutes | Hours/days | Days/weeks |
| RAN fidelity | Low | High (LTE) | Highest (5G) |
| Rows (300 s, 200 runs) | ~1.68M | ~1.68M | ~1.68M |
| Best for | ML + MAPE-K scale | RAN credibility | True 5G claim |
| Script | `thesis-fault-sim` | `thesis-fault-sim-lte` | `thesis-fault-sim-nr` |

---

## 6. Key Numbers to Know (Table 3.1)

| Parameter | Value |
|-----------|-------|
| Macro cells | 7 (hexagonal, ISD 500 m) |
| Small cells | 21 (3 per macro) |
| **Total cells** | **28** |
| UEs (target / stable) | 500 / 280 |
| Sim time | 300 s (120 s in batch) |
| Trials | 50 × 4 faults = 200 runs |
| KPI interval | 1 s |
| Window size | 10 s, stride 1 |
| ML split | 70 / 15 / 15 % |
| Classes | 0 Normal, 1 Power, 2 Congestion, 3 HW |

**Raw CSV rows ≠ ML windows.** ~1.68M raw rows → ~51k windows after Table 4.1 subsampling. Both are correct.

---

## 7. Results You Present (Observed vs Targets)

**Default pipeline:** `METRIC_BLEND_WEIGHT = 0` — report **observed** values only.

| Artefact | File | Use in thesis |
|----------|------|---------------|
| ML accuracy, F1, AUC | `reports/ml_metrics.json` → `observed_*` fields | Tables §4.3–4.5 |
| MTTR, availability | `reports/mapek_summary.json` | Tables §4.6–4.7 |
| Full Ch. 4 bundle | `reports/chapter4_results.json` | Appendix |
| Approved draft targets | `thesis_constants.py` → `THESIS_*` | Compare only; not forced |

**Headline targets (if observed is close):**

| Model | Accuracy | Macro F1 | MTTR (min) | Availability |
|-------|----------|----------|------------|--------------|
| LSTM | ~0.96 | ~0.96 | ~102 | ~99.0% |
| RF | ~0.95 | ~0.94 | ~119 | ~98.1% |
| SVM | ~0.81 | ~0.80 | ~187 | ~96.7% |
| Reactive | — | — | ~312 | ~94.2% |

*If observed differs, report observed and discuss gap — that is academically stronger.*

---

## 8. Top 10 Examiner Questions & Answers

**Q1. Did you use MTN or Airtel live data?**  
No. Operationally realistic **simulation** with fault types and remediation times grounded in Nigerian MNO practice.

**Q2. Why LTE if the title mentions 5G?**  
LTE-A LENA is the mature NS-3 RAN stack for HetNet KPI studies. NR (`thesis-fault-sim-nr`) extends the same schema to 5G-LENA. Dense small-cell fault **KPI semantics** transfer across LTE/NR.

**Q3. How do you know fault labels are correct?**  
Labels come from **simulation ground truth** (`fault_start_s`, `fault_end_s`, `fault_label`) — not from ML. ML is trained on these labels.

**Q4. Is SMOTE cheating?**  
SMOTE applies **only on the training split** after windowing. Test set is untouched. Standard for imbalanced fault detection.

**Q5. What is novel?**  
The **integrated pipeline**: HetNet fault sim → labelled KPI corpus → MAPE-K closed loop with operational MTTR model vs reactive baseline.

**Q6. 500 UEs or 280?**  
Table 3.1 target is 500. 280 is the NS-3 stability-validated count on this hardware — stated as platform limitation.

**Q7. Are Chapter 4 numbers fabricated?**  
No. Pipeline defaults to observed-only. `--blend-thesis-metrics` exists for drafts only.

**Q8. What does MAPE-K Execute actually do?**  
It **models** remediation duration from Table 4.6 policies — not live OSS API calls. Valid for comparative autonomic networking research.

**Q9. Why LSTM beats RF?**  
Temporal KPI degradation patterns (pre-fault ramps) favour sequence models; RF is strong but memoryless on tabular aggregates.

**Q10. Can others reproduce this?**  
Yes: `setup.sh`, fixed seeds, `run_all_trials.py`, committed CSV schema, `check_environment.py`.

---

## 9. Red Lines — Never Claim

- Live deployment on MTN, Airtel, or any MNO network  
- Field drive-test or OSS-exported proprietary KPIs  
- All results from NR unless NR batch actually completed  
- Exact Table 4.x match without showing `observed_*` JSON  
- Schematic Ch. 3 figures (3.1, 3.2, 3.3, 3.5) as measured data plots  

---

## 10. Evidence Checklist Before Viva

Bring or have open:

- [ ] `output/kpi_master_dataset.csv` — row count + class distribution  
- [ ] One raw trial CSV (e.g. `kpi_trial0_power.csv`) — show fault window rows  
- [ ] `reports/ml_metrics.json` — observed metrics  
- [ ] `reports/mapek_summary.json` — MTTR + availability  
- [ ] `fig3_4_timeline.png` — fault injection (data-driven if raw exists)  
- [ ] `fig4_5_mttr.png` or `fig4_2b_availability.png`  
- [ ] Statement of which track: **KPI / LTE / Both**  
- [ ] `thesis_constants.py` — show `METRIC_BLEND_WEIGHT = 0`  

---

## 11. Supervisor Meeting — Slide Outline (10 slides)

Use for pre-viva supervisor sign-off or internal review.

| Slide | Title | Bullet content |
|-------|-------|----------------|
| **1** | Title & candidate | Thesis title, name, department, “Simulation-based HetNet fault management” |
| **2** | Problem | Reactive OSS monitoring detects late; high MTTR in Nigerian ops; NCC 99% expectation |
| **3** | Research question | ML proactive detection + MAPE-K vs reactive baseline; MTTR + availability |
| **4** | Real-world mapping | MTN/Airtel scenario table (outage / congestion / HW) → sim proxy (1 slide, §3 above) |
| **5** | Methodology | 28-cell HetNet, 50 trials, 4 faults, 8 KPIs, 10 s windows; **state dataset track** |
| **6** | RAN backend | KPI / LTE / Both diagram; why not live network |
| **7** | ML results | RF vs LSTM vs SVM — observed accuracy, F1, AUC; LSTM best |
| **8** | MAPE-K results | MTTR + availability bar chart; reactive vs LSTM; NCC 99% line |
| **9** | Limitations | Simulation not live; UE scaling; MAPE-K modelled execute; LTE surrogate |
| **10** | Contribution & future | Reproducible framework; NR extension; live OSS integration as future work |

**Supervisor ask at end:**  
*“Which dataset track are we defending — KPI, LTE, or both? Are all Ch. 4 numbers from `observed_*` JSON?”*

---

## 12. Track-Specific Closing Statements

### If defending KPI data only

> *The contribution is a validated ML and MAPE-K evaluation framework over a large labelled KPI corpus with operationally grounded fault semantics. RAN PHY validation via LENA LTE is identified as future corroboration.*

### If defending LTE data only

> *RAN KPIs originate from NS-3 LENA/EPC with real PHY traces. ML and MAPE-K results demonstrate proactive detection benefit on physically simulated HetNet traffic.*

### If defending both (recommended)

> *Monte Carlo KPI-scale experiments establish classifier and MAPE-K comparative conclusions; LENA LTE/EPC trials confirm that fault signatures and KPI degradation patterns are consistent with a real RAN simulation stack under the same topology and labelling scheme.*

---

## 13. One-Page Cheat Sheet (Cut Here for Exam Day)

```
THESIS IN ONE PAGE
──────────────────
WHAT:  NS-3 28-cell HetNet → labelled KPIs → RF/LSTM/SVM → MAPE-K vs reactive
WHY:   Proactive fault detection reduces MTTR vs late OSS thresholds (MTN/Airtel-style ops)
NOT:   Live MNO network; fabricated metrics (blend weight = 0)

TOPOLOGY:  7 macro + 21 small = 28 cells | 280–500 UEs | 50 trials × 4 faults
FAULTS:    Power (outage) | Congestion (peak load) | HW (RRU/transport)
ML:        10 s windows, 8 KPIs, SMOTE train-only, LSTM best temporal model
MAPE-K:    Monitor → Analyse (ML≥0.7) → Plan → Execute (Table 4.6 min)
BASELINE:  Severe thresholds only → 240–300 min reactive MTTR

DATASET TRACK (circle one):  [ KPI generator ]  [ LTE ]  [ Both ]
EVIDENCE:  kpi_master_dataset.csv | ml_metrics.json | mapek_summary.json

KEY LINE:  "Operationally realistic simulation with simulation-derived ground truth."
```

---

## Related files

| Doc | Purpose |
|-----|---------|
| [README.md](README.md) | Full project + chapter map + integrity guide |
| [INSTALL.md](INSTALL.md) | Install, execution, MTN/Airtel fault table |
| [thesis_constants.py](thesis_constants.py) | Approved parameters & targets |

---

*Prepared for MSc viva and supervisor review. Update the **dataset track** checkbox before each meeting.*
