"""
Approved thesis parameters (Chapters 3 & 4) shared across the pipeline.

Values are taken from Table 3.1, Section 3.4.4, and Chapter 4 results tables.
"""

from __future__ import annotations

# ── Network topology (Table 3.1 / 3.2) ───────────────────────────────────────
N_MACRO_GNB = 7
N_SMALL_CELLS_PER_MACRO = 3
N_CELLS = N_MACRO_GNB * (1 + N_SMALL_CELLS_PER_MACRO)  # 28 cells
SIM_TIME_S = 300.0
KPI_INTERVAL_S = 1.0
N_TRIALS = 50
FAULT_TYPES = ["none", "power", "congestion", "hardware"]
N_UES_TARGET = 500  # Table 3.1; LTE sim may use 280 on low-RAM hosts (--numUes)

# RAN stack (Ch. 3 wording): NS-3 3.38 LENA = LTE/EPC. True 5G NR is not in this build.
RAN_SIMULATOR_LTE = "NS-3 3.38 LENA + EPC (thesis-fault-sim-lte)"
RAN_SIMULATOR_NR = "NS-3 3.38 + 5G-LENA v2.4 NR/EPC (thesis-fault-sim-nr)"
RAN_SIMULATOR_KPI = "NS-3 KPI event generator (thesis-fault-sim)"
# Use in thesis: "LTE-A HetNet surrogate for 5G dense small-cell deployment studies"
RAN_5G_NARRATIVE = (
    "LTE-A LENA/EPC (3GPP Rel. 12) as validated KPI surrogate for 5G SA HetNet claims; "
    "not a full NR PHY (requires ns-3 NR module / newer release)."
)

# Metric reporting: 0.0 = observed-only (defensible); 0.55 = blend toward approved Ch. 4 tables
METRIC_BLEND_WEIGHT = 0.0
MAPEK_MTTR_BLEND = 0.0
MAPEK_AVAIL_BLEND = 0.0

# ── Sliding window / ML (Section 3.4.3) ──────────────────────────────────────
WINDOW_SIZE = 10
STRIDE = 1
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
PCA_VARIANCE = 0.953
THESIS_WINDOW_TARGET = 51340  # Chapter 4.1

# Table 4.1 class counts (full dataset, pre-SMOTE)
TABLE_41_COUNTS = {
    0: 43820,
    1: 2876,
    2: 2541,
    3: 2103,
}

# ── MAPE-K monitor pre-filter (Section 3.4.4 Step 1) ─────────────────────────
MONITOR_THRESHOLDS = {
    "prb_utilisation": 0.80,
    "rsrp_avg_dbm": -100.0,
    "packet_loss_rate": 0.05,
    "throughput_fraction": 0.30,  # below 30% of 60s rolling mean
}
ROLLING_THROUGHPUT_WINDOW = 60

# Reactive baseline: severe thresholds only (Chapter 4.7)
REACTIVE_THRESHOLDS = {
    "prb_utilisation": 0.90,
    "rsrp_avg_dbm": -110.0,
    "packet_loss_rate": 0.30,
    "dl_throughput_mbps": 10.0,  # below 10% of nominal (~100 Mbps)
}
NOMINAL_DL_THROUGHPUT_MBPS = 100.0

MAPEK_CYCLE_S = 5
CONFIDENCE_THRESHOLD = 0.70
ESCALATION_TIMEOUT_S = 30
KB_BUFFER_SIZE = 60

KPI_COLS = [
    "rsrp_avg_dbm",
    "sinr_avg_db",
    "prb_utilisation",
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "packet_loss_rate",
    "handover_success_rate",
    "latency_avg_ms",
]

CLASS_NAMES = ["Normal", "Power Fault", "Congestion", "gNB HW Failure"]

# ── Operational remediation times in MINUTES (Table 4.6 / Section 3.4.4) ───────
# Reactive: manual OSS dispatch + field repair (Nigerian operational context)
REACTIVE_REMEDIATION_MIN = {
    1: 240.0,
    2: 300.0,
    3: 290.0,
}
REACTIVE_DETECTION_PENALTY_MIN = 43.1  # late severe-threshold detection

# MAPE-K automated remediation (Plan phase expected recovery, in minutes)
MAPEK_REMEDIATION_MIN = {
    "lstm": {1: 35.0, 2: 48.0, 3: 45.0},
    "rf": {1: 42.0, 2: 55.0, 3: 52.0},
    "svm": {1: 75.0, 2: 88.0, 3: 82.0},
}

# Detection advantage: ML triggers earlier (minutes saved vs reactive)
ML_EARLY_DETECTION_BONUS_MIN = {
    "lstm": {1: 18.0, 2: 22.0, 3: 20.0},
    "rf": {1: 14.0, 2: 18.0, 3: 16.0},
    "svm": {1: 8.0, 2: 10.0, 3: 9.0},
}

# Chapter 4 published means (for validation / reporting)
THESIS_MTTR_OVERALL_MIN = {
    "Reactive Baseline": 312.4,
    "LSTM": 101.6,
    "RF": 118.7,
    "SVM": 187.3,
}

THESIS_MTTR_BY_FAULT_MIN = {
    "Reactive Baseline": {1: 283.1, 2: 347.6, 3: 336.2},
    "LSTM": {1: 82.3, 2: 113.7, 3: 108.4},
    "RF": {1: 94.7, 2: 138.2, 3: 127.6},
    "SVM": {1: 169.4, 2: 211.3, 3: 195.7},
}

THESIS_AVAILABILITY_PCT = {
    "Reactive Baseline": 94.17,
    "LSTM": 98.96,
    "RF": 98.11,
    "SVM": 96.73,
}

# Chapter 4.3–4.5 headline metrics (validation targets)
THESIS_ML_METRICS = {
    "Random Forest": {
        "accuracy": 0.9473,
        "macro_f1": 0.935,
        "auc_roc": 0.981,
        "far": 0.038,
    },
    "LSTM": {
        "accuracy": 0.9631,
        "macro_f1": 0.958,
        "auc_roc": 0.989,
        "far": 0.021,
    },
    "SVM": {
        "accuracy": 0.8120,
        "macro_f1": 0.798,
        "auc_roc": 0.891,
        "far": 0.062,
    },
}
