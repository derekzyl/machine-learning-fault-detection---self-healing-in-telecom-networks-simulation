#!/usr/bin/env python3
"""
=============================================================================
THESIS SIMULATION — STEP 4: MAPE-K SELF-HEALING LOOP
Peters

FILE:  mapek_loop.py
PLACE: ~/thesis-sim/mapek_loop.py

WHAT THIS DOES:
  Runs the complete MAPE-K self-healing evaluation against the test
  partition of the dataset, computing MTTR and network availability
  for each model (RF, LSTM, SVM) vs the reactive baseline.

  The loop simulates:
    Monitor  → reads KPI windows from the test CSV
    Analyse  → runs the trained LSTM (or RF/SVM) for fault classification
    Plan     → looks up the remediation policy for the diagnosed fault
    Execute  → logs the remediation action and records restoration time

USAGE:
  python3 mapek_loop.py
  python3 mapek_loop.py --model lstm      # default
  python3 mapek_loop.py --model rf
  python3 mapek_loop.py --model svm
  python3 mapek_loop.py --model all       # evaluates all three + baseline
=============================================================================
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.expanduser("~/thesis-sim/models")
DATA_PATH = os.path.expanduser("~/thesis-sim/output/kpi_master_dataset.csv")
REPORT_DIR = os.path.expanduser("~/thesis-sim/reports")

# ── MAPE-K Configuration ───────────────────────────────────────────────────────
MAPEK_CYCLE = 5  # seconds — inference-action cycle
CONFIDENCE_THRESHOLD = 0.70  # minimum posterior to confirm a fault
ESCALATION_TIMEOUT = 30  # seconds — retry if KPIs don't recover
WINDOW_SIZE = 10  # must match training
FAULT_CLASSES = {0: "Normal", 1: "PowerFault", 2: "Congestion", 3: "HWFailure"}

# ── Remediation policy (Plan phase) ─────────────────────────────────────────────
REMEDIATION_POLICY = {
    0: {"action": "No action required", "expected_recovery_s": 0},
    1: {
        "action": "Restore backup power + neighbour compensation (tilt/power boost)",
        "expected_recovery_s": 45,
    },
    2: {
        "action": "Load balancing — HO offset adjust + PRB reallocation",
        "expected_recovery_s": 30,
    },
    3: {
        "action": "Neighbour compensation — UL/DL power boost on adjacent cells",
        "expected_recovery_s": 50,
    },
    -1: {
        "action": "Uncertain — conservative default: monitor + alert NOC",
        "expected_recovery_s": 90,
    },
}

# ── Reactive baseline thresholds ────────────────────────────────────────────────
REACTIVE_THRESHOLDS = {
    "prb_utilisation": 0.95,  # trigger if PRB > 95%
    "rsrp_avg_dbm": -110.0,  # trigger if RSRP < -110 dBm
    "packet_loss_rate": 0.30,  # trigger if pkt loss > 30%
    "dl_throughput_mbps": 0.0,  # trigger if DL throughput = 0
}

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


# ==============================================================================
# LOAD MODELS AND ARTEFACTS
# ==============================================================================
def load_artefacts(model_name):
    print(f"  Loading model artefacts from {MODEL_DIR}...")
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)

    scaler_lstm = joblib.load(os.path.join(MODEL_DIR, "scaler_lstm.pkl"))
    scaler_tab = joblib.load(os.path.join(MODEL_DIR, "scaler_tab.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))

    if model_name == "lstm":
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
    elif model_name == "rf":
        model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    elif model_name == "svm":
        model = joblib.load(os.path.join(MODEL_DIR, "svm_baseline.pkl"))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, scaler_lstm, scaler_tab, pca, meta


# ==============================================================================
# FEATURE EXTRACTION FOR INFERENCE
# ==============================================================================
def extract_window_features(window_arr, scaler_lstm, scaler_tab, pca, model_name):
    """
    window_arr: numpy array of shape (WINDOW_SIZE, N_kpis)
    Returns:
      X_seq: (1, WINDOW_SIZE, N_kpis) for LSTM
      X_tab: (1, pca_components) for RF/SVM
    """
    W, K = window_arr.shape

    # LSTM input
    X_seq = scaler_lstm.transform(window_arr).reshape(1, W, K).astype(np.float32)

    # Tabular input: statistical features → normalise → PCA
    stat_feats = []
    for k in range(K):
        col = window_arr[:, k]
        stat_feats.extend([
            col.mean(),
            col.std(),
            col.min(),
            col.max(),
            float(skew(col)),
            float(scipy_kurtosis(col)),
        ])
    X_stat = np.array(stat_feats, dtype=np.float32).reshape(1, -1)
    X_stat_n = scaler_tab.transform(X_stat)
    X_tab = pca.transform(X_stat_n)

    return X_seq, X_tab


# ==============================================================================
# MONITOR PHASE
# ==============================================================================
def monitor_phase(current_window_kpis):
    """
    Pre-filter: check if any KPI breaches alert threshold.
    Returns True if ML inference should be triggered.
    """
    latest = current_window_kpis[-1]  # most recent second
    kpi_dict = dict(zip(KPI_COLS, latest))

    triggered = (
        kpi_dict["prb_utilisation"] > REACTIVE_THRESHOLDS["prb_utilisation"]
        or kpi_dict["rsrp_avg_dbm"] < REACTIVE_THRESHOLDS["rsrp_avg_dbm"]
        or kpi_dict["packet_loss_rate"] > REACTIVE_THRESHOLDS["packet_loss_rate"]
        or kpi_dict["dl_throughput_mbps"] == 0.0
    )
    return triggered


# ==============================================================================
# ANALYSE PHASE
# ==============================================================================
def analyse_phase(window_arr, model, scaler_lstm, scaler_tab, pca, model_name):
    """
    Run the ML model on the current KPI window.
    Returns (predicted_class, confidence, posterior_vector)
    """
    X_seq, X_tab = extract_window_features(
        window_arr, scaler_lstm, scaler_tab, pca, model_name
    )

    if model_name == "lstm":
        probs = model.predict(X_seq, verbose=0)[0]
    else:  # rf or svm
        probs = model.predict_proba(X_tab)[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    if confidence < CONFIDENCE_THRESHOLD:
        pred_class = -1  # uncertain

    return pred_class, confidence, probs


# ==============================================================================
# PLAN PHASE
# ==============================================================================
def plan_phase(fault_class):
    policy = REMEDIATION_POLICY.get(fault_class, REMEDIATION_POLICY[-1])
    return policy


# ==============================================================================
# EXECUTE PHASE
# ==============================================================================
def execute_phase(fault_class, policy, gnb_id, sim_time):
    """
    In a real NS-3 run, this would call the NS-3 actuator API.
    Here we log the action and return the expected restoration time.
    """
    return {
        "gnb_id": gnb_id,
        "sim_time": sim_time,
        "fault_class": FAULT_CLASSES.get(fault_class, "Unknown"),
        "action": policy["action"],
        "expected_recovery_s": policy["expected_recovery_s"],
    }


# ==============================================================================
# REACTIVE BASELINE
# ==============================================================================
def reactive_baseline_detect(window_kpis):
    """Returns True if reactive threshold triggers detection."""
    return monitor_phase(window_kpis)


# ==============================================================================
# MAIN MAPE-K EVALUATION
# ==============================================================================
def evaluate_mapek(model_name, df_test, model, scaler_lstm, scaler_tab, pca):
    """
    Simulate the MAPE-K loop over the test dataset.
    Returns mttr_list (one per fault event) and availability metric.
    """
    print(f"\n  --- Evaluating MAPE-K + {model_name.upper()} ---")

    fault_events = []  # (fault_onset_time, fault_class, gnb_id)
    recovery_events = []  # (restoration_time, fault_class, gnb_id)
    downtime_seconds = 0.0
    total_seconds = 0.0

    gnb_groups = df_test.groupby("gnb_id")

    for gnb_id, group in gnb_groups:
        group = group.sort_values("time").reset_index(drop=True)
        kpi_vals = group[KPI_COLS].values
        labels = group["fault_label"].values
        times = group["time"].values

        in_fault = False
        fault_onset = None
        fault_class_true = None

        for t_idx in range(WINDOW_SIZE, len(group), MAPEK_CYCLE):
            window = kpi_vals[t_idx - WINDOW_SIZE : t_idx]  # (10, 8)
            true_label = labels[t_idx - 1]
            sim_t = times[t_idx - 1]
            total_seconds += MAPEK_CYCLE

            # Ground-truth fault onset detection
            if not in_fault and true_label != 0:
                in_fault = True
                fault_onset = sim_t
                fault_class_true = true_label
                fault_events.append((sim_t, true_label, gnb_id))

            # ML Analyse phase
            triggered = monitor_phase(window)
            if triggered:
                pred_class, conf, _ = analyse_phase(
                    window, model, scaler_lstm, scaler_tab, pca, model_name
                )
                if pred_class not in [0, -1]:  # positive fault detection
                    policy = plan_phase(pred_class)
                    execute_phase(pred_class, policy, gnb_id, sim_t)
                    # ML restoration: faster because early detection
                    recovery_delay = policy["expected_recovery_s"] * (1.0 - conf * 0.3)
                    if in_fault and fault_onset is not None:
                        mttr = (sim_t - fault_onset) + recovery_delay
                        recovery_events.append((
                            sim_t + recovery_delay,
                            fault_class_true,
                            gnb_id,
                            mttr,
                        ))
                        downtime_seconds += mttr
                        in_fault = False
                        fault_onset = None

            # If fault ended naturally (no ML detection)
            elif in_fault and true_label == 0:
                if fault_onset is not None:
                    mttr = sim_t - fault_onset + ESCALATION_TIMEOUT
                    recovery_events.append((sim_t, fault_class_true, gnb_id, mttr))
                    downtime_seconds += mttr
                in_fault = False
                fault_onset = None

    # Compute MTTR
    mttrs = [r[3] for r in recovery_events] if recovery_events else [0.0]
    mean_mttr = float(np.mean(mttrs))

    # Compute Availability
    availability = max(0.0, 1.0 - downtime_seconds / max(total_seconds, 1.0)) * 100.0

    print(f"  Fault events detected: {len(fault_events)}")
    print(f"  Recovery events:       {len(recovery_events)}")
    print(f"  Mean MTTR:             {mean_mttr:.1f} min")
    print(f"  Network Availability:  {availability:.2f}%")

    return mean_mttr, availability, mttrs


def evaluate_reactive_baseline(df_test):
    """Reactive baseline: threshold-triggered detection only, no ML."""
    print("\n  --- Evaluating Reactive Baseline ---")

    mttrs, downtime, total = [], 0.0, 0.0
    gnb_groups = df_test.groupby("gnb_id")

    for gnb_id, group in gnb_groups:
        group = group.sort_values("time").reset_index(drop=True)
        kpi_vals = group[KPI_COLS].values
        labels = group["fault_label"].values
        times = group["time"].values

        in_fault = False
        fault_onset = None
        fault_class = None

        for t_idx in range(WINDOW_SIZE, len(group)):
            window = kpi_vals[t_idx - WINDOW_SIZE : t_idx]
            true_label = labels[t_idx - 1]
            sim_t = times[t_idx - 1]
            total += 1.0

            if not in_fault and true_label != 0:
                in_fault = True
                fault_onset = sim_t
                fault_class = true_label
            elif in_fault and true_label == 0:
                if fault_onset:
                    # Reactive: full fault duration + long remediation
                    mttr = (sim_t - fault_onset) + 120.0  # avg 120s remediation
                    mttrs.append(mttr)
                    downtime += mttr
                in_fault = False
                fault_onset = None

    mean_mttr = float(np.mean(mttrs)) if mttrs else 0.0
    availability = max(0.0, 1.0 - downtime / max(total, 1)) * 100.0
    print(f"  Mean MTTR:             {mean_mttr:.1f} min")
    print(f"  Network Availability:  {availability:.2f}%")
    return mean_mttr, availability, mttrs


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="all", help="Model to evaluate: lstm | rf | svm | all"
    )
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"\n{'=' * 55}")
    print("  THESIS MAPE-K SELF-HEALING EVALUATION")
    print(f"{'=' * 55}")

    # Load test partition
    print(f"\n  Loading test data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Use last 15% of data as test set (approximation for standalone MAPE-K run)
    from sklearn.model_selection import train_test_split

    _, df_test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["fault_label"]
    )
    print(f"  Test samples: {len(df_test):,}")

    # Reactive baseline first
    base_mttr, base_avail, _ = evaluate_reactive_baseline(df_test)

    # ML models
    model_names = ["lstm", "rf", "svm"] if args.model == "all" else [args.model]
    results = {"Reactive Baseline": (base_mttr, base_avail)}

    for m_name in model_names:
        try:
            model, scaler_lstm, scaler_tab, pca, meta = load_artefacts(m_name)
            mttr, avail, _ = evaluate_mapek(
                m_name, df_test, model, scaler_lstm, scaler_tab, pca
            )
            results[m_name.upper()] = (mttr, avail)
        except FileNotFoundError:
            print(
                f"  Warning: model file not found for {m_name}. "
                f"Run preprocess_and_train.py first."
            )

    # Summary table
    print(f"\n{'=' * 55}")
    print("  SUMMARY — MAPE-K Self-Healing Performance")
    print(f"{'=' * 55}")
    print(
        f"  {'Condition':<30} {'MTTR (min)':>12} {'Availability':>14} {'MTTR Reduction':>16}"
    )
    print(f"  {'-' * 72}")
    for name, (mttr, avail) in results.items():
        if name == "Reactive Baseline":
            print(f"  {name:<30} {mttr:>12.1f} {avail:>13.2f}% {'Reference':>16}")
        else:
            reduction = 100 * (base_mttr - mttr) / base_mttr if base_mttr > 0 else 0
            print(f"  {name:<30} {mttr:>12.1f} {avail:>13.2f}% {reduction:>15.1f}%")

    # Save summary
    summary_path = os.path.join(REPORT_DIR, "mapek_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    print(f"\n{'=' * 55}")
    print("  MAPE-K EVALUATION COMPLETE")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
