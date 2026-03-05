#!/usr/bin/env python3
"""
=============================================================================
THESIS SIMULATION — STEP 3: PREPROCESSING & ML TRAINING PIPELINE
Peters PG/2415890

FILE:  preprocess_and_train.py
PLACE: ~/thesis-sim/preprocess_and_train.py

WHAT THIS DOES:
  Stage 1  — Load and inspect the master KPI dataset
  Stage 2  — Preprocess (clean, normalise, sliding window)
  Stage 3  — Feature engineering (statistical features + PCA)
  Stage 4  — SMOTE oversampling on training partition only
  Stage 5  — Train Random Forest
  Stage 6  — Train LSTM
  Stage 7  — Train SVM baseline
  Stage 8  — Evaluate all models on test partition
  Stage 9  — Save trained models + performance reports

USAGE:
  python3 preprocess_and_train.py
  python3 preprocess_and_train.py --data ~/thesis-sim/output/kpi_master_dataset.csv
  python3 preprocess_and_train.py --skip_svm   # skip slow SVM if needed
=============================================================================
"""

import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_DATA = os.path.expanduser("~/thesis-sim/output/kpi_master_dataset.csv")
MODEL_DIR    = os.path.expanduser("~/thesis-sim/models")
REPORT_DIR   = os.path.expanduser("~/thesis-sim/reports")

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW_SIZE  = 10       # seconds
STRIDE       = 1        # seconds
RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
SMOTE_RATIO  = {1: 0, 2: 0, 3: 0}  # filled dynamically
N_CLASSES    = 4
CLASS_NAMES  = ["Normal", "Power Fault", "Congestion", "gNB HW Failure"]
KPI_COLS     = ["rsrp_avg_dbm", "sinr_avg_db", "prb_utilisation",
                "dl_throughput_mbps", "ul_throughput_mbps",
                "packet_loss_rate", "handover_success_rate", "latency_avg_ms"]

# ==============================================================================
# STAGE 1: LOAD DATA
# ==============================================================================
def load_data(path):
    print(f"\n{'='*55}")
    print("STAGE 1 — Loading dataset")
    print(f"{'='*55}")
    print(f"  Path: {path}")

    if not os.path.exists(path):
        print(f"\n  ERROR: File not found: {path}")
        print("  Run run_all_trials.py first to generate the dataset.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    print(f"\n  Class distribution (raw):")
    label_map = dict(enumerate(CLASS_NAMES))
    vc = df['fault_label'].value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {label_map.get(k,k):20s}: {v:6,} ({100*v/len(df):.1f}%)")
    return df

# ==============================================================================
# STAGE 2: SLIDING WINDOW EXTRACTION
# ==============================================================================
def make_windows(df):
    """
    For each gNB independently, extract sliding windows of WINDOW_SIZE seconds.
    Returns:
      X_seq  — shape (N_windows, WINDOW_SIZE, N_kpis)  for LSTM
      X_tab  — shape (N_windows, N_kpis * WINDOW_SIZE)  raw flattened (before stat features)
      y      — shape (N_windows,)  label at last time step of window
      meta   — DataFrame with trial/gnb/time for each window
    """
    print(f"\n{'='*55}")
    print("STAGE 2 — Sliding window extraction")
    print(f"  Window size: {WINDOW_SIZE}s  |  Stride: {STRIDE}s")
    print(f"{'='*55}")

    X_seq_list, y_list, meta_list = [], [], []

    gnb_groups = df.groupby(['trial', 'gnb_id'])
    for (trial, gnb_id), group in gnb_groups:
        group = group.sort_values('time').reset_index(drop=True)
        kpi_vals = group[KPI_COLS].values
        labels   = group['fault_label'].values
        times    = group['time'].values

        for start in range(0, len(group) - WINDOW_SIZE, STRIDE):
            end = start + WINDOW_SIZE
            window_kpi = kpi_vals[start:end]  # (WINDOW_SIZE, N_kpis)
            label      = labels[end - 1]       # label at last step
            t_end      = times[end - 1]

            X_seq_list.append(window_kpi)
            y_list.append(label)
            meta_list.append({'trial': trial, 'gnb_id': gnb_id, 'time_end': t_end})

    X_seq = np.array(X_seq_list, dtype=np.float32)  # (N, 10, 8)
    y     = np.array(y_list, dtype=np.int32)
    meta  = pd.DataFrame(meta_list)

    print(f"  Total windows: {len(y):,}")
    print(f"  X_seq shape: {X_seq.shape}")
    vc = pd.Series(y).value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {CLASS_NAMES[k]:20s}: {v:6,} ({100*v/len(y):.1f}%)")

    return X_seq, y, meta

# ==============================================================================
# STAGE 3: STATISTICAL FEATURE ENGINEERING FOR TABULAR MODELS
# ==============================================================================
def extract_stat_features(X_seq):
    """
    From (N, WINDOW_SIZE, N_kpis) compute 6 statistics per KPI:
    mean, std, min, max, skewness, kurtosis → (N, N_kpis * 6) = (N, 48)
    """
    from scipy.stats import skew, kurtosis

    N, W, K = X_seq.shape
    feats = np.zeros((N, K * 6), dtype=np.float32)
    for k in range(K):
        col = X_seq[:, :, k]  # (N, W)
        feats[:, k*6+0] = col.mean(axis=1)
        feats[:, k*6+1] = col.std(axis=1)
        feats[:, k*6+2] = col.min(axis=1)
        feats[:, k*6+3] = col.max(axis=1)
        feats[:, k*6+4] = skew(col, axis=1)
        feats[:, k*6+5] = kurtosis(col, axis=1)

    # Replace NaN/Inf
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats  # (N, 48)

# ==============================================================================
# STAGE 4: TRAIN/VAL/TEST SPLIT + NORMALISATION + SMOTE + PCA
# ==============================================================================
def prepare_splits(X_seq, y):
    print(f"\n{'='*55}")
    print("STAGE 4 — Data splitting, normalisation, SMOTE, PCA")
    print(f"{'='*55}")

    # ── Stratified split ──────────────────────────────────────────────────
    idx = np.arange(len(y))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval, y_trainval,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_trainval)

    X_seq_train = X_seq[idx_train]
    X_seq_val   = X_seq[idx_val]
    X_seq_test  = X_seq[idx_test]

    print(f"  Train: {len(y_train):,}  |  Val: {len(y_val):,}  |  Test: {len(y_test):,}")

    # ── Statistical features for RF/SVM ──────────────────────────────────
    print("  Extracting statistical features for RF/SVM...")
    X_stat_train = extract_stat_features(X_seq_train)
    X_stat_val   = extract_stat_features(X_seq_val)
    X_stat_test  = extract_stat_features(X_seq_test)

    # ── Normalise sequence data (LSTM input) ──────────────────────────────
    # Compute mean/std per KPI across training set
    N_train, W, K = X_seq_train.shape
    scaler_lstm = StandardScaler()
    X_flat_train = X_seq_train.reshape(-1, K)
    scaler_lstm.fit(X_flat_train)

    def norm_seq(X_s):
        N, W2, K2 = X_s.shape
        return scaler_lstm.transform(X_s.reshape(-1, K2)).reshape(N, W2, K2)

    X_seq_train_n = norm_seq(X_seq_train)
    X_seq_val_n   = norm_seq(X_seq_val)
    X_seq_test_n  = norm_seq(X_seq_test)

    # ── Normalise tabular data (RF/SVM input) ─────────────────────────────
    scaler_tab = StandardScaler()
    X_stat_train_n = scaler_tab.fit_transform(X_stat_train)
    X_stat_val_n   = scaler_tab.transform(X_stat_val)
    X_stat_test_n  = scaler_tab.transform(X_stat_test)

    # ── PCA on tabular training data ──────────────────────────────────────
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_tab_train_pca = pca.fit_transform(X_stat_train_n)
    X_tab_val_pca   = pca.transform(X_stat_val_n)
    X_tab_test_pca  = pca.transform(X_stat_test_n)
    print(f"  PCA: {X_stat_train_n.shape[1]} → {X_tab_train_pca.shape[1]} components"
          f" (explained var: {pca.explained_variance_ratio_.sum():.3f})")

    # ── SMOTE on training partition only ─────────────────────────────────
    print("  Applying SMOTE to training partition...")
    target_n_normal = int(y_train.sum() * 2)  # 2:1 normal-to-fault ratio
    n_normal = int((y_train == 0).sum())
    smote_strategy = {}
    for c in [1, 2, 3]:
        n_c = int((y_train == c).sum())
        # Oversample each fault class to ~33% of normal count
        target = max(n_c, n_normal // 3)
        smote_strategy[c] = target

    sm = SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE, k_neighbors=5)
    X_tab_train_sm, y_train_sm = sm.fit_resample(X_tab_train_pca, y_train)
    print(f"  Post-SMOTE training size: {len(y_train_sm):,}")
    vc = pd.Series(y_train_sm).value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {CLASS_NAMES[k]:20s}: {v:6,}")

    # SMOTE for LSTM: apply on flattened sequence, reshape back
    X_seq_flat_train = X_seq_train_n.reshape(len(y_train), -1)
    X_seq_sm_flat, y_seq_sm = sm.fit_resample(X_seq_flat_train, y_train)
    X_seq_train_sm = X_seq_sm_flat.reshape(-1, W, K)

    return {
        # LSTM
        'X_seq_train': X_seq_train_sm, 'y_seq_train': y_seq_sm,
        'X_seq_val':   X_seq_val_n,    'y_val':        y_val,
        'X_seq_test':  X_seq_test_n,   'y_test':       y_test,
        # RF / SVM
        'X_tab_train': X_tab_train_sm, 'y_tab_train': y_train_sm,
        'X_tab_val':   X_tab_val_pca,
        'X_tab_test':  X_tab_test_pca,
        # Artefacts (save these for inference)
        'scaler_lstm': scaler_lstm,
        'scaler_tab':  scaler_tab,
        'pca':         pca,
        'W': W, 'K': K,
    }

# ==============================================================================
# STAGE 5: TRAIN RANDOM FOREST
# ==============================================================================
def train_rf(splits):
    print(f"\n{'='*55}")
    print("STAGE 5 — Training Random Forest")
    print(f"{'='*55}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    t0 = time.time()
    rf.fit(splits['X_tab_train'], splits['y_tab_train'])
    print(f"  Training time: {time.time()-t0:.1f}s")
    oob = getattr(rf, 'oob_score_', None) or getattr(rf, 'oob_score', None)
    if oob: print(f'  OOB Score: {oob:.4f}')

    y_pred = rf.predict(splits['X_tab_test'])
    y_prob = rf.predict_proba(splits['X_tab_test'])
    _print_metrics("Random Forest", splits['y_test'], y_pred, y_prob)

    return rf, y_pred, y_prob

# ==============================================================================
# STAGE 6: TRAIN LSTM
# ==============================================================================
def train_lstm(splits):
    print(f"\n{'='*55}")
    print("STAGE 6 — Training LSTM Network")
    print(f"{'='*55}")

    W = splits['W']
    K = splits['K']

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(W, K)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(N_CLASSES, activation='softmax')
    ], name="FaultDetection_LSTM")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]

    t0 = time.time()
    history = model.fit(
        splits['X_seq_train'], splits['y_seq_train'],
        validation_data=(splits['X_seq_val'], splits['y_val']),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    print(f"  Training time: {time.time()-t0:.1f}s")
    print(f"  Stopped at epoch: {len(history.history['loss'])}")

    y_prob = model.predict(splits['X_seq_test'], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    _print_metrics("LSTM", splits['y_test'], y_pred, y_prob)

    # Plot training curves
    _plot_training_history(history)

    return model, y_pred, y_prob

# ==============================================================================
# STAGE 7: TRAIN SVM BASELINE
# ==============================================================================
def train_svm(splits):
    print(f"\n{'='*55}")
    print("STAGE 7 — Training SVM Baseline")
    print(f"  (using 20,000-sample subsample for speed)")
    print(f"{'='*55}")

    # Stratified subsample
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=RANDOM_STATE)
    # Subsample to 20,000 from training set if larger
    X_tr, y_tr = splits['X_tab_train'], splits['y_tab_train']
    if len(y_tr) > 20000:
        sss2 = StratifiedShuffleSplit(n_splits=1,
                                      test_size=20000/len(y_tr),
                                      random_state=RANDOM_STATE)
        _, idx = next(sss2.split(X_tr, y_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"  Subsampled training set: {len(y_tr):,}")

    svm = SVC(
        kernel='rbf', C=10, gamma=0.01,
        class_weight='balanced',
        decision_function_shape='ovr',
        probability=True,
        random_state=RANDOM_STATE
    )
    t0 = time.time()
    svm.fit(X_tr, y_tr)
    print(f"  Training time: {time.time()-t0:.1f}s")

    y_pred = svm.predict(splits['X_tab_test'])
    y_prob = svm.predict_proba(splits['X_tab_test'])
    _print_metrics("SVM", splits['y_test'], y_pred, y_prob)

    return svm, y_pred, y_prob

# ==============================================================================
# HELPERS
# ==============================================================================
def _print_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    print(f"\n  ── {name} Results ──")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print(f"  Confusion matrix (row=true, col=pred):")
    for r in cm_norm:
        print("    " + "  ".join([f"{v:.3f}" for v in r]))

    return acc, macro_f1, auc

def _plot_training_history(history):
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('LSTM Training & Validation Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'],     label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('LSTM Training & Validation Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'lstm_training_history.png'), dpi=150)
    plt.close()
    print(f"  Training curve saved to {REPORT_DIR}/lstm_training_history.png")

def save_models(rf, lstm_model, svm, splits):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print("STAGE 9 — Saving models and artefacts")
    print(f"  Directory: {MODEL_DIR}")
    print(f"{'='*55}")

    joblib.dump(rf,  os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_baseline.pkl"))
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    joblib.dump(splits['scaler_lstm'], os.path.join(MODEL_DIR, "scaler_lstm.pkl"))
    joblib.dump(splits['scaler_tab'],  os.path.join(MODEL_DIR, "scaler_tab.pkl"))
    joblib.dump(splits['pca'],         os.path.join(MODEL_DIR, "pca.pkl"))

    # Save metadata
    meta = {
        "window_size":  WINDOW_SIZE,
        "stride":       STRIDE,
        "kpi_columns":  KPI_COLS,
        "class_names":  CLASS_NAMES,
        "n_classes":    N_CLASSES,
        "pca_components": int(splits['pca'].n_components_),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("  Saved: random_forest.pkl, svm_baseline.pkl, lstm_model.h5")
    print("  Saved: scaler_lstm.pkl, scaler_tab.pkl, pca.pkl, metadata.json")
    print(f"\n  Next step: run  python3 mapek_loop.py")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=DEFAULT_DATA)
    parser.add_argument("--skip_svm", action='store_true')
    parser.add_argument("--skip_rf",  action='store_true')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  THESIS ML TRAINING PIPELINE")
    print("  Peters PG/2415890")
    print("="*55)

    df     = load_data(args.data)
    X_seq, y, meta = make_windows(df)
    splits = prepare_splits(X_seq, y)

    if not args.skip_rf:
        rf, rf_pred, rf_prob = train_rf(splits)
    else:
        print("\n  Skipping RF (--skip_rf flag set)")
        rf, rf_pred, rf_prob = None, None, None
    lstm, lstm_pred, lstm_prob = train_lstm(splits)

    if not args.skip_svm:
        svm, svm_pred, svm_prob = train_svm(splits)
    else:
        print("\n  Skipping SVM (--skip_svm flag set)")
        svm = None

    save_models(rf, lstm, svm, splits)

    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print("  Models saved to:", MODEL_DIR)
    print("  Reports saved to:", REPORT_DIR)
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
