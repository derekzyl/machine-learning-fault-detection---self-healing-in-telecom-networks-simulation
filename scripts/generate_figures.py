#!/usr/bin/env python3
"""
generate_figures.py
===================
Generates five publication-quality thesis figures directly from the
simulation parameters in thesis-fault-sim.cc / preprocess_and_train.py:

  Figure 3.1 — NS-3 hexagonal macro-cell topology
  Figure 3.2 — End-to-end ML pipeline flowchart
  Figure 3.3 — MAPE-K closed-loop self-healing block diagram
  Figure 3.4 — Fault injection timeline across simulation trials
  Figure 3.5 — LSTM network architecture diagram

Output: ~/thesis-sim/reports/fig3_1_topology.png
        ~/thesis-sim/reports/fig3_2_pipeline.png
        ~/thesis-sim/reports/fig3_3_mapek.png
        ~/thesis-sim/reports/fig3_4_timeline.png
        ~/thesis-sim/reports/fig3_5_lstm_arch.png

Usage:
  python3 scripts/generate_figures.py"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

REPORT_DIR = os.path.expanduser("~/thesis-sim/reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── colour palette (consistent across all figures) ──────────────────────────
C_MACRO = "#1A73E8"  # blue  – macro gNB
C_SMALL = "#34A853"  # green – small cell
C_UE = "#FBBC04"  # amber – UE
C_FAULT = "#EA4335"  # red   – fault injection
C_BG = "#F8F9FA"
C_BORDER = "#DADCE0"
C_ARROW = "#5F6368"
C_BOX_A = "#E8F0FE"  # light blue  – data processing boxes
C_BOX_B = "#E6F4EA"  # light green – ML model boxes
C_BOX_C = "#FEF7E0"  # light amber – evaluation boxes
C_BOX_D = "#FCE8E6"  # light red   – MAPE-K boxes

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})


# =============================================================================
# FIGURE 3.1 — NS-3 HEXAGONAL MACRO-CELL TOPOLOGY
# =============================================================================
def fig3_1_topology():
    """
    7-cell hexagonal layout (1 centre + 6 surrounding), each macro gNB has
    3 small cells at 1/3 radius, and random UEs scattered across the area.
    One gNB is annotated as the fault-injection point.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_aspect("equal")
    ax.axis("off")

    R = 1.0  # macro-cell radius (normalised)
    sr = R * 0.33  # small-cell offset from macro
    hex_r = R * 0.96  # hexagon draw radius (slightly smaller than cell radius)

    # Hexagonal macro-gNB positions (flat-top hex grid)
    macro_pos = [(0.0, 0.0)]  # centre
    for i in range(6):
        angle = np.radians(60 * i)
        macro_pos.append((
            np.sqrt(3) * R * np.cos(angle),
            np.sqrt(3) * R * np.sin(angle),
        ))

    # Fault is injected at gNB index 3 (matching g_faultGnb logic)
    FAULT_GNB = 3

    # ── draw hexagonal cell boundaries ──────────────────────────────────────
    for gid, (cx, cy) in enumerate(macro_pos):
        hex_verts = []
        for k in range(6):
            a = np.radians(30 + 60 * k)  # 30° offset → flat-top
            hex_verts.append((cx + hex_r * np.cos(a), cy + hex_r * np.sin(a)))
        hex_verts.append(hex_verts[0])
        xs, ys = zip(*hex_verts)
        color = "#FCE8E6" if gid == FAULT_GNB else "white"
        ax.fill(xs, ys, color=color, alpha=0.55, zorder=1)
        ax.plot(xs, ys, color=C_BORDER, lw=1.2, zorder=2)
        ax.text(
            cx,
            cy - 0.62,
            f"Cell {gid}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="#5F6368",
            zorder=10,
        )

    # ── draw small cells (3 per macro) ──────────────────────────────────────
    sc_angles = [90, 210, 330]
    for gid, (cx, cy) in enumerate(macro_pos):
        for sa in sc_angles:
            sx = cx + sr * np.cos(np.radians(sa))
            sy = cy + sr * np.sin(np.radians(sa))
            sc = Circle((sx, sy), radius=0.085, color=C_SMALL, alpha=0.85, zorder=5)
            ax.add_patch(sc)
            ax.plot(
                [cx, sx],
                [cy, sy],
                color=C_SMALL,
                lw=0.6,
                alpha=0.4,
                linestyle="--",
                zorder=4,
            )

    # ── draw macro gNBs ──────────────────────────────────────────────────────
    for gid, (cx, cy) in enumerate(macro_pos):
        color = C_FAULT if gid == FAULT_GNB else C_MACRO
        marker = ax.plot(
            cx,
            cy,
            marker="^",
            markersize=18,
            color=color,
            zorder=8,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        label = f"gNB {gid}"
        if gid == FAULT_GNB:
            label += "\n(Fault)"
        ax.text(
            cx,
            cy + 0.22,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=C_FAULT if gid == FAULT_GNB else C_MACRO,
            zorder=9,
        )

    # ── scatter UEs randomly within the simulation area ──────────────────────
    rng = np.random.default_rng(42)
    n_ue = 70
    ue_xs, ue_ys = [], []
    attempts = 0
    while len(ue_xs) < n_ue and attempts < 5000:
        x = rng.uniform(-2.1, 2.1)
        y = rng.uniform(-2.1, 2.1)
        # only place within the outer hex boundary (rough circle check)
        if np.sqrt(x**2 + y**2) < 2.1:
            ue_xs.append(x)
            ue_ys.append(y)
        attempts += 1
    ax.scatter(
        ue_xs,
        ue_ys,
        s=18,
        color=C_UE,
        zorder=7,
        edgecolors="#E37400",
        linewidths=0.4,
        alpha=0.85,
    )

    # ── fault injection annotation ────────────────────────────────────────────
    fx, fy = macro_pos[FAULT_GNB]
    ax.annotate(
        "Fault\nInjection\nPoint",
        xy=(fx, fy),
        xytext=(fx + 0.85, fy + 0.75),
        fontsize=8.5,
        color=C_FAULT,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_FAULT, lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#FCE8E6", edgecolor=C_FAULT, alpha=0.9
        ),
        zorder=12,
    )

    # ── scale bar: 500 m = R/√3 normalised units ≈ 0.577 ────────────────────
    # 1.75 km² area → ≈ 1.32 km across; 7-cell layout ≈ 2√3 R wide
    # We label 500 m as reference
    bar_x, bar_y = -1.9, -1.85
    scale = 0.577  # normalised ≈ 500 m
    ax.annotate(
        "",
        xy=(bar_x + scale, bar_y),
        xytext=(bar_x, bar_y),
        arrowprops=dict(arrowstyle="<->", color="#5F6368", lw=1.5),
    )
    ax.text(
        bar_x + scale / 2,
        bar_y - 0.13,
        "500 m",
        ha="center",
        fontsize=8,
        color="#5F6368",
    )

    # ── legend ───────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=C_MACRO,
            markersize=12,
            label="Macro gNB (7 cells)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=C_FAULT,
            markersize=12,
            label="Faulty gNB (fault injected)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_SMALL,
            markersize=10,
            label="Small Cell (21 total, 3 per macro)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_UE,
            markersize=8,
            label=f"UE (n={n_ue}, uniform dist.)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
        edgecolor=C_BORDER,
    )

    ax.set_title(
        "Figure 3.1 — NS-3 Network Topology: 7-Cell Hexagonal Macro Layout\n"
        "with Small Cells and UE Spatial Distribution (1.75 km² area)",
        fontsize=11,
        fontweight="bold",
        pad=14,
        color="#202124",
    )

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "fig3_1_topology.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIGURE 3.2 — END-TO-END ML PIPELINE FLOWCHART
# =============================================================================
def fig3_2_pipeline():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor=C_BORDER,
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#202124",
            zorder=4,
            wrap=True,
            multialignment="center",
        )

    def arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->", color=C_ARROW, lw=1.4, connectionstyle="arc3,rad=0.0"
            ),
            zorder=2,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx + 0.06,
                my,
                label,
                fontsize=7,
                color="#5F6368",
                ha="left",
                va="center",
                style="italic",
            )

    # ── Row 1: NS-3 Data Generation ─────────────────────────────────────────
    box(
        ax,
        2.0,
        8.2,
        3.2,
        0.7,
        "NS-3 Simulation\n(50 trials × 4 fault types)",
        C_BOX_A,
        bold=True,
    )
    box(
        ax,
        5.8,
        8.2,
        2.8,
        0.7,
        "Per-trial KPI CSVs\n(kpi_trial{N}_{fault}.csv)",
        C_BOX_A,
    )
    box(
        ax,
        9.4,
        8.2,
        2.8,
        0.7,
        "Merged Master Dataset\n(kpi_master_dataset.csv)",
        C_BOX_A,
    )
    arrow(ax, 3.6, 8.2, 4.4, 8.2, "200 files")
    arrow(ax, 7.2, 8.2, 8.0, 8.2, "concat")

    # ── Row 2: Preprocessing ─────────────────────────────────────────────────
    box(
        ax,
        2.0,
        6.8,
        3.0,
        0.7,
        "Sliding Window Extraction\n(W=10 s, stride=1 s)",
        C_BOX_A,
    )
    box(
        ax,
        5.8,
        6.8,
        2.8,
        0.7,
        "Statistical Feature Eng.\n(6 stats × 8 KPIs = 48 feats)",
        C_BOX_A,
    )
    box(ax, 9.4, 6.8, 2.8, 0.7, "Normalisation\n(StandardScaler per split)", C_BOX_A)
    arrow(ax, 9.4, 8.55 + 0.02, 9.4, 7.16)  # master dataset → window (vertical skip)
    # re-route: master → sliding window
    ax.annotate(
        "",
        xy=(2.0, 7.16),
        xytext=(2.0, 7.85),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.4),
    )
    # Actually route from master dataset down
    ax.annotate(
        "",
        xy=(2.0, 7.16),
        xytext=(9.4, 7.84),
        arrowprops=dict(
            arrowstyle="->", color=C_ARROW, lw=1.4, connectionstyle="arc3,rad=0.25"
        ),
    )
    arrow(ax, 3.5, 6.8, 4.4, 6.8)
    arrow(ax, 7.2, 6.8, 8.0, 6.8)

    # ── Row 3: Splitting / SMOTE / PCA ──────────────────────────────────────
    box(
        ax,
        2.0,
        5.4,
        3.0,
        0.7,
        "Stratified Train/Val/Test Split\n(70% / 15% / 15%)",
        C_BOX_A,
    )
    box(
        ax, 5.8, 5.4, 2.8, 0.7, "SMOTE Oversampling\n(training partition only)", C_BOX_A
    )
    box(
        ax,
        9.4,
        5.4,
        2.8,
        0.7,
        "PCA Dimensionality Reduction\n(95% variance retained)",
        C_BOX_A,
    )
    arrow(ax, 9.4, 6.45, 9.4, 5.75)
    arrow(ax, 2.0, 6.45, 2.0, 5.75)
    arrow(ax, 3.5, 5.4, 4.4, 5.4)
    arrow(ax, 7.2, 5.4, 8.0, 5.4)

    # ── Row 4: Model Training ────────────────────────────────────────────────
    box(
        ax,
        2.0,
        4.0,
        2.8,
        0.7,
        "Random Forest\n(200 trees, max_depth=20)",
        C_BOX_B,
        bold=True,
    )
    box(
        ax,
        5.8,
        4.0,
        2.8,
        0.7,
        "LSTM Network\n(2×128 units, Dropout 0.3)",
        C_BOX_B,
        bold=True,
    )
    box(ax, 9.4, 4.0, 2.8, 0.7, "SVM Baseline\n(RBF kernel, C=10)", C_BOX_B, bold=True)
    # arrows from SMOTE/PCA to models
    ax.annotate(
        "",
        xy=(2.0, 4.36),
        xytext=(2.0, 5.04),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.4),
    )
    ax.annotate(
        "",
        xy=(5.8, 4.36),
        xytext=(5.8, 5.04),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.4),
    )
    ax.annotate(
        "",
        xy=(9.4, 4.36),
        xytext=(9.4, 5.04),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.4),
    )

    # ── Row 5: Serialisation / Evaluation ────────────────────────────────────
    box(
        ax,
        5.8,
        2.7,
        3.4,
        0.7,
        "Model Serialisation\n(random_forest.pkl | lstm_model.h5 | svm_baseline.pkl\n"
        "+ scaler_lstm.pkl | scaler_tab.pkl | pca.pkl | metadata.json)",
        C_BOX_B,
        fontsize=7.5,
    )
    arrow(ax, 2.0, 3.64, 5.1, 2.9)
    arrow(ax, 5.8, 3.64, 5.8, 3.04)
    arrow(ax, 9.4, 3.64, 6.5, 2.9)

    # ── Row 6: MAPE-K loop ────────────────────────────────────────────────────
    for i, (label, xc) in enumerate([
        ("Monitor\n(KPI window\npre-filter)", 1.8),
        ("Analyse\n(ML inference\nfault class + conf.)", 4.7),
        ("Plan\n(remediation\npolicy lookup)", 7.6),
        ("Execute\n(action log +\nrecovery timing)", 10.5),
    ]):
        box(ax, xc, 1.35, 2.4, 0.9, label, C_BOX_D, fontsize=7.8, bold=(i == 1))

    arrow(ax, 3.0, 1.35, 3.5, 1.35)
    arrow(ax, 5.9, 1.35, 6.4, 1.35)
    arrow(ax, 8.8, 1.35, 9.3, 1.35)

    # feedback loop arrow (Execute → Monitor)
    ax.annotate(
        "",
        xy=(1.8, 0.85),
        xytext=(10.5, 0.85),
        arrowprops=dict(
            arrowstyle="->", color=C_FAULT, lw=1.4, connectionstyle="arc3,rad=0.0"
        ),
    )
    ax.annotate(
        "",
        xy=(1.8, 0.9),
        xytext=(1.8, 0.93),
        arrowprops=dict(arrowstyle="->", color=C_FAULT, lw=1.4),
    )
    ax.text(
        6.15,
        0.67,
        "Closed-loop feedback (MAPE-K cycle = 5 s)",
        ha="center",
        fontsize=7.5,
        color=C_FAULT,
        style="italic",
    )

    # serialisation → analyse
    arrow(ax, 5.8, 2.35, 4.7, 1.8)

    # ── section labels ────────────────────────────────────────────────────────
    ax.text(
        0.18,
        7.5,
        "DATA\nGENERATION",
        fontsize=7.5,
        color="#5F6368",
        rotation=90,
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.18,
        6.1,
        "PRE-\nPROCESSING",
        fontsize=7.5,
        color="#5F6368",
        rotation=90,
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.18,
        4.0,
        "MODEL\nTRAINING",
        fontsize=7.5,
        color="#5F6368",
        rotation=90,
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.18,
        1.35,
        "MAPE-K\nINFERENCE",
        fontsize=7.5,
        color=C_FAULT,
        rotation=90,
        va="center",
        fontweight="bold",
    )

    # horizontal dividers
    for yline in [7.85, 6.17, 4.76, 2.15]:
        ax.axhline(yline, color=C_BORDER, lw=0.8, linestyle="--", alpha=0.6)

    ax.set_title(
        "Figure 3.2 — End-to-End Machine Learning Pipeline for Fault Detection\n"
        "and MAPE-K Self-Healing Integration",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#202124",
    )

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "fig3_2_pipeline.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIGURE 3.5 — LSTM ARCHITECTURE DIAGRAM
# =============================================================================
def fig3_5_lstm():
    """
    Draws the stacked LSTM architecture as a vertical layer diagram.
    Architecture (from preprocess_and_train.py):
      Input  → (10, 8)
      LSTM 1 → 128 units, return_sequences=True
      Dropout 0.3
      LSTM 2 → 128 units
      Dropout 0.3
      Dense  → 64 units ReLU
      Output → 4 units Softmax
    """
    layers = [
        ("Input", "(10 timesteps × 8 KPI features)", "#FFFFFF", C_MACRO),
        ("LSTM Layer 1", "128 hidden units\nreturn_sequences=True", C_BOX_A, C_MACRO),
        ("Dropout", "rate = 0.3", C_BOX_C, "#E37400"),
        ("LSTM Layer 2", "128 hidden units\nreturn_sequences=False", C_BOX_A, C_MACRO),
        ("Dropout", "rate = 0.3", C_BOX_C, "#E37400"),
        ("Dense (ReLU)", "64 units · activation: ReLU", C_BOX_B, "#1E7E34"),
        (
            "Output (Softmax)",
            "4 units · activation: Softmax\n"
            "[Normal | Power Fault | Congestion | HW Failure]",
            C_BOX_D,
            C_FAULT,
        ),
    ]

    n = len(layers)
    fig_h = 1.1 * n + 1.4
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    box_w = 6.5
    box_h = 0.75
    x_centre = 4.5
    y_top = fig_h - 0.9
    gap = (fig_h - 1.2) / n  # vertical spacing

    prev_y = None
    for i, (name, detail, facecolor, edgecolor) in enumerate(layers):
        y = y_top - i * gap

        rect = FancyBboxPatch(
            (x_centre - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.1",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.8,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            x_centre - 2.0,
            y,
            name,
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#202124",
            zorder=4,
        )
        ax.text(
            x_centre - 1.8,
            y,
            detail,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#5F6368",
            zorder=4,
        )

        # dimension annotation on right
        dims = {
            "Input": "shape: (batch, 10, 8)",
            "LSTM Layer 1": "output: (batch, 10, 128)",
            "Dropout": "",
            "LSTM Layer 2": "output: (batch, 128)",
            "Dense (ReLU)": "output: (batch, 64)",
            "Output (Softmax)": "output: (batch, 4)  →  argmax → class",
        }
        dim_text = dims.get(name, "")
        if dim_text:
            ax.text(
                x_centre + box_w / 2 + 0.12,
                y,
                dim_text,
                ha="left",
                va="center",
                fontsize=7.5,
                color="#5F6368",
                style="italic",
                zorder=4,
            )

        # draw arrow from previous box
        if prev_y is not None:
            ax.annotate(
                "",
                xy=(x_centre, y + box_h / 2 + 0.02),
                xytext=(x_centre, prev_y - box_h / 2 - 0.02),
                arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5),
                zorder=2,
            )
        prev_y = y

    # ── training config annotation (right column) ─────────────────────────────
    cfg_x = 0.3
    cfg = [
        ("Optimiser:", "Adam (lr = 0.001)"),
        ("Loss:", "Sparse Categorical\nCrossentropy"),
        ("Epochs:", "up to 100\n(EarlyStopping, patience=10)"),
        ("Batch:", "64 samples"),
        ("Reg.:", "ReduceLROnPlateau\n(factor=0.5, patience=5)"),
        ("Input:", "SMOTE-augmented\nnormalised windows"),
    ]
    cfg_y0 = y_top - 0.1
    ax.text(
        cfg_x + 0.85,
        cfg_y0 + 0.25,
        "Training Configuration",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#202124",
    )
    for j, (k, v) in enumerate(cfg):
        ky = cfg_y0 - j * (fig_h - 1.5) / len(cfg)
        ax.text(
            cfg_x, ky, k, ha="left", fontsize=7.8, fontweight="bold", color="#202124"
        )
        ax.text(cfg_x, ky - 0.28, v, ha="left", fontsize=7.5, color="#5F6368")

    ax.set_title(
        "Figure 3.5 — LSTM Network Architecture for Fault Classification\n"
        "(Input: 10 timesteps × 8 KPIs → Output: 4-class fault label)",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#202124",
    )

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "fig3_5_lstm_arch.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIGURE 3.3 — MAPE-K CLOSED-LOOP SELF-HEALING BLOCK DIAGRAM
# =============================================================================
def fig3_3_mapek():
    """
    Draws the four MAPE-K phase boxes (Monitor, Analyse, Plan, Execute)
    in a circular arrangement with a central Knowledge Base, surrounded
    by an NS-3 simulation environment boundary.
    Data flows and the bidirectional Execute↔NS-3 interface are annotated.
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── NS-3 environment boundary (dashed outer rectangle) ────────────────────
    env_rect = FancyBboxPatch(
        (-5.2, -4.6),
        10.4,
        9.8,
        boxstyle="round,pad=0.2",
        facecolor="#EAF4FB",
        edgecolor="#1A73E8",
        linewidth=2.0,
        linestyle="--",
        zorder=0,
        alpha=0.4,
    )
    ax.add_patch(env_rect)
    ax.text(
        -4.9,
        4.8,
        "NS-3 Simulation Environment",
        fontsize=9,
        color=C_MACRO,
        fontweight="bold",
        style="italic",
    )

    # ── Four MAPE-K phase boxes at cardinal positions ─────────────────────────
    phases = [
        # (label, detail, x,  y,  face_color,  edge_color)
        (
            "MONITOR",
            "KPI window collection\n(1 s interval, 7 gNBs)\nThreshold pre-filter",
            0.0,
            3.2,
            "#E8F0FE",
            C_MACRO,
        ),
        (
            "ANALYSE",
            "ML fault classification\n(RF | LSTM | SVM)\nconf. threshold = 0.70",
            3.5,
            0.0,
            "#E6F4EA",
            "#1E7E34",
        ),
        (
            "PLAN",
            "Remediation policy lookup\nPower / Congestion\nHW Failure actions",
            0.0,
            -3.2,
            "#FEF7E0",
            "#E37400",
        ),
        (
            "EXECUTE",
            "Action dispatch +\nrecovery timer start\nMAPE-K cycle = 5 s",
            -3.5,
            0.0,
            "#FCE8E6",
            C_FAULT,
        ),
    ]

    box_w, box_h = 3.0, 1.55
    centers = {}
    for name, detail, x, y, fc, ec in phases:
        rect = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.15",
            facecolor=fc,
            edgecolor=ec,
            linewidth=2.0,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y + 0.38,
            name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#202124",
            zorder=4,
        )
        ax.text(
            x,
            y - 0.28,
            detail,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#5F6368",
            zorder=4,
            linespacing=1.4,
        )
        centers[name] = (x, y)

    # ── Central Knowledge Base ────────────────────────────────────────────────
    kb_rect = FancyBboxPatch(
        (-1.45, -0.9),
        2.9,
        1.8,
        boxstyle="round,pad=0.12",
        facecolor="#F3E8FD",
        edgecolor="#8430CE",
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(kb_rect)
    ax.text(
        0,
        0.35,
        "Knowledge Base",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#6A0DAD",
        zorder=4,
    )
    ax.text(
        0,
        -0.2,
        "KPI history · fault patterns\nremediation policies · thresholds",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#5F6368",
        zorder=4,
        linespacing=1.4,
    )

    # ── Directed data-flow arrows between phases ──────────────────────────────
    def phase_arrow(p1, p2, label, rad=0.0, color=C_ARROW):
        x1, y1 = centers[p1]
        x2, y2 = centers[p2]
        # Compute edge points (stop at box boundary, not centre)
        dx, dy = x2 - x1, y2 - y1
        # rough offset to box edge
        ox1 = (box_w / 2 if abs(dx) > abs(dy) else 0) * np.sign(dx)
        oy1 = (box_h / 2 if abs(dy) >= abs(dx) else 0) * np.sign(dy)
        ox2, oy2 = -ox1, -oy1
        ax.annotate(
            "",
            xy=(x2 + ox2 * 0.55, y2 + oy2 * 0.55),
            xytext=(x1 + ox1 * 0.55, y1 + oy1 * 0.55),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=1.8, connectionstyle=f"arc3,rad={rad}"
            ),
            zorder=2,
        )
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            bbox=dict(facecolor=C_BG, edgecolor="none", pad=1.5),
        )

    phase_arrow("MONITOR", "ANALYSE", "KPI window\n+ alert flag", rad=-0.25)
    phase_arrow("ANALYSE", "PLAN", "fault class\n+ confidence", rad=-0.25)
    phase_arrow("PLAN", "EXECUTE", "remediation\naction", rad=-0.25)
    phase_arrow("EXECUTE", "MONITOR", "cycle\ncomplete", rad=-0.25, color="#E37400")

    # ── Knowledge Base ↔ each phase (thin dashed lines) ──────────────────────
    kb_cx, kb_cy = 0.0, 0.0
    for name, detail, px, py, fc, ec in phases:
        ax.plot(
            [kb_cx, px * 0.48],
            [kb_cy, py * 0.48],
            color="#8430CE",
            lw=1.0,
            linestyle=":",
            alpha=0.6,
            zorder=1,
        )

    # ── Execute ↔ NS-3 bidirectional annotation ───────────────────────────────
    ex, ey = centers["EXECUTE"]
    ax.annotate(
        "",
        xy=(ex - 1.55, ey - 1.5),
        xytext=(ex - 0.5, ey - box_h / 2 - 0.05),
        arrowprops=dict(arrowstyle="<->", color=C_FAULT, lw=1.6, linestyle="dashed"),
        zorder=5,
    )
    ax.text(
        ex - 2.45,
        ey - 2.1,
        "Actuation commands\n& KPI feedback",
        ha="center",
        fontsize=7.8,
        color=C_FAULT,
        fontweight="bold",
        bbox=dict(
            facecolor="#FCE8E6", edgecolor=C_FAULT, boxstyle="round,pad=0.3", alpha=0.9
        ),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#E8F0FE", edgecolor=C_MACRO, label="Monitor phase"),
        mpatches.Patch(
            facecolor="#E6F4EA",
            edgecolor="#1E7E34",
            label="Analyse phase  (ML inference)",
        ),
        mpatches.Patch(facecolor="#FEF7E0", edgecolor="#E37400", label="Plan phase"),
        mpatches.Patch(facecolor="#FCE8E6", edgecolor=C_FAULT, label="Execute phase"),
        mpatches.Patch(
            facecolor="#F3E8FD", edgecolor="#8430CE", label="Knowledge Base"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
        edgecolor=C_BORDER,
    )

    ax.set_title(
        "Figure 3.3 — MAPE-K Closed-Loop Autonomous Self-Healing Architecture\n"
        "(Monitor → Analyse → Plan → Execute, with central Knowledge Base)",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color="#202124",
    )

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "fig3_3_mapek.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIGURE 3.4 — FAULT INJECTION TIMELINE
# =============================================================================
def fig3_4_timeline():
    """
    Gantt-style timeline chart for 6 representative trials.
    Fault parameters are derived from thesis-fault-sim.cc:
      - SIM_TIME = 300 s
      - Fault onset: uniform ~ [30, 250] s
      - Duration:    uniform ~ [15,  45] s
      - MAPE-K detection delay: MAPEK_CYCLE = 5 s + model inference
      - Remediation times per fault type from REMEDIATION_POLICY
    Reproducible random seed matching the NS-3 RNG seed formula (1000 + trial).
    """
    SIM_TIME = 300
    MAPEK_CYCLE = 5  # seconds
    CONF_MEAN = 0.82  # representative mean confidence
    REMED_TIME = {1: 45, 2: 30, 3: 50}  # from REMEDIATION_POLICY
    FAULT_COLORS = {
        1: ("#EA4335", "Power Fault"),
        2: ("#FBBC04", "Congestion"),
        3: ("#1A73E8", "gNB HW Failure"),
    }
    N_TRIALS = 6  # show 6 representative trials

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor("white")

    yticks, ylabels = [], []
    rng = np.random.default_rng(1000)  # matches NS-3 seed base

    for trial_idx in range(N_TRIALS):
        y = N_TRIALS - trial_idx  # top row = Trial 0
        fault_type = (trial_idx % 3) + 1  # cycle 1→2→3→1…
        color, fname = FAULT_COLORS[fault_type]

        # Reproduce NS-3 stochastic fault window
        onset = 30.0 + rng.uniform() * 220.0
        duration = 15.0 + rng.uniform() * 30.0
        end = min(onset + duration, SIM_TIME)

        # Detection delay: MAPE-K cycle latency (uniform 1–2 cycles)
        detect_delay = MAPEK_CYCLE * rng.integers(1, 3)
        detect_t = min(onset + detect_delay, end)

        # Remediation action dispatched at detection; recovery = remed_time × (1 - conf×0.3)
        remed_s = REMED_TIME[fault_type] * (1.0 - CONF_MEAN * 0.3)
        restore_t = min(detect_t + remed_s, SIM_TIME)

        # ── Normal (background) ───────────────────────────────────────────
        ax.barh(
            y,
            SIM_TIME,
            left=0,
            height=0.55,
            color="#E8F0FE",
            edgecolor="#DADCE0",
            linewidth=0.5,
            zorder=1,
        )

        # ── Fault active window ───────────────────────────────────────────
        ax.barh(
            y,
            end - onset,
            left=onset,
            height=0.55,
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
            label=fname if trial_idx < 3 else "",
        )

        # ── Post-detection recovery zone (lighter shade) ──────────────────
        if detect_t < end:
            ax.barh(
                y,
                restore_t - detect_t,
                left=detect_t,
                height=0.55,
                color=color,
                alpha=0.30,
                zorder=3,
                hatch="////",
                edgecolor="white",
                linewidth=0,
            )

        # ── Fault onset marker ────────────────────────────────────────────
        ax.plot(
            onset,
            y,
            marker="|",
            markersize=14,
            color=color,
            markeredgewidth=2.5,
            zorder=5,
        )
        ax.text(
            onset,
            y + 0.36,
            f"t={onset:.0f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

        # ── MAPE-K detection marker ───────────────────────────────────────
        ax.plot(
            detect_t,
            y,
            marker="D",
            markersize=7,
            color="#8430CE",
            zorder=6,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        ax.annotate(
            f"Detect\nt={detect_t:.0f}s",
            xy=(detect_t, y),
            xytext=(detect_t + 4, y - 0.42),
            fontsize=6.5,
            color="#6A0DAD",
            arrowprops=dict(arrowstyle="->", color="#8430CE", lw=0.9),
        )

        # ── Restore marker ────────────────────────────────────────────────
        ax.plot(
            restore_t,
            y,
            marker="*",
            markersize=11,
            color="#34A853",
            zorder=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.text(
            restore_t + 1,
            y + 0.36,
            f"Restore\nt={restore_t:.0f}s",
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#34A853",
        )

        # ── MTTR brace ────────────────────────────────────────────────────
        mttr = restore_t - onset
        ax.annotate(
            "",
            xy=(restore_t, y - 0.38),
            xytext=(onset, y - 0.38),
            arrowprops=dict(arrowstyle="<->", color="#5F6368", lw=1.0),
        )
        ax.text(
            (onset + restore_t) / 2,
            y - 0.48,
            f"MTTR≈{mttr:.0f}s",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#5F6368",
            style="italic",
        )

        yticks.append(y)
        ylabels.append(f"Trial {trial_idx}  |  {fname}  |  gNB {trial_idx % 7}")

    # ── Phase labels at top ───────────────────────────────────────────────────
    for t, lbl in [(0, "Normal"), (150, "Normal"), (SIM_TIME, "")]:
        ax.axvline(t, color=C_BORDER, lw=0.8, linestyle=":")

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlim(0, SIM_TIME)
    ax.set_ylim(0.2, N_TRIALS + 0.8)
    ax.set_xlabel("Simulation Time (seconds)", fontsize=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xticks(range(0, SIM_TIME + 1, 30))
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C_BORDER, lw=0.6, alpha=0.7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#EA4335", alpha=0.75, label="Power Fault (class 1)"),
        mpatches.Patch(color="#FBBC04", alpha=0.75, label="Congestion (class 2)"),
        mpatches.Patch(color="#1A73E8", alpha=0.75, label="gNB HW Failure (class 3)"),
        mpatches.Patch(
            facecolor="white",
            edgecolor="#5F6368",
            hatch="////",
            label="Detection → Recovery window",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#8430CE",
            markersize=8,
            label="MAPE-K Detection point",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#34A853",
            markersize=10,
            label="Restoration point",
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=7.8,
        framealpha=0.93,
        edgecolor=C_BORDER,
        ncol=2,
    )

    ax.set_title(
        "Figure 3.4 — Representative Fault Injection Timeline Across 6 Simulation Trials\n"
        "(300 s window · stochastic onset [30–250 s] · duration [15–45 s] · MAPE-K cycle = 5 s)",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color="#202124",
    )

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "fig3_4_timeline.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  THESIS FIGURE GENERATOR")
    print("=" * 55)

    print("\n[1/5] Figure 3.1 — Network Topology...")
    fig3_1_topology()

    print("[2/5] Figure 3.2 — ML Pipeline Flowchart...")
    fig3_2_pipeline()

    print("[3/5] Figure 3.3 — MAPE-K Block Diagram...")
    fig3_3_mapek()

    print("[4/5] Figure 3.4 — Fault Injection Timeline...")
    fig3_4_timeline()

    print("[5/5] Figure 3.5 — LSTM Architecture...")
    fig3_5_lstm()

    print("\n" + "=" * 55)
    print(f"  All figures saved to: {REPORT_DIR}")
    print("=" * 55 + "\n")
