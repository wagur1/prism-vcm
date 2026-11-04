#!/usr/bin/env python3
"""Generate publication-grade RD-curves specifically contrasted against the Stock x265 Anchor.

Produces:
1. docs/paper_artifacts/rd_curves_vs_anchor.png & .pdf
   - 2x2 grid (YOLOv11-L & Faster R-CNN x AP50 & mAP) with shaded bitrate saving regions
     and explicit BD-rate delta callouts.
2. docs/paper_artifacts/rd_delta_vs_anchor.png & .pdf
   - Direct Delta Bitrate (%) and Delta AP (%) against the Anchor across CRFs.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO / "docs/reports/tier1b_val5k_final_report.json"
OUTPUT_DIR = REPO / "docs/paper_artifacts"

CRFS = (20, 24, 28, 32, 36)
DETECTORS = ("yolo11l", "faster_rcnn")
DETECTOR_NAMES = {
    "yolo11l": "YOLOv11-L (Target Detector)",
    "faster_rcnn": "Faster R-CNN (Cross-Detector Transfer)"
}

MODEL_CONFIGS = {
    "M1": {
        "label": "Anchor: Stock x265",
        "color": "#111111",
        "linestyle": "--",
        "linewidth": 1.8,
        "marker": "s",
        "markerfacecolor": "#111111",
        "zorder": 4,
    },
    "M2": {
        "label": "M2: Spatial QP (Zero-mean)",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.4,
        "marker": "o",
        "markerfacecolor": "#0072B2",
        "zorder": 5,
    },
    "M2_asym": {
        "label": "M2: Asymmetric [-4, 6]",
        "color": "#009E73",
        "linestyle": "-.",
        "linewidth": 1.5,
        "marker": "^",
        "markerfacecolor": "#009E73",
        "zorder": 6,
    },
    "M2_asym_prefilter": {
        "label": "M2: Asym + Prefilter (Ours)",
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 2.2,
        "marker": "D",
        "markerfacecolor": "#D55E00",
        "zorder": 7,
    },
}


def load_data():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["results"]["metrics"]
    bd_rates = report["results"]["bd_rates"]
    
    # Organize data per detector -> model -> crf -> bytes, ap50, ap_50_95
    data = {det: {m: {} for m in MODEL_CONFIGS} for det in DETECTORS}
    
    for det in DETECTORS:
        for crf in CRFS:
            # M1
            m1_entry = metrics[det][f"M1_crf{crf}"]
            data[det]["M1"][crf] = {
                "mb": m1_entry["bytes"] / (1024 * 1024),
                "ap50": m1_entry["ap50"] * 100,
                "map": m1_entry["ap_50_95"] * 100,
            }
            # M2, M2_asym, M2_asym_prefilter (average over seeds 0, 1, 2)
            for model in ("M2", "M2_asym", "M2_asym_prefilter"):
                seed_entries = [metrics[det][f"{model}_seed{s}_crf{crf}"] for s in (0, 1, 2)]
                data[det][model][crf] = {
                    "mb": np.mean([e["bytes"] for e in seed_entries]) / (1024 * 1024),
                    "ap50": np.mean([e["ap50"] for e in seed_entries]) * 100,
                    "map": np.mean([e["ap_50_95"] for e in seed_entries]) * 100,
                }
    return data, bd_rates


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
        "savefig.dpi": 300,
    })


def plot_rd_curves_vs_anchor(data, bd_rates, output_dir: Path):
    """2x2 figure: Top=YOLOv11-L (AP50 & mAP), Bottom=Faster R-CNN (AP50 & mAP) with shaded savings."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8), dpi=300)
    
    tasks = [
        (0, 0, "yolo11l", "ap50", "YOLOv11-L: AP50 vs. Bitstream Size", "AP50 (%)"),
        (0, 1, "yolo11l", "map", "YOLOv11-L: mAP@[.50:.95] vs. Bitstream Size", "mAP@[.50:.95] (%)"),
        (1, 0, "faster_rcnn", "ap50", "Faster R-CNN: AP50 vs. Bitstream Size", "AP50 (%)"),
        (1, 1, "faster_rcnn", "map", "Faster R-CNN: mAP@[.50:.95] vs. Bitstream Size", "mAP@[.50:.95] (%)"),
    ]
    
    for row, col, det, metric, title, ylabel in tasks:
        ax = axes[row, col]
        
        # Plot anchor first
        m1_pts = sorted(data[det]["M1"].values(), key=lambda p: p["mb"])
        m1_x = [p["mb"] for p in m1_pts]
        m1_y = [p[metric] for p in m1_pts]
        cfg_m1 = MODEL_CONFIGS["M1"]
        ax.plot(m1_x, m1_y, color=cfg_m1["color"], linestyle=cfg_m1["linestyle"],
                linewidth=cfg_m1["linewidth"], marker=cfg_m1["marker"],
                markersize=5, zorder=cfg_m1["zorder"], label=cfg_m1["label"])
        
        # Plot candidates
        cand_curves = {}
        for model in ("M2", "M2_asym", "M2_asym_prefilter"):
            cfg = MODEL_CONFIGS[model]
            pts = sorted(data[det][model].values(), key=lambda p: p["mb"])
            x = [p["mb"] for p in pts]
            y = [p[metric] for p in pts]
            cand_curves[model] = (x, y)
            ax.plot(x, y, color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], marker=cfg["marker"],
                    markersize=4.5, zorder=cfg["zorder"], label=cfg["label"])
            
        # Shaded region between Anchor and M2_asym_prefilter (showing rate savings)
        pref_x, pref_y = cand_curves["M2_asym_prefilter"]
        # Interpolate Anchor to candidate AP levels or vice versa for shaded visual
        ax.fill_betweenx(
            pref_y, pref_x, np.interp(pref_y, m1_y, m1_x),
            color="#D55E00", alpha=0.12, label="Bitrate Saved vs Anchor"
        )
        
        # Annotation text box for BD-rates
        bds = bd_rates[det]
        m2_bd = bds["M2"]["mean"].get(f"bjontegaard_bd_rate_{metric}", 0.0)
        asym_bd = bds["M2_asym"]["mean"].get(f"bjontegaard_bd_rate_{metric}", 0.0)
        pref_bd = bds["M2_asym_prefilter"]["mean"].get(f"bjontegaard_bd_rate_{metric}", 0.0)
        
        info_text = (
            f"BD-rate vs. Anchor:\n"
            f"• M2 (Spatial): {m2_bd:+.2f}%\n"
            f"• M2_asym: {asym_bd:+.2f}%\n"
            f"• Prefilter: {pref_bd:+.2f}%"
        )
        
        # Position box: lower right for AP curves
        ax.text(0.97, 0.05, info_text, transform=ax.transAxes,
                fontsize=7.5, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))
        
        ax.set_title(title, fontweight="bold", pad=7)
        ax.set_xlabel("Total Bitstream Size (MB on 4,737 images)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(150, 970)
        
        # Add CRF labels near Anchor points
        for crf in CRFS:
            pt = data[det]["M1"][crf]
            ax.annotate(f"CRF{crf}", (pt["mb"], pt[metric]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6.5, color="#555555")

    # Global legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.015), frameon=True)
    
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"rd_curves_vs_anchor.{ext}", dpi=300)
    plt.close(fig)
    print(f"Generated {output_dir / 'rd_curves_vs_anchor.png'}")


def plot_rd_delta_vs_anchor(data, bd_rates, output_dir: Path):
    """Direct delta plot: X=CRF, Y=Delta Bitrate (%) and Delta AP (%) against Anchor."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), dpi=300)
    
    # Plot 1: Bitstream Size Reduction (%) vs CRF
    ax1 = axes[0]
    ax1.axhline(0, color="#111111", linestyle="--", linewidth=1.5, label="Anchor: Stock x265 (0%)")
    
    models = ("M2", "M2_asym", "M2_asym_prefilter")
    for model in models:
        cfg = MODEL_CONFIGS[model]
        # Bitstream reduction is detector-independent (codec output bytes are identical)
        deltas = []
        for crf in CRFS:
            m1_b = data["yolo11l"]["M1"][crf]["mb"]
            cand_b = data["yolo11l"][model][crf]["mb"]
            rel_delta = ((cand_b - m1_b) / m1_b) * 100.0
            deltas.append(rel_delta)
        ax1.plot(CRFS, deltas, color=cfg["color"], linestyle=cfg["linestyle"],
                 linewidth=cfg["linewidth"], marker=cfg["marker"], markersize=5,
                 label=cfg["label"])
        
        # Print numbers on the full proposal line
        if model == "M2_asym_prefilter":
            for crf, d in zip(CRFS, deltas):
                ax1.annotate(f"{d:.1f}%", (crf, d), textcoords="offset points",
                             xytext=(0, -11), ha="center", fontsize=7, color=cfg["color"], fontweight="bold")

    ax1.set_title("Bitstream Size Delta vs. Anchor (%)", fontweight="bold")
    ax1.set_xlabel("Constant Rate Factor (CRF)")
    ax1.set_ylabel("Size Delta relative to Stock x265 (%)")
    ax1.set_xticks(CRFS)
    ax1.set_ylim(-8.5, 2.5)
    ax1.axhspan(-8.5, 0, color="#2ca02c", alpha=0.06, label="Bitrate Saving Zone")
    
    # Plot 2: Accuracy Delta (mAP@[.50:.95]) on YOLOv11-L vs CRF
    ax2 = axes[1]
    ax2.axhline(0, color="#111111", linestyle="--", linewidth=1.5, label="Anchor (Stock x265)")
    for model in models:
        cfg = MODEL_CONFIGS[model]
        map_deltas = []
        for crf in CRFS:
            m1_map = data["yolo11l"]["M1"][crf]["map"]
            cand_map = data["yolo11l"][model][crf]["map"]
            map_deltas.append(cand_map - m1_map)
        ax2.plot(CRFS, map_deltas, color=cfg["color"], linestyle=cfg["linestyle"],
                 linewidth=cfg["linewidth"], marker=cfg["marker"], markersize=5,
                 label=cfg["label"])

    ax2.set_title("YOLOv11-L mAP Delta vs. Anchor (pp)", fontweight="bold")
    ax2.set_xlabel("Constant Rate Factor (CRF)")
    ax2.set_ylabel(r"$\Delta$ mAP@[.50:.95] (percentage points)")
    ax2.set_xticks(CRFS)
    ax2.set_ylim(-0.20, 0.25)
    ax2.axhspan(0, 0.25, color="#1f77b4", alpha=0.06, label="Accuracy Improvement Zone")
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=True)
    
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"rd_delta_vs_anchor.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {output_dir / 'rd_delta_vs_anchor.png'}")


def main():
    setup_style()
    data, bd_rates = load_data()
    plot_rd_curves_vs_anchor(data, bd_rates, OUTPUT_DIR)
    plot_rd_delta_vs_anchor(data, bd_rates, OUTPUT_DIR)


if __name__ == "__main__":
    main()
