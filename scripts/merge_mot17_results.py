#!/usr/bin/env python3
"""Aggregate MOT17 video tracking evaluation results across sequences.

Computes mean tracking metrics (HOTA, MOTA, IDF1, DetA, AssA), Bjontegaard and
PAVA-PCHIP BD-rates against anchor, paired sequence bootstrap 95% confidence
intervals, publication-grade RD curves, and journal LaTeX tables.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from prism.evaluation.metrics import RDPoint, bjontegaard_bd_rate, pava_pchip_bd_rate
from tier1a.common import read_json, sha256, write_json

DEFAULT_REPORTS_DIR = REPO / "docs/reports/mot17"
DEFAULT_OUTPUT_DIR = REPO / "docs/reports/mot17/merged"

SEQUENCES = (
    "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
    "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN",
)
CRFS = (22, 27, 32, 37)
VARIANTS = (
    "anchor", "M2_zero_mean", "M2_asym", "M2_asym_ratecomp",
    "M2_filter_only", "M2_asym_prefilter",
)

VARIANT_LABELS = {
    "anchor": "Anchor (Stock x265 LDP)",
    "M2_zero_mean": "M2: Zero-Mean Spatial QP",
    "M2_asym": "M2: Asymmetric [-4, +6]",
    "M2_asym_ratecomp": "M2: Asym + Rate-Comp",
    "M2_filter_only": "M2: Background Filter Only",
    "M2_asym_prefilter": "M2: Asym + Prefilter (PRISM)",
}

VARIANT_COLORS = {
    "anchor": "#222222",
    "M2_zero_mean": "#0072B2",
    "M2_asym": "#009E73",
    "M2_asym_ratecomp": "#E69F00",
    "M2_filter_only": "#56B4E9",
    "M2_asym_prefilter": "#D55E00",
}

VARIANT_MARKERS = {
    "anchor": "o",
    "M2_zero_mean": "s",
    "M2_asym": "^",
    "M2_asym_ratecomp": "v",
    "M2_filter_only": "D",
    "M2_asym_prefilter": "P",
}


def find_sequence_reports(reports_dir: Path) -> dict[str, Path]:
    """Discover valid mot17_seq_report.json files by sequence name."""
    found: dict[str, Path] = {}
    for path in sorted(reports_dir.glob("**/mot17_seq_report.json")):
        try:
            data = read_json(path)
            seq = data.get("sequence")
            if seq in SEQUENCES and data.get("status") == "complete":
                if seq in found and found[seq] != path:
                    # Prefer deeper or canonical directory docs/reports/mot17/<seq>/...
                    if path.parent.name == seq:
                        found[seq] = path
                else:
                    found[seq] = path
        except Exception:
            continue
    return found


def validate_report(report: dict, seq_name: str) -> None:
    if report.get("sequence") != seq_name:
        raise ValueError(f"Sequence mismatch: expected {seq_name}, got {report.get('sequence')}")
    if report.get("status") != "complete":
        raise ValueError(f"Report status is not complete: {report.get('status')}")
    obs = report.get("observations", [])
    if len(obs) != len(VARIANTS) * len(CRFS):
        raise ValueError(f"Expected {len(VARIANTS) * len(CRFS)} observations, got {len(obs)}")


def aggregate_observations(reports: dict[str, dict]) -> dict:
    """Aggregate per-sequence data into structured table and overall RD points."""
    # seq -> variant -> crf -> obs
    indexed: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    seq_metadata: dict[str, dict] = {}

    for seq_name, rep in reports.items():
        seq_metadata[seq_name] = {
            "fps": rep.get("fps", 30),
            "num_frames": rep.get("num_frames", 0),
            "duration_seconds": rep.get("duration_seconds", 0.0),
            "width": rep.get("width"),
            "height": rep.get("height"),
        }
        for ob in rep.get("observations", []):
            indexed[seq_name][ob["variant"]][ob["crf"]] = ob

    # Mean across sequences for each (variant, crf)
    variant_crf_summary: dict[str, dict[int, dict]] = defaultdict(dict)
    seq_list = sorted(reports.keys())
    total_duration = sum(seq_metadata[s]["duration_seconds"] for s in seq_list)

    for v in VARIANTS:
        for c in CRFS:
            bitrates = []
            total_bits = 0
            hotas, motas, idf1s, detas, assas = [], [], [], [], []
            for s in seq_list:
                ob = indexed[s][v][c]
                bitrates.append(ob["bitrate_kbps"])
                total_bits += ob.get("bits", ob.get("total_bytes", 0) * 8)
                m = ob["metrics"]
                hotas.append(m["HOTA"])
                motas.append(m["MOTA"])
                idf1s.append(m["IDF1"])
                detas.append(m.get("DetA", 0.0))
                assas.append(m.get("AssA", 0.0))

            weighted_bitrate = (total_bits / (total_duration * 1000.0)) if total_duration > 0 else float(np.mean(bitrates))
            variant_crf_summary[v][c] = {
                "mean_bitrate_kbps": float(np.mean(bitrates)),
                "weighted_bitrate_kbps": float(weighted_bitrate),
                "std_bitrate_kbps": float(np.std(bitrates)),
                "HOTA": float(np.mean(hotas)),
                "MOTA": float(np.mean(motas)),
                "IDF1": float(np.mean(idf1s)),
                "DetA": float(np.mean(detas)),
                "AssA": float(np.mean(assas)),
                "HOTA_std": float(np.std(hotas)),
                "MOTA_std": float(np.std(motas)),
                "IDF1_std": float(np.std(idf1s)),
            }

    return {
        "indexed": indexed,
        "seq_metadata": seq_metadata,
        "variant_crf_summary": variant_crf_summary,
        "seq_list": seq_list,
        "total_duration": total_duration,
    }


def compute_sequence_bd_rates(agg: dict) -> dict[str, dict[str, dict]]:
    """Compute per-sequence BD-rates against anchor for each variant and metric."""
    indexed = agg["indexed"]
    seq_list = agg["seq_list"]
    results: dict[str, dict[str, dict]] = defaultdict(dict)

    metrics = ("HOTA", "MOTA", "IDF1")

    for seq in seq_list:
        for v in VARIANTS:
            if v == "anchor":
                continue
            results[seq][v] = {}
            for met in metrics:
                anchor_points = [
                    RDPoint(qp=c, score=indexed[seq]["anchor"][c]["metrics"][met],
                            bitstream_bytes=int(indexed[seq]["anchor"][c]["bitrate_kbps"] * 1000))
                    for c in CRFS
                ]
                cand_points = [
                    RDPoint(qp=c, score=indexed[seq][v][c]["metrics"][met],
                            bitstream_bytes=int(indexed[seq][v][c]["bitrate_kbps"] * 1000))
                    for c in CRFS
                ]
                bj = bjontegaard_bd_rate(anchor_points, cand_points)
                pchip = pava_pchip_bd_rate(anchor_points, cand_points)
                results[seq][v][met] = {
                    "bjontegaard": bj.get("bd_rate") if bj.get("valid") else None,
                    "pava_pchip": pchip.get("bd_rate") if pchip.get("valid") else None,
                }
    return results


def compute_overall_bd_rates(agg: dict) -> dict[str, dict[str, dict]]:
    """Compute overall BD-rates using sequence-pooled average RD curves."""
    summary = agg["variant_crf_summary"]
    results: dict[str, dict[str, dict]] = {}
    metrics = ("HOTA", "MOTA", "IDF1")

    for v in VARIANTS:
        if v == "anchor":
            continue
        results[v] = {}
        for met in metrics:
            anchor_points = [
                RDPoint(qp=c, score=summary["anchor"][c][met],
                        bitstream_bytes=int(summary["anchor"][c]["weighted_bitrate_kbps"] * 1000))
                for c in CRFS
            ]
            cand_points = [
                RDPoint(qp=c, score=summary[v][c][met],
                        bitstream_bytes=int(summary[v][c]["weighted_bitrate_kbps"] * 1000))
                for c in CRFS
            ]
            bj = bjontegaard_bd_rate(anchor_points, cand_points)
            pchip = pava_pchip_bd_rate(anchor_points, cand_points)
            results[v][met] = {
                "bjontegaard": bj.get("bd_rate") if bj.get("valid") else None,
                "pava_pchip": pchip.get("bd_rate") if pchip.get("valid") else None,
            }
    return results


def compute_bootstrap_ci(loaded_reports: dict[str, dict],
                         seq_list: list[str],
                         resamples: int = 2000,
                         seed: int = 42) -> dict[str, dict[str, dict]]:
    """Compute paired bootstrap 95% confidence intervals by resampling sequences."""
    rng = np.random.default_rng(seed)
    ci_results: dict[str, dict[str, dict]] = defaultdict(dict)
    metrics = ("HOTA", "MOTA", "IDF1")

    for v in VARIANTS:
        if v == "anchor":
            continue
        ci_results[v] = {}
        for met in metrics:
            draws_bj = []
            draws_pc = []
            for _ in range(resamples):
                sampled_seqs = rng.choice(seq_list, size=len(seq_list), replace=True)
                tot_bits_a, tot_bits_c = {c: 0 for c in CRFS}, {c: 0 for c in CRFS}
                tot_dur = sum(loaded_reports[s].get("duration_seconds", 1.0) for s in sampled_seqs)
                scores_a, scores_c = {c: [] for c in CRFS}, {c: [] for c in CRFS}
                for s in sampled_seqs:
                    for ob in loaded_reports[s]["observations"]:
                        c = ob["crf"]
                        if ob["variant"] == "anchor":
                            tot_bits_a[c] += ob.get("bits", ob["total_bytes"] * 8)
                            scores_a[c].append(ob["metrics"][met])
                        elif ob["variant"] == v:
                            tot_bits_c[c] += ob.get("bits", ob["total_bytes"] * 8)
                            scores_c[c].append(ob["metrics"][met])

                pts_a = [RDPoint(qp=c, score=float(np.mean(scores_a[c])),
                                 bitstream_bytes=int(tot_bits_a[c] / tot_dur / 1000)) for c in CRFS]
                pts_c = [RDPoint(qp=c, score=float(np.mean(scores_c[c])),
                                 bitstream_bytes=int(tot_bits_c[c] / tot_dur / 1000)) for c in CRFS]

                bj = bjontegaard_bd_rate(pts_a, pts_c)
                if bj.get("valid") and np.isfinite(bj["bd_rate"]) and abs(bj["bd_rate"]) < 500:
                    draws_bj.append(bj["bd_rate"])
                pc = pava_pchip_bd_rate(pts_a, pts_c)
                if pc.get("valid") and np.isfinite(pc["bd_rate"]):
                    draws_pc.append(pc["bd_rate"])

            ci_bj = np.quantile(draws_bj, [0.025, 0.975]).tolist() if len(draws_bj) > 50 else [None, None]
            ci_pc = np.quantile(draws_pc, [0.025, 0.975]).tolist() if len(draws_pc) > 50 else [None, None]

            ci_results[v][met] = {
                "bjontegaard": {"mean": float(np.mean(draws_bj)) if draws_bj else None, "ci95": ci_bj},
                "pava_pchip": {"mean": float(np.mean(draws_pc)) if draws_pc else None, "ci95": ci_pc},
            }
    return ci_results


def plot_rd_curves(agg: dict, output_dir: Path) -> list[Path]:
    """Generate publication-quality RD curves for HOTA and MOTA vs Bitrate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = agg["variant_crf_summary"]
    saved: list[Path] = []

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
    })

    plots_cfg = [
        ("HOTA", "mot17_rd_hota", "HOTA Score (%)", "Higher is better (Tracking Accuracy & Association)"),
        ("MOTA", "mot17_rd_mota", "MOTA Score (%)", "Higher is better (Multi-Object Tracking Accuracy)"),
        ("IDF1", "mot17_rd_idf1", "IDF1 Score (%)", "Higher is better (ID F1-Score)"),
    ]

    for metric, filename_base, ylabel, subtitle in plots_cfg:
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
        ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

        for v in VARIANTS:
            pts = [(summary[v][c]["weighted_bitrate_kbps"], summary[v][c][metric]) for c in CRFS]
            pts.sort(key=lambda p: p[0])
            rates, scores = zip(*pts)

            ax.plot(
                rates, scores,
                marker=VARIANT_MARKERS.get(v, "o"),
                color=VARIANT_COLORS.get(v, "black"),
                label=VARIANT_LABELS.get(v, v),
                linestyle="--" if v == "anchor" else "-",
                alpha=0.95,
            )

        ax.set_xlabel("Bitrate (kbps) [Weighted Sequence Average]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"MOT17 Multi-Object Tracking Rate-Distortion: {metric}\n{subtitle}", pad=10)
        ax.legend(frameon=True, loc="lower right", facecolor="white", framealpha=0.9)
        fig.tight_layout()

        for ext in (".png", ".pdf"):
            out_file = output_dir / f"{filename_base}{ext}"
            fig.savefig(out_file, bbox_inches="tight")
            saved.append(out_file)
        plt.close(fig)

    # Bar chart of BD-rates (HOTA and MOTA)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc", axis="y")

    candidates = [v for v in VARIANTS if v != "anchor"]
    x = np.arange(len(candidates))
    width = 0.35

    # Compute sequence mean BD-rate for bar chart
    hota_bds, mota_bds = [], []
    for c in candidates:
        pts_a_h = [RDPoint(qp=crf, score=summary["anchor"][crf]["HOTA"],
                           bitstream_bytes=int(summary["anchor"][crf]["weighted_bitrate_kbps"] * 1000)) for crf in CRFS]
        pts_c_h = [RDPoint(qp=crf, score=summary[c][crf]["HOTA"],
                           bitstream_bytes=int(summary[c][crf]["weighted_bitrate_kbps"] * 1000)) for crf in CRFS]
        b_h = bjontegaard_bd_rate(pts_a_h, pts_c_h)
        hota_bds.append(b_h.get("bd_rate") if b_h.get("valid") else 0.0)

        pts_a_m = [RDPoint(qp=crf, score=summary["anchor"][crf]["MOTA"],
                           bitstream_bytes=int(summary["anchor"][crf]["weighted_bitrate_kbps"] * 1000)) for crf in CRFS]
        pts_c_m = [RDPoint(qp=crf, score=summary[c][crf]["MOTA"],
                           bitstream_bytes=int(summary[c][crf]["weighted_bitrate_kbps"] * 1000)) for crf in CRFS]
        b_m = bjontegaard_bd_rate(pts_a_m, pts_c_m)
        mota_bds.append(b_m.get("bd_rate") if b_m.get("valid") else 0.0)

    rects1 = ax.bar(x - width / 2, hota_bds, width, label="BD-Rate (HOTA)", color="#0072B2", alpha=0.85)
    rects2 = ax.bar(x + width / 2, mota_bds, width, label="BD-Rate (MOTA)", color="#D55E00", alpha=0.85)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("BD-Rate (%) [Negative = Bitrate Savings at Same Score]")
    ax.set_title("Bjøntegaard BD-Rate Relative to Stock x265 Anchor (MOT17 Val)")
    ax.set_xticks(x)
    short_labels = [
        "M2: Zero-Mean", "M2: Asym [-4,+6]", "M2: Asym+RateComp",
        "M2: Filter Only", "M2: Asym+Prefilter"
    ]
    ax.set_xticklabels(short_labels, rotation=15, ha="right")
    ax.legend(frameon=True, loc="upper right", facecolor="white")

    # Add value labels
    for rect in rects1 + rects2:
        h = rect.get_height()
        va = "bottom" if h < 0 else "top"
        ax.annotate(f"{h:+.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, -12 if h < 0 else 3),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=8, fontweight="bold")

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out_file = output_dir / f"mot17_bdrate_bar{ext}"
        fig.savefig(out_file, bbox_inches="tight")
        saved.append(out_file)
    plt.close(fig)

    return saved


def generate_latex_tables(agg: dict,
                          overall_bd: dict[str, dict[str, dict]],
                          bootstrap_ci: dict[str, dict[str, dict]],
                          output_dir: Path) -> list[Path]:
    """Render journal-ready LaTeX tables for MOT17 benchmark ablation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = agg["variant_crf_summary"]
    saved: list[Path] = []

    # Table 1: Main Ablation Summary
    tex_path = output_dir / "table_mot17_ablation.tex"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation of PRISM-VCM spatial-QP and prefiltering enhancements on MOT17 tracking benchmark (YOLOv11-L + ByteTrack, GOP 32, Low-Delay P). BD-rates computed via monotonic PAVA-PCHIP relative to stock x265 anchor; negative values indicate rate reduction at equivalent tracking accuracy. 95\% confidence intervals via 2,000 paired bootstrap resamples across sequences.}",
        r"\label{tab:mot17_ablation}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Bitrate (kbps)} & \textbf{HOTA (\%)} & \textbf{MOTA (\%)} & \textbf{IDF1 (\%)} & \textbf{BD-Rate HOTA (\%)} & \textbf{BD-Rate MOTA (\%)} \\",
        r"\midrule",
    ]

    for v in VARIANTS:
        mean_rate = np.mean([summary[v][c]["weighted_bitrate_kbps"] for c in CRFS])
        mean_hota = np.mean([summary[v][c]["HOTA"] for c in CRFS])
        mean_mota = np.mean([summary[v][c]["MOTA"] for c in CRFS])
        mean_idf1 = np.mean([summary[v][c]["IDF1"] for c in CRFS])

        label = VARIANT_LABELS.get(v, v)
        if v == "anchor":
            bd_h_str = r"\textemdash"
            bd_m_str = r"\textemdash"
        else:
            bd_h = overall_bd.get(v, {}).get("HOTA", {}).get("pava_pchip")
            bd_m = overall_bd.get(v, {}).get("MOTA", {}).get("pava_pchip")
            ci_h = bootstrap_ci.get(v, {}).get("HOTA", {}).get("pava_pchip", {}).get("ci95", [None, None])
            ci_m = bootstrap_ci.get(v, {}).get("MOTA", {}).get("pava_pchip", {}).get("ci95", [None, None])

            if bd_h is not None and ci_h[0] is not None:
                bd_h_str = f"{bd_h:+.2f} [{ci_h[0]:+.2f}, {ci_h[1]:+.2f}]"
            elif bd_h is not None:
                bd_h_str = f"{bd_h:+.2f}"
            else:
                bd_h_str = "N/A"

            if bd_m is not None and ci_m[0] is not None:
                bd_m_str = f"{bd_m:+.2f} [{ci_m[0]:+.2f}, {ci_m[1]:+.2f}]"
            elif bd_m is not None:
                bd_m_str = f"{bd_m:+.2f}"
            else:
                bd_m_str = "N/A"

        row = f"{label} & {mean_rate:.1f} & {mean_hota:.2f} & {mean_mota:.2f} & {mean_idf1:.2f} & {bd_h_str} & {bd_m_str} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    saved.append(tex_path)

    # Table 2: Per-Sequence Breakdown for Key Models
    tex_seq_path = output_dir / "table_mot17_per_sequence.tex"
    lines2 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-sequence performance breakdown comparing Stock x265 Anchor with PRISM Asymmetric Prefilter on MOT17 validation sequences across CRF $\in \{22, 27, 32, 37\}$.}",
        r"\label{tab:mot17_per_sequence}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Stock x265 Anchor}} & \multicolumn{3}{c}{\textbf{PRISM (M2\_asym\_prefilter)}} & \multicolumn{2}{c}{\textbf{BD-Rate Savings}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9}",
        r"\textbf{Sequence} & \textbf{kbps} & \textbf{HOTA} & \textbf{MOTA} & \textbf{kbps} & \textbf{HOTA} & \textbf{MOTA} & \textbf{HOTA (\%)} & \textbf{MOTA (\%)} \\",
        r"\midrule",
    ]

    indexed = agg["indexed"]
    for s in agg["seq_list"]:
        r_anc = np.mean([indexed[s]["anchor"][c]["bitrate_kbps"] for c in CRFS])
        h_anc = np.mean([indexed[s]["anchor"][c]["metrics"]["HOTA"] for c in CRFS])
        m_anc = np.mean([indexed[s]["anchor"][c]["metrics"]["MOTA"] for c in CRFS])

        r_pri = np.mean([indexed[s]["M2_asym_prefilter"][c]["bitrate_kbps"] for c in CRFS])
        h_pri = np.mean([indexed[s]["M2_asym_prefilter"][c]["metrics"]["HOTA"] for c in CRFS])
        m_pri = np.mean([indexed[s]["M2_asym_prefilter"][c]["metrics"]["MOTA"] for c in CRFS])

        # Sequence BDR
        pts_a_h = [RDPoint(qp=c, score=indexed[s]["anchor"][c]["metrics"]["HOTA"],
                           bitstream_bytes=int(indexed[s]["anchor"][c]["bitrate_kbps"] * 1000)) for c in CRFS]
        pts_p_h = [RDPoint(qp=c, score=indexed[s]["M2_asym_prefilter"][c]["metrics"]["HOTA"],
                           bitstream_bytes=int(indexed[s]["M2_asym_prefilter"][c]["bitrate_kbps"] * 1000)) for c in CRFS]
        b_h = bjontegaard_bd_rate(pts_a_h, pts_p_h)
        bdr_h_str = f"{b_h['bd_rate']:+.2f}\\%" if b_h.get("valid") else "N/A"

        pts_a_m = [RDPoint(qp=c, score=indexed[s]["anchor"][c]["metrics"]["MOTA"],
                           bitstream_bytes=int(indexed[s]["anchor"][c]["bitrate_kbps"] * 1000)) for c in CRFS]
        pts_p_m = [RDPoint(qp=c, score=indexed[s]["M2_asym_prefilter"][c]["metrics"]["MOTA"],
                           bitstream_bytes=int(indexed[s]["M2_asym_prefilter"][c]["bitrate_kbps"] * 1000)) for c in CRFS]
        b_m = bjontegaard_bd_rate(pts_a_m, pts_p_m)
        bdr_m_str = f"{b_m['bd_rate']:+.2f}\\%" if b_m.get("valid") else "N/A"

        lines2.append(f"{s} & {r_anc:.1f} & {h_anc:.2f} & {m_anc:.2f} & {r_pri:.1f} & {h_pri:.2f} & {m_pri:.2f} & {bdr_h_str} & {bdr_m_str} \\\\")

    lines2.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    tex_seq_path.write_text("\n".join(lines2) + "\n", encoding="utf-8")
    saved.append(tex_seq_path)

    return saved


def merge_and_report(reports_dir: Path, output_dir: Path,
                     resamples: int = 2000,
                     allow_partial: bool = False) -> Path:
    """Run full aggregation pipeline and produce final JSON report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    found_reports = find_sequence_reports(reports_dir)
    print(f"[Merge] Found {len(found_reports)} MOT17 reports in {reports_dir}", flush=True)

    if not allow_partial and len(found_reports) < len(SEQUENCES):
        missing = set(SEQUENCES) - set(found_reports.keys())
        raise ValueError(f"Missing {len(missing)} sequence reports: {missing}")

    loaded_reports = {}
    for s, p in found_reports.items():
        data = read_json(p)
        validate_report(data, s)
        loaded_reports[s] = data

    agg = aggregate_observations(loaded_reports)
    seq_bd = compute_sequence_bd_rates(agg)
    overall_bd = compute_overall_bd_rates(agg)
    bootstrap_ci = compute_bootstrap_ci(loaded_reports, agg["seq_list"], resamples=resamples)

    # Render plots & LaTeX tables
    plots = plot_rd_curves(agg, output_dir)
    tables = generate_latex_tables(agg, overall_bd, bootstrap_ci, output_dir)

    final_report = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sequences_evaluated": agg["seq_list"],
        "num_sequences": len(agg["seq_list"]),
        "total_duration_seconds": agg["total_duration"],
        "crfs": list(CRFS),
        "variants": list(VARIANTS),
        "overall_summary": agg["variant_crf_summary"],
        "overall_bd_rates": overall_bd,
        "bootstrap_ci95": bootstrap_ci,
        "per_sequence_bd_rates": seq_bd,
        "artifacts_generated": [str(p.relative_to(output_dir).as_posix()) for p in plots + tables],
    }

    final_report_path = output_dir / "mot17_final_report.json"
    write_json(final_report_path, final_report)
    print(f"[Merge] Completed successfully. Summary written to {final_report_path}", flush=True)
    return final_report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    merge_and_report(args.reports_dir, args.output_dir, args.resamples, args.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
