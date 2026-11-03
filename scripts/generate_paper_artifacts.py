#!/usr/bin/env python3
"""Render pooled COCO results from Step 3 or a merged Tier 1b report.

Rates are whole-bitstream totals, never inferred kbps. RD points are seed means;
BD-rates are computed per seed against the shared M1, then averaged. MPEG and
PAVA-PCHIP are supplemental analyses, not replacements for the raw BD gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from prism.evaluation.metrics import RDPoint, bjontegaard_bd_rate, pava_pchip_bd_rate

DEFAULT_REPORT = REPO / "docs/reports/step3_report.json"
DEFAULT_OUTPUT_DIR = REPO / "docs/paper_artifacts"
ARTIFACT_NAMES = (
    "rd_curves.pdf", "rd_curves.png", "bd_rate_comparison.pdf",
    "bd_rate_comparison.png", "table_m1_m4_coco.tex", "table_cross_detector.tex",
)
CRFS = (20, 24, 28, 32, 36)
MODELS = ("M1", "M2", "M3", "M4")
DETECTORS = ("yolo11l", "faster_rcnn")
DETECTOR_NAMES = ("YOLOv11-L", "Faster R-CNN ResNet50-FPN v2")
LABELS = ("M1: Stock x265", "M2: Spatial QP only", r"M3: $\Delta_x$ only", "M4: PRISM-VCM Coupled")
COLORS = ("black", "#0072B2", "#009E73", "#C62828")
METHODS = {"bjontegaard": bjontegaard_bd_rate, "pava_pchip": pava_pchip_bd_rate}
METRICS = {"map": "ap_50_95", "ap50": "ap50"}

MODEL_LABELS = {
    "M1": "M1: Stock x265",
    "M2": "M2: Spatial QP (Zero-mean)",
    "M2_asym": "M2: Asymmetric [-4, 6]",
    "M2_asym_prefilter": "M2: Asym + Prefilter",
    "M3": r"M3: $\Delta_x$ only",
    "M4": "M4: PRISM-VCM Coupled",
}
MODEL_COLORS = {
    "M1": "black",
    "M2": "#0072B2",
    "M2_asym": "#009E73",
    "M2_asym_prefilter": "#D55E00",
    "M3": "#009E73",
    "M4": "#C62828",
}


def detect_models(report: dict) -> tuple[str, ...]:
    if "detectors" in report:
        return MODELS
    if "metrics" in report.get("results", {}):
        found = set()
        for det in DETECTORS:
            for key in report["results"]["metrics"].get(det, {}).keys():
                match = re.fullmatch(r"([A-Za-z0-9_]+?)(?:_seed\d+)?_crf\d+", key)
                if match:
                    found.add(match.group(1))
        order = ("M1", "M2", "M2_asym", "M2_asym_prefilter", "M3", "M4")
        res = tuple(m for m in order if m in found)
        return res if res else tuple(sorted(found))
    return MODELS


def finite_number(value):
    """JSON-safe numeric value; invalid BD result objects remain unavailable."""
    if isinstance(value, dict):
        if not value.get("valid", True):
            return None
        value = value.get("bd_rate", value.get("percent"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def normalize_report(report: dict) -> dict:
    """Return detector -> model -> seed -> CRF -> pooled metrics for both schemas."""
    active_models = detect_models(report)
    curves = {det: {model: {} for model in active_models} for det in DETECTORS}
    if "detectors" in report:
        for det in DETECTORS:
            source = report["detectors"][det]
            curves[det]["M1"][None] = source["m1_pooled_by_crf"]
            for model in active_models[1:]:
                curves[det][model] = dict(source[f"{model.lower()}_pooled_by_seed_and_crf"])
    elif "metrics" in report.get("results", {}):
        for det in DETECTORS:
            for key, row in report["results"]["metrics"][det].items():
                match = re.fullmatch(r"([A-Za-z0-9_]+?)(?:_seed(\d+))?_crf(\d+)", key)
                if not match:
                    raise ValueError(f"Unrecognized metric key: {det}/{key}")
                model, seed, crf = match.groups()
                if model not in active_models:
                    continue
                if (model == "M1") != (seed is None):
                    raise ValueError(f"Incorrect seed for {key}")
                curves[det][model].setdefault(seed, {})[crf] = row
    else:
        raise ValueError("Expected Step 3 detectors or Tier 1b results.metrics")

    for det in DETECTORS:
        seed_sets = []
        for model in active_models:
            seeds = curves[det][model]
            if not seeds:
                raise ValueError(f"No pooled metrics for {det}/{model}")
            if model != "M1":
                seed_sets.append(set(seeds))
            for seed, points in seeds.items():
                if {int(crf) for crf in points} != set(CRFS):
                    raise ValueError(f"Expected all five CRFs for {det}/{model}/{seed}")
                normalized = {}
                for crf, row in points.items():
                    size = finite_number(row.get("whole_bitstream_bytes", row.get("bytes")))
                    if size is None or size <= 0:
                        raise ValueError(f"Invalid bitstream bytes for {det}/{model}/{crf}")
                    result = {"bytes": size}
                    for metric in ("ap_50_95", "ap50", "ap_small"):
                        value = finite_number(row.get(metric))
                        # COCOeval uses -1 for an undefined area/category metric.
                        if value is not None and value != -1 and not 0 <= value <= 1:
                            raise ValueError(f"Expected COCO AP fraction for {metric}")
                        result[metric] = None if value == -1 else value
                    normalized[int(crf)] = result
                seeds[seed] = normalized
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            raise ValueError(f"Models have inconsistent seeds for {det}")
    if any(set(curves[DETECTORS[0]][m]) != set(curves[DETECTORS[1]][m]) for m in active_models):
        raise ValueError("Detectors have inconsistent seeds")
    return curves


def mean_or_none(values):
    values = list(values)
    # Do not silently turn a partial set of seeds into a complete mean.
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def mean_curves(curves: dict) -> dict:
    return {det: {model: {crf: {
        metric: mean_or_none(points[crf][metric] for points in seeds.values())
        for metric in ("bytes", "ap_50_95", "ap50", "ap_small")
    } for crf in CRFS} for model, seeds in models.items()} for det, models in curves.items()}


def compute_bd_rates(curves: dict, report: dict) -> dict:
    """Use recorded Tier 1b values when present; otherwise compute from pooled AP."""
    output = {}
    recorded = report.get("results", {}).get("bd_rates", {})
    active_models = tuple(curves[DETECTORS[0]].keys())
    for det in DETECTORS:
        output[det] = {}
        for model in [m for m in active_models if m != "M1"]:
            output[det][model] = {}
            saved = recorded.get(det, {}).get(model, {}).get("mean", {})
            for method, function in METHODS.items():
                for suffix, metric in METRICS.items():
                    key = f"{method}_bd_rate_{suffix}"
                    if key in saved:
                        output[det][model][key] = finite_number(saved[key])
                        continue
                    values = []
                    for candidate in curves[det][model].values():
                        anchor = curves[det]["M1"][None]
                        if any(points[c][metric] is None for points in (anchor, candidate) for c in CRFS):
                            values.append(None)
                            continue
                        a, b = ([RDPoint(c, points[c][metric], points[c]["bytes"]) for c in CRFS]
                                for points in (anchor, candidate))
                        try:
                            # Ill-conditioned polynomial fits must not become paper claims.
                            with warnings.catch_warnings():
                                warnings.simplefilter("error")
                                result = function(a, b)
                            values.append(finite_number(result))
                        except (ValueError, ArithmeticError, Warning, np.linalg.LinAlgError):
                            values.append(None)
                    output[det][model][key] = mean_or_none(values)
    return output


def publication_style():
    return {
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.18,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    }


def save_figure(fig, directory: Path, stem: str):
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight")


def plot_rd_curves(means: dict, output_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    active_models = tuple(means[DETECTORS[0]].keys())
    colors = [MODEL_COLORS.get(m, "#333333") for m in active_models]
    labels = [MODEL_LABELS.get(m, m) for m in active_models]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.5), sharey=True)
    for ax, det, title in zip(axes, DETECTORS,
                              ("YOLOv11-L (Target Detector)", "Faster R-CNN (Cross-Detector Transfer)")):
        for i, model in enumerate(active_models):
            points = sorted(means[det][model].values(), key=lambda p: p["bytes"])
            for metric, marker, fill in (("ap50", "o", colors[i]), ("ap_50_95", "s", "white")):
                ax.plot([p["bytes"] * 8 for p in points],
                        [p[metric] * 100 if p[metric] is not None else np.nan for p in points],
                        color=colors[i], linestyle="--" if model == "M1" else "-",
                        linewidth=2 if ("4" in model or "prefilter" in model) else 1.1,
                        marker=marker, markerfacecolor=fill, markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("Bitrate (bits over evaluation set)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    axes[0].set_ylabel("Detection AP (%)")
    handles = [Line2D([], [], color=c, linestyle="--" if i == 0 else "-",
                      linewidth=2 if ("4" in active_models[i] or "prefilter" in active_models[i]) else 1.1,
                      label=label)
               for i, (c, label) in enumerate(zip(colors, labels))]
    handles += [Line2D([], [], color="gray", marker="o", linestyle="", label="AP50"),
                Line2D([], [], color="gray", marker="s", markerfacecolor="white", linestyle="",
                       label="AP@[.50:.95]")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save_figure(fig, output_dir, "rd_curves")
    plt.close(fig)


def plot_bd_rates(bd_rates: dict, output_dir: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cand_models = tuple(m for m in bd_rates[DETECTORS[0]].keys() if m != "M1")
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.6), sharex=True)
    for row, (suffix, metric_label) in enumerate((("map", "AP@[.50:.95]"), ("ap50", "AP50"))):
        for col, (det, det_label) in enumerate(zip(DETECTORS, ("YOLOv11-L", "Faster R-CNN"))):
            ax = axes[row, col]
            for i, model in enumerate(cand_models):
                color = MODEL_COLORS.get(model, "#333333")
                for j, method in enumerate(METHODS):
                    value = bd_rates[det][model][f"{method}_bd_rate_{suffix}"]
                    x = i + (j - 0.5) * 0.34
                    if value is None:
                        ax.text(x, 0, "N/A", ha="center", va="bottom", rotation=90, fontsize=6)
                    else:
                        ax.bar(x, value, width=0.30, color=color,
                               hatch="///" if j else None, edgecolor="black", linewidth=0.4)
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set_xticks(range(len(cand_models)), cand_models)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(f"{det_label}: {metric_label}")
            ax.margins(y=0.25)
            if col == 0:
                ax.set_ylabel("BD-rate vs. M1 (%)")
    fig.legend(handles=[Patch(facecolor="white", edgecolor="black", label="MPEG Bjontegaard"),
                        Patch(facecolor="white", edgecolor="black", hatch="///", label="PAVA-PCHIP")],
               loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2)
    fig.text(0.5, 0.008, "Negative = rate saving; N/A = undefined. PAVA-PCHIP is a sensitivity analysis.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, output_dir, "bd_rate_comparison")
    plt.close(fig)


def latex_number(value, *, percent=False, reduction=False):
    if value is None:
        return r"\textemdash"
    value = value * 100 if percent else value
    return f"{-value if reduction else value:.2f}"


def write_tables(means: dict, bd_rates: dict, report: dict, output_dir: Path):
    count = report.get("summary", {}).get("total_unique_images")
    if count is None:
        count = report.get("split_hygiene", {}).get("counts_by_split", {}).get("pilot", {}).get("total")
    dataset = f"COCO val2017 ({int(count):,} images)" if count else "COCO evaluation set"
    note = (r"AP values are percentages. M2--M4 entries are equal-weight seed means; "
            r"M1 is shared across seeds. BD values compare full five-CRF curves with M1, "
            r"averaged over seeds; an undefined seed makes the mean unavailable (\textemdash). "
            r"MPEG and PAVA-PCHIP are supplemental analyses, not raw-gate decisions. ")
    gate = report.get("calibration_gate_audit", {})
    if gate.get("delta_x_forced_zero"):
        note += r"The calibration gate forced $\Delta_x=0$ in this run. "
    start = [r"% Generated by scripts/generate_paper_artifacts.py; requires \usepackage{booktabs}.",
             r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}"]
    lines = start + [r"\caption{" + dataset + r" with YOLOv11-L. Bitstream bytes are totals over the "
                     r"evaluation set. BD reduction (positive = saving) is shown once per model "
                     r"for AP@[.50:.95], in MPEG / PAVA-PCHIP order. " + note + "}",
                     r"\label{tab:m1-m4-coco}", r"\begin{tabular}{llrrrrr}", r"\toprule",
                     r"Model & CRF & Bitstream bytes & mAP@[.50:.95] & AP50 & AP$_{\rm small}$ & BD reduction (\%) \\",
                     r"\midrule"]
    active_models = tuple(means["yolo11l"].keys())
    for i, model in enumerate(active_models):
        if i:
            lines.append(r"\midrule")
        for crf in CRFS:
            p = means["yolo11l"][model][crf]
            bd = ""
            if crf == CRFS[0]:
                bd = "0.00 / 0.00" if model == "M1" else " / ".join(
                    latex_number(bd_rates["yolo11l"][model][f"{method}_bd_rate_map"], reduction=True)
                    for method in METHODS)
            cells = [r"\textbf{" + model + r"}" if ("4" in model or "prefilter" in model) else model, str(crf), f"{p['bytes']:,.0f}"]
            cells += [latex_number(p[m], percent=True) for m in ("ap_50_95", "ap50", "ap_small")]
            lines.append(" & ".join(cells + [bd]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output_dir / "table_m1_m4_coco.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = start + [r"\caption{Cross-detector transfer from YOLOv11-L to "
                     r"Faster R-CNN ResNet50-FPN v2 on " + dataset + ". "
                     r"AP columns are arithmetic means over the five CRFs (20, 24, 28, 32, 36). "
                     r"BD-rate columns use AP@[.50:.95]; negative values indicate savings. " + note + "}",
                     r"\label{tab:cross-detector}", r"\begin{tabular}{llrrrrr}", r"\toprule",
                     r"Detector & Model & mAP@[.50:.95] & AP50 & AP$_{\rm small}$ & MPEG (\%) & PAVA (\%) \\",
                     r"\midrule"]
    for i, (det, name) in enumerate(zip(DETECTORS, ("YOLOv11-L", "Faster R-CNN"))):
        if i:
            lines.append(r"\midrule")
        for model in active_models:
            cells = [name, r"\textbf{" + model + r"}" if ("4" in model or "prefilter" in model) else model]
            cells += [latex_number(mean_or_none(means[det][model][c][m] for c in CRFS), percent=True)
                      for m in ("ap_50_95", "ap50", "ap_small")]
            cells += ["0.00" if model == "M1" else latex_number(bd_rates[det][model][f"{method}_bd_rate_map"])
                      for method in METHODS]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output_dir / "table_cross_detector.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_artifacts(report_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(report_path.read_text(encoding="utf-8"))
    curves = normalize_report(report)
    means = mean_curves(curves)
    bd_rates = compute_bd_rates(curves, report)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(publication_style()):
        plot_rd_curves(means, output_dir)
        plot_bd_rates(bd_rates, output_dir)
    write_tables(means, bd_rates, report, output_dir)
    return [output_dir / name for name in ARTIFACT_NAMES]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        paths = generate_artifacts(args.report.expanduser(), args.output_dir.expanduser())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"[Paper artifacts] {exc}\n")
    print(f"[Paper artifacts] Wrote {len(paths)} files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
