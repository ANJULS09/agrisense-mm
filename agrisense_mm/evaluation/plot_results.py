"""
AgriSense-MM: Automated Plot Generation.
Generates:
1. Ablation comparison bar chart (Accuracy, F1, Severity MAE)
2. Normalized Confusion Matrix heatmap
3. ECE Reliability / Calibration curve
4. Missing modality robustness retention chart
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from agrisense_mm.data.constants import STRESS_CLASSES


def generate_all_evaluation_plots(
    results_json_path: str = "evaluation/results/ablation_results.json",
    output_dir: str = "evaluation/plots",
):
    """Reads evaluation results and generates publication-grade visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_json_path):
        print(f"Results file {results_json_path} not found. Skipping plot generation.")
        return

    with open(results_json_path, "r") as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Ablation Comparison Bar Chart
    configs = ["1_Vision_Only", "2_Vision_Plus_Env", "3_Vision_Env_Motion", "4_Full_AgriSense_MM"]
    labels = ["Vision Only\n(B0)", "Vision + Env\n(Bi-LSTM)", "+ Motion Gating\n(1D-CNN)", "Full AgriSense\n(+GNN)"]

    f1_scores = []
    acc_scores = []
    sev_maes = []

    for c in configs:
        if c in data:
            f1_scores.append(data[c]["classification"]["f1_macro"] * 100.0)
            acc_scores.append(data[c]["classification"]["accuracy"] * 100.0)
            sev_maes.append(data[c]["severity"]["mae"])
        else:
            f1_scores.append(0.0)
            acc_scores.append(0.0)
            sev_maes.append(0.0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.28

    rects1 = ax1.bar(x - width/2, acc_scores, width, label="Accuracy (%)", color="#2ecc71")
    rects2 = ax1.bar(x + width/2, f1_scores, width, label="Macro F1 (%)", color="#3498db")

    ax1.set_ylabel("Classification Score (%)", fontweight="bold")
    ax1.set_title("AgriSense-MM: Ablation Study Across Model Configurations", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper left")

    # Annotate values
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ablation_plot = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(ablation_plot, dpi=200)
    plt.close()
    print(f"[+] Saved ablation chart to {ablation_plot}")

    # 2. Confusion Matrix Heatmap (from Full Model)
    full_cfg = data.get("4_Full_AgriSense_MM") or data.get("1_Vision_Only")
    if full_cfg and "confusion_matrix" in full_cfg["classification"]:
        cm = np.array(full_cfg["classification"]["confusion_matrix"])
        # Normalize by row
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm.astype(float)), where=row_sums != 0)

        plt.figure(figsize=(8, 6.5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Greens",
            xticklabels=[c.replace(" ", "\n") for c in STRESS_CLASSES],
            yticklabels=STRESS_CLASSES,
            cbar=True,
        )
        plt.title("AgriSense-MM: Normalized Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Class", fontweight="bold")
        plt.ylabel("Ground Truth Class", fontweight="bold")
        plt.tight_layout()
        cm_plot = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_plot, dpi=200)
        plt.close()
        print(f"[+] Saved confusion matrix to {cm_plot}")

    # 3. Calibration Curve / ECE Diagram
    if full_cfg and "calibration_bins" in full_cfg["classification"]:
        bins = full_cfg["classification"]["calibration_bins"]
        confs = np.array(bins["confidences"])
        accs = np.array(bins["accuracies"])
        ece = full_cfg["classification"]["ece"]

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        plt.plot(confs, accs, "s-", color="#e74c3c", linewidth=2, label=f"AgriSense-MM (ECE={ece:.3f})")
        plt.bar(confs, accs, width=0.08, alpha=0.25, color="#e74c3c", edgecolor="black")

        plt.xlabel("Confidence (Predicted Probability)", fontweight="bold")
        plt.ylabel("Empirical Accuracy", fontweight="bold")
        plt.title("Uncertainty Reliability Diagram (ECE)", fontsize=13, fontweight="bold")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc="upper left")
        plt.tight_layout()
        cal_plot = os.path.join(output_dir, "calibration_curve.png")
        plt.savefig(cal_plot, dpi=200)
        plt.close()
        print(f"[+] Saved calibration curve to {cal_plot}")

    # 4. Missing Modality Robustness Chart
    if "missing_modality_robustness" in data:
        mod_data = data["missing_modality_robustness"]
        scenarios = list(mod_data.keys())
        scen_labels = [s.replace("_", " ").title() for s in scenarios]
        accs = [mod_data[s]["accuracy"] * 100.0 for s in scenarios]

        plt.figure(figsize=(9, 4.5))
        bars = plt.barh(scen_labels, accs, color=["#2ecc71", "#e67e22", "#3498db", "#9b59b6"], height=0.55)
        plt.xlim(0, 105)
        plt.xlabel("Retained Accuracy (%)", fontweight="bold")
        plt.title("Missing Modality Robustness (Sensor Drop Resistance)", fontsize=13, fontweight="bold")

        for bar in bars:
            w = bar.get_width()
            plt.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontweight="bold")

        plt.tight_layout()
        mod_plot = os.path.join(output_dir, "modality_robustness.png")
        plt.savefig(mod_plot, dpi=200)
        plt.close()
        print(f"[+] Saved modality robustness plot to {mod_plot}")


if __name__ == "__main__":
    generate_all_evaluation_plots()

