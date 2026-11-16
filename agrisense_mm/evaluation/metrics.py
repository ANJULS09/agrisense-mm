"""
AgriSense-MM: Quantitative Evaluation Metrics & Calibration.
Computes:
- Classification: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix
- Regression: MAE, RMSE, R^2
- Uncertainty Calibration: Expected Calibration Error (ECE) & Reliability Diagrams
- Routing Efficiency: Distance, Time, and High-Risk Discovery Rate
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Dict, List, Tuple, Any


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    Measures reliability of model confidence vs empirical accuracy.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2.0)

    return float(ece), np.array(bin_accs), np.array(bin_confs)


def evaluate_multitask_predictions(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_probs_cls: np.ndarray,
    y_true_sev: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_fc3: np.ndarray,
    y_pred_fc3: np.ndarray,
    y_true_fc7: np.ndarray,
    y_pred_fc7: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-task performance metrics.
    """
    # 1. Stress Classification
    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true_cls, y_pred_cls).tolist()
    ece, bin_accs, bin_confs = compute_expected_calibration_error(y_probs_cls, y_true_cls)

    # 2. Severity Regression
    mae = float(mean_absolute_error(y_true_sev, y_pred_sev))
    rmse = float(np.sqrt(mean_squared_error(y_true_sev, y_pred_sev)))
    r2 = float(r2_score(y_true_sev, y_pred_sev)) if len(y_true_sev) > 2 else 0.0

    # 3. Forecast Heads
    fc3_acc = float(accuracy_score(y_true_fc3, y_pred_fc3))
    fc3_f1 = float(f1_score(y_true_fc3, y_pred_fc3, average="macro", zero_division=0))
    fc7_acc = float(accuracy_score(y_true_fc7, y_pred_fc7))
    fc7_f1 = float(f1_score(y_true_fc7, y_pred_fc7, average="macro", zero_division=0))

    return {
        "classification": {
            "accuracy": round(acc, 4),
            "precision_macro": round(float(p), 4),
            "recall_macro": round(float(r), 4),
            "f1_macro": round(float(f1), 4),
            "ece": round(ece, 4),
            "confusion_matrix": cm,
            "calibration_bins": {
                "accuracies": bin_accs.tolist(),
                "confidences": bin_confs.tolist(),
            },
        },
        "severity": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
        },
        "forecasting": {
            "t3_acc": round(fc3_acc, 4),
            "t3_f1": round(fc3_f1, 4),
            "t7_acc": round(fc7_acc, 4),
            "t7_f1": round(fc7_f1, 4),
        },
    }
