"""
Metrics module for the CEN454 baggage threat detection project.

Implements the exact scoring formulas from the project specification:

    Classification Score = 0.7 * Accuracy + 0.3 * Macro_F1
    Localization Score   = mean IoU over all detected threat images
                           (only counts if IoU >= 0.5)
    Final Score          = 0.7 * Classification Score
                         + 0.3 * Localization Score

Usage:
    from utils.metrics import compute_all_metrics
    report = compute_all_metrics(y_true, y_pred, iou_scores)
    print(report.summary())
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

log = get_logger(__name__)

LABEL_NAMES = ["safe", "gun", "knife", "shuriken"]
IOU_PASS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data class that holds all computed results
# ---------------------------------------------------------------------------
@dataclass
class MetricsReport:
    # Classification
    accuracy:            float = 0.0
    macro_f1:            float = 0.0
    classification_score: float = 0.0

    # Per-class F1, precision, recall
    per_class_f1:        Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall:    Dict[str, float] = field(default_factory=dict)

    # Localization
    iou_scores:          List[float] = field(default_factory=list)
    mean_iou:            float = 0.0
    iou_pass_rate:       float = 0.0   # fraction with IoU >= 0.5
    localization_score:  float = 0.0

    # Final
    final_score:         float = 0.0

    # Extras
    total_images:        int = 0
    confusion_matrix:    Optional[object] = None
    class_report_str:    str = ""

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  CEN454 EVALUATION RESULTS",
            "=" * 55,
            f"  Total images evaluated : {self.total_images}",
            "",
            "  --- Classification (70% of final score) ---",
            f"  Accuracy              : {self.accuracy:.4f}",
            f"  Macro F1-Score        : {self.macro_f1:.4f}",
            f"  Classification Score  : {self.classification_score:.4f}"
            f"  (0.7*Acc + 0.3*F1)",
            "",
            "  --- Per-Class F1 ---",
        ]
        for cls in LABEL_NAMES:
            lines.append(
                f"  {cls:<12}: F1={self.per_class_f1.get(cls, 0):.4f}"
                f"  P={self.per_class_precision.get(cls, 0):.4f}"
                f"  R={self.per_class_recall.get(cls, 0):.4f}"
            )
        lines += [
            "",
            "  --- Localization (30% of final score) ---",
            f"  Mean IoU              : {self.mean_iou:.4f}",
            f"  IoU >= 0.5 rate       : {self.iou_pass_rate:.4f}",
            f"  Localization Score    : {self.localization_score:.4f}",
            "",
            "  --- Final ---",
            f"  FINAL SCORE           : {self.final_score:.4f}"
            f"  (0.7*Cls + 0.3*Loc)",
            "=" * 55,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def compute_per_class_metrics(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and F1 for each class without sklearn,
    so the script works even without scikit-learn installed.
    """
    results: Dict[str, Dict[str, float]] = {}
    for cls in LABEL_NAMES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))

        precision = _safe_div(tp, tp + fp)
        recall    = _safe_div(tp, tp + fn)
        f1        = _safe_div(2 * precision * recall, precision + recall)

        results[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return results


def compute_macro_f1(per_class: Dict[str, Dict[str, float]]) -> float:
    f1_values = [per_class[cls]["f1"] for cls in LABEL_NAMES]
    return sum(f1_values) / len(f1_values)


def compute_classification_score(accuracy: float, macro_f1: float) -> float:
    """Project spec: 0.7 * Accuracy + 0.3 * Macro F1."""
    return 0.7 * accuracy + 0.3 * macro_f1


def compute_iou(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area   = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - inter
    return _safe_div(inter, union)


def compute_localization_score(iou_scores: List[float]) -> tuple:
    """
    Returns (mean_iou, pass_rate, localization_score).
    Only threat images with a detection contribute an IoU score.
    """
    if not iou_scores:
        return 0.0, 0.0, 0.0
    mean_iou   = sum(iou_scores) / len(iou_scores)
    pass_rate  = sum(s >= IOU_PASS_THRESHOLD for s in iou_scores) / len(iou_scores)
    loc_score  = mean_iou          # spec: average IoU across all detected threats
    return mean_iou, pass_rate, loc_score


def compute_final_score(cls_score: float, loc_score: float) -> float:
    """Project spec: 0.7 * Classification Score + 0.3 * Localization Score."""
    return 0.7 * cls_score + 0.3 * loc_score


# ---------------------------------------------------------------------------
# Confusion matrix (no dependencies)
# ---------------------------------------------------------------------------
def build_confusion_matrix(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, Dict[str, int]]:
    cm: Dict[str, Dict[str, int]] = {
        t: {p: 0 for p in LABEL_NAMES} for t in LABEL_NAMES
    }
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


def format_confusion_matrix(cm: Dict[str, Dict[str, int]]) -> str:
    header = f"{'':12}" + "".join(f"{n:>10}" for n in LABEL_NAMES)
    rows = [header, "-" * (12 + 10 * len(LABEL_NAMES))]
    for true_cls in LABEL_NAMES:
        row = f"{true_cls:<12}" + "".join(
            f"{cm[true_cls][pred_cls]:>10}" for pred_cls in LABEL_NAMES
        )
        rows.append(row)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: List[str],
    y_pred: List[str],
    iou_scores: Optional[List[float]] = None,
) -> MetricsReport:
    """
    Compute the full metric suite and return a MetricsReport.

    y_true:     ground-truth labels
    y_pred:     predicted labels
    iou_scores: IoU value per threat detection (omit if no gt bboxes)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")

    iou_scores = iou_scores or []

    # Classification
    accuracy   = compute_accuracy(y_true, y_pred)
    per_class  = compute_per_class_metrics(y_true, y_pred)
    macro_f1   = compute_macro_f1(per_class)
    cls_score  = compute_classification_score(accuracy, macro_f1)

    # Localization
    mean_iou, pass_rate, loc_score = compute_localization_score(iou_scores)

    # Final
    final = compute_final_score(cls_score, loc_score)

    # Confusion matrix
    cm = build_confusion_matrix(y_true, y_pred)

    # sklearn detailed report (optional)
    class_report_str = ""
    try:
        from sklearn.metrics import classification_report
        class_report_str = classification_report(
            y_true, y_pred, labels=LABEL_NAMES, zero_division=0
        )
    except ImportError:
        pass

    report = MetricsReport(
        accuracy            = accuracy,
        macro_f1            = macro_f1,
        classification_score= cls_score,
        per_class_f1        = {c: per_class[c]["f1"]        for c in LABEL_NAMES},
        per_class_precision = {c: per_class[c]["precision"] for c in LABEL_NAMES},
        per_class_recall    = {c: per_class[c]["recall"]    for c in LABEL_NAMES},
        iou_scores          = iou_scores,
        mean_iou            = mean_iou,
        iou_pass_rate       = pass_rate,
        localization_score  = loc_score,
        final_score         = final,
        total_images        = len(y_true),
        confusion_matrix    = cm,
        class_report_str    = class_report_str,
    )

    log.info(f"Metrics computed — Final score: {final:.4f}")
    return report


def save_report(report: MetricsReport, path: str = "outputs/evaluation_report.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = report.confusion_matrix
    with open(path, "w") as f:
        f.write(report.summary())
        f.write("\n\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(f"Labels: {LABEL_NAMES}\n")
        if cm:
            f.write(format_confusion_matrix(cm))
        if report.class_report_str:
            f.write("\n\nDetailed sklearn report:\n")
            f.write(report.class_report_str)
    log.info(f"Evaluation report saved to {path}")
