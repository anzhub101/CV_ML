"""
Submission formatter — generates, validates, and summarises the final CSV.

Project spec format:
    Image Name,Predicted Label
    img1.jpg,gun
    img2.jpg,safe

Usage:
    from inference.submission import SubmissionWriter
    sw = SubmissionWriter()
    sw.write(predictions, "outputs/predictions.csv")
    sw.validate("outputs/predictions.csv")
    sw.print_summary("outputs/predictions.csv")
"""

import csv
import os
import sys
from collections import Counter
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

log = get_logger(__name__)

VALID_LABELS   = {"safe", "gun", "knife", "shuriken"}
CSV_FIELDNAMES = ["Image Name", "Predicted Label"]


class SubmissionWriter:

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, predictions: List[dict], path: str = "outputs/predictions.csv"):
        """
        Write predictions to the submission CSV.
        predictions: list of dicts with keys 'image_name' and 'pred_label'.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for p in predictions:
                writer.writerow({
                    "Image Name":      p["image_name"],
                    "Predicted Label": p["pred_label"],
                })

        log.info(f"Submission CSV written -> {path}  ({len(predictions)} rows)")

    # ── Validate ──────────────────────────────────────────────────────────────

    def validate(self, path: str) -> bool:
        """
        Check the CSV is correctly formatted.
        Returns True if valid, False + logs errors otherwise.
        """
        if not os.path.exists(path):
            log.error(f"Submission file not found: {path}")
            return False

        errors = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_FIELDNAMES:
                errors.append(
                    f"Wrong header: got {reader.fieldnames}, "
                    f"expected {CSV_FIELDNAMES}"
                )
            for i, row in enumerate(reader, start=2):
                label = row.get("Predicted Label", "").strip().lower()
                if label not in VALID_LABELS:
                    errors.append(
                        f"Row {i}: invalid label '{label}' "
                        f"(valid: {sorted(VALID_LABELS)})"
                    )
                if not row.get("Image Name", "").strip():
                    errors.append(f"Row {i}: empty Image Name")

        if errors:
            log.error(f"Submission validation FAILED ({len(errors)} errors):")
            for e in errors[:10]:
                log.error(f"  {e}")
            return False

        log.info(f"Submission validation PASSED: {path}")
        return True

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self, path: str):
        """Print a breakdown of how many images were predicted per class."""
        if not os.path.exists(path):
            log.error(f"File not found: {path}")
            return

        labels = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row.get("Predicted Label", "").strip())

        counts = Counter(labels)
        total  = sum(counts.values())

        print("\n" + "=" * 40)
        print("  SUBMISSION SUMMARY")
        print("=" * 40)
        print(f"  Total images: {total}")
        for lbl in ["safe", "gun", "knife", "shuriken"]:
            n = counts.get(lbl, 0)
            pct = 100 * n / total if total else 0
            print(f"  {lbl:<12}: {n:>4}  ({pct:5.1f}%)")
        print("=" * 40)
