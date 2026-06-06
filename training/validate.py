"""
Evaluate the fine-tuned model on the held-out test split.

Reports mAP50, mAP50-95, precision, recall, and per-class mAP50, so you know
your real detection performance before evaluation day.

Run from project root:
    python training/validate.py
"""

import os
import sys

from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import IMG_SIZE, CONF_LOW, IOU_THRESHOLD

WEIGHTS = 'weights/best.pt'
DATA_YAML = 'training/data.yaml'


def validate():
    if not os.path.exists(WEIGHTS):
        print(f"[ERROR] No weights at {WEIGHTS}. Train first.")
        sys.exit(1)

    model = YOLO(WEIGHTS)
    metrics = model.val(
        data    = DATA_YAML,
        split   = 'test',
        imgsz   = IMG_SIZE,
        conf    = CONF_LOW,
        iou     = IOU_THRESHOLD,
        verbose = True,
    )

    print("\n" + "=" * 40)
    print("TEST SET RESULTS")
    print("=" * 40)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("\nPer-class mAP50:")
    for i, name in model.names.items():
        try:
            print(f"  {name}: {metrics.box.ap50[i]:.4f}")
        except (IndexError, TypeError):
            print(f"  {name}: n/a")


if __name__ == '__main__':
    validate()
