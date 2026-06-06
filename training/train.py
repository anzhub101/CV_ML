"""
Fine-tune YOLO26 on the baggage threat dataset.

Strategy:
  - Start from pretrained yolo26s.pt (COCO transfer learning).
  - Set a HIGH epoch ceiling and let `patience` (early stopping) decide when
    to actually stop — the optimal epoch count depends on dataset size and
    complexity, not the model, so we don't hard-code a magic number.
  - batch=-1 auto-selects the largest batch that fits the GPU.
  - Optionally freeze the backbone (freeze=10) for very small datasets.

Run from project root:
    python training/train.py

If YOLO26 is unavailable in your environment, change MODEL_WEIGHTS to
'yolo11s.pt' — the rest of the script is identical.
"""

import os
import shutil
import sys

import torch
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import IMG_SIZE


# ----- Configuration ---------------------------------------------------------
MODEL_WEIGHTS = 'yolo26s.pt'        # fallback: 'yolo11s.pt'
DATA_YAML     = 'training/data.yaml'
EPOCHS        = 150                 # HIGH CEILING — patience ends the run
PATIENCE      = 25                  # stop if no val improvement for 25 epochs
BATCH         = -1                  # auto-pick batch size
PROJECT_DIR   = 'training/runs'
RUN_NAME      = 'baggage_v1'
FREEZE        = None                # set to 10 to freeze the backbone


def train():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Device:        {device}")
    print(f"Base weights:  {MODEL_WEIGHTS}")
    print(f"Data config:   {DATA_YAML}")
    print(f"Epoch ceiling: {EPOCHS}  (patience={PATIENCE})\n")

    model = YOLO(MODEL_WEIGHTS)

    train_kwargs = dict(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        patience    = PATIENCE,
        imgsz       = IMG_SIZE,
        batch       = BATCH,
        device      = device,
        pretrained  = True,
        optimizer   = 'auto',       # AdamW for short runs, MuSGD for long
        project     = PROJECT_DIR,
        name        = RUN_NAME,
        exist_ok    = True,
        save        = True,
        save_period = 10,
        workers     = 4,
        verbose     = True,
        # Built-in augmentation (stacks on top of albumentations copies)
        mosaic      = 1.0,
        mixup       = 0.1,
        degrees     = 10.0,
        flipud      = 0.2,
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        scale       = 0.5,
    )
    if FREEZE is not None:
        train_kwargs['freeze'] = FREEZE

    results = model.train(**train_kwargs)

    # Copy the best checkpoint to a stable, easy-to-find location
    best_src = os.path.join(PROJECT_DIR, RUN_NAME, 'weights', 'best.pt')
    os.makedirs('weights', exist_ok=True)
    if os.path.exists(best_src):
        shutil.copy2(best_src, 'weights/best.pt')

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("Best weights -> weights/best.pt")
    try:
        print(f"Val mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    except Exception:
        pass
    print("=" * 50)


if __name__ == '__main__':
    train()
