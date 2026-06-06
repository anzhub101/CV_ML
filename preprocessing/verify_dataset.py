"""
Verify the YOLO dataset is correctly formatted BEFORE training.

Checks, per split (train/val/test):
  - image and label folders exist
  - every image has a matching .txt label
  - label lines have exactly 5 columns
  - class_id is within range
  - all coordinates are within [0, 1]
  - counts empty (safe) labels

Run from project root:
    python preprocessing/verify_dataset.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import CLASS_MAP

DATASET_ROOT = 'data/dataset'
SPLITS = ('train', 'val', 'test')
NUM_CLASSES = len(CLASS_MAP)


def verify():
    all_ok = True

    for split in SPLITS:
        img_dir = os.path.join(DATASET_ROOT, 'images', split)
        lbl_dir = os.path.join(DATASET_ROOT, 'labels', split)

        print(f"\n=== {split.upper()} ===")

        if not os.path.exists(img_dir):
            print(f"  [FAIL] Missing folder: {img_dir}")
            all_ok = False
            continue
        if not os.path.exists(lbl_dir):
            print(f"  [FAIL] Missing folder: {lbl_dir}")
            all_ok = False
            continue

        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")

        # Every image must have a matching label
        missing = [
            img for img in images
            if not os.path.exists(
                os.path.join(lbl_dir, os.path.splitext(img)[0] + '.txt')
            )
        ]
        if missing:
            print(f"  [FAIL] {len(missing)} images have NO label file")
            print(f"         e.g. {missing[:3]}")
            all_ok = False
        else:
            print("  [OK] Every image has a matching label file")

        # Validate label contents
        bad, empty = [], 0
        for lbl in labels:
            with open(os.path.join(lbl_dir, lbl)) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    bad.append((lbl, "wrong column count")); break
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    bad.append((lbl, "non-numeric value")); break
                if not (0 <= cls < NUM_CLASSES):
                    bad.append((lbl, f"class_id {cls} out of range")); break
                if any(c < 0 or c > 1 for c in coords):
                    bad.append((lbl, "coordinate outside [0,1]")); break

        print(f"  Empty labels (safe images): {empty}")
        if bad:
            print(f"  [FAIL] {len(bad)} malformed label files")
            for name, reason in bad[:3]:
                print(f"         {name}: {reason}")
            all_ok = False
        else:
            print("  [OK] All label files well-formed")

    print("\n" + "=" * 40)
    print("DATASET VERIFIED — ready to train" if all_ok
          else "PROBLEMS FOUND — fix before training")
    print("=" * 40)
    return all_ok


if __name__ == '__main__':
    ok = verify()
    sys.exit(0 if ok else 1)
