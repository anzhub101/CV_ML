"""
Make the (near-black) instance masks visible, and confirm they line up with
the source images.

The masks store instance ids 0,1,2,3..., which look black in a viewer. This
script renders, for a few samples per class:
    outputs/mask_<name>.png     - the mask recolored so each instance id is a
                                  distinct bright color (this is the proper way
                                  to "see" the mask, instead of multiplying by
                                  255 which would overflow for ids > 1)
    outputs/overlay_<name>.png  - the colored mask blended over the source image
                                  with the derived YOLO boxes drawn on top

Use this to sanity-check the annotations BEFORE converting/training.

Run from project root:
    python preprocessing/visualize_masks.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import (
    CLASS_MAP, RAW_DIR, ANNOTATION_DIR, resolve_class_dir,
)
from preprocessing.convert_annotations import mask_to_yolo_bbox

N_PER_CLASS = 3
OUT_DIR = 'outputs'


def colorize_mask(mask):
    """Map instance ids -> distinct bright colors (0 stays black)."""
    if mask.max() == 0:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Scale ids across the full 0-255 range, then apply a color map so each
    # id is clearly distinguishable regardless of how small the raw values are.
    scaled = (mask.astype(np.float32) / mask.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colored[mask == 0] = (0, 0, 0)
    return colored


def visualize_one(img_path, mask_path, out_stem):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  [WARN] could not read {out_stem}")
        return
    h, w = img.shape[:2]
    if (mask.shape[1], mask.shape[0]) != (w, h):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ids = sorted(int(v) for v in np.unique(mask) if v > 0)
    colored = colorize_mask(mask)
    cv2.imwrite(os.path.join(OUT_DIR, f'mask_{out_stem}.png'), colored)

    # Overlay + derived boxes
    overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    for (xc, yc, bw, bh) in mask_to_yolo_bbox(mask_path, w, h):
        x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, f'overlay_{out_stem}.png'), overlay)
    print(f"  {out_stem}: instance ids {ids} -> {len(ids)} object(s)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for class_name in CLASS_MAP:
        img_dir  = resolve_class_dir(RAW_DIR, class_name)
        mask_dir = resolve_class_dir(ANNOTATION_DIR, class_name)
        if img_dir is None or mask_dir is None:
            print(f"[SKIP] {class_name}: staged data not found "
                  f"(run preprocessing/setup_data.py first)")
            continue

        print(f"\n=== {class_name} ===")
        masks = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))
        for mfile in masks[:N_PER_CLASS]:
            stem = os.path.splitext(mfile)[0]
            # Source image may be .png/.jpg/.jpeg
            img_path = None
            for ext in ('.png', '.jpg', '.jpeg'):
                cand = os.path.join(img_dir, stem + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is None:
                print(f"  [WARN] no source image for {stem}")
                continue
            visualize_one(img_path, os.path.join(mask_dir, mfile),
                          f'{class_name}_{stem}')

    print(f"\nWrote mask_* and overlay_* images to {OUT_DIR}/ — open them to "
          f"confirm the boxes sit on the weapons.")


if __name__ == '__main__':
    main()
