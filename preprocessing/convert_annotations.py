"""
Convert instance-indexed PNG segmentation masks into YOLO bounding-box labels.

IMPORTANT — mask format
-----------------------
The annotation PNGs are NOT 0/255 binary images. They are INSTANCE-INDEXED
masks: pixel value 0 is background and each separate object instance is painted
with its own small integer id (1, 2, 3, ...). Those ids are tiny next to 255,
so the masks look almost solid black when opened in an image viewer — but they
are not empty.

A naive `cv2.threshold(mask, 127, 255)` would wipe every value 1-4 to zero and
produce empty labels (this was the original bug). Instead we treat any pixel
>= MASK_FOREGROUND_MIN (1) as foreground and convert EACH unique non-zero id
into its own bounding box, so two overlapping/adjacent objects of the same
class stay as two boxes. The class id comes from the folder the mask lives in.

Output per image (one line per object):
    class_id  x_center  y_center  width  height   (all normalized to [0, 1])

Safe images receive an empty .txt file (no detections = background = safe).

Run from project root:
    python preprocessing/convert_annotations.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import (
    CLASS_MAP, RAW_DIR, ANNOTATION_DIR, ALL_LABELS_DIR, SAFE_CLASS,
    MASK_FOREGROUND_MIN, MIN_CONTOUR_AREA, resolve_class_dir,
)


def get_image_dimensions(image_path):
    """Return (width, height) of an image, or raise if unreadable."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img.shape[1], img.shape[0]


def find_multiclass_stems(raw_dir):
    """
    Return the set of filename stems that appear under MORE THAN ONE threat
    class folder.

    In this dataset, a multi-threat scan (e.g. one bag holding both a gun and a
    knife) is filed under every class folder it belongs to, but ships a single
    shared instance mask that does NOT say which instance is which class. Those
    images are therefore ambiguous to label per-class, so we skip them.
    """
    stem_classes = {}
    for class_name in CLASS_MAP:
        img_dir = resolve_class_dir(raw_dir, class_name)
        if img_dir is None:
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(f)[0]
                stem_classes.setdefault(stem, set()).add(class_name)
    return {stem for stem, classes in stem_classes.items() if len(classes) > 1}


def _instance_bbox(instance_mask, min_contour_area):
    """
    Given a uint8 binary mask for a SINGLE instance, return one (x1, y1, x2, y2)
    pixel box covering all of its (cleaned) contours, or None if it is only
    speckle noise below the area threshold.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(instance_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    if not contours:
        return None

    x1 = y1 = np.inf
    x2 = y2 = -np.inf
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1, y1 = min(x1, x), min(y1, y)
        x2, y2 = max(x2, x + w), max(y2, y + h)
    return int(x1), int(y1), int(x2), int(y2)


def mask_to_yolo_bbox(mask_path, image_w, image_h,
                      min_contour_area=MIN_CONTOUR_AREA):
    """
    Convert a single instance-indexed mask into normalized YOLO bboxes.

    Returns a list of (x_center, y_center, width, height) tuples, one per
    object instance, all normalized to the [0, 1] range.
    """
    # Read as grayscale; the three RGB channels are identical, so channel 0 is
    # enough and the instance ids survive (IMREAD_GRAYSCALE keeps the values).
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Align mask to the source image resolution. NEAREST keeps the integer ids
    # crisp (no interpolation that would invent in-between values).
    if (mask.shape[1], mask.shape[0]) != (image_w, image_h):
        mask = cv2.resize(mask, (image_w, image_h),
                          interpolation=cv2.INTER_NEAREST)

    bboxes = []
    instance_ids = [v for v in np.unique(mask) if v >= MASK_FOREGROUND_MIN]
    for inst_id in instance_ids:
        instance_mask = np.where(mask == inst_id, 255, 0).astype(np.uint8)
        box = _instance_bbox(instance_mask, min_contour_area)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        bboxes.append((
            (x1 + w / 2) / image_w,
            (y1 + h / 2) / image_h,
            w / image_w,
            h / image_h,
        ))
    return bboxes


def convert_all_annotations(raw_dir=RAW_DIR,
                            annotation_dir=ANNOTATION_DIR,
                            output_label_dir=ALL_LABELS_DIR):
    """Convert every class's masks to YOLO labels and write safe empties."""
    os.makedirs(output_label_dir, exist_ok=True)
    total = 0
    total_boxes = 0
    skipped_empty = 0

    # Ambiguous multi-threat scans (filed under >1 class) are skipped — see
    # find_multiclass_stems(). A skipped image gets NO label file, so the
    # dataset builder leaves it out entirely.
    multiclass_stems = find_multiclass_stems(raw_dir)
    if multiclass_stems:
        print(f"Skipping {len(multiclass_stems)} ambiguous multi-threat "
              f"images (appear under more than one class folder).\n")

    for class_name, class_id in CLASS_MAP.items():
        # Folders ship as 'GUN'/'knife'/'shuriken'; resolve case-insensitively.
        img_dir  = resolve_class_dir(raw_dir, class_name)
        mask_dir = resolve_class_dir(annotation_dir, class_name)

        if img_dir is None:
            print(f"[SKIP] No image folder for class: {class_name} "
                  f"(looked in {raw_dir})")
            continue
        if mask_dir is None:
            print(f"[SKIP] No annotation folder for class: {class_name} "
                  f"(looked in {annotation_dir})")
            continue

        files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        converted = 0
        class_boxes = 0
        for img_file in files:
            stem      = os.path.splitext(img_file)[0]
            img_path  = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, stem + '.png')
            label_out = os.path.join(output_label_dir, stem + '.txt')

            # Skip ambiguous multi-threat scans (no label -> excluded later).
            if stem in multiclass_stems:
                continue

            try:
                w, h = get_image_dimensions(img_path)
            except FileNotFoundError:
                continue

            if not os.path.exists(mask_path):
                print(f"  [WARN] No mask found for {img_file}")
                continue

            bboxes = mask_to_yolo_bbox(mask_path, w, h)
            # Empty mask (nothing annotated): skip rather than mislabel a known
            # threat image as background/safe.
            if not bboxes:
                skipped_empty += 1
                continue

            with open(label_out, 'w') as f:
                for (xc, yc, bw, bh) in bboxes:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            converted += 1
            class_boxes += len(bboxes)

        print(f"  {class_name}: {converted}/{len(files)} labels written "
              f"({class_boxes} boxes)")
        total += converted
        total_boxes += class_boxes

    # Safe class — empty label files (no detections = background = safe)
    safe_dir = resolve_class_dir(raw_dir, SAFE_CLASS)
    safe_count = 0
    if safe_dir is not None:
        for img_file in os.listdir(safe_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            stem = os.path.splitext(img_file)[0]
            open(os.path.join(output_label_dir, stem + '.txt'), 'w').close()
            safe_count += 1
        print(f"  safe: {safe_count} empty labels written")

    print(f"\nDone. Threat labels: {total} ({total_boxes} boxes), "
          f"safe labels: {safe_count}")
    print(f"Skipped: {len(multiclass_stems)} ambiguous multi-threat, "
          f"{skipped_empty} empty-mask images.")
    if total > 0 and total_boxes == 0:
        print("[ERROR] Labels were written but contain ZERO boxes. The masks "
              "may not be instance-indexed as expected — inspect them with "
              "preprocessing/visualize_masks.py before training.")


if __name__ == '__main__':
    convert_all_annotations()
