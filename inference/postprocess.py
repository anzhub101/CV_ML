"""
Post-processing for raw YOLO detections.

Implements the three robustness filters and the label-resolution logic:
  1. Confidence-band filtering (reject low; validate medium; accept high)
  2. Bounding-box size filtering (drop tiny keychain/toy detections)
  3. Metal-density validation (real metal weapons are dense/bright in X-ray)
  4. Multi-object resolution by threat priority

Also provides compute_iou() for localization scoring.

Note: YOLO26 is NMS-free, so no explicit Non-Maximum-Suppression step is
needed here — the model already returns final detections.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import (
    THREAT_PRIORITY, CONF_HIGH, CONF_LOW,
    MIN_BBOX_AREA_RATIO, METAL_DENSITY_THRESHOLD,
)


def validate_metal_density(image, bbox_xyxy, threshold=METAL_DENSITY_THRESHOLD):
    """Real metallic weapons appear dense (bright) in the blue X-ray channel."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    roi = image[y1:y2, x1:x2]
    blue = roi[:, :, 0]
    return float(np.mean(blue)) >= threshold


def validate_size(bbox_xyxy, image_shape):
    """Reject objects too small to be a real weapon (keychains, jewelry)."""
    x1, y1, x2, y2 = bbox_xyxy
    box_area = (x2 - x1) * (y2 - y1)
    img_area = image_shape[0] * image_shape[1]
    if img_area == 0:
        return False
    return (box_area / img_area) >= MIN_BBOX_AREA_RATIO


def filter_detections(detections, image, conf_low=CONF_LOW,
                      conf_high=CONF_HIGH, metal_threshold=METAL_DENSITY_THRESHOLD):
    """
    Apply the three robustness filters and return ALL detections that survive.

    detections: list of dicts with keys
        class_id, class_name, confidence, bbox_xyxy
    conf_low / conf_high / metal_threshold default to the config values but can
    be overridden (e.g. to trade recall vs. precision on missed threats).
    Returns: list of the kept detection dicts (may be empty = safe).
    """
    valid = []
    for det in detections:
        conf = det['confidence']
        bbox = det['bbox_xyxy']

        if conf < conf_low:
            continue
        if not validate_size(bbox, image.shape):
            continue
        # medium-confidence detections get the extra density check
        if conf < conf_high and not validate_metal_density(image, bbox,
                                                            metal_threshold):
            continue
        valid.append(det)
    return valid


def count_by_class(detections):
    """Return {class_name: count} for a list of (kept) detections."""
    counts = {}
    for det in detections:
        counts[det['class_name']] = counts.get(det['class_name'], 0) + 1
    return counts


def resolve_detections(detections, image, conf_low=CONF_LOW,
                       conf_high=CONF_HIGH,
                       metal_threshold=METAL_DENSITY_THRESHOLD):
    """
    Apply all filters and collapse to a single (label, bbox) for the
    image-level classification submission (highest-priority threat).

    Returns: (label_str, bbox_xyxy or None)
    """
    if not detections:
        return 'safe', None

    valid = filter_detections(detections, image, conf_low, conf_high,
                              metal_threshold)
    if not valid:
        return 'safe', None

    best = max(
        valid,
        key=lambda d: (THREAT_PRIORITY[d['class_name']], d['confidence']),
    )
    return best['class_name'], best['bbox_xyxy']


def compute_iou(pred_bbox, gt_bbox):
    """IoU between two [x1, y1, x2, y2] boxes."""
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area   = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - inter
    return inter / union if union > 0 else 0.0
