"""
Localization module — bounding box extraction and refinement.

Responsibilities:
  1. Refine a YOLO bounding box using morphological operations (Topic 6)
     on the image region to produce a tighter fit around the weapon.
  2. Convert between bbox formats (YOLO normalized, pixel xyxy, pixel xywh).
  3. Merge multiple bboxes from the same class into a single union box.
  4. Validate a predicted bbox against a ground-truth mask or bbox (IoU).
  5. Generate a segmentation mask for the detected region, which can be used
     to produce more accurate IoU when ground-truth masks are available.

The morphological refinement is the classical CV contribution:
  - Threshold the ROI on the high-density channel (metallic content)
  - Apply OPEN (erode then dilate) to remove pepper noise
  - Apply CLOSE (dilate then erode) to fill gaps in the mask
  - Find the tightest bounding contour

Usage:
    from inference.localization import LocalizationModule
    loc = LocalizationModule()
    refined_bbox = loc.refine(image, yolo_bbox_xyxy)
    iou = loc.compute_iou(refined_bbox, gt_bbox_xyxy)
"""

import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

log = get_logger(__name__)

BBox = List[float]   # [x1, y1, x2, y2] in pixels unless noted


class LocalizationModule:
    """Refines and validates bounding boxes for the detected threat objects."""

    def __init__(
        self,
        density_threshold: int = 110,
        min_contour_area: int  = 50,
        padding: float         = 0.05,
    ):
        """
        density_threshold: min blue-channel intensity to count as metal.
        min_contour_area: ignore tiny contours (px²).
        padding: fractional padding added to the refined box.
        """
        self.density_threshold = density_threshold
        self.min_contour_area  = min_contour_area
        self.padding           = padding

    # ── Format converters ───────────────────────────────────────────────────

    @staticmethod
    def yolo_to_xyxy(
        xc: float, yc: float, w: float, h: float,
        img_w: int, img_h: int
    ) -> BBox:
        """Normalized YOLO [xc, yc, w, h] -> pixel [x1, y1, x2, y2]."""
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        return [x1, y1, x2, y2]

    @staticmethod
    def xyxy_to_yolo(
        x1: float, y1: float, x2: float, y2: float,
        img_w: int, img_h: int
    ) -> Tuple[float, float, float, float]:
        """Pixel [x1, y1, x2, y2] -> normalized YOLO [xc, yc, w, h]."""
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        return xc, yc, w, h

    @staticmethod
    def clip_bbox(bbox: BBox, img_w: int, img_h: int) -> BBox:
        x1, y1, x2, y2 = bbox
        return [
            max(0, min(img_w, x1)),
            max(0, min(img_h, y1)),
            max(0, min(img_w, x2)),
            max(0, min(img_h, y2)),
        ]

    # ── Morphological refinement (Topic 6) ──────────────────────────────────

    def _extract_roi(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        return image[y1:y2, x1:x2]

    def _metal_mask(self, roi: np.ndarray) -> np.ndarray:
        """
        Create a binary mask highlighting dense (metallic) regions.
        In pseudo-colored X-ray, metals are brightest in the blue channel.
        """
        blue = roi[:, :, 0]
        _, mask = cv2.threshold(
            blue, self.density_threshold, 255, cv2.THRESH_BINARY
        )
        # Also consider LAB lightness
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, l_mask = cv2.threshold(l_channel, 180, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_or(mask, l_mask)

    def _morphological_clean(self, mask: np.ndarray) -> np.ndarray:
        """OPEN (remove noise) then CLOSE (fill gaps) — Topic 6."""
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
        return mask

    def refine(self, image: np.ndarray, bbox: BBox) -> BBox:
        """
        Use morphological operations on the ROI to produce a tighter bbox.
        Falls back to the original YOLO bbox if refinement fails.
        """
        h_img, w_img = image.shape[:2]

        roi = self._extract_roi(image, bbox)
        if roi.size == 0:
            return bbox

        mask = self._metal_mask(roi)
        mask = self._morphological_clean(mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            log.debug("Localization: no contours after morph — using YOLO bbox")
            return bbox

        # Keep the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_contour_area:
            return bbox

        rx, ry, rw, rh = cv2.boundingRect(largest)

        # Map ROI-relative coordinates back to full-image space
        ox1, oy1 = int(bbox[0]), int(bbox[1])
        ox1, oy1 = max(0, ox1), max(0, oy1)

        pad_x = int(rw * self.padding)
        pad_y = int(rh * self.padding)

        refined = [
            ox1 + rx - pad_x,
            oy1 + ry - pad_y,
            ox1 + rx + rw + pad_x,
            oy1 + ry + rh + pad_y,
        ]
        refined = self.clip_bbox(refined, w_img, h_img)
        log.debug(f"Localization: refined {[int(v) for v in bbox]} "
                  f"-> {[int(v) for v in refined]}")
        return refined

    def generate_mask(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """
        Generate a full-image binary mask for the detected weapon region.
        Useful for pixel-level IoU against ground-truth PNG masks.
        """
        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        roi = self._extract_roi(image, bbox)
        if roi.size == 0:
            return full_mask

        mask = self._metal_mask(roi)
        mask = self._morphological_clean(mask)

        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        # Place ROI mask back into full image space
        roi_h = min(y2 - y1, mask.shape[0])
        roi_w = min(x2 - x1, mask.shape[1])
        if roi_h > 0 and roi_w > 0:
            full_mask[y1:y1 + roi_h, x1:x1 + roi_w] = mask[:roi_h, :roi_w]

        return full_mask

    # ── IoU helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_iou(pred: BBox, gt: BBox) -> float:
        """Intersection over Union for two [x1,y1,x2,y2] boxes."""
        px1, py1, px2, py2 = pred
        gx1, gy1, gx2, gy2 = gt

        ix1, iy1 = max(px1, gx1), max(py1, gy1)
        ix2, iy2 = min(px2, gx2), min(py2, gy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        pred_area = (px2 - px1) * (py2 - py1)
        gt_area   = (gx2 - gx1) * (gy2 - gy1)
        union = pred_area + gt_area - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Pixel-level IoU between two binary masks."""
        pred_bin = (pred_mask > 127).astype(np.uint8)
        gt_bin   = (gt_mask   > 127).astype(np.uint8)

        intersection = float(np.logical_and(pred_bin, gt_bin).sum())
        union        = float(np.logical_or(pred_bin,  gt_bin).sum())
        return intersection / union if union > 0 else 0.0

    # ── Multi-detection helpers ──────────────────────────────────────────────

    @staticmethod
    def merge_boxes(boxes: List[BBox]) -> BBox:
        """Compute the union bounding box over a list of boxes."""
        if not boxes:
            raise ValueError("boxes list is empty")
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [x1, y1, x2, y2]
