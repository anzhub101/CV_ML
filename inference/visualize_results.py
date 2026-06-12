"""
Visualization module — draw predictions on images and save annotated results.

Produces annotated copies of test images showing:
  - Bounding box around the detected threat
  - Class label and confidence score
  - Quality issues detected (if QualityReport provided)
  - "SAFE" stamp for safe images

Usage:
    from inference.visualize_results import Visualizer
    viz = Visualizer(output_dir="outputs/annotated")
    viz.draw_prediction(image, image_name, label, bbox, confidence)
    viz.draw_batch(predictions_list, image_dir)
    viz.save_grid(image_names, nrows=4)  # summary grid
"""

import os
import sys
from typing import List, Optional

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import CLASS_NAMES
from utils.logger import get_logger

log = get_logger(__name__)

# Color per class (BGR)
CLASS_COLORS = {
    "gun":      (0,   60,  255),   # red-orange
    "knife":    (0,   200, 80),    # green
    "shuriken": (255, 140, 0),     # blue-orange
    "safe":     (80,  80,  80),    # grey
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
THICKNESS  = 2


class Visualizer:
    def __init__(self, output_dir: str = "outputs/annotated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Single image ─────────────────────────────────────────────────────────

    def draw_prediction(
        self,
        image:       np.ndarray,
        image_name:  str,
        label:       str,
        bbox:        Optional[List[float]] = None,
        confidence:  float                 = 0.0,
        gt_bbox:     Optional[List[float]] = None,
        iou:         Optional[float]       = None,
    ) -> np.ndarray:
        """
        Draw label, bbox, and optionally the ground-truth box + IoU.
        Returns the annotated image array and saves it to output_dir.
        """
        img = image.copy()
        color = CLASS_COLORS.get(label, (200, 200, 200))

        if label == "safe" or bbox is None:
            self._stamp_safe(img)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Prediction box (solid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)

            # Ground-truth box (dashed, white) if provided
            if gt_bbox is not None:
                gx1, gy1, gx2, gy2 = [int(v) for v in gt_bbox]
                self._draw_dashed_rect(img, gx1, gy1, gx2, gy2,
                                       (255, 255, 255), THICKNESS)

            # Label tag
            iou_str = f"  IoU={iou:.2f}" if iou is not None else ""
            tag = f"{label}  {confidence:.2f}{iou_str}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, FONT_SCALE, THICKNESS)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(img, tag, (x1 + 2, ty - 2),
                        FONT, FONT_SCALE, (255, 255, 255), THICKNESS)

        out_path = os.path.join(self.output_dir, image_name)
        cv2.imwrite(out_path, img)
        return img

    # ── Multi-object drawing ─────────────────────────────────────────────────

    def draw_detections(
        self,
        image:       np.ndarray,
        image_name:  str,
        detections:  List[dict],
    ) -> np.ndarray:
        """
        Draw EVERY detected object with its own class-colored box + confidence,
        and a per-class count banner across the top. `detections` is a list of
        {class_name, confidence, bbox_xyxy}. Saves to output_dir.
        """
        img = image.copy()
        if not detections:
            self._stamp_safe(img)
        else:
            counts = {}
            for det in detections:
                cls   = det["class_name"]
                color = CLASS_COLORS.get(cls, (200, 200, 200))
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
                tag = f"{cls} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, FONT, FONT_SCALE, THICKNESS)
                ty = max(y1 - 6, th + 4)
                cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty),
                              color, -1)
                cv2.putText(img, tag, (x1 + 2, ty - 2),
                            FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
                counts[cls] = counts.get(cls, 0) + 1

            summary = "  ".join(f"{c}:{n}" for c, n in sorted(counts.items()))
            banner  = f"{sum(counts.values())} object(s)   {summary}"
            (bw, bh), _ = cv2.getTextSize(banner, FONT, FONT_SCALE, THICKNESS)
            cv2.rectangle(img, (0, 0), (bw + 10, bh + 12), (0, 0, 0), -1)
            cv2.putText(img, banner, (5, bh + 4),
                        FONT, FONT_SCALE, (255, 255, 255), THICKNESS)

        cv2.imwrite(os.path.join(self.output_dir, image_name), img)
        return img

    def draw_detections_batch(self, predictions: List[dict], image_dir: str):
        """Draw all detected objects for each result dict (multi-object mode)."""
        log.info(f"Visualizing {len(predictions)} multi-object results "
                 f"-> {self.output_dir}")
        for p in predictions:
            img = cv2.imread(os.path.join(image_dir, p["image_name"]))
            if img is None:
                continue
            self.draw_detections(img, p["image_name"], p.get("detections", []))

    def _stamp_safe(self, img: np.ndarray):
        h, w = img.shape[:2]
        text = "SAFE"
        (tw, th), _ = cv2.getTextSize(text, FONT, 2.0, 3)
        tx, ty = (w - tw) // 2, (h + th) // 2
        cv2.putText(img, text, (tx, ty), FONT, 2.0,
                    (0, 200, 60), 3, cv2.LINE_AA)

    def _draw_dashed_rect(
        self, img, x1, y1, x2, y2, color, thickness, dash=10
    ):
        pts = [(x1, y1, x2, y1), (x2, y1, x2, y2),
               (x2, y2, x1, y2), (x1, y2, x1, y1)]
        for (sx, sy, ex, ey) in pts:
            length = max(abs(ex - sx), abs(ey - sy))
            steps  = max(1, length // (2 * dash))
            for i in range(steps):
                t0 = (2 * i) / (2 * steps)
                t1 = (2 * i + 1) / (2 * steps)
                p0 = (int(sx + t0 * (ex - sx)), int(sy + t0 * (ey - sy)))
                p1 = (int(sx + t1 * (ex - sx)), int(sy + t1 * (ey - sy)))
                cv2.line(img, p0, p1, color, thickness)

    # ── Batch processing ─────────────────────────────────────────────────────

    def draw_batch(
        self,
        predictions: List[dict],
        image_dir:   str,
    ) -> None:
        """
        Draw predictions for a list of result dicts.
        Each dict: {image_name, pred_label, pred_bbox, confidence,
                    gt_bbox (opt), iou (opt)}
        """
        log.info(f"Visualizing {len(predictions)} predictions -> {self.output_dir}")
        for p in predictions:
            img_path = os.path.join(image_dir, p["image_name"])
            img = cv2.imread(img_path)
            if img is None:
                log.warning(f"Cannot read {img_path}, skipping viz")
                continue
            self.draw_prediction(
                image      = img,
                image_name = p["image_name"],
                label      = p["pred_label"],
                bbox       = p.get("pred_bbox"),
                confidence = p.get("confidence", 0.0),
                gt_bbox    = p.get("gt_bbox"),
                iou        = p.get("iou"),
            )

    # ── Summary grid ─────────────────────────────────────────────────────────

    def save_grid(
        self,
        image_names: List[str],
        nrows:       int = 4,
        thumb_size:  tuple = (300, 200),
    ) -> str:
        """
        Tile the annotated images into a grid and save as grid.png.
        Only includes images that were already saved by draw_prediction.
        """
        frames = []
        for name in image_names:
            path = os.path.join(self.output_dir, name)
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is not None:
                frames.append(cv2.resize(img, thumb_size))

        if not frames:
            log.warning("No annotated frames found for grid.")
            return ""

        ncols   = nrows
        n_blank = (ncols - len(frames) % ncols) % ncols
        blank   = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
        frames += [blank] * n_blank

        rows = []
        for i in range(0, len(frames), ncols):
            rows.append(np.hstack(frames[i:i + ncols]))
        grid = np.vstack(rows)

        grid_path = os.path.join(self.output_dir, "grid.png")
        cv2.imwrite(grid_path, grid)
        log.info(f"Grid saved -> {grid_path}")
        return grid_path
