"""
run_inference.py — Master inference script for evaluation day.

This is the ONLY file you need to run after training is done.

What it does end-to-end:
  1. Loads the fine-tuned YOLO26 weights from weights/best.pt
  2. For every image in the test folder:
       a. Adaptive quality fixing  (quality_handler)
       b. Classical CV preprocessing (preprocess)
       c. YOLO26 detection
       d. Post-processing: confidence / size / metal-density filters
       e. Morphological bbox refinement (localization)
       f. Optional Test-Time Augmentation for uncertain cases
  3. Writes outputs/predictions.csv  (submission format)
  4. Saves annotated images to outputs/annotated/
  5. If a ground-truth CSV is provided, computes and prints all metrics
     (accuracy, macro F1, classification score, IoU, localization score,
      final score) and saves outputs/evaluation_report.txt

Quick start:
    # Evaluation day — hidden test, no ground truth (writes predictions.csv):
    python run_inference.py --images hidden_test

    # Development — FULL project criteria on the test split
    # (accuracy, macro F1, mean IoU, final score):
    python run_inference.py --images data/dataset/images/test \
                            --labels data/dataset/labels/test

    # Classification-only metrics from a label CSV:
    python run_inference.py --images data/dataset/images/test --gt gt.csv

    # Skip TTA / visualization (faster):
    python run_inference.py --images hidden_test --no-tta --no-viz
"""

import argparse
import os
import sys
import time

import cv2
from ultralytics import YOLO

# ── project imports ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from preprocessing.config import (
    CLASS_NAMES, CLASS_MAP, CONF_LOW, CONF_HIGH,
    THREAT_PRIORITY, MIN_BBOX_AREA_RATIO,
)
from preprocessing.preprocess     import preprocess
from inference.quality_handler    import QualityHandler
from inference.postprocess        import resolve_detections, validate_metal_density, validate_size
from inference.localization       import LocalizationModule
from inference.submission         import SubmissionWriter
from inference.visualize_results  import Visualizer
from utils.metrics                import compute_all_metrics, compute_iou, save_report
from utils.logger                 import get_logger

log = get_logger("run_inference")

WEIGHTS    = "weights/best.pt"
OUTPUT_CSV = "outputs/predictions.csv"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ── Model loader (cached) ────────────────────────────────────────────────────

_model: YOLO | None = None

def load_model(weights: str = WEIGHTS) -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(weights):
            log.error(f"No weights found at {weights}. Run training/train.py first.")
            sys.exit(1)
        _model = YOLO(weights)
        log.info(f"Model loaded: {weights}")
    return _model


# ── Single-pass YOLO detection ───────────────────────────────────────────────

def _yolo_detect(model: YOLO, image: cv2.Mat) -> list:
    """Run YOLO on a preprocessed image; return raw detection dicts."""
    results = model(image, conf=CONF_LOW, verbose=False)
    boxes   = results[0].boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CLASS_NAMES:
                continue
            detections.append({
                "class_id":   cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": float(box.conf[0]),
                "bbox_xyxy":  box.xyxy[0].tolist(),
            })
    return detections


# ── Test-Time Augmentation ───────────────────────────────────────────────────

def _predict_tta(model: YOLO, processed: cv2.Mat, original: cv2.Mat) -> tuple:
    """5-variant TTA with majority vote on label."""
    variants = [
        processed,
        cv2.flip(processed, 1),
        cv2.flip(processed, 0),
        cv2.convertScaleAbs(processed, alpha=1.2, beta=10),
        cv2.convertScaleAbs(processed, alpha=0.8, beta=-10),
    ]
    labels, bboxes, confs = [], [], []
    for v in variants:
        dets = _yolo_detect(model, v)
        lbl, bb = resolve_detections(dets, original)
        labels.append(lbl)
        if bb is not None:
            bboxes.append(bb)
        if dets:
            confs.append(max(d["confidence"] for d in dets))

    final_label = max(set(labels), key=labels.count)
    final_bbox  = bboxes[0] if (bboxes and final_label != "safe") else None
    final_conf  = sum(confs) / len(confs) if confs else 0.0
    return final_label, final_bbox, final_conf


# ── Full per-image pipeline ──────────────────────────────────────────────────

def process_image(
    image_path:  str,
    model:       YOLO,
    quality_handler: QualityHandler,
    localizer:   LocalizationModule,
    use_tta:     bool = True,
) -> dict:
    """
    Run the full pipeline on one image.
    Returns a result dict:
      image_name, pred_label, pred_bbox, confidence, quality_report
    """
    image_name = os.path.basename(image_path)

    raw = cv2.imread(image_path)
    if raw is None:
        log.warning(f"Cannot read {image_path} — defaulting to safe")
        return {
            "image_name": image_name, "pred_label": "safe",
            "pred_bbox": None, "confidence": 0.0, "quality_report": None,
        }

    # Step 1: Adaptive quality fixing
    fixed, qr = quality_handler.process(raw)

    # Step 2: Classical CV preprocessing (Topics 3–6)
    processed = preprocess(image_array=fixed)

    # Step 3: Detect
    if use_tta:
        label, bbox, conf = _predict_tta(model, processed, raw)
    else:
        dets  = _yolo_detect(model, processed)
        label, bbox = resolve_detections(dets, raw)
        conf  = max((d["confidence"] for d in dets), default=0.0)

    # Step 4: Morphological bbox refinement (Topic 6)
    if bbox is not None:
        bbox = localizer.refine(raw, bbox)

    log.debug(f"{image_name}: {label} (conf={conf:.3f})  bbox={bbox}  {qr}")

    return {
        "image_name":     image_name,
        "pred_label":     label,
        "pred_bbox":      bbox,
        "confidence":     conf,
        "quality_report": qr,
    }


# ── Metrics against ground-truth CSV ─────────────────────────────────────────

def _load_ground_truth(gt_csv: str) -> dict:
    import csv
    gt = {}
    with open(gt_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = list(row.values())[0]
            val = list(row.values())[1]
            gt[key] = val.strip().lower()
    return gt


def _evaluate(predictions: list, gt_csv: str):
    """Classification-only metrics from a label CSV (no ground-truth boxes)."""
    gt_map = _load_ground_truth(gt_csv)
    y_true, y_pred = [], []
    for p in predictions:
        true = gt_map.get(p["image_name"])
        if true is None:
            continue
        y_true.append(true)
        y_pred.append(p["pred_label"])

    if not y_true:
        log.warning("No matching ground-truth entries found.")
        return

    report = compute_all_metrics(y_true, y_pred, iou_scores=[])
    print(report.summary())
    save_report(report)


def _build_gt_from_labels(image_dir: str, label_dir: str) -> dict:
    """
    Build ground truth (image-level label + boxes) from YOLO .txt labels — the
    same format build_dataset.py produces for the test split. Returns
        {image_name: {"label": str, "boxes": [[x1,y1,x2,y2], ...]}}
    boxes are in ORIGINAL-image pixel coordinates (to match predicted boxes).
    The image-level label is 'safe' for an empty file, otherwise the
    highest-priority threat class present (gun > knife > shuriken).
    """
    gt = {}
    for fname in os.listdir(image_dir):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        stem     = os.path.splitext(fname)[0]
        lbl_path = os.path.join(label_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(os.path.join(image_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes, classes = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                xc, yc, bw, bh = (float(x) for x in parts[1:])
                x1 = (xc - bw / 2) * w; y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w; y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                classes.append(CLASS_NAMES.get(cid, "safe"))

        if not classes:
            label = "safe"
        else:
            label = max(classes, key=lambda c: THREAT_PRIORITY.get(c, 0))
        gt[fname] = {"label": label, "boxes": boxes}
    return gt


def _evaluate_full(predictions: list, image_dir: str, label_dir: str):
    """
    Full project-criteria evaluation against YOLO ground-truth labels:
    classification (accuracy, macro F1) AND localization (mean IoU, >=0.5).
    """
    gt = _build_gt_from_labels(image_dir, label_dir)
    if not gt:
        log.warning(f"No ground-truth labels found in {label_dir}.")
        return

    y_true, y_pred, iou_scores = [], [], []
    for p in predictions:
        g = gt.get(p["image_name"])
        if g is None:
            continue
        y_true.append(g["label"])
        y_pred.append(p["pred_label"])

        # Localization counts only on images that ARE a threat and where we
        # predicted a threat with a box (per spec: detected threat images).
        if g["label"] != "safe" and p["pred_label"] != "safe" \
                and p.get("pred_bbox") and g["boxes"]:
            iou_scores.append(
                max(compute_iou(p["pred_bbox"], gb) for gb in g["boxes"])
            )

    if not y_true:
        log.warning("No predictions matched the ground-truth labels.")
        return

    report = compute_all_metrics(y_true, y_pred, iou_scores=iou_scores)
    print(report.summary())
    save_report(report)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CEN454 baggage threat detector — inference pipeline"
    )
    parser.add_argument(
        "--images", required=True,
        help="Folder containing test images"
    )
    parser.add_argument(
        "--gt", default=None,
        help="Ground-truth label CSV (classification-only metrics)"
    )
    parser.add_argument(
        "--labels", default=None,
        help="YOLO ground-truth label folder (e.g. data/dataset/labels/test). "
             "Computes the FULL project criteria: classification + localization "
             "(mean IoU) + final score."
    )
    parser.add_argument(
        "--weights", default=WEIGHTS,
        help=f"Path to fine-tuned weights (default: {WEIGHTS})"
    )
    parser.add_argument(
        "--out", default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})"
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable Test-Time Augmentation (faster, slightly less robust)"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip saving annotated images"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        log.error(f"Image folder not found: {args.images}")
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        log.error(f"No images found in {args.images}")
        sys.exit(1)

    use_tta = not args.no_tta
    log.info(f"Images: {len(image_files)}  |  TTA: {'ON' if use_tta else 'OFF'}")

    # Initialise components
    model           = load_model(args.weights)
    quality_handler = QualityHandler()
    localizer       = LocalizationModule()
    submitter       = SubmissionWriter()
    visualizer      = Visualizer() if not args.no_viz else None

    predictions = []
    t0 = time.time()

    for i, fname in enumerate(image_files, 1):
        img_path = os.path.join(args.images, fname)
        result   = process_image(img_path, model, quality_handler,
                                 localizer, use_tta)
        predictions.append(result)

        status = f"[{i:>4}/{len(image_files)}]  {fname:<35}  {result['pred_label']}"
        if result["pred_label"] != "safe":
            status += f"  (conf={result['confidence']:.3f})"
        log.info(status)

    elapsed = time.time() - t0
    log.info(f"\nInference done in {elapsed:.1f}s  "
             f"({elapsed / len(image_files):.2f}s per image)")

    # Write submission CSV
    submitter.write(predictions, args.out)
    submitter.validate(args.out)
    submitter.print_summary(args.out)

    # Save annotated images
    if visualizer:
        visualizer.draw_batch(predictions, args.images)
        visualizer.save_grid([p["image_name"] for p in predictions])

    # Metrics (development mode)
    if args.labels:
        _evaluate_full(predictions, args.images, args.labels)
    elif args.gt:
        _evaluate(predictions, args.gt)


if __name__ == "__main__":
    main()
