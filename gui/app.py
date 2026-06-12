"""
Simple GUI for the CEN454 baggage threat detector (Gradio).

Lets a user upload one or more images — or point at a folder of images — and
view the detection results: every detected object boxed and labelled, a
per-class object count, and the image-level label. Reuses the exact inference
pipeline from run_inference.py (classical preprocessing -> YOLO26 ->
post-processing), so what the GUI shows matches the batch tool.

Run from the project root (needs weights/best.pt and `pip install gradio`):

    python gui/app.py
    # then open the printed local URL (e.g. http://127.0.0.1:7860)
"""

import os
import sys

import cv2
import numpy as np
import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing.config import CONF_LOW
from run_inference import load_model, process_image, WEIGHTS
from inference.quality_handler import QualityHandler
from inference.localization import LocalizationModule
from inference.visualize_results import Visualizer, CLASS_COLORS
from baseline import hog_svm

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

YOLO_DETECTOR = "YOLO26  (detection + per-object counts)"
SVM_DETECTOR  = "Classical HOG+SVM  (image-level label only)"

# Build the heavy components once (model is cached inside load_model).
_quality   = QualityHandler()
_localizer = LocalizationModule()
_viz       = Visualizer(output_dir="outputs/gui_annotated")
_svm_bundle = None      # lazily loaded baseline model


def _load_svm():
    """Load + cache the HOG+SVM baseline bundle (pipeline + config + threshold)."""
    global _svm_bundle
    if _svm_bundle is None:
        if not os.path.exists(hog_svm.MODEL_PATH):
            raise gr.Error(
                f"No baseline model at {hog_svm.MODEL_PATH}. Train it first: "
                f"python baseline/hog_svm.py train")
        # The bundle is pickled by hog_svm.py running as __main__, so its
        # FeatureConfig is referenced as __main__.FeatureConfig. Register it so
        # joblib can resolve it when loading from any other entry point.
        import __main__
        __main__.FeatureConfig = hog_svm.FeatureConfig
        import joblib
        _svm_bundle = joblib.load(hog_svm.MODEL_PATH)
    return _svm_bundle


def _label_banner(img, label, prefix="HOG+SVM"):
    """Stamp a colored label banner across the top (classification-only view)."""
    out = img.copy()
    color = CLASS_COLORS.get(label, (80, 80, 80))
    text = f"{prefix}: {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 14), color, -1)
    cv2.putText(out, text, (6, th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def _collect_paths(files, folder_path):
    """Gather image paths from uploaded files and/or a folder path."""
    paths = []
    for f in files or []:
        p = getattr(f, "name", f)          # gradio gives a temp path/obj
        if str(p).lower().endswith(IMAGE_EXTS):
            paths.append(p)
    folder_path = (folder_path or "").strip()
    if folder_path and os.path.isdir(folder_path):
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(folder_path, name))
    return paths


def _run_yolo(paths, use_tta, use_refine, conf_low):
    if not os.path.exists(WEIGHTS):
        raise gr.Error(f"No model weights at {WEIGHTS}. Train first "
                       f"(python training/train.py) or drop in best.pt.")
    model = load_model()
    gallery, rows = [], []
    totals = {"gun": 0, "knife": 0, "shuriken": 0}
    for path in paths:
        name = os.path.basename(path)
        raw = cv2.imread(path)
        if raw is None:
            continue
        res = process_image(path, model, _quality, _localizer,
                            use_tta=use_tta, refine=use_refine,
                            conf_low=float(conf_low))
        annotated = _viz.draw_detections(raw, name, res["detections"])
        gallery.append(
            (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             f"{name} — {res['pred_label']} ({sum(res['counts'].values())} obj)")
        )
        c = res["counts"]
        for k in totals:
            totals[k] += c.get(k, 0)
        rows.append([name, res["pred_label"], sum(c.values()),
                     c.get("gun", 0), c.get("knife", 0), c.get("shuriken", 0)])

    n_threat = sum(1 for r in rows if r[1] != "safe")
    summary = (f"YOLO26 · {len(rows)} image(s): {n_threat} with threats, "
               f"{len(rows) - n_threat} safe.  Objects detected — "
               f"gun:{totals['gun']}  knife:{totals['knife']}  "
               f"shuriken:{totals['shuriken']}.")
    return gallery, rows, summary


def _run_svm(paths):
    """Classical HOG+SVM: one image-level label per image (no localization)."""
    bundle = _load_svm()
    cfg = bundle["config"]
    hog = hog_svm.make_hog(cfg)
    gallery, rows = [], []
    counts = {"gun": 0, "knife": 0, "shuriken": 0, "safe": 0}
    for path in paths:
        name = os.path.basename(path)
        raw = cv2.imread(path)
        if raw is None:
            continue
        feat = hog_svm.extract_features(path, cfg, hog)
        if feat is None:
            continue
        label = str(hog_svm.predict_labels(bundle, np.asarray([feat]))[0])
        counts[label] = counts.get(label, 0) + 1
        gallery.append((cv2.cvtColor(_label_banner(raw, label),
                                     cv2.COLOR_BGR2RGB), f"{name} — {label}"))
        # classification-only: mark a 1 in the predicted class column
        rows.append([name, label, 0 if label == "safe" else 1,
                     int(label == "gun"), int(label == "knife"),
                     int(label == "shuriken")])

    summary = (f"HOG+SVM (classification only, no boxes) · {len(rows)} image(s) — "
               f"gun:{counts['gun']}  knife:{counts['knife']}  "
               f"shuriken:{counts['shuriken']}  safe:{counts['safe']}.")
    return gallery, rows, summary


def detect(files, folder_path, detector, use_tta, use_refine, conf_low):
    paths = _collect_paths(files, folder_path)
    if not paths:
        return [], [], "No images provided. Upload files or enter a folder path."
    if detector == SVM_DETECTOR:
        return _run_svm(paths)
    return _run_yolo(paths, use_tta, use_refine, conf_low)


def build_ui():
    with gr.Blocks(title="CEN454 Baggage Threat Detector") as demo:
        gr.Markdown(
            "# 🛄 CEN454 — Baggage Threat Detector\n"
            "Upload image(s) **or** enter a folder path, pick a detector, then "
            "**Detect**.\n"
            "- **YOLO26** boxes and counts every object per class "
            "(safe / gun / knife / shuriken).\n"
            "- **HOG+SVM** is the classical baseline: one image-level label, "
            "no boxes or object counts."
        )
        with gr.Row():
            with gr.Column(scale=1):
                detector = gr.Radio(
                    choices=[YOLO_DETECTOR, SVM_DETECTOR],
                    value=YOLO_DETECTOR, label="Detector")
                files = gr.File(label="Upload image(s)", file_count="multiple",
                                file_types=["image"])
                folder = gr.Textbox(
                    label="…or a folder path on this machine",
                    placeholder="e.g. data/dataset/images/test")
                gr.Markdown("**YOLO options** (ignored by HOG+SVM):")
                use_tta = gr.Checkbox(value=True, label="Test-Time Augmentation "
                                      "(more robust label, slower)")
                use_refine = gr.Checkbox(value=False, label="Morphological box "
                                         "refinement (off — it lowers IoU here)")
                conf_low = gr.Slider(0.05, 0.9, value=CONF_LOW, step=0.05,
                                     label="Min confidence (lower = catch more)")
                run_btn = gr.Button("Detect", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Results", columns=3, height=520)
                summary = gr.Textbox(label="Summary", interactive=False)
                table = gr.Dataframe(
                    headers=["image", "label", "total", "gun", "knife", "shuriken"],
                    label="Per-image results", interactive=False, wrap=True)

        run_btn.click(detect,
                      inputs=[files, folder, detector, use_tta, use_refine, conf_low],
                      outputs=[gallery, table, summary])
    return demo


if __name__ == "__main__":
    build_ui().launch()
