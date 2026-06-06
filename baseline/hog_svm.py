"""
Classical baseline — HOG features + SVM classifier (course Topics 3 & ML).

This is a deliberately small, fully classical alternative to the YOLO26
detector, included to demonstrate the classical computer-vision and machine
learning techniques required by the project brief:

    Histogram of Oriented Gradients (HOG)  ->  gradient/edge feature descriptor
    CLAHE contrast normalization           ->  Topic 5 (color/intensity)
    Support Vector Machine (SVM, RBF)      ->  classical supervised classifier

It performs the CLASSIFICATION task only (safe / gun / knife / shuriken) — HOG
+ SVM does not localize — so it complements the YOLO pipeline rather than
replacing it. It reuses the same dataset split and the same scoring code, so
its Accuracy / Macro-F1 / Classification Score are directly comparable to the
detector's.

Run from the project root, AFTER the data-prep steps (setup_data ->
convert_annotations -> build_dataset) have produced data/dataset/:

    python baseline/hog_svm.py train       # fit on the train split, save model
    python baseline/hog_svm.py eval        # score on the test split
    python baseline/hog_svm.py predict --images hidden_test   # -> CSV

Dependencies: opencv-python, numpy, scikit-learn (all already in
requirements.txt) — no deep-learning framework needed.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import CLASS_NAMES, THREAT_PRIORITY, DATASET_DIR
from utils.metrics import (
    LABEL_NAMES,
    compute_accuracy, compute_per_class_metrics, compute_macro_f1,
    compute_classification_score, build_confusion_matrix, format_confusion_matrix,
)
from inference.submission import SubmissionWriter

IMG_EXTS  = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
MODEL_PATH = "baseline/hog_svm.joblib"

# ── HOG descriptor (fixed 128x128 window) ────────────────────────────────────
WIN = 128
_HOG = cv2.HOGDescriptor(
    _winSize=(WIN, WIN), _blockSize=(16, 16), _blockStride=(8, 8),
    _cellSize=(8, 8), _nbins=9,
)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(image_path):
    """Grayscale -> resize 128x128 -> CLAHE -> HOG descriptor (1-D vector)."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    gray = cv2.resize(gray, (WIN, WIN))
    gray = _CLAHE.apply(gray)                 # contrast normalization (Topic 5)
    return _HOG.compute(gray).ravel()         # gradient histogram (HOG)


# ── Image-level label from a YOLO .txt file ──────────────────────────────────
def label_from_yolo(label_path):
    """'safe' for an empty/missing file, else the highest-priority class."""
    if not os.path.exists(label_path):
        return None
    classes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if parts:
                classes.append(CLASS_NAMES.get(int(parts[0]), "safe"))
    if not classes:
        return "safe"
    return max(classes, key=lambda c: THREAT_PRIORITY.get(c, 0))


def load_split(split, include_aug=False):
    """Load (X, y, names) for a dataset split, deriving labels from YOLO txt."""
    image_dir = os.path.join(DATASET_DIR, "images", split)
    label_dir = os.path.join(DATASET_DIR, "labels", split)
    if not os.path.isdir(image_dir):
        print(f"[ERROR] Missing {image_dir}. Run the data-prep steps first "
              f"(setup_data -> convert_annotations -> build_dataset).")
        sys.exit(1)

    files = sorted(f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    X, y, names = [], [], []
    t0 = time.time()
    for i, fname in enumerate(files, 1):
        stem = os.path.splitext(fname)[0]
        if "_aug" in stem and not include_aug:     # skip augmented copies
            continue
        label = label_from_yolo(os.path.join(label_dir, stem + ".txt"))
        if label is None:
            continue
        feat = extract_features(os.path.join(image_dir, fname))
        if feat is None:
            continue
        X.append(feat); y.append(label); names.append(fname)
        if i % 200 == 0:
            print(f"  ...{i}/{len(files)} images")
    print(f"  loaded {len(X)} images from '{split}' "
          f"({time.time() - t0:.1f}s, feature dim={len(X[0]) if X else 0})")
    return np.asarray(X, dtype=np.float32), y, names


# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    """StandardScaler -> RBF SVM (class_weight balanced for imbalance)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                    class_weight="balanced")),
    ])


# ── Classification-only report (no localization for this baseline) ───────────
def report_classification(y_true, y_pred, title):
    acc       = compute_accuracy(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred)
    macro_f1  = compute_macro_f1(per_class)
    cls_score = compute_classification_score(acc, macro_f1)

    lines = [
        "=" * 55, f"  {title}", "=" * 55,
        f"  Images               : {len(y_true)}",
        f"  Accuracy             : {acc:.4f}",
        f"  Macro F1-Score       : {macro_f1:.4f}",
        f"  Classification Score : {cls_score:.4f}  (0.7*Acc + 0.3*F1)",
        "", "  --- Per-Class F1 ---",
    ]
    for c in LABEL_NAMES:
        lines.append(f"  {c:<12}: F1={per_class[c]['f1']:.4f}"
                     f"  P={per_class[c]['precision']:.4f}"
                     f"  R={per_class[c]['recall']:.4f}")
    lines += ["", "  Confusion matrix (rows=true, cols=pred):",
              format_confusion_matrix(build_confusion_matrix(y_true, y_pred)),
              "=" * 55,
              "  NOTE: HOG+SVM is classification-only; localization (IoU) is",
              "        handled by the YOLO pipeline (run_inference.py).",
              "=" * 55]
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_hog_svm_report.txt", "w") as f:
        f.write(text + "\n")
    print("\nReport -> outputs/baseline_hog_svm_report.txt")
    return cls_score


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_train(args):
    import joblib
    print("Extracting HOG features (train split)...")
    X, y, _ = load_split("train", include_aug=args.include_aug)
    if len(set(y)) < 2:
        print("[ERROR] Need at least two classes to train."); sys.exit(1)

    print(f"\nTraining RBF-SVM on {len(X)} samples...")
    t0 = time.time()
    model = build_model()
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  done in {time.time() - t0:.1f}s -> {MODEL_PATH}")

    # Quick train-set sanity score
    report_classification(y, list(model.predict(X)),
                          "BASELINE (HOG+SVM) — TRAIN-SET FIT")


def cmd_eval(args):
    import joblib
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No model at {MODEL_PATH}. Run: "
              f"python baseline/hog_svm.py train"); sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}\nExtracting HOG features (test split)...")
    X, y, _ = load_split(args.split)
    report_classification(y, list(model.predict(X)),
                          f"BASELINE (HOG+SVM) — {args.split.upper()} SET")


def cmd_predict(args):
    import joblib
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No model at {MODEL_PATH}. Run: "
              f"python baseline/hog_svm.py train"); sys.exit(1)
    if not os.path.isdir(args.images):
        print(f"[ERROR] Folder not found: {args.images}"); sys.exit(1)
    model = joblib.load(MODEL_PATH)

    files = sorted(f for f in os.listdir(args.images)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    predictions = []
    for i, fname in enumerate(files, 1):
        feat = extract_features(os.path.join(args.images, fname))
        label = "safe" if feat is None else str(model.predict([feat])[0])
        predictions.append({"image_name": fname, "pred_label": label})
        print(f"  [{i:>4}/{len(files)}] {fname:<32} -> {label}")

    writer = SubmissionWriter()
    writer.write(predictions, args.out)
    writer.validate(args.out)
    writer.print_summary(args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Fit HOG+SVM on the train split")
    pt.add_argument("--include-aug", action="store_true",
                    help="Also use augmented (_aug) training images")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Score on a split (default: test)")
    pe.add_argument("--split", default="test", choices=["train", "val", "test"])
    pe.set_defaults(func=cmd_eval)

    pp = sub.add_parser("predict", help="Predict a folder -> submission CSV")
    pp.add_argument("--images", required=True, help="Folder of images")
    pp.add_argument("--out", default="outputs/predictions_hog_svm.csv")
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
