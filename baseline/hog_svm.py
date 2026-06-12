"""
Classical baseline — multi-feature (HOG + LBP + intensity) + SVM classifier.

A fully classical alternative to the YOLO26 detector, demonstrating the
classical computer-vision + machine-learning techniques required by the brief:

    Preprocessing : grayscale -> CLAHE contrast (Topic 5)
                              -> unsharp sharpen / HPF (Topic 3)
    Features      : HOG (gradient/edge descriptor)
                  + LBP (local binary pattern texture, 8-neighbour)
                  + intensity histogram
    Model         : StandardScaler -> [optional PCA] -> SVM (RBF,
                    class_weight='balanced')

It performs the CLASSIFICATION task only (safe / gun / knife / shuriken) — it
does not localize — so it complements the YOLO pipeline. It reuses the same
dataset split and the same scoring code, so its Accuracy / Macro-F1 /
Classification Score are directly comparable to the detector's.

Run from the project root, AFTER the data-prep steps (setup_data ->
convert_annotations -> build_dataset) have produced data/dataset/:

    python baseline/hog_svm.py train          # fit defaults, save model
    python baseline/hog_svm.py eval            # score on the test split
    python baseline/hog_svm.py tune            # grid-search HOG + SVM + threshold
    python baseline/hog_svm.py predict --images hidden_test   # -> CSV

Useful flags:
    train/tune --pca            add PCA (0.95 variance) before the SVM
    train      --no-lbp/--no-hist/--no-sharpen   ablate a component
    tune       --quick          smaller grid (faster)

Dependencies: opencv-python, numpy, scikit-learn (all in requirements.txt) —
no deep-learning framework and no scikit-image needed (LBP is implemented with
numpy).
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict

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

IMG_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
MODEL_PATH = "baseline/hog_svm.joblib"


# ── Feature configuration ────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    win:          int  = 128      # working window size (square)
    use_clahe:    bool = True     # CLAHE contrast normalization (Topic 5)
    use_sharpen:  bool = True     # unsharp mask / HPF (Topic 3)
    # HOG
    hog_ppc:      int  = 8        # pixels per cell
    hog_cpb:      int  = 2        # cells per block
    hog_orient:   int  = 9        # orientation bins
    use_hog:      bool = True
    # LBP (8-neighbour, 256-bin)
    use_lbp:      bool = True
    # Intensity histogram
    use_hist:     bool = True
    hist_bins:    int  = 32


# ── Preprocessing: grayscale -> resize -> CLAHE -> unsharp sharpen ────────────
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_gray(image_path, cfg):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    gray = cv2.resize(gray, (cfg.win, cfg.win))
    if cfg.use_clahe:
        gray = _CLAHE.apply(gray)
    if cfg.use_sharpen:                              # unsharp mask (HPF, Topic 3)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return gray


# ── Feature extractors ───────────────────────────────────────────────────────
def make_hog(cfg):
    """Build a cv2.HOGDescriptor for the given config (cell stride = 1 cell)."""
    ppc, cpb = cfg.hog_ppc, cfg.hog_cpb
    return cv2.HOGDescriptor(
        _winSize=(cfg.win, cfg.win),
        _blockSize=(ppc * cpb, ppc * cpb),
        _blockStride=(ppc, ppc),
        _cellSize=(ppc, ppc),
        _nbins=cfg.hog_orient,
    )


def lbp_hist(gray):
    """8-neighbour Local Binary Pattern, returned as a 256-bin normalized hist."""
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                  (1, 1), (1, 0), (1, -1), (0, -1)]
    h, w = g.shape
    for i, (dy, dx) in enumerate(neighbours):
        shifted = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code |= ((shifted >= center).astype(np.uint8) << i)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    return hist / (hist.sum() + 1e-7)


def intensity_hist(gray, bins):
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()
    return hist / (hist.sum() + 1e-7)


def extract_features(image_path, cfg, hog):
    """Concatenate the enabled feature blocks into one 1-D vector."""
    gray = preprocess_gray(image_path, cfg)
    if gray is None:
        return None
    parts = []
    if cfg.use_hog:
        parts.append(hog.compute(gray).ravel())
    if cfg.use_lbp:
        parts.append(lbp_hist(gray))
    if cfg.use_hist:
        parts.append(intensity_hist(gray, cfg.hist_bins))
    return np.concatenate(parts).astype(np.float32)


# ── Image-level label from a YOLO .txt file ──────────────────────────────────
def label_from_yolo(label_path):
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


def load_split(split, cfg, include_aug=False):
    """Load (X, y, names) for a dataset split, deriving labels from YOLO txt."""
    image_dir = os.path.join(DATASET_DIR, "images", split)
    label_dir = os.path.join(DATASET_DIR, "labels", split)
    if not os.path.isdir(image_dir):
        print(f"[ERROR] Missing {image_dir}. Run the data-prep steps first "
              f"(setup_data -> convert_annotations -> build_dataset).")
        sys.exit(1)

    hog   = make_hog(cfg)
    files = sorted(f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    X, y, names = [], [], []
    t0 = time.time()
    for fname in files:
        stem = os.path.splitext(fname)[0]
        if "_aug" in stem and not include_aug:
            continue
        label = label_from_yolo(os.path.join(label_dir, stem + ".txt"))
        if label is None:
            continue
        feat = extract_features(os.path.join(image_dir, fname), cfg, hog)
        if feat is None:
            continue
        X.append(feat); y.append(label); names.append(fname)
    print(f"  '{split}': {len(X)} images, feature dim "
          f"{len(X[0]) if X else 0}  ({time.time() - t0:.1f}s)")
    return np.asarray(X, dtype=np.float32), y, names


# ── Model ────────────────────────────────────────────────────────────────────
def build_pipeline(C=10.0, gamma="scale", use_pca=False, pca_var=0.95,
                   probability=False):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    steps = [("scaler", StandardScaler())]
    if use_pca:
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    steps.append(("svm", SVC(kernel="rbf", C=C, gamma=gamma,
                             class_weight="balanced", probability=probability)))
    from sklearn.pipeline import Pipeline as _P
    return _P(steps)


# ── Decision-threshold helpers (safe vs threat) ──────────────────────────────
def apply_threshold(proba, classes, tau):
    """Predict 'safe' only if P(safe) >= tau, else the most likely threat."""
    safe_i = classes.index("safe")
    threat_idx = [i for i in range(len(classes)) if i != safe_i]
    out = []
    for row in proba:
        if row[safe_i] >= tau:
            out.append("safe")
        else:
            out.append(classes[max(threat_idx, key=lambda i: row[i])])
    return out


def predict_labels(bundle, X):
    pipe = bundle["pipeline"]
    tau  = bundle.get("threshold")
    classes = list(pipe.classes_)
    if tau is None or "safe" not in classes or not hasattr(pipe, "predict_proba"):
        return list(pipe.predict(X))
    try:
        proba = pipe.predict_proba(X)
    except Exception:
        return list(pipe.predict(X))
    return apply_threshold(proba, classes, tau)


# ── Reporting ────────────────────────────────────────────────────────────────
def report_classification(y_true, y_pred, title, save=True):
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
              "  NOTE: classification-only; localization (IoU) is handled by",
              "        the YOLO pipeline (run_inference.py).", "=" * 55]
    text = "\n".join(lines)
    print("\n" + text)
    if save:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/baseline_hog_svm_report.txt", "w") as f:
            f.write(text + "\n")
        print("\nReport -> outputs/baseline_hog_svm_report.txt")
    return cls_score, macro_f1


def save_bundle(pipeline, cfg, threshold):
    import joblib
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "config": cfg, "threshold": threshold},
                MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")


def load_bundle():
    import joblib
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No model at {MODEL_PATH}. Run: "
              f"python baseline/hog_svm.py train"); sys.exit(1)
    return joblib.load(MODEL_PATH)


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_train(args):
    cfg = FeatureConfig(
        use_sharpen=not args.no_sharpen,
        use_lbp=not args.no_lbp,
        use_hist=not args.no_hist,
    )
    print("Extracting features (train split)...")
    X, y, _ = load_split("train", cfg, include_aug=args.include_aug)
    if len(set(y)) < 2:
        print("[ERROR] Need at least two classes to train."); sys.exit(1)

    print(f"\nTraining RBF-SVM on {len(X)} samples "
          f"(PCA={'on' if args.pca else 'off'})...")
    t0 = time.time()
    pipe = build_pipeline(use_pca=args.pca, probability=False)
    pipe.fit(X, y)
    print(f"  done in {time.time() - t0:.1f}s")
    save_bundle(pipe, cfg, threshold=None)
    report_classification(y, list(pipe.predict(X)),
                          "BASELINE (HOG+LBP+HIST / SVM) — TRAIN-SET FIT")


def cmd_eval(args):
    bundle = load_bundle()
    cfg = bundle["config"]
    print(f"Loaded model: {MODEL_PATH} (threshold={bundle.get('threshold')})\n"
          f"Extracting features ({args.split} split)...")
    X, y, _ = load_split(args.split, cfg)
    report_classification(y, predict_labels(bundle, X),
                          f"BASELINE (HOG+LBP+HIST / SVM) — {args.split.upper()} SET")


def cmd_predict(args):
    bundle = load_bundle()
    cfg = bundle["config"]
    if not os.path.isdir(args.images):
        print(f"[ERROR] Folder not found: {args.images}"); sys.exit(1)
    hog = make_hog(cfg)
    files = sorted(f for f in os.listdir(args.images)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    feats, names = [], []
    for fname in files:
        feat = extract_features(os.path.join(args.images, fname), cfg, hog)
        if feat is not None:
            feats.append(feat); names.append(fname)
    preds = predict_labels(bundle, np.asarray(feats, dtype=np.float32))

    predictions = [{"image_name": n, "pred_label": p}
                   for n, p in zip(names, preds)]
    for n, p in zip(names, preds):
        print(f"  {n:<32} -> {p}")
    writer = SubmissionWriter()
    writer.write(predictions, args.out)
    writer.validate(args.out)
    writer.print_summary(args.out)


def cmd_tune(args):
    from sklearn.model_selection import GridSearchCV

    # ---- search grids -------------------------------------------------------
    if args.quick:
        hog_ppc_grid = [16]
        C_grid, gamma_grid = [1.0, 10.0], ["scale"]
    else:
        hog_ppc_grid = [8, 16]
        C_grid, gamma_grid = [1.0, 10.0, 100.0], ["scale", 0.001]

    base_cfg = FeatureConfig(use_sharpen=not args.no_sharpen,
                             use_lbp=not args.no_lbp,
                             use_hist=not args.no_hist)

    print("=" * 55)
    print("  GRID SEARCH — HOG params x SVM (C, gamma)"
          + ("  [+PCA]" if args.pca else ""))
    print("=" * 55)

    best = {"score": -1.0}
    log_lines = ["GRID SEARCH RESULTS", "=" * 55]
    for ppc in hog_ppc_grid:
        cfg = FeatureConfig(**{**asdict(base_cfg), "hog_ppc": ppc})
        print(f"\n[HOG ppc={ppc}] extracting train features...")
        Xtr, ytr, _ = load_split("train", cfg, include_aug=args.include_aug)

        pipe = build_pipeline(use_pca=args.pca, probability=False)
        grid = GridSearchCV(
            pipe,
            {"svm__C": C_grid, "svm__gamma": gamma_grid},
            scoring="f1_macro", cv=args.cv, n_jobs=-1, refit=True,
        )
        t0 = time.time()
        grid.fit(Xtr, ytr)
        msg = (f"  best CV macro-F1={grid.best_score_:.4f} "
               f"params={grid.best_params_}  ({time.time() - t0:.1f}s)")
        print(msg)
        log_lines.append(f"HOG ppc={ppc}: {msg.strip()}")
        if grid.best_score_ > best["score"]:
            best = {"score": grid.best_score_, "cfg": cfg,
                    "params": grid.best_params_}

    print("\n" + "=" * 55)
    print(f"  BEST: HOG ppc={best['cfg'].hog_ppc}  {best['params']}  "
          f"(CV macro-F1={best['score']:.4f})")
    print("=" * 55)

    # ---- refit best on full train (with probabilities for thresholding) -----
    cfg = best["cfg"]
    Xtr, ytr, _ = load_split("train", cfg, include_aug=args.include_aug)
    final = build_pipeline(C=best["params"]["svm__C"],
                           gamma=best["params"]["svm__gamma"],
                           use_pca=args.pca, probability=True)
    final.fit(Xtr, ytr)
    classes = list(final.classes_)

    # ---- tune the safe-vs-threat decision threshold on the VAL split --------
    best_tau = None
    if "safe" in classes:
        print("\nTuning safe/threat decision threshold on the val split...")
        Xva, yva, _ = load_split("val", cfg)
        proba = final.predict_proba(Xva)
        # baseline (argmax) macro-F1
        base_f1 = compute_macro_f1(
            compute_per_class_metrics(yva, list(final.predict(Xva))))
        best_tau, best_f1 = None, base_f1
        for tau in np.round(np.arange(0.10, 0.91, 0.05), 2):
            preds = apply_threshold(proba, classes, tau)
            f1 = compute_macro_f1(compute_per_class_metrics(yva, preds))
            log_lines.append(f"  tau={tau:.2f}  val macroF1={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)
        if best_tau is None:
            print(f"  no threshold beat argmax (val macroF1={base_f1:.4f}); "
                  f"using argmax.")
        else:
            print(f"  best threshold tau={best_tau:.2f} "
                  f"(val macroF1 {base_f1:.4f} -> {best_f1:.4f})")

    save_bundle(final, cfg, threshold=best_tau)

    # ---- final evaluation on the TEST split ---------------------------------
    bundle = {"pipeline": final, "config": cfg, "threshold": best_tau}
    Xte, yte, _ = load_split("test", cfg)
    report_classification(yte, predict_labels(bundle, Xte),
                          "BASELINE (TUNED) — TEST SET")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_tuning_report.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("Tuning log -> outputs/baseline_tuning_report.txt")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_feature_flags(sp):
        sp.add_argument("--pca", action="store_true",
                        help="Add PCA (0.95 variance) before the SVM")
        sp.add_argument("--no-sharpen", action="store_true")
        sp.add_argument("--no-lbp", action="store_true")
        sp.add_argument("--no-hist", action="store_true")
        sp.add_argument("--include-aug", action="store_true",
                        help="Also use augmented (_aug) training images")

    pt = sub.add_parser("train", help="Fit on the train split")
    add_feature_flags(pt)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="Score on a split (default: test)")
    pe.add_argument("--split", default="test", choices=["train", "val", "test"])
    pe.set_defaults(func=cmd_eval)

    pp = sub.add_parser("predict", help="Predict a folder -> submission CSV")
    pp.add_argument("--images", required=True)
    pp.add_argument("--out", default="outputs/predictions_hog_svm.csv")
    pp.set_defaults(func=cmd_predict)

    pn = sub.add_parser("tune", help="Grid-search HOG + SVM + decision threshold")
    add_feature_flags(pn)
    pn.add_argument("--cv", type=int, default=3, help="CV folds (default 3)")
    pn.add_argument("--quick", action="store_true", help="Smaller, faster grid")
    pn.set_defaults(func=cmd_tune)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
