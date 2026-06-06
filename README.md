# CEN454 — Baggage Threat Detection

A computer-vision framework that classifies baggage X-ray images as **safe**,
**gun**, **knife**, or **shuriken**, and localizes any detected threat with a
bounding box.

The system combines **classical CV preprocessing** (course Topics 3–6) with a
fine-tuned **YOLO26** detector. The classical layer adapts raw X-ray images to
the detector's expected input domain (denoising, contrast enhancement, edge
sharpening, color normalization); YOLO26 performs classification and
localization in a single pass; a post-processing layer filters false positives
and resolves multi-object images.

---

## Architecture

```
Raw X-ray image
      |
      v
Classical CV preprocessing  (Topics 3-6)
  denoise -> CLAHE -> sharpen (HPF) -> color normalize -> resize
      |
      v
YOLO26 detector  (fine-tuned, NMS-free)
  -> label + bounding box
      |
      v
Post-processing
  confidence bands -> size filter -> metal-density check -> priority resolve
      |
      v
Output: {safe | gun | knife | shuriken}  + bbox
```

---

## Project Structure

```
Project_CEN454/
├── TrainData/                  # source dataset (images + masks; not committed)
├── preprocessing/
│   ├── config.py               # class map, paths, thresholds, constants
│   ├── setup_data.py           # stage TrainData/ -> data/raw + annotations
│   ├── convert_annotations.py  # instance-mask PNGs -> YOLO .txt labels
│   ├── build_dataset.py        # train/val/test split assembly
│   ├── augment.py              # albumentations augmentation (train only)
│   ├── preprocess.py           # classical CV pipeline (Topics 3-6)
│   ├── verify_dataset.py       # pre-training format checks
│   ├── visualize_masks.py      # render the near-black masks + derived boxes
│   └── visualize_labels.py     # draw boxes to sanity-check labels
├── training/
│   ├── data.yaml               # YOLO dataset config
│   ├── download_weights.py     # pre-fetch pretrained weights (offline prep)
│   ├── train.py                # fine-tune YOLO26
│   └── validate.py             # test-split evaluation
├── inference/
│   ├── postprocess.py          # filters, priority resolution, IoU
│   ├── quality_handler.py      # adaptive quality fixing
│   ├── localization.py         # morphological bbox refinement
│   ├── submission.py           # submission CSV writer/validator
│   └── visualize_results.py    # annotated prediction images
├── utils/
│   ├── metrics.py              # accuracy, macro F1, IoU, final score
│   └── logger.py
├── baseline/
│   └── hog_svm.py              # classical HOG + SVM classifier baseline
├── run_inference.py            # MASTER inference + evaluation entrypoint
├── data/                       # all auto-staged/generated (not committed)
│   ├── raw/{GUN,knife,shuriken,safe}/      # staged from TrainData/
│   ├── annotations/{GUN,knife,shuriken}/   # instance-indexed mask PNGs
│   ├── all_labels/                         # generated YOLO labels
│   └── dataset/                            # final YOLO structure
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── weights/                    # best.pt saved here after training
├── outputs/                    # predictions.csv, reports, visual checks
├── run_all.sh                  # end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pre-download pretrained weights (while you have internet)
python training/download_weights.py
```

---

## Data

The dataset ships in a **`TrainData/` folder at the project root**:

```
TrainData/
├── GUN/  knife/  shuriken/      # source X-ray images
├── safe/                        # safe bags (no threat)
└── annotations/
    ├── GUN/  knife/  shuriken/  # one mask PNG per image (same filename stem)
```

You do **not** copy these in by hand — `setup_data.py` stages them into
`data/raw/` and `data/annotations/` (symlinks by default; `--copy` to copy).
If your `TrainData` lives elsewhere, point to it:

```bash
export CEN454_SOURCE_DATA=/path/to/TrainData      # optional
python preprocessing/setup_data.py                # or: --copy / --source PATH
```

### About the annotation masks (important)

The mask PNGs are **instance-indexed**, not 0/255 binary. Background is 0 and
each separate object instance gets a tiny integer id (1, 2, 3, …). Because
those ids are small next to 255, the masks look almost solid **black** in an
image viewer — but they are not empty. The converter treats any pixel `> 0` as
a threat and turns each instance id into its own YOLO box; the **class comes
from the folder** the mask lives in. (Multiplying by 255 only makes them
*visible* — run `visualize_masks.py` to see them properly colored.)

Two cases are skipped automatically, to keep labels clean:

- **Ambiguous multi-threat scans** — the same bag filed under two class folders
  (e.g. `gun` *and* `knife`) with one shared mask that doesn't say which object
  is which class.
- **Empty masks** — a threat-folder image whose mask has no annotated region.

---

## Usage

Everything runs from the project root (`Project_CEN454/`).

### Option A — the whole thing in one command

```bash
bash run_all.sh
```

This runs every step below in order: stage data → build labels → split →
augment → preprocess → **train YOLO26** → **evaluate the test split with the
full project criteria**.

### Option B — step by step

**1. Prepare the dataset** (turns `TrainData/` into a YOLO dataset):

```bash
python preprocessing/setup_data.py            # stage TrainData -> data/
python preprocessing/convert_annotations.py   # instance masks -> YOLO labels
python preprocessing/visualize_masks.py       # (optional) eyeball the masks
python preprocessing/build_dataset.py         # split into train/val/test
python preprocessing/augment.py               # expand the training split
python preprocessing/preprocess.py            # classical CV on all images
python preprocessing/verify_dataset.py        # confirm the format is valid
```

**2. Train the YOLO26 model**

```bash
python training/download_weights.py   # once, while online: fetch yolo26s.pt
python training/train.py              # fine-tune; saves weights/best.pt
```

Training uses a high epoch ceiling (150) with early stopping (`patience=25`),
so it stops itself at the best validation mAP — see *How Training Stops* below.
Tune `EPOCHS`, `BATCH`, `FREEZE`, etc. at the top of `training/train.py`.
Optionally inspect raw detector metrics on the test split:

```bash
python training/validate.py           # YOLO mAP50, mAP50-95, precision/recall
```

**3. Run the full framework and compute the testing criteria**

This is the end-to-end inference pipeline (quality-fix → classical CV →
YOLO26 → post-processing → localization refinement → TTA). Point it at the
test images **and** their YOLO ground-truth labels to score the exact project
criteria:

```bash
python run_inference.py \
    --images data/dataset/images/test \
    --labels data/dataset/labels/test
```

It prints and saves (`outputs/evaluation_report.txt`):

```
Accuracy, Macro F1   -> Classification Score = 0.7*Acc + 0.3*F1   (70%)
Mean IoU (>=0.5)     -> Localization Score   = mean IoU            (30%)
                        FINAL SCORE          = 0.7*Cls + 0.3*Loc
```

### Evaluation day — predict on the hidden test set (no labels)

```bash
python run_inference.py --images hidden_test
# -> writes outputs/predictions.csv  (Image Name, Predicted Label)
# add --no-tta for speed, --no-viz to skip annotated images
```

---

## Classical baseline (HOG + SVM)

A small, fully classical alternative to the YOLO detector, included to
demonstrate the course's classical computer-vision + machine-learning
techniques: **Histogram of Oriented Gradients (HOG)** edge/gradient descriptors
+ **CLAHE** contrast normalization + a **Support Vector Machine** classifier.
It does the **classification** task only (HOG+SVM does not localize), reuses the
same dataset split, and is scored with the same metrics — so its numbers are
directly comparable to the detector's classification score. No deep-learning
framework needed (just OpenCV + scikit-learn).

Run after the data-prep steps have produced `data/dataset/`:

```bash
python baseline/hog_svm.py train     # fit on the train split -> baseline/hog_svm.joblib
python baseline/hog_svm.py eval      # score on the test split (acc, macro F1)
python baseline/hog_svm.py predict --images hidden_test   # -> CSV submission
```

Reference result on this dataset's test split (yours will vary with the data):
`Accuracy ≈ 0.80, Macro F1 ≈ 0.83, Classification Score ≈ 0.81`. Use it as a
sanity floor — the fine-tuned YOLO26 pipeline should beat it, and it also gives
you a classical fallback that needs no GPU.

---

## How Training Stops (epochs vs. patience)

The optimal number of epochs depends on the dataset's size and complexity, not
on the model size, so it is not hard-coded. Training uses a **high epoch
ceiling** (150) together with **early stopping** (`patience=25`): after each
epoch YOLO checks validation mAP, and if it does not improve for 25 consecutive
epochs the run stops and keeps the best checkpoint. This finds the sweet spot
automatically and avoids both underfitting and overfitting.

For very small datasets, set `FREEZE = 10` in `training/train.py` to freeze the
backbone and train only the detection head.

---

## Notes

- **YOLO26 is NMS-free**, so there is no Non-Maximum-Suppression step in
  post-processing. If you swap in an older model (e.g. `yolo11s.pt` via
  `MODEL_WEIGHTS` in `training/train.py`), the pipeline still works.
- **"safe" is not a YOLO class.** Safe images carry empty label files
  (background). When the model detects nothing above threshold, the output is
  `safe`.
- **Scoring** (per project spec):
  `Classification = 0.7*Accuracy + 0.3*MacroF1`,
  `Localization = mean IoU (≥0.5 counts)`,
  `Final = 0.7*Classification + 0.3*Localization`.

---

## License

Coursework project for CEN454 Computer Vision and Machine Learning.
