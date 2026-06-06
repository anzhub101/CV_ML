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
cen454-baggage-detection/
├── preprocessing/
│   ├── config.py               # class map, thresholds, shared constants
│   ├── convert_annotations.py  # PNG masks -> YOLO .txt labels
│   ├── build_dataset.py        # train/val/test split assembly
│   ├── augment.py              # albumentations augmentation (train only)
│   ├── preprocess.py           # classical CV pipeline (Topics 3-6)
│   ├── verify_dataset.py       # pre-training format checks
│   └── visualize_labels.py     # draw boxes to sanity-check labels
├── training/
│   ├── data.yaml               # YOLO dataset config
│   ├── download_weights.py     # pre-fetch pretrained weights (offline prep)
│   ├── train.py                # fine-tune YOLO26
│   └── validate.py             # test-split evaluation
├── inference/
│   ├── postprocess.py          # filters, priority resolution, IoU
│   ├── predict.py              # single-image pipeline (+ TTA)
│   └── evaluate.py             # batch inference + submission CSV
├── data/
│   ├── raw/{GUN,knife,shuriken,safe}/      # original images (not committed)
│   ├── annotations/{GUN,knife,shuriken}/   # binary PNG masks
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

## Data Placement

Put your raw files here before running anything:

```
data/raw/GUN/         # gun X-ray images
data/raw/knife/       # knife X-ray images
data/raw/shuriken/    # shuriken X-ray images
data/raw/safe/        # safe bag images

data/annotations/GUN/         # binary mask PNGs (same filename stem as image)
data/annotations/knife/
data/annotations/shuriken/
```

Image and mask filenames must share the same stem, e.g.
`data/raw/GUN/P00096.png` ↔ `data/annotations/GUN/P00096.png`.

---

## Usage

### Full pipeline (one command)

```bash
bash run_all.sh
```

### Or step by step

```bash
python preprocessing/convert_annotations.py   # masks -> labels
python preprocessing/build_dataset.py         # split into train/val/test
python preprocessing/augment.py               # expand training set
python preprocessing/preprocess.py            # classical CV on all images
python preprocessing/verify_dataset.py        # confirm format is valid
python training/train.py                       # fine-tune YOLO26
python training/validate.py                    # evaluate on test split
```

### Generate predictions on a hidden test set (evaluation day)

```bash
python inference/evaluate.py --images hidden_test
# -> writes outputs/predictions.csv
```

### Predict on a single image

```bash
python inference/predict.py path/to/image.png --tta
```

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

---q