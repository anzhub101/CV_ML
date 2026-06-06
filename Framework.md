# CEN454 — Baggage Threat Detection Framework
## Complete Pipeline: Classical CV Preprocessing + YOLOv8 Fine-tuning

---

## Framework Overview

```
RAW IMAGE
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA PREPARATION                                      │
│  Collect → Annotate → Convert → Augment → Split                │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: CLASSICAL CV PREPROCESSING  (Topics 3–6)             │
│  Denoise → Enhance → Sharpen → Normalize                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: YOLO FINE-TUNING                                      │
│  Configure → Train → Validate → Save Best Weights              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: INFERENCE PIPELINE                                    │
│  Preprocess → Detect → Post-process → Edge Case Handling       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: EVALUATION + CSV GENERATION                           │
│  Accuracy → Macro F1 → IoU → Final Score → predictions.csv    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
TrainData/                          # SOURCE data, ships next to the project
│   ├── GUN/ knife/ shuriken/       # source X-ray images
│   ├── safe/                       # safe bags (no threat)
│   └── annotations/
│       └── GUN/ knife/ shuriken/   # instance-indexed mask PNGs

CEN454_Project/                     # (= cen454-baggage-detection/)
│
├── data/                           # AUTO-staged + generated (gitignored)
│   ├── raw/                        # staged from ../TrainData by setup_data.py
│   │   ├── GUN/                    # source images (~800)
│   │   ├── knife/                  # source images (~900)
│   │   ├── shuriken/               # source images (~90)
│   │   └── safe/                   # source images (~110)
│   │
│   ├── annotations/                # staged from ../TrainData/annotations
│   │   ├── GUN/                    # instance-indexed PNG masks (NOT 0/255)
│   │   ├── knife/                  # instance-indexed PNG masks
│   │   └── shuriken/               # instance-indexed PNG masks
│   │
│   ├── all_labels/                 # generated YOLO .txt labels
│   └── dataset/                    # YOLO-formatted, auto-generated
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── preprocessing/
│   ├── config.py                   # class map, data paths, thresholds
│   ├── setup_data.py               # stage ../TrainData → data/raw + annotations
│   ├── convert_annotations.py     # instance-mask PNG → YOLO .txt
│   ├── build_dataset.py            # train/val/test split assembly
│   ├── preprocess.py               # Classical CV pipeline
│   ├── augment.py                  # Augmentation pipeline
│   ├── verify_dataset.py           # pre-training format checks
│   ├── visualize_masks.py          # render the near-black masks + boxes
│   └── visualize_labels.py         # draw boxes to sanity-check labels
│
├── training/
│   ├── data.yaml                   # YOLO dataset config
│   ├── train.py                    # Fine-tuning script
│   └── runs/                       # YOLO training outputs (auto)
│
├── inference/
│   ├── predict.py                  # Single image inference
│   ├── evaluate.py                 # Batch evaluation + CSV
│   └── postprocess.py              # Label resolution, edge cases
│
├── weights/
│   └── best.pt                     # Best fine-tuned model
│
└── outputs/
    ├── predictions.csv             # Final submission file
    └── evaluation_report.txt       # Accuracy, F1, IoU results
```

---

## PHASE 1 — Data Preparation

### 1.1 Class Map and Label Assignment

```python
# preprocessing/config.py

CLASS_MAP = {
    'gun':      0,
    'knife':    1,
    'shuriken': 2
    # safe = no label file (empty .txt = background class)
}

# Threat priority for multi-detection resolution
THREAT_PRIORITY = {
    'gun':      3,
    'knife':    2,
    'shuriken': 1,
    'safe':     0
}

# Confidence thresholds
CONF_HIGH   = 0.65   # accept detection
CONF_LOW    = 0.35   # reject as safe
# Between 0.35-0.65: apply secondary validation

# Minimum bounding box area relative to image
# Smaller than this = likely keychain/toy, not real weapon
MIN_BBOX_AREA_RATIO = 0.005

# Annotation masks are INSTANCE-INDEXED, not 0/255 binary:
#   0 = background, 1,2,3,... = separate object instances.
# Any pixel >= 1 is foreground (a threat); each id becomes its own box.
MASK_FOREGROUND_MIN = 1
MIN_CONTOUR_AREA    = 50   # drop speckle noise below this many mask pixels
```

`config.py` also defines the data-path constants (`SOURCE_DATA_DIR`, `RAW_DIR`,
`ANNOTATION_DIR`, `ALL_LABELS_DIR`, `DATASET_DIR`) and a case-insensitive
`resolve_class_dir()` helper, because the dataset ships its folders as `GUN`
(uppercase) while the class map uses `gun`.

---

### 1.1b Data Staging — TrainData → data/

The original dataset ships in a sibling **`TrainData/`** folder (next to the
project), laid out as `TrainData/{GUN,knife,shuriken,safe}` for images and
`TrainData/annotations/{GUN,knife,shuriken}` for masks. `setup_data.py` mirrors
that into the `data/raw` and `data/annotations` layout the rest of the pipeline
expects — symlinks by default (instant, no duplicated disk), or `--copy`.

```python
# preprocessing/setup_data.py  (run first)
#   python preprocessing/setup_data.py            # symlink
#   python preprocessing/setup_data.py --copy     # physical copy
#   CEN454_SOURCE_DATA=/path/to/TrainData python preprocessing/setup_data.py
```

---

### 1.2 Annotation Conversion — Instance Mask PNG to YOLO Format

> **The masks are NOT 0/255 binary images — they are instance-indexed.**
> Background is `0`; each separate object instance is painted with a tiny
> integer id (`1, 2, 3, …`). Those ids look almost solid **black** in a viewer
> because they are tiny next to 255, but the masks are not empty. A naive
> `cv2.threshold(mask, 127, 255)` wipes every value 1–4 to zero and produces
> empty labels (the original bug). Instead we treat any pixel `>= 1` as
> foreground and convert **each unique non-zero id into its own box**; the
> **class comes from the folder** the mask lives in. Run
> `preprocessing/visualize_masks.py` to see the masks recolored per instance.
>
> Two cases are skipped to keep labels clean: **ambiguous multi-threat scans**
> (the same bag filed under two class folders with one shared mask that doesn't
> say which instance is which class) and **empty masks** (a threat-folder image
> with no annotated region).

```python
# preprocessing/convert_annotations.py

import cv2
import numpy as np
import os
from config import CLASS_MAP, resolve_class_dir

def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height


def find_multiclass_stems(raw_dir):
    """Stems that appear under MORE THAN ONE class folder = ambiguous
    multi-threat scans (shared mask doesn't say which object is which class)."""
    stem_classes = {}
    for class_name in CLASS_MAP:
        img_dir = resolve_class_dir(raw_dir, class_name)
        if img_dir is None:
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(f)[0]
                stem_classes.setdefault(stem, set()).add(class_name)
    return {s for s, classes in stem_classes.items() if len(classes) > 1}


def mask_to_yolo_bbox(mask_path, image_w, image_h, min_contour_area=50):
    """
    Convert an INSTANCE-INDEXED mask into YOLO boxes — one per object instance.
    Returns list of (x_center, y_center, width, height) normalized to [0,1].
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # NEAREST keeps the integer ids crisp (no invented in-between values)
    if (mask.shape[1], mask.shape[0]) != (image_w, image_h):
        mask = cv2.resize(mask, (image_w, image_h),
                          interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bboxes = []
    # Each unique non-zero id is one object — NOT a 127 threshold.
    for inst_id in [v for v in np.unique(mask) if v >= 1]:
        inst = np.where(mask == inst_id, 255, 0).astype(np.uint8)
        inst = cv2.morphologyEx(inst, cv2.MORPH_CLOSE, kernel)  # fill gaps
        inst = cv2.morphologyEx(inst, cv2.MORPH_OPEN, kernel)   # drop speckle
        cnts, _ = cv2.findContours(inst, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_contour_area]
        if not cnts:
            continue
        # Union box covering all of this instance's contours
        xs, ys, xe, ye = [], [], [], []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
        x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
        bw, bh = x2 - x1, y2 - y1
        bboxes.append(((x1 + bw / 2) / image_w, (y1 + bh / 2) / image_h,
                       bw / image_w, bh / image_h))
    return bboxes


def convert_all_annotations(raw_dir, annotation_dir, output_label_dir):
    """
    Convert all class instance-masks to YOLO .txt labels.
    Safe images get an empty .txt; ambiguous multi-threat scans and empty masks
    are skipped (no label file -> excluded from the dataset).
    """
    os.makedirs(output_label_dir, exist_ok=True)
    multiclass_stems = find_multiclass_stems(raw_dir)

    for class_name, class_id in CLASS_MAP.items():
        # Folders ship as 'GUN'/'knife'/'shuriken' — resolve case-insensitively
        img_dir  = resolve_class_dir(raw_dir, class_name)
        mask_dir = resolve_class_dir(annotation_dir, class_name)
        if img_dir is None or mask_dir is None:
            print(f"[SKIP] missing folder for class: {class_name}")
            continue

        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            stem = os.path.splitext(img_file)[0]
            if stem in multiclass_stems:            # ambiguous -> skip
                continue

            mask_path = os.path.join(mask_dir, stem + '.png')
            label_out = os.path.join(output_label_dir, stem + '.txt')
            if not os.path.exists(mask_path):
                continue
            image_w, image_h = get_image_dimensions(
                os.path.join(img_dir, img_file))

            bboxes = mask_to_yolo_bbox(mask_path, image_w, image_h)
            if not bboxes:                          # empty mask -> skip
                continue
            with open(label_out, 'w') as f:
                for (xc, yc, w, h) in bboxes:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Safe class — empty label files (no detections = background = safe)
    safe_dir = resolve_class_dir(raw_dir, 'safe')
    if safe_dir is not None:
        for img_file in os.listdir(safe_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            stem = os.path.splitext(img_file)[0]
            open(os.path.join(output_label_dir, stem + '.txt'), 'w').close()


if __name__ == '__main__':
    convert_all_annotations(
        raw_dir        = 'data/raw',
        annotation_dir = 'data/annotations',
        output_label_dir = 'data/all_labels'
    )
    print("Annotation conversion complete.")
```

---

### 1.3 Dataset Split and YOLO Folder Assembly

```python
# preprocessing/build_dataset.py

import os
import shutil
import random
from pathlib import Path

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

def build_yolo_dataset(raw_dir, label_dir, output_dir, seed=42):
    random.seed(seed)
    splits = ['train', 'val', 'test']

    for split in splits:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    all_classes = list(os.listdir(raw_dir))
    all_classes.append('safe')

    for class_name in all_classes:
        class_img_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_img_dir):
            continue

        images = [
            f for f in os.listdir(class_img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val   = int(n * SPLIT_RATIOS[1])

        split_assignments = (
            [(img, 'train') for img in images[:n_train]] +
            [(img, 'val')   for img in images[n_train:n_train + n_val]] +
            [(img, 'test')  for img in images[n_train + n_val:]]
        )

        for img_file, split in split_assignments:
            stem      = os.path.splitext(img_file)[0]
            label_src = os.path.join(label_dir, stem + '.txt')
            img_src   = os.path.join(class_img_dir, img_file)
            img_dst   = os.path.join(output_dir, 'images', split, img_file)
            lbl_dst   = os.path.join(output_dir, 'labels', split, stem + '.txt')

            shutil.copy2(img_src, img_dst)
            if os.path.exists(label_src):
                shutil.copy2(label_src, lbl_dst)
            else:
                open(lbl_dst, 'w').close()

    print(f"Dataset built at: {output_dir}")


if __name__ == '__main__':
    build_yolo_dataset(
        raw_dir    = 'data/raw',
        label_dir  = 'data/all_labels',
        output_dir = 'data/dataset'
    )
```

---

### 1.4 Data Augmentation Pipeline

```python
# preprocessing/augment.py
# Run this on training split only before YOLO training

import cv2
import numpy as np
import os
import albumentations as A

# Combined augmentation for X-ray baggage domain
AUGMENTATION_PIPELINE = A.Compose([

    # --- Geometric ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.4),
    A.RandomScale(scale_limit=0.2, p=0.3),

    # --- Quality degradation (simulate low-quality scanners) ---
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
        A.MedianBlur(blur_limit=5),
    ], p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ImageCompression(quality_lower=40, quality_upper=85, p=0.25),
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),

    # --- Photometric (simulate different scanner color profiles) ---
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.4
    ),
    A.ToGray(p=0.15),           # simulate grayscale X-ray scanners
    A.InvertImg(p=0.10),        # simulate inverted X-ray output

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.3          # discard bbox if <30% visible after crop
))


def augment_training_set(image_dir, label_dir, multiplier=3):
    """
    Augment training images by `multiplier` times.
    Writes augmented images and updated labels back to same directory.
    """
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for img_file in image_files:
        stem      = os.path.splitext(img_file)[0]
        img_path  = os.path.join(image_dir, img_file)
        lbl_path  = os.path.join(label_dir, stem + '.txt')

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Read YOLO labels
        bboxes, class_labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

        for i in range(multiplier):
            try:
                result = AUGMENTATION_PIPELINE(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            except Exception:
                continue

            aug_img   = result['image']
            aug_boxes = result['bboxes']
            aug_cls   = result['class_labels']

            aug_name  = f"{stem}_aug{i}"
            aug_img_path = os.path.join(image_dir, aug_name + '.png')
            aug_lbl_path = os.path.join(label_dir,  aug_name + '.txt')

            cv2.imwrite(aug_img_path, aug_img)
            with open(aug_lbl_path, 'w') as f:
                for cls, box in zip(aug_cls, aug_boxes):
                    f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
```

---

## PHASE 2 — Classical CV Preprocessing Pipeline

```python
# preprocessing/preprocess.py
# Applied to every image before YOLO training and at inference time

import cv2
import numpy as np


# ─────────────────────────────────────────────────────
# STEP 1: Image Quality Assessment
# ─────────────────────────────────────────────────────
def assess_blur(image):
    """Laplacian variance — low value = blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def assess_noise(image):
    """Standard deviation of high-frequency components."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blur.astype(float)
    return np.std(noise)


def is_grayscale(image, threshold=5):
    """Check if image is effectively grayscale."""
    return np.std(image[:,:,0].astype(float) - image[:,:,1].astype(float)) < threshold


# ─────────────────────────────────────────────────────
# STEP 2: Denoising   (Topic 4 — Image Restoration)
# ─────────────────────────────────────────────────────
def denoise(image, noise_level):
    """
    Apply appropriate denoising based on measured noise level.
    - Low noise  → light Gaussian LPF
    - High noise → stronger median filter
    """
    if noise_level < 5:
        return image  # clean image, skip denoising

    if noise_level < 15:
        # Gaussian LPF for mild noise
        return cv2.GaussianBlur(image, (3, 3), 0)
    else:
        # Median filter for heavy/impulse noise
        return cv2.medianBlur(image, 5)


# ─────────────────────────────────────────────────────
# STEP 3: Contrast Enhancement   (Topic 5 — Color)
# ─────────────────────────────────────────────────────
def enhance_contrast(image):
    """
    CLAHE applied in LAB space to improve local contrast
    without blowing out highlights.
    Enhances visibility of metallic threat objects in X-ray.
    """
    lab   = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────
# STEP 4: Edge Sharpening   (Topic 3 — HPF)
# ─────────────────────────────────────────────────────
def sharpen_edges(image, blur_score):
    """
    Unsharp masking (approximates HPF) to enhance weapon contours.
    Only applied when image is blurry.
    """
    if blur_score >= 100:
        return image  # image is sharp enough, skip

    # Unsharp mask: original + (original - blurred) * amount
    blurred  = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


# ─────────────────────────────────────────────────────
# STEP 5: Color Normalization   (Topic 5)
# ─────────────────────────────────────────────────────
def normalize_color_domain(image):
    """
    Standardize image to X-ray-like appearance.
    Handles: grayscale input, inverted images, different scanner palettes.
    """
    # Case 1: Grayscale → apply pseudo-color mapping
    if is_grayscale(image):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Case 2: Detect and correct inverted X-ray
    # In standard X-ray, background is dark (low mean)
    mean_val = np.mean(image)
    if mean_val > 180:
        image = cv2.bitwise_not(image)

    # Case 3: Normalize to [0,255] range
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return image.astype(np.uint8)


# ─────────────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────
def preprocess(image_path=None, image_array=None):
    """
    Full classical CV preprocessing pipeline.
    Accepts either a file path or a numpy array.
    Returns preprocessed numpy array ready for YOLO.
    """
    if image_array is not None:
        image = image_array.copy()
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

    # Step 1: Assess
    blur_score  = assess_blur(image)
    noise_level = assess_noise(image)

    # Step 2: Denoise (Topic 4)
    image = denoise(image, noise_level)

    # Step 3: Contrast enhancement (Topic 5)
    image = enhance_contrast(image)

    # Step 4: Sharpen edges (Topic 3 — HPF)
    image = sharpen_edges(image, blur_score)

    # Step 5: Color normalization (Topic 5)
    image = normalize_color_domain(image)

    # Step 6: Final resize to YOLO input size
    image = cv2.resize(image, (640, 640))

    return image


# ─────────────────────────────────────────────────────
# Preprocess entire training dataset
# ─────────────────────────────────────────────────────
def preprocess_dataset(image_dir, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            processed = preprocess(image_path=os.path.join(image_dir, fname))
            cv2.imwrite(os.path.join(output_dir, fname), processed)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
```

---

## PHASE 3 — YOLO Fine-tuning

### 3.1 YOLO Dataset Config

```yaml
# training/data.yaml

path: ./data/dataset
train: images/train
val:   images/val
test:  images/test

nc: 3
names:
  0: gun
  1: knife
  2: shuriken

# safe class = no label file (empty .txt)
```

---

### 3.2 Training Script

```python
# training/train.py

from ultralytics import YOLO
import torch
import os

# ─────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────
CONFIG = {
    'model':       'yolov8s.pt',    # pretrained on COCO — start here
    'data':        'training/data.yaml',
    'epochs':      150,
    'imgsz':       640,
    'batch':       16,              # reduce to 8 if GPU memory is limited
    'lr0':         0.001,           # initial learning rate
    'lrf':         0.01,            # final LR as fraction of lr0
    'momentum':    0.937,
    'weight_decay':0.0005,
    'warmup_epochs':3.0,
    'patience':    25,              # early stopping patience
    'augment':     True,            # YOLO built-in augmentation ON
    'mosaic':      1.0,             # mosaic augmentation (4 images combined)
    'mixup':       0.1,             # mixup augmentation
    'degrees':     10.0,            # rotation augmentation
    'flipud':      0.2,
    'fliplr':      0.5,
    'hsv_h':       0.015,
    'hsv_s':       0.7,
    'hsv_v':       0.4,
    'scale':       0.5,
    'device':      '0' if torch.cuda.is_available() else 'cpu',
    'project':     'training/runs',
    'name':        'baggage_v1',
    'exist_ok':    True,
    'save':        True,
    'save_period': 10,              # save checkpoint every 10 epochs
    'workers':     4,
    'verbose':     True,
}


def train():
    model = YOLO(CONFIG['model'])

    results = model.train(
        data         = CONFIG['data'],
        epochs       = CONFIG['epochs'],
        imgsz        = CONFIG['imgsz'],
        batch        = CONFIG['batch'],
        lr0          = CONFIG['lr0'],
        lrf          = CONFIG['lrf'],
        momentum     = CONFIG['momentum'],
        weight_decay = CONFIG['weight_decay'],
        warmup_epochs= CONFIG['warmup_epochs'],
        patience     = CONFIG['patience'],
        augment      = CONFIG['augment'],
        mosaic       = CONFIG['mosaic'],
        mixup        = CONFIG['mixup'],
        degrees      = CONFIG['degrees'],
        flipud       = CONFIG['flipud'],
        fliplr       = CONFIG['fliplr'],
        hsv_h        = CONFIG['hsv_h'],
        hsv_s        = CONFIG['hsv_s'],
        hsv_v        = CONFIG['hsv_v'],
        scale        = CONFIG['scale'],
        device       = CONFIG['device'],
        project      = CONFIG['project'],
        name         = CONFIG['name'],
        exist_ok     = CONFIG['exist_ok'],
        save         = CONFIG['save'],
        save_period  = CONFIG['save_period'],
        workers      = CONFIG['workers'],
        verbose      = CONFIG['verbose'],
    )

    # Copy best weights to central location
    best_src = f"training/runs/{CONFIG['name']}/weights/best.pt"
    os.makedirs('weights', exist_ok=True)
    import shutil
    shutil.copy2(best_src, 'weights/best.pt')
    print(f"Best weights saved to weights/best.pt")
    print(f"Val mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


if __name__ == '__main__':
    train()
```

---

### 3.3 Training Process Flow

```
START
  │
  ▼
Load YOLOv8s pretrained weights (COCO)
  │
  ▼
┌─────────────────────────────────────────┐
│  For each epoch (1 → 150):             │
│                                         │
│  1. Load batch of preprocessed images  │
│  2. Apply YOLO built-in augmentation   │
│     (mosaic, flip, HSV jitter)         │
│  3. Forward pass through backbone      │
│     C2f blocks → feature pyramid      │
│  4. Compute losses:                    │
│     - Box loss (bounding box regression│
│     - Classification loss              │
│     - Distribution Focal Loss (DFL)   │
│  5. Backpropagate + update weights     │
│  6. Every epoch: validate on val set  │
│     - Compute mAP50, mAP50-95         │
│     - Save if best val mAP so far     │
│  7. Check early stopping (patience=25)│
└─────────────────────────────────────────┘
  │
  ▼
Training ends (epoch 150 or early stop)
  │
  ▼
Load best.pt (highest val mAP checkpoint)
  │
  ▼
Evaluate on held-out test set
  │
  ▼
Save weights/best.pt
```

---

## PHASE 4 — Inference Pipeline

### 4.1 Post-processing Module

```python
# inference/postprocess.py

import numpy as np
import cv2
from config import THREAT_PRIORITY, CONF_HIGH, CONF_LOW, MIN_BBOX_AREA_RATIO

CLASS_NAMES = {0: 'gun', 1: 'knife', 2: 'shuriken'}


def validate_detection_density(image, bbox_xyxy):
    """
    Classical CV validation — Topic 5 (Color Processing).
    Real metal weapons appear as high-density (bright) regions
    in the blue channel of pseudo-colored X-ray.
    Rejects low-density detections (plastic toys, keychains).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return False

    roi  = image[y1:y2, x1:x2]
    blue = roi[:, :, 0]  # blue channel = high density in X-ray pseudo-color
    mean_density = np.mean(blue)
    return mean_density >= 100  # calibrate this threshold on your training data


def validate_bbox_size(bbox_xyxy, image_shape):
    """
    Filter out tiny detections (keychains, toys, jewelry).
    A real weapon occupies at least MIN_BBOX_AREA_RATIO of the image.
    """
    x1, y1, x2, y2 = bbox_xyxy
    box_area  = (x2 - x1) * (y2 - y1)
    img_area  = image_shape[0] * image_shape[1]
    return (box_area / img_area) >= MIN_BBOX_AREA_RATIO


def resolve_detections(detections, image):
    """
    Given a list of YOLO detections, resolve to a single label + bbox.

    detections: list of dicts with keys:
        class_id, class_name, confidence, bbox_xyxy

    Returns: (label, bbox) or ('safe', None)
    """
    if not detections:
        return 'safe', None

    valid = []
    for det in detections:
        conf  = det['confidence']
        bbox  = det['bbox_xyxy']
        cls   = det['class_name']

        # Reject low-confidence detections
        if conf < CONF_LOW:
            continue

        # Size check — filter keychains/toys
        if not validate_bbox_size(bbox, image.shape):
            continue

        # Metal density check — only in medium-confidence band
        if conf < CONF_HIGH:
            if not validate_detection_density(image, bbox):
                continue

        valid.append(det)

    if not valid:
        return 'safe', None

    # Resolve to highest-priority, highest-confidence detection
    best = max(valid, key=lambda d: (
        THREAT_PRIORITY[d['class_name']],
        d['confidence']
    ))

    return best['class_name'], best['bbox_xyxy']


def compute_iou(pred_bbox, gt_bbox):
    """
    Compute IoU between predicted and ground truth bounding boxes.
    Both in [x1, y1, x2, y2] format.
    """
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area   = (gx2 - gx1) * (gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
```

---

### 4.2 Main Inference Script

```python
# inference/predict.py

import cv2
import os
import numpy as np
from ultralytics import YOLO
from preprocessing.preprocess import preprocess
from inference.postprocess import resolve_detections, CLASS_NAMES

MODEL_PATH = 'weights/best.pt'
CONF_THRESHOLD = 0.35  # minimum confidence to consider a detection


def load_model():
    model = YOLO(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    return model


def predict_single(model, image_path=None, image_array=None, use_tta=False):
    """
    Full inference pipeline for a single image.
    Returns: label (str), bbox (list [x1,y1,x2,y2] or None)
    """
    # Step 1: Load image
    if image_array is not None:
        raw = image_array
    else:
        raw = cv2.imread(image_path)
        if raw is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

    # Step 2: Classical CV preprocessing (Phase 2)
    processed = preprocess(image_array=raw)

    # Step 3: Run YOLO inference
    if use_tta:
        label, bbox = predict_with_tta(model, processed, raw)
    else:
        label, bbox = run_yolo(model, processed, raw)

    return label, bbox


def run_yolo(model, processed_image, original_image):
    """Single-pass YOLO inference."""
    results = model(processed_image, conf=CONF_THRESHOLD, verbose=False)
    boxes   = results[0].boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].tolist()

            if cls_id in CLASS_NAMES:
                detections.append({
                    'class_id':   cls_id,
                    'class_name': CLASS_NAMES[cls_id],
                    'confidence': conf,
                    'bbox_xyxy':  xyxy
                })

    return resolve_detections(detections, original_image)


def predict_with_tta(model, processed_image, original_image):
    """
    Test-Time Augmentation — run 5 variants, take majority vote on label.
    Significantly improves robustness on edge cases.
    """
    variants = [
        processed_image,
        cv2.flip(processed_image, 1),
        cv2.flip(processed_image, 0),
        cv2.convertScaleAbs(processed_image, alpha=1.2, beta=10),
        cv2.convertScaleAbs(processed_image, alpha=0.8, beta=-10),
    ]

    all_labels = []
    all_bboxes = []

    for variant in variants:
        label, bbox = run_yolo(model, variant, original_image)
        all_labels.append(label)
        if bbox is not None:
            all_bboxes.append(bbox)

    # Majority vote on label
    final_label = max(set(all_labels), key=all_labels.count)

    # Use first bbox if threat detected
    final_bbox = all_bboxes[0] if all_bboxes and final_label != 'safe' else None

    return final_label, final_bbox
```

---

## PHASE 5 — Evaluation and CSV Generation

```python
# inference/evaluate.py

import os
import csv
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, classification_report
from inference.predict import load_model, predict_single
from inference.postprocess import compute_iou

LABEL_NAMES = ['safe', 'gun', 'knife', 'shuriken']
IOU_THRESHOLD = 0.5


def run_evaluation(test_image_dir, ground_truth_csv=None, use_tta=True):
    """
    Run full evaluation on test images.
    Generates predictions.csv for submission.
    Optionally computes metrics if ground truth is available.
    """
    model = load_model()

    image_files = sorted([
        f for f in os.listdir(test_image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    predictions = []

    print(f"Running inference on {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(test_image_dir, img_file)
        try:
            label, bbox = predict_single(
                model,
                image_path=img_path,
                use_tta=use_tta
            )
        except Exception as e:
            print(f"[ERROR] {img_file}: {e}")
            label, bbox = 'safe', None

        predictions.append({
            'image_name':  img_file,
            'pred_label':  label,
            'pred_bbox':   bbox
        })

    # ── Write submission CSV ──────────────────────────────
    write_submission_csv(predictions, 'outputs/predictions.csv')

    # ── Compute metrics if ground truth available ─────────
    if ground_truth_csv:
        compute_metrics(predictions, ground_truth_csv)

    return predictions


def write_submission_csv(predictions, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Image Name', 'Predicted Label'])
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                'Image Name':     p['image_name'],
                'Predicted Label': p['pred_label']
            })
    print(f"Submission CSV written to: {output_path}")


def compute_metrics(predictions, ground_truth_csv):
    """
    Compute full evaluation metrics against ground truth.
    """
    import pandas as pd
    gt_df = pd.read_csv(ground_truth_csv)
    gt_map = dict(zip(gt_df['Image Name'], gt_df['True Label']))

    y_true, y_pred = [], []

    for p in predictions:
        true_label = gt_map.get(p['image_name'])
        if true_label is None:
            continue
        y_true.append(true_label)
        y_pred.append(p['pred_label'])

    # ── Classification Metrics ────────────────────────────
    accuracy   = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average='macro', labels=LABEL_NAMES, zero_division=0)
    class_score = 0.7 * accuracy + 0.3 * macro_f1

    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Macro F1-Score:       {macro_f1:.4f}")
    print(f"Classification Score: {class_score:.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=LABEL_NAMES, zero_division=0))

    # ── Localization Metrics ──────────────────────────────
    # (Only for images where prediction and truth are both threat)
    # Requires ground truth bounding boxes in a separate file
    print("="*50)
    print("LOCALIZATION RESULTS")
    print("="*50)
    print("(IoU computation requires ground truth bbox file)")
    print(f"IoU Threshold: {IOU_THRESHOLD}")

    # Final Score
    # Assuming localization_score will be filled with actual IoU
    localization_score = 0.0  # placeholder
    final_score = 0.7 * class_score + 0.3 * localization_score

    print(f"\nFinal Score (with placeholder IoU=0): {final_score:.4f}")
    print("="*50)

    # Save report
    with open('outputs/evaluation_report.txt', 'w') as f:
        f.write(f"Accuracy:             {accuracy:.4f}\n")
        f.write(f"Macro F1-Score:       {macro_f1:.4f}\n")
        f.write(f"Classification Score: {class_score:.4f}\n")
        f.write(f"Final Score:          {final_score:.4f}\n")


if __name__ == '__main__':
    run_evaluation(
        test_image_dir  = 'data/test_images',
        ground_truth_csv = None,
        use_tta          = True
    )
```

---

## PHASE 6 — Edge Case Handling Summary

```
INCOMING TEST IMAGE
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  QUALITY CHECK                                       │
│  blur_score = Laplacian variance                     │
│  noise_level = HPF std dev                          │
│                                                      │
│  Blurry?  → Unsharp mask sharpening (Topic 3 HPF)  │
│  Noisy?   → Median / Gaussian filter (Topic 4)      │
│  Grayscale? → Apply pseudo-color map (Topic 5)      │
│  Inverted?  → Invert (Topic 5)                      │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  YOLO DETECTION                                      │
│  conf_threshold = 0.35                              │
│  Returns: [(class, conf, bbox), ...]                │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  POST-PROCESSING CHECKS                              │
│                                                      │
│  For each detection:                                 │
│                                                      │
│  1. Confidence band check:                           │
│     conf < 0.35 → discard                           │
│     0.35–0.65  → apply secondary validation         │
│     conf > 0.65 → accept directly                   │
│                                                      │
│  2. Size check:                                      │
│     bbox_area / img_area < 0.005 → discard          │
│     (filters keychains, toys, jewelry)              │
│                                                      │
│  3. Metal density check (medium conf only):          │
│     blue_channel_mean < 100 → discard               │
│     (plastic toys don't show up as dense in X-ray)  │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  LABEL RESOLUTION                                    │
│                                                      │
│  No valid detections?    → output 'safe'            │
│  Single detection?       → output that class        │
│  Multiple detections?    → priority order:          │
│                            gun > knife > shuriken   │
│  Still uncertain (TTA)?  → majority vote of         │
│                            5 augmented variants     │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  OUTPUT                                              │
│  label ∈ {safe, gun, knife, shuriken}               │
│  bbox  = [x1, y1, x2, y2] or None                  │
└──────────────────────────────────────────────────────┘
```

---

## Complete Process Flow — End to End

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA PREPARATION                              │
│                                                                     │
│  TrainData/  →  setup_data.py  →  data/raw + data/annotations      │
│       ↓                                                             │
│  Raw images (GUN, knife, shuriken, safe)                           │
│       ↓                                                             │
│  Instance-indexed mask PNGs (0=bg, 1,2,3..=objects)                │
│       ↓  convert_annotations.py  (skip ambiguous / empty masks)    │
│  YOLO .txt label files  (class_id xc yc w h)                       │
│       ↓  build_dataset.py                                           │
│  Train / Val / Test split  (70/15/15)                              │
│       ↓  augment.py                                                 │
│  Augmented training set  (~4× more images)                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  CLASSICAL CV PREPROCESSING                         │
│                                                                     │
│  Every image → preprocess()                                         │
│    1. Denoise     (Gaussian LPF / Median)          Topic 4         │
│    2. Enhance     (CLAHE in LAB space)             Topic 5         │
│    3. Sharpen     (Unsharp mask / HPF)             Topic 3         │
│    4. Normalize   (Color domain standardization)   Topic 5         │
│    5. Resize      (640 × 640)                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      YOLO FINE-TUNING                               │
│                                                                     │
│  Base model: yolov8s.pt (COCO pretrained)                          │
│  Classes: gun(0), knife(1), shuriken(2)                            │
│  Safe: empty label files (background)                              │
│                                                                     │
│  Epochs: 150   Batch: 16   Img size: 640                           │
│  LR: 0.001 → 0.00001 (cosine decay)                               │
│  Early stopping patience: 25                                        │
│  Monitor: val mAP50                                                 │
│                                                                     │
│  Output: weights/best.pt                                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                              │
│                                                                     │
│  Input image                                                        │
│       ↓  preprocess()                                               │
│  Preprocessed image (640×640)                                       │
│       ↓  YOLO(best.pt)                                             │
│  Raw detections [(class, conf, bbox)]                               │
│       ↓  postprocess()                                              │
│  Confidence filter → Size filter → Density filter                  │
│       ↓  resolve_detections()                                       │
│  Final label + bounding box                                         │
│       ↓  (optional) TTA if uncertain                               │
│  Final output                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        EVALUATION                                   │
│                                                                     │
│  Classification (70%):                                             │
│    Accuracy + Macro F1                                             │
│    Score = 0.7 × Acc + 0.3 × F1                                   │
│                                                                     │
│  Localization (30%):                                               │
│    IoU per threat image                                            │
│    Score = Average IoU (threshold ≥ 0.5)                          │
│                                                                     │
│  Final Score = 0.7 × Class Score + 0.3 × Loc Score               │
│                                                                     │
│  Output: predictions.csv                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Execution Order — Step by Step

```
Step 0:   python preprocessing/setup_data.py
           → Stages ../TrainData into data/raw + data/annotations
             (symlink by default; --copy to copy)

Step 1:   python preprocessing/convert_annotations.py
           → Converts instance-mask PNGs to YOLO .txt labels
             (skips ambiguous multi-threat + empty-mask images)

Step 1b:  python preprocessing/visualize_masks.py        # optional
           → Renders the near-black masks + derived boxes to outputs/

Step 2:   python preprocessing/build_dataset.py
           → Assembles train/val/test folder structure
             (only images that have a label; no train/val leakage)

Step 3:   python preprocessing/augment.py
           → Expands training set with augmented copies

Step 4:   python preprocessing/preprocess.py
           → Applies classical CV pipeline to all images

Step 4b:  python preprocessing/verify_dataset.py
           → Confirms every image has a well-formed label

Step 5:   python training/train.py
           → Fine-tunes YOLO26, saves weights/best.pt

Step 6:   python run_inference.py --images data/dataset/images/test \
                                  --labels data/dataset/labels/test
           → Full framework on the test split → prints/saves the project
             criteria (Accuracy, Macro F1, mean IoU, Final Score)

          # Evaluation day (hidden test, no labels) → predictions.csv:
          python run_inference.py --images hidden_test
```

Or just run everything: `bash run_all.sh`

> **Note:** `run_inference.py` is the single master inference + evaluation
> entrypoint. The PHASE 4–5 `predict.py` / `evaluate.py` blocks below document
> the underlying logic, which now lives inside `run_inference.py` (with the
> helper modules `inference/{quality_handler,postprocess,localization,
> submission,visualize_results}.py` and `utils/metrics.py`).

---

## Key Hyperparameters to Tune

| Parameter | Default | Try if Overfitting | Try if Underfitting |
|---|---|---|---|
| lr0 | 0.001 | 0.0005 | 0.005 |
| epochs | 150 | 80 | 200 |
| batch | 16 | 16 | 32 |
| mosaic | 1.0 | 0.5 | 1.0 |
| dropout | 0.0 | 0.1–0.3 | 0.0 |
| conf threshold | 0.35 | 0.5 | 0.25 |
| imgsz | 640 | 640 | 960 |

---

## Classical Baseline — HOG + SVM

Alongside the YOLO26 detector, the project ships a **fully classical
classification baseline** (`baseline/hog_svm.py`) to demonstrate the course's
classical CV + ML techniques directly on the task:

```
Grayscale image
   → CLAHE contrast normalization        (Topic 5)
   → Histogram of Oriented Gradients     (HOG edge/gradient descriptor)
   → Support Vector Machine (RBF kernel)  → {safe | gun | knife | shuriken}
```

It is **classification-only** (HOG+SVM does not localize), reuses the same
`data/dataset/` split, and is scored with the same `utils/metrics.py`, so its
Accuracy / Macro-F1 / Classification Score sit on the same scale as the
detector's. It needs no GPU or deep-learning framework — only OpenCV +
scikit-learn.

```
python baseline/hog_svm.py train     # fit on train split -> baseline/hog_svm.joblib
python baseline/hog_svm.py eval      # score on the test split
python baseline/hog_svm.py predict --images hidden_test    # -> submission CSV
```

Use it as a sanity floor and a no-GPU fallback; the fine-tuned YOLO26 pipeline
should outperform it on classification while also providing localization.

---

## Dependencies

```
# requirements.txt

ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
albumentations>=1.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
PyYAML>=6.0
tqdm>=4.65.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

*CEN454 Computer Vision and Machine Learning — Term Project Framework*
*Classical CV Preprocessing (Topics 3–6) + YOLOv8s Fine-tuning*
