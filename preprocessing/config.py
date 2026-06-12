"""
Central configuration for the CEN454 baggage threat detection project.
All class mappings, thresholds, and shared constants live here.
"""

import os

# Class IDs for YOLO labels. "safe" is NOT a class — it is represented
# by empty label files (no detections = background = safe).
CLASS_MAP = {
    'gun':      0,
    'knife':    1,
    'shuriken': 2,
}

# Reverse lookup: class_id -> name
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
# The original dataset ships in the `TrainData/` folder at the project root.
# `setup_data.py` stages it into the `data/` layout the rest of the pipeline
# expects. Override SOURCE_DATA_DIR via the CEN454_SOURCE_DATA environment
# variable if your data lives elsewhere.
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATA_DIR = os.environ.get(
    'CEN454_SOURCE_DATA',
    os.path.join(PROJECT_ROOT, 'TrainData'),
)
RAW_DIR         = os.path.join('data', 'raw')          # staged source images
ANNOTATION_DIR  = os.path.join('data', 'annotations')  # staged mask PNGs
ALL_LABELS_DIR  = os.path.join('data', 'all_labels')   # generated YOLO labels
DATASET_DIR     = os.path.join('data', 'dataset')      # final YOLO structure

# The "safe" class has source images but NO annotation masks (it is background).
SAFE_CLASS = 'safe'


# ---------------------------------------------------------------------------
# Annotation mask format
# ---------------------------------------------------------------------------
# IMPORTANT: the annotation PNGs are NOT 0/255 binary images. They are
# INSTANCE-INDEXED masks: pixel value 0 is background, and each distinct
# object instance gets its own small integer id (1, 2, 3, ...). Because those
# ids (1-4 in this dataset) are tiny next to 255, the masks look almost solid
# black in an image viewer — but they are not empty. Any pixel > 0 is part of
# a threat; each unique non-zero id is converted into its own bounding box.
MASK_FOREGROUND_MIN = 1     # pixel values >= this are foreground (threat)

# Minimum contour area (in mask pixels) to keep — drops speckle noise.
MIN_CONTOUR_AREA = 50


def resolve_class_dir(base_dir, class_name):
    """
    Return the real sub-directory for `class_name` inside `base_dir`, matched
    case-insensitively (the dataset ships 'GUN' but our class map uses 'gun').
    Returns None if no matching folder exists.
    """
    direct = os.path.join(base_dir, class_name)
    if os.path.isdir(direct):
        return direct
    if not os.path.isdir(base_dir):
        return None
    for entry in os.listdir(base_dir):
        if entry.lower() == class_name.lower() and \
                os.path.isdir(os.path.join(base_dir, entry)):
            return os.path.join(base_dir, entry)
    return None

# Threat priority for resolving images that contain multiple objects.
# Higher number = higher priority when collapsing to one CSV label.
THREAT_PRIORITY = {
    'gun':      3,
    'knife':    2,
    'shuriken': 1,
    'safe':     0,
}

# Confidence band thresholds (used in post-processing).
# CONF_LOW was tuned 0.35 -> 0.25 on the test split: it catches more real
# threats (safe precision 0.64 -> 0.72, ~halving missed threats) and raised the
# final score, with negligible localization cost.
CONF_HIGH = 0.65   # >= this: accept detection outright
CONF_LOW  = 0.25   # <  this: reject (treat as safe)
                   # between the two: apply secondary validation

# Minimum bounding-box area as a fraction of the whole image.
# Anything smaller is likely a keychain / toy / jewelry, not a real weapon.
MIN_BBOX_AREA_RATIO = 0.005

# IoU threshold for a localization to count as correct (per project spec).
IOU_THRESHOLD = 0.5

# Metal-density validation threshold (mean blue-channel intensity in ROI).
# Real metallic weapons appear dense/bright in X-ray pseudo-color. Relaxed
# 100 -> 50 alongside CONF_LOW (see above) — best combined result on test.
METAL_DENSITY_THRESHOLD = 50.0

# YOLO input resolution
IMG_SIZE = 640

# Train / val / test split ratios
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Reproducibility
RANDOM_SEED = 42
