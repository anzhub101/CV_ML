"""
Stage the original dataset into the layout the pipeline expects.

The dataset ships in the `TrainData/` folder at the project root:

    TrainData/
      GUN/ knife/ shuriken/ safe/        # source images
      annotations/GUN/ knife/ shuriken/  # instance-indexed mask PNGs

This script links (or copies) those into:

    data/raw/GUN/ knife/ shuriken/ safe/
    data/annotations/GUN/ knife/ shuriken/

By default it creates SYMLINKS (instant, no duplicated disk usage). Pass
--copy to physically copy instead (use this on Windows or if you plan to edit
the staged files without touching the originals).

Run from project root:
    python preprocessing/setup_data.py            # symlink
    python preprocessing/setup_data.py --copy     # copy
    CEN454_SOURCE_DATA=/path/to/TrainData python preprocessing/setup_data.py
"""

import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import (
    SOURCE_DATA_DIR, RAW_DIR, ANNOTATION_DIR, CLASS_MAP, SAFE_CLASS,
)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def _stage_dir(src, dst, copy):
    """Mirror image files from src -> dst via symlink (default) or copy."""
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTS)]
    for f in files:
        s = os.path.abspath(os.path.join(src, f))
        d = os.path.join(dst, f)
        if os.path.lexists(d):
            os.remove(d)
        if copy:
            shutil.copy2(s, d)
        else:
            os.symlink(s, d)
    return len(files)


def setup_data(source=SOURCE_DATA_DIR, copy=False):
    if not os.path.isdir(source):
        print(f"[ERROR] Source data folder not found: {source}")
        print("        Set CEN454_SOURCE_DATA to the path of your TrainData "
              "folder, or place it at the project root.")
        return False

    mode = "Copying" if copy else "Symlinking"
    print(f"{mode} data from: {source}\n")

    # Threat classes: source images + their masks. Folders may be 'GUN' etc.,
    # so match case-insensitively and stage under the original folder name.
    src_ann_root = os.path.join(source, 'annotations')
    ok = True
    for class_name in list(CLASS_MAP.keys()):
        src_img = _find(source, class_name)
        src_ann = _find(src_ann_root, class_name) if os.path.isdir(src_ann_root) else None

        if src_img is None:
            print(f"  [WARN] No source images for class '{class_name}'")
            ok = False
            continue
        folder = os.path.basename(src_img)
        n_img = _stage_dir(src_img, os.path.join(RAW_DIR, folder), copy)
        print(f"  raw/{folder}: {n_img} images")

        if src_ann is None:
            print(f"  [WARN] No annotation masks for class '{class_name}'")
            ok = False
            continue
        n_ann = _stage_dir(src_ann, os.path.join(ANNOTATION_DIR, folder), copy)
        print(f"  annotations/{folder}: {n_ann} masks")

    # Safe class: images only, no masks.
    src_safe = _find(source, SAFE_CLASS)
    if src_safe is not None:
        folder = os.path.basename(src_safe)
        n_safe = _stage_dir(src_safe, os.path.join(RAW_DIR, folder), copy)
        print(f"  raw/{folder}: {n_safe} images")
    else:
        print(f"  [WARN] No '{SAFE_CLASS}' folder found in source")

    print("\nData staged into data/raw and data/annotations.")
    if not ok:
        print("[WARN] Some classes were missing — check the warnings above.")
    return ok


def _find(base_dir, name):
    """Case-insensitive lookup of a sub-directory; returns full path or None."""
    if not os.path.isdir(base_dir):
        return None
    direct = os.path.join(base_dir, name)
    if os.path.isdir(direct):
        return direct
    for entry in os.listdir(base_dir):
        if entry.lower() == name.lower() and \
                os.path.isdir(os.path.join(base_dir, entry)):
            return os.path.join(base_dir, entry)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy', action='store_true',
                        help='Physically copy files instead of symlinking')
    parser.add_argument('--source', default=SOURCE_DATA_DIR,
                        help='Path to the TrainData folder')
    args = parser.parse_args()

    success = setup_data(source=args.source, copy=args.copy)
    sys.exit(0 if success else 1)
