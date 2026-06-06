"""
Classical computer-vision preprocessing pipeline (course Topics 3-6).

Applied to every image before YOLO sees it, both at training time (run once
over the dataset folders) and at inference time (called per-image inside the
prediction code).

Pipeline:
    1. Assess blur and noise levels
    2. Denoise        -> Gaussian LPF / Median filter      (Topic 4)
    3. Enhance        -> CLAHE contrast in LAB space        (Topic 5)
    4. Sharpen        -> Unsharp mask (HPF equivalent)       (Topic 3)
    5. Normalize      -> color-domain standardization        (Topic 5)
    6. Resize         -> 640 x 640 for YOLO

Run from project root to preprocess all splits in place:
    python preprocessing/preprocess.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import IMG_SIZE


# ----- 1. Quality assessment -------------------------------------------------
def assess_blur(image):
    """Laplacian variance — lower means blurrier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def assess_noise(image):
    """Std-dev of the high-frequency residual — higher means noisier."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.std(gray.astype(float) - blur.astype(float)))


def is_grayscale(image, threshold=5):
    """True if the color channels are nearly identical."""
    b = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    return np.std(b - g) < threshold


# ----- 2. Denoising (Topic 4) ------------------------------------------------
def denoise(image, noise_level):
    if noise_level < 5:
        return image
    if noise_level < 15:
        return cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.medianBlur(image, 5)


# ----- 3. Contrast enhancement (Topic 5) -------------------------------------
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----- 4. Edge sharpening (Topic 3, HPF) -------------------------------------
def sharpen_edges(image, blur_score):
    if blur_score >= 100:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


# ----- 5. Color normalization (Topic 5) --------------------------------------
def normalize_color_domain(image):
    # grayscale -> pseudo-color
    if is_grayscale(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # invert if background appears bright
    if np.mean(image) > 180:
        image = cv2.bitwise_not(image)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image.astype(np.uint8)


# ----- Full pipeline ---------------------------------------------------------
def preprocess(image_path=None, image_array=None, target_size=(IMG_SIZE, IMG_SIZE)):
    """Run the full classical CV pipeline. Returns a preprocessed BGR array."""
    if image_array is not None:
        image = image_array.copy()
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

    blur_score  = assess_blur(image)
    noise_level = assess_noise(image)

    image = denoise(image, noise_level)
    image = enhance_contrast(image)
    image = sharpen_edges(image, blur_score)
    image = normalize_color_domain(image)
    image = cv2.resize(image, target_size)
    return image


def preprocess_split(image_dir, output_dir):
    """Preprocess every image in a folder (writes to output_dir)."""
    os.makedirs(output_dir, exist_ok=True)
    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    success = 0
    for fname in files:
        src = os.path.join(image_dir, fname)
        dst = os.path.join(output_dir, fname)
        try:
            cv2.imwrite(dst, preprocess(image_path=src))
            success += 1
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
    print(f"  Preprocessed {success}/{len(files)} in {image_dir}")


if __name__ == '__main__':
    for split in ('train', 'val', 'test'):
        print(f"\nPreprocessing {split} ...")
        preprocess_split(
            image_dir  = f'data/dataset/images/{split}',
            output_dir = f'data/dataset/images/{split}',  # overwrite in place
        )
    print("\nPreprocessing complete.")
