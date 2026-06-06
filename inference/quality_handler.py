"""
Adaptive image quality handler.

Detects quality issues in each incoming test image BEFORE it enters the
classical CV preprocessing pipeline, and applies targeted fixes:

    Issue detected          Fix applied
    ─────────────────────   ──────────────────────────────────────
    Heavy blur              Wiener-style unsharp sharpening
    Heavy noise             Aggressive median denoising
    Grayscale input         Pseudo-color (JET) mapping
    Inverted X-ray          Bitwise invert to restore normal polarity
    Over-/under-exposed     Gamma correction
    JPEG block artefacts    Mild Gaussian smoothing
    Wrong resolution        Resize to 640x640

All checks are non-destructive measurements first; fixes are only applied
when a threshold is exceeded to avoid degrading already-good images.

Usage:
    from inference.quality_handler import QualityHandler
    handler = QualityHandler()
    fixed_image, report = handler.process(image)
"""

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

log = get_logger(__name__)

# Tunable thresholds
BLUR_THRESHOLD      = 80.0    # Laplacian variance below this → blurry
NOISE_THRESHOLD     = 18.0    # high-freq std-dev above this → noisy
GRAY_THRESHOLD      = 6.0     # channel diff std-dev below this → grayscale
INVERT_THRESHOLD    = 180.0   # mean pixel value above this → likely inverted
DARK_THRESHOLD      = 40.0    # mean pixel value below this → underexposed
BRIGHT_THRESHOLD    = 215.0   # mean pixel value above this → overexposed
BLOCK_THRESHOLD     = 12.0    # DCT block artefact indicator


class QualityReport:
    """Records which issues were detected and which fixes were applied."""

    def __init__(self):
        self.blur_score    = 0.0
        self.noise_level   = 0.0
        self.mean_intensity= 0.0
        self.is_grayscale  = False
        self.is_inverted   = False
        self.is_blurry     = False
        self.is_noisy      = False
        self.is_dark       = False
        self.is_bright     = False
        self.fixes_applied: list = []

    def __str__(self):
        issues = []
        if self.is_blurry:    issues.append(f"blurry (score={self.blur_score:.1f})")
        if self.is_noisy:     issues.append(f"noisy (level={self.noise_level:.1f})")
        if self.is_grayscale: issues.append("grayscale")
        if self.is_inverted:  issues.append("inverted")
        if self.is_dark:      issues.append(f"dark (mean={self.mean_intensity:.1f})")
        if self.is_bright:    issues.append(f"bright (mean={self.mean_intensity:.1f})")
        issue_str = ", ".join(issues) if issues else "none"
        fix_str   = ", ".join(self.fixes_applied) if self.fixes_applied else "none"
        return f"issues=[{issue_str}]  fixes=[{fix_str}]"


class QualityHandler:
    """Detects and corrects quality issues in a single X-ray image."""

    # ── Assessment ──────────────────────────────────────────────────────────

    def _blur_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _noise_level(self, image: np.ndarray) -> float:
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        return float(np.std(gray.astype(float) - smooth.astype(float)))

    def _is_grayscale(self, image: np.ndarray) -> bool:
        b = image[:, :, 0].astype(float)
        g = image[:, :, 1].astype(float)
        return float(np.std(b - g)) < GRAY_THRESHOLD

    def _mean_intensity(self, image: np.ndarray) -> float:
        return float(np.mean(image))

    def _has_block_artefacts(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # sample 8x8 DCT blocks and look for coefficient spikes
        scores = []
        for r in range(0, min(h, 64), 8):
            for c in range(0, min(w, 64), 8):
                block = gray[r:r+8, c:c+8].astype(np.float32)
                if block.shape == (8, 8):
                    dct = cv2.dct(block)
                    scores.append(float(np.std(dct)))
        return (sum(scores) / len(scores)) > BLOCK_THRESHOLD if scores else False

    # ── Fixes ────────────────────────────────────────────────────────────────

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Unsharp mask to recover edges in blurry X-rays."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1.6, blurred, -0.6, 0)

    def _denoise(self, image: np.ndarray, level: float) -> np.ndarray:
        """Median for high noise; mild Gaussian for low noise."""
        if level > 25:
            return cv2.medianBlur(image, 5)
        return cv2.GaussianBlur(image, (3, 3), 0)

    def _pseudo_color(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    def _invert(self, image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(image)

    def _gamma_correct(self, image: np.ndarray, gamma: float) -> np.ndarray:
        table = np.array([
            min(255, int((i / 255.0) ** gamma * 255))
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(image, table)

    def _smooth_artefacts(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (3, 3), 0.5)

    # ── Main entry point ─────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> tuple:
        """
        Assess and fix an image.
        Returns (fixed_image, QualityReport).
        """
        qr = QualityReport()
        img = image.copy()

        # Assessment
        qr.blur_score     = self._blur_score(img)
        qr.noise_level    = self._noise_level(img)
        qr.mean_intensity = self._mean_intensity(img)
        qr.is_grayscale   = self._is_grayscale(img)
        qr.is_blurry      = qr.blur_score < BLUR_THRESHOLD
        qr.is_noisy       = qr.noise_level > NOISE_THRESHOLD
        qr.is_dark        = qr.mean_intensity < DARK_THRESHOLD
        qr.is_bright      = qr.mean_intensity > BRIGHT_THRESHOLD
        # re-check invert after potential grayscale conversion
        qr.is_inverted    = qr.mean_intensity > INVERT_THRESHOLD

        # Order matters: grayscale first, then invert check, then quality

        if qr.is_grayscale:
            img = self._pseudo_color(img)
            qr.fixes_applied.append("pseudo-color")
            # re-measure mean after colorization
            qr.mean_intensity = self._mean_intensity(img)
            qr.is_inverted    = qr.mean_intensity > INVERT_THRESHOLD

        if qr.is_inverted:
            img = self._invert(img)
            qr.fixes_applied.append("invert")

        if qr.is_dark:
            img = self._gamma_correct(img, gamma=0.6)
            qr.fixes_applied.append("gamma-brighten")
        elif qr.is_bright:
            img = self._gamma_correct(img, gamma=1.5)
            qr.fixes_applied.append("gamma-darken")

        if self._has_block_artefacts(img):
            img = self._smooth_artefacts(img)
            qr.fixes_applied.append("smooth-artefacts")

        if qr.is_noisy:
            img = self._denoise(img, qr.noise_level)
            qr.fixes_applied.append(f"denoise(level={qr.noise_level:.1f})")

        if qr.is_blurry:
            img = self._sharpen(img)
            qr.fixes_applied.append(f"sharpen(score={qr.blur_score:.1f})")

        # Final normalize
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        log.debug(f"QualityHandler: {qr}")
        return img, qr
