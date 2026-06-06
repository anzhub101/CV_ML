"""
Augment the TRAINING split only, to expand a small dataset and to teach the
model to cope with low-quality / differently-colored test images.

For each training image this produces `multiplier` augmented copies, with
bounding boxes transformed alongside the image. Augmentations include
geometric transforms, quality degradation (blur, noise, compression,
downscale) and photometric shifts (brightness, hue, grayscale, invert).

Run from project root AFTER build_dataset.py:
    python preprocessing/augment.py
"""

import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import albumentations as A


AUGMENTATION_PIPELINE = A.Compose(
    [
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
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=40, quality_upper=85, p=0.25),
        A.Downscale(scale_min=0.5, scale_max=0.9,
                    interpolation=cv2.INTER_LINEAR, p=0.2),

        # --- Photometric (simulate different scanner color profiles) ---
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.4),
        A.ToGray(p=0.15),       # simulate grayscale X-ray scanners
        A.InvertImg(p=0.10),    # simulate inverted X-ray output
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,     # drop a box if <30% remains after a crop
    ),
)


def _read_yolo_label(lbl_path):
    bboxes, class_labels = [], []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
    return bboxes, class_labels


def augment_training_set(image_dir, label_dir, multiplier=3):
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_aug' not in f
    ]

    print(f"Augmenting {len(image_files)} images x{multiplier} ...")
    skipped = 0

    for img_file in image_files:
        stem     = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, stem + '.txt')

        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        bboxes, class_labels = _read_yolo_label(lbl_path)

        for i in range(multiplier):
            try:
                result = AUGMENTATION_PIPELINE(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
            except Exception as e:
                print(f"  [WARN] augmentation failed for {img_file}: {e}")
                continue

            aug_name    = f"{stem}_aug{i}"
            aug_img_out = os.path.join(image_dir, aug_name + '.png')
            aug_lbl_out = os.path.join(label_dir,  aug_name + '.txt')

            cv2.imwrite(aug_img_out, result['image'])
            with open(aug_lbl_out, 'w') as f:
                for cls, box in zip(result['class_labels'], result['bboxes']):
                    f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f}\n")

    total_after = len([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"  Done. Training images now: {total_after} (skipped {skipped}).")


if __name__ == '__main__':
    augment_training_set(
        image_dir  = 'data/dataset/images/train',
        label_dir  = 'data/dataset/labels/train',
        multiplier = 3,
    )
