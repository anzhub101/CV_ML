"""
Assemble the YOLO dataset folder structure from raw images + labels.

Splits each class into train/val/test (default 70/15/15) and copies images
and their matching label files into:
    data/dataset/images/{train,val,test}/
    data/dataset/labels/{train,val,test}/

Run from project root:
    python preprocessing/build_dataset.py
"""

import os
import shutil
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import (
    SPLIT_RATIOS, RANDOM_SEED, RAW_DIR, ALL_LABELS_DIR, DATASET_DIR,
)


def build_yolo_dataset(raw_dir=RAW_DIR, label_dir=ALL_LABELS_DIR,
                       output_dir=DATASET_DIR,
                       ratios=SPLIT_RATIOS, seed=RANDOM_SEED):
    random.seed(seed)

    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    counts  = {'train': 0, 'val': 0, 'test': 0}
    skipped = 0

    classes = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    for class_name in classes:
        class_img_dir = os.path.join(raw_dir, class_name)
        all_imgs = [
            f for f in os.listdir(class_img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        # Only include images that have a label file. convert_annotations.py
        # deliberately omits labels for skipped images (ambiguous multi-threat
        # or empty-mask); safe images carry an empty label. This also keeps
        # every kept stem globally unique, so an image lands in exactly one
        # split — no cross-folder train/val leakage.
        images = [
            f for f in all_imgs
            if os.path.exists(
                os.path.join(label_dir, os.path.splitext(f)[0] + '.txt')
            )
        ]
        skipped += len(all_imgs) - len(images)
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])

        assignments = (
            [(f, 'train') for f in images[:n_train]] +
            [(f, 'val')   for f in images[n_train:n_train + n_val]] +
            [(f, 'test')  for f in images[n_train + n_val:]]
        )

        for img_file, split in assignments:
            stem    = os.path.splitext(img_file)[0]
            src_img = os.path.join(class_img_dir, img_file)
            src_lbl = os.path.join(label_dir, stem + '.txt')
            dst_img = os.path.join(output_dir, 'images', split, img_file)
            dst_lbl = os.path.join(output_dir, 'labels', split, stem + '.txt')

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)  # exists by construction
            counts[split] += 1

    print("\nDataset split complete:")
    print(f"  Train: {counts['train']} images")
    print(f"  Val:   {counts['val']} images")
    print(f"  Test:  {counts['test']} images")
    print(f"  Total: {sum(counts.values())} images")
    print(f"  Skipped (no label — ambiguous/empty-mask): {skipped}")


if __name__ == '__main__':
    build_yolo_dataset()
