"""
Draw YOLO bounding boxes onto a few training images so you can confirm,
by eye, that the boxes land on the actual weapons.

Outputs annotated copies to outputs/check_*.png

Run from project root:
    python preprocessing/visualize_labels.py
"""

import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.config import CLASS_NAMES

IMG_DIR = 'data/dataset/images/train'
LBL_DIR = 'data/dataset/labels/train'
N_SAMPLES = 8


def draw_one(img_name):
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, CLASS_NAMES.get(cls, '?'), (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs('outputs', exist_ok=True)
    out = os.path.join('outputs', 'check_' + img_name)
    cv2.imwrite(out, img)
    print(f"Saved {out}")


if __name__ == '__main__':
    files = [f for f in os.listdir(IMG_DIR)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files[:N_SAMPLES]:
        draw_one(f)
    print("\nOpen the images in outputs/ and confirm boxes sit on the weapons.")
