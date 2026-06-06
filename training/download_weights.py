"""
Pre-download the pretrained YOLO weights while you have internet, so that
evaluation day (no internet) can load them from disk.

Run from project root:
    python training/download_weights.py
"""

from ultralytics import YOLO

MODELS = ['yolo26s.pt']   # add 'yolo11s.pt' here if you want the fallback too

if __name__ == '__main__':
    for m in MODELS:
        print(f"Downloading {m} ...")
        YOLO(m)
        print(f"  {m} ready.")
    print("\nAll weights downloaded.")
