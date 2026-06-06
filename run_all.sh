#!/usr/bin/env bash
#
# End-to-end pipeline runner for the CEN454 baggage threat detector.
# Run from the project root:  bash run_all.sh
#
set -e  # stop on first error

echo "=================================================="
echo " STEP 0/7  Stage TrainData -> data/raw + data/annotations"
echo "=================================================="
python preprocessing/setup_data.py

echo
echo "=================================================="
echo " STEP 1/7  Convert instance masks -> YOLO labels"
echo "=================================================="
python preprocessing/convert_annotations.py

echo
echo "=================================================="
echo " STEP 2/7  Build train/val/test dataset structure"
echo "=================================================="
python preprocessing/build_dataset.py

echo
echo "=================================================="
echo " STEP 3/7  Augment the training split"
echo "=================================================="
python preprocessing/augment.py

echo
echo "=================================================="
echo " STEP 4/7  Classical CV preprocessing (all splits)"
echo "=================================================="
python preprocessing/preprocess.py

echo
echo "=================================================="
echo " STEP 4b   Verify dataset before training"
echo "=================================================="
python preprocessing/verify_dataset.py

echo
echo "=================================================="
echo " STEP 5/7  Fine-tune YOLO26"
echo "=================================================="
python training/train.py

echo
echo "=================================================="
echo " STEP 6/7  Evaluate on the test split"
echo "=================================================="
python training/validate.py

echo
echo "=================================================="
echo " STEP 7/7  Full framework eval on test split"
echo "          (classification + localization criteria)"
echo "=================================================="
python run_inference.py \
    --images data/dataset/images/test \
    --labels data/dataset/labels/test

echo
echo "All done. Best weights at weights/best.pt"
echo "Test-split scores -> outputs/evaluation_report.txt"
echo
echo "On evaluation day, predict on the hidden test folder with:"
echo "  python run_inference.py --images hidden_test"
echo "  -> writes outputs/predictions.csv (submission format)"
