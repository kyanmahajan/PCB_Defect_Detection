"""
inference.py

Run inference using a YOLOv8 model on PCB test dataset.

Author: Your Name
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2


SCRIPT_DIR = Path(__file__).parent.resolve()
print(SCRIPT_DIR)

# Model folder is outside, e.g., ../models/model.pt
MODEL_PATH = SCRIPT_DIR.parent / "model" / "model.pt"  # Path to your trained model
print(MODEL_PATH)
TEST_IMAGES_DIR = SCRIPT_DIR.parent/"test_images"
RESULTS_DIR = Path("results")
SAVE_IMAGES = True        # Set False if you don't want images saved
PRINT_OUTPUT = True       # Set True to print boxes, classes, confidences

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = YOLO(str(MODEL_PATH))
    model = model

    for img_path in TEST_IMAGES_DIR.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        res = model(str(img_path))[0]  # batch size 1

        # Print boxes, classes, confidences
        print(f"\nImage: {img_path.name}")
        print("Boxes:", res.boxes.xyxy)
        print("Classes:", res.boxes.cls)
        print("Confidences:", res.boxes.conf)

        # Plot the predictions
        pred_img = res.plot()  # returns a NumPy array with boxes drawn

        # Save image using OpenCV
        save_path = RESULTS_DIR / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        print(f" Saved predicted image to {save_path}")

if __name__ == "__main__":
    main()
