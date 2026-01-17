"""
train_yolo.py

Train a YOLOv8 model on the PCB defect dataset.

Author: Your Name
"""

import torch
from ultralytics import YOLO
from pathlib import Path


DATA_ROOT = Path("/kaggle/working/dataset") 
NUM_CLASSES = 6
CLASS_NAMES = [
    "mouse_bite",
    "spur",
    "open_circuit",
    "short",
    "missing_hole",
    "spurious_copper"
]

# Training hyperparameters
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
IOU_THRESH = 0.5
MOSAIC = 0.5
MIXUP = 0.0
ERASING = 0.0
AUTO_AUGMENT = None
DEVICE = 0  # GPU index


YOLO_MODEL = "yolov8n.pt"


# CREATE dataset.yaml

dataset_yaml = f"""
path: {DATA_ROOT}

train: train
val: val
test: test

nc: {NUM_CLASSES}

names:
"""
for cname in CLASS_NAMES:
    dataset_yaml += f"  - {cname}\n"

yaml_path = Path("dataset.yaml")
yaml_path.write_text(dataset_yaml)
print(f" dataset.yaml created at {yaml_path.resolve()}")

def main():
    # Load the YOLOv8 model
    model = YOLO(YOLO_MODEL)

    # Move model to GPU
    model = model.to("cuda")

    # Train the model
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        iou=IOU_THRESH,
        mosaic=MOSAIC,
        mixup=MIXUP,
        erasing=ERASING,
        auto_augment=AUTO_AUGMENT,
        device=DEVICE
    )

    print(" Training complete!")

if __name__ == "__main__":
    main()
