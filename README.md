# PCB Defect Detection using YOLOv8

This repository implements a **Printed Circuit Board (PCB) defect detection system**
using **YOLOv8**. The model detects multiple types of PCB defects from images using
a custom-trained YOLOv8 object detection pipeline.

The project supports **training, inference, and evaluation**, and is structured
to ensure **clean experimentation and reproducibility**.



## Folder Structure

```
PCB_Defect_Detection
    -> data
        -> train
            -> images
            -> labels
        -> val
            -> images
            -> labels
        -> test
            -> images
            -> labels
    -> models
        -> model.pt            # trained YOLOv8 weights (not uploaded)
    -> results                # inference output images
    -> scripts
        -> inference.py        # run inference on images
        -> evaluate.py         # compute Precision, Recall, F1, mIoU
    -> dataset.yaml            # YOLO dataset configuration
    -> .gitignore
    -> README.md
```




perl
Copy code

 ⚠️ **Note:** Do not push `data/` or `models/` to GitHub to avoid large files.  

## Dataset Preparation

1. Organize your PCB dataset in YOLO format:

data/train/images
data/train/labels
data/val/images
data/val/labels
data/test/images
data/test/labels

less
Copy code

2. **Label Format:** Each label file (`.txt`) should have lines in the format:

<class_id> <x_center> <y_center> <width> <height>

vbnet
Copy code

- Normalized coordinates in `[0,1]`
- `class_id` corresponds to defect type (e.g., mouse_bite=0, spur=1, etc.)

3. Update `dataset.yaml` to match your dataset:

```yaml
path: data
train: train
val: val
test: test
nc: 6
names:
  - mouse_bite
  - spur
  - open_circuit
  - short
  - missing_hole
  - spurious_copper
Environment Setup
Create a Python environment and install dependencies:
```
bash
Copy code
conda create -n pcb-yolo python=3.10 -y
conda activate pcb-yolo
pip install ultralytics tqdm opencv-python numpy
Training
Train YOLOv8 on your dataset:

python
Copy code
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt / yolov8m.pt
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    iou=0.5,
    mosaic=0.5,
    mixup=0.0,
    erasing=0.0,
    device=0
)
This saves the trained weights in runs/train/ by default. You can move your best weights to models/model.pt.

## Inference

Run inference on test images:

```bash
python scripts/inference.py
Saves predicted images to results/.
```

Prints bounding boxes, detected classes, and confidence scores.

Ensure models/model.pt exists outside the scripts/ folder.

Evaluation
Run evaluation to compute Precision, Recall, F1, mIoU:

```bash

python scripts/evaluate.py
```
Evaluates predictions against ground truth labels in data/test/labels/.

Prints a table with per-class metrics and overall averages.

Notes
Make sure all images and labels follow YOLO format.


Recommended GPU usage for training large datasets.

Works with YOLOv8 (Ultralytics) library..
