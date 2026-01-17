"""
evaluate.py

Run inference and evaluate YOLOv8 model on PCB dataset.

Author: Your Name
"""

import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pt"

# Test dataset
TEST_IMAGES_DIR = Path(__file__).parent.parent / "data/test/images"
TEST_LABELS_DIR = Path(__file__).parent.parent / "data/test/labels"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1,y1,x2,y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def yolo_to_xyxy(xc, yc, w, h, H, W):
    """Convert YOLO normalized box to absolute xyxy coordinates."""
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return [x1, y1, x2, y2]

def evaluate_detection(results, label_dir, class_names,
                       iou_thr=0.5, conf_thr=0.25):
    stats = defaultdict(lambda: {"TP":0, "FP":0, "FN":0, "IoU":[]})

    for res in results:
        H, W = res.orig_img.shape[:2]
        label_path = os.path.join(
            label_dir, os.path.basename(res.path).rsplit(".",1)[0] + ".txt"
        )

        # ---------- Load GT ----------
        gt_boxes = []
        with open(label_path) as f:
            for line in f:
                c, xc, yc, w, h = map(float, line.split())
                gt_boxes.append([*yolo_to_xyxy(xc, yc, w, h, H, W), int(c)])

        matched_gt = [False]*len(gt_boxes)

        # ---------- Predictions ----------
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        mask = confs >= conf_thr
        boxes, classes = boxes[mask], classes[mask]

        # ---------- Match predictions ----------
        for pb, pc in zip(boxes, classes):
            best_iou, best_idx = 0, -1
            for i, gt in enumerate(gt_boxes):
                if pc == gt[4]:
                    iou = compute_iou(pb, gt[:4])
                    if iou > best_iou:
                        best_iou, best_idx = iou, i

            if best_iou >= iou_thr and best_idx != -1:
                if not matched_gt[best_idx]:
                    stats[pc]["TP"] += 1
                    stats[pc]["IoU"].append(best_iou)
                    matched_gt[best_idx] = True
                else:
                    stats[pc]["FP"] += 1
            else:
                stats[pc]["FP"] += 1

        # ---------- FN ----------
        for i, gt in enumerate(gt_boxes):
            if not matched_gt[i]:
                stats[gt[4]]["FN"] += 1

    # -----------------------------
    # Print table
    # -----------------------------
    print("\n=== Detection Metrics ===")
    print(f"{'Class':18s} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'mIoU':>6}")

    P_all, R_all, F1_all = [], [], []

    for cid, cname in enumerate(class_names):
        TP = stats[cid]["TP"]
        FP = stats[cid]["FP"]
        FN = stats[cid]["FN"]

        prec = TP / (TP + FP + 1e-9)
        rec  = TP / (TP + FN + 1e-9)
        f1   = 2*prec*rec / (prec + rec + 1e-9)
        miou = np.mean(stats[cid]["IoU"]) if stats[cid]["IoU"] else 0

        print(f"{cname:18s} {TP:4d} {FP:4d} {FN:4d} {prec:6.2f} {rec:6.2f} {f1:6.2f} {miou:6.2f}")

        P_all.append(prec)
        R_all.append(rec)
        F1_all.append(f1)

    print("\n=== Overall ===")
    print(f"Mean Precision: {np.mean(P_all):.3f}")
    print(f"Mean Recall   : {np.mean(R_all):.3f}")
    print(f"Mean F1       : {np.mean(F1_all):.3f}")


def main():
    # Load model
    model = YOLO(str(MODEL_PATH))
    model = model.to("cuda")

    # Run inference on all test images
    results = []
    for img_path in TEST_IMAGES_DIR.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        res = model(str(img_path))[0]  # batch size 1
        results.append(res)

    # Evaluate
    evaluate_detection(
        results,
        label_dir=str(TEST_LABELS_DIR),
        class_names=list(model.names.values()),
        iou_thr=IOU_THRESHOLD,
        conf_thr=CONF_THRESHOLD
    )

if __name__ == "__main__":
    main()
