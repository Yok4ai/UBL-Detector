#!/usr/bin/env python3
"""
Test script to compare inference between annotate.py and app.py methods
"""
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("models/DA_YOLO11X.pt")

# Load image
img_path = "examples/DA/13.jpg"

print("="*60)
print("Method 1: Direct from file path (like annotate.py)")
print("="*60)
results1 = model.predict(
    source=img_path,
    conf=0.5,
    iou=0.5,
    imgsz=640,
    verbose=True
)
r1 = results1[0]
print(f"Detections: {len(r1.boxes) if r1.boxes is not None else 0}")
if r1.boxes is not None and len(r1.boxes) > 0:
    print("Confidences:", r1.boxes.conf.cpu().numpy())
    print("Classes:", r1.boxes.cls.cpu().numpy())

print("\n" + "="*60)
print("Method 2: Via PIL then numpy (like app.py)")
print("="*60)
pil_img = Image.open(img_path).convert("RGB")
print(f"PIL Image size: {pil_img.size}, mode: {pil_img.mode}")
results2 = model.predict(
    source=np.array(pil_img),
    conf=0.5,
    iou=0.5,
    imgsz=640,
    verbose=True
)
r2 = results2[0]
print(f"Detections: {len(r2.boxes) if r2.boxes is not None else 0}")
if r2.boxes is not None and len(r2.boxes) > 0:
    print("Confidences:", r2.boxes.conf.cpu().numpy())
    print("Classes:", r2.boxes.cls.cpu().numpy())

print("\n" + "="*60)
print("Comparison")
print("="*60)
print(f"Method 1 (file): {len(r1.boxes) if r1.boxes is not None else 0} detections")
print(f"Method 2 (PIL):  {len(r2.boxes) if r2.boxes is not None else 0} detections")
print(f"Match: {len(r1.boxes) == len(r2.boxes) if r1.boxes and r2.boxes else False}")
