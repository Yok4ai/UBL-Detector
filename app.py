import os
import io
import glob
import json
import uuid
import torch
import shutil
import tempfile
import numpy as np
import gradio as gr
import cv2

from PIL import Image
from ultralytics import YOLO
from gradio.themes.utils import colors
from typing import List, Tuple, Optional
from collections import defaultdict

# Import category analysis functions and constants
from category_analysis import (
    CATEGORY_MAPPING,
    CATEGORY_COLORS,
    CATEGORY_DISPLAY_NAMES,
    categorize_detections,
    draw_category_results,
    draw_category_pie_chart
)

# Import shelf detection module
try:
    from shelf_detector import get_shelf_detector, detect_and_crop_shelves
    SHELF_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Shelf detector not available: {e}")
    SHELF_DETECTOR_AVAILABLE = False
    get_shelf_detector = None
    detect_and_crop_shelves = None

# Import product clustering module
try:
    from product_clustering import detect_shelves_from_products
    PRODUCT_CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Product clustering not available: {e}")
    PRODUCT_CLUSTERING_AVAILABLE = False
    detect_shelves_from_products = None





# Redirect Ultralytics config to a writable folder
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# Set Gradio theme colors - Unilever brand colors
theme = gr.themes.Default(
    primary_hue=colors.blue,
    secondary_hue=colors.slate,
    neutral_hue=colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#0057B8",
    button_primary_background_fill_hover="#004494",
    button_primary_text_color="white",
)

# Default model inference parameters
DEFAULT_CONF  = float(os.environ.get("YOLO_CONF", 0.50))
DEFAULT_IOU   = float(os.environ.get("YOLO_IOU", 0.50))
DEFAULT_IMGSZ = int(os.environ.get("YOLO_IMGSZ", 640))

# Limit CPU threads for stability if GPU is unavailable
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() // 2))

# Load available models automatically
MODEL_DIR = "models"
model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
if not model_files:
    raise FileNotFoundError(f"🚫 No YOLO weight files found in {MODEL_DIR}/")

# Load the first one by default
current_model_path = model_files[0]
MODEL = YOLO(current_model_path)
print(f"✅ Loaded model: {current_model_path}")

# Load models for shelf share analysis
SKU110K_MODEL_PATH = os.path.join(MODEL_DIR, "SKU110k_YOLO11X.pt")
UBL_MODEL_PATH = os.path.join(MODEL_DIR, "DA_YOLO11X.pt")
SHELFTALKER_MODEL_PATH = os.path.join(MODEL_DIR, "Shelftalker.pt")

if os.path.exists(SKU110K_MODEL_PATH) and os.path.exists(UBL_MODEL_PATH):
    SKU110K_MODEL = YOLO(SKU110K_MODEL_PATH)
    UBL_MODEL = YOLO(UBL_MODEL_PATH)
    print(f"✅ Loaded SKU-110K model: {SKU110K_MODEL_PATH}")
    print(f"✅ Loaded UBL model: {UBL_MODEL_PATH}")
    SHELF_ANALYSIS_AVAILABLE = True
else:
    SKU110K_MODEL = None
    UBL_MODEL = None
    SHELF_ANALYSIS_AVAILABLE = False
    print("⚠️ Shelf analysis models not found. Shelf Share Analysis tab will be disabled.")

# Load Shelftalker model for Fixed Shelf Analysis
if os.path.exists(SHELFTALKER_MODEL_PATH):
    SHELFTALKER_MODEL = YOLO(SHELFTALKER_MODEL_PATH)
    print(f"✅ Loaded Shelftalker model: {SHELFTALKER_MODEL_PATH}")
    FIXED_SHELF_ANALYSIS_AVAILABLE = True
else:
    SHELFTALKER_MODEL = None
    FIXED_SHELF_ANALYSIS_AVAILABLE = False
    print("⚠️ Shelftalker model not found. Fixed Shelf Analysis tab will be disabled.")

# Function to reload the model dynamically
def select_model(model_name):
    global MODEL
    model_path = os.path.join(MODEL_DIR, model_name)
    MODEL = YOLO(model_path)
    print(f"🔁 Switched to model: {model_name}")
    return (f"**Active model:** {model_name}", gr.update(choices=_class_names(), value=[]))





# Get sorted list of class names from the loaded YOLO model
def _class_names() -> List[str]:
    names = getattr(MODEL, "names", None)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return []

# Convert YOLO prediction boxes to normalized YOLO TXT format
def _yolo_box_txt(r, include_conf: bool = False) -> List[str]:
    if r.boxes is None or len(r.boxes) == 0:
        return []

    H, W = r.orig_shape  # original image size
    xyxy = r.boxes.xyxy.detach().cpu().numpy()  # box corners
    clss = r.boxes.cls.detach().cpu().numpy().astype(int)  # class IDs
    confs = r.boxes.conf.detach().cpu().numpy() if include_conf else None  # confidences if needed

    lines = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        # convert corner coords to normalized center format
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H

        # clip values between 0 and 1
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w  = min(max(w,  0.0), 1.0)
        h  = min(max(h,  0.0), 1.0)

        # build one YOLO-format line: class x_center y_center width height [conf]
        parts = [str(int(clss[i])), f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
        if include_conf:
            parts.append(f"{float(confs[i]):.6f}")
        lines.append(" ".join(parts))
    return lines


# ============================================================================
# Shelf Share Analysis Functions
# ============================================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_detections(all_products_boxes, ubl_boxes, iou_threshold=0.5):
    """
    Match UBL detections to all product detections.

    Returns:
        ubl_matched: Set of indices from all_products that match UBL products
        non_ubl: Set of indices from all_products that don't match any UBL products
    """
    ubl_matched = set()

    for ubl_box in ubl_boxes:
        best_iou = 0
        best_idx = -1

        for idx, product_box in enumerate(all_products_boxes):
            iou = calculate_iou(ubl_box, product_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_idx = idx

        if best_idx != -1:
            ubl_matched.add(best_idx)

    all_indices = set(range(len(all_products_boxes)))
    non_ubl = all_indices - ubl_matched

    return ubl_matched, non_ubl


def draw_shelf_analysis_results(image_rgb, all_products_boxes, ubl_indices, non_ubl_indices,
                                 total_products, ubl_count, non_ubl_count, share_of_shelf):
    """
    Draw bounding boxes on image with different colors for UBL and Non-UBL products.

    Green: UBL products (Unilever)
    Red: Non-UBL products
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    img_draw = image_bgr.copy()

    # Draw Non-UBL products in red
    for idx in non_ubl_indices:
        box = all_products_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_draw, 'Non-Unilever', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw UBL products in green
    for idx in ubl_indices:
        box = all_products_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, 'Unilever', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add text overlay with metrics
    overlay = img_draw.copy()
    height, width = img_draw.shape[:2]

    # Draw semi-transparent background for text
    cv2.rectangle(overlay, (10, 10), (450, 130), (0, 0, 0), -1)
    img_draw = cv2.addWeighted(img_draw, 0.7, overlay, 0.3, 0)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_draw, f'Total Products: {total_products}',
               (20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img_draw, f'Unilever Products: {ubl_count}',
               (20, 75), font, 0.8, (0, 255, 0), 2)
    cv2.putText(img_draw, f'Non-Unilever Products: {non_ubl_count}',
               (20, 110), font, 0.8, (0, 0, 255), 2)

    # Add share of shelf prominently
    share_text = f'Unilever Share: {share_of_shelf:.1f}%'
    text_size = cv2.getTextSize(share_text, font, 1.5, 3)[0]
    text_x = width - text_size[0] - 20
    text_y = 60
    cv2.rectangle(img_draw, (text_x-10, text_y-45),
                 (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
    cv2.putText(img_draw, share_text, (text_x, text_y),
               font, 1.5, (0, 255, 255), 3)

    # Convert BGR back to RGB for PIL
    result_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def run_shelf_analysis(image: Image.Image, conf: float, iou_threshold: float):
    """
    Run two-stage detection pipeline and calculate share of shelf.

    Stage 1: Detect all products using SKU-110K model
    Stage 2: Detect UBL products using DA YOLO model
    Result: Calculate Unilever share of shelf
    """
    if not SHELF_ANALYSIS_AVAILABLE:
        raise gr.Error("Shelf analysis models are not available.")

    if image is None:
        raise gr.Error("Please upload an image.")

    # Save PIL image to temporary file for consistent preprocessing
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    image.save(temp_path, quality=95)

    try:
        # Stage 1: Detect all products
        results_all = SKU110K_MODEL(temp_path, conf=conf, verbose=False)[0]
        all_products_boxes = results_all.boxes.xyxy.cpu().numpy() if results_all.boxes is not None else np.array([])

        # Stage 2: Detect UBL products
        results_ubl = UBL_MODEL(temp_path, conf=conf, verbose=False)[0]
        ubl_boxes = results_ubl.boxes.xyxy.cpu().numpy() if results_ubl.boxes is not None else np.array([])
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Match detections
    if len(all_products_boxes) > 0 and len(ubl_boxes) > 0:
        ubl_indices, non_ubl_indices = match_detections(all_products_boxes, ubl_boxes, iou_threshold)
    elif len(all_products_boxes) > 0:
        ubl_indices = set()
        non_ubl_indices = set(range(len(all_products_boxes)))
    else:
        ubl_indices = set()
        non_ubl_indices = set()

    # Calculate metrics
    total_products = len(all_products_boxes)
    ubl_count = len(ubl_indices)
    non_ubl_count = len(non_ubl_indices)
    share_of_shelf = (ubl_count / total_products * 100) if total_products > 0 else 0

    # Create visualization
    result_image = draw_shelf_analysis_results(
        image, all_products_boxes, ubl_indices, non_ubl_indices,
        total_products, ubl_count, non_ubl_count, share_of_shelf
    )

    # Create metrics text
    metrics = {
        "total_products": total_products,
        "unilever_products": ubl_count,
        "non_unilever_products": non_ubl_count,
        "unilever_share_percentage": round(share_of_shelf, 2)
    }

    metrics_text = json.dumps(metrics, indent=2)

    return result_image, metrics_text


def detect_shelf_rows(
    image: Image.Image,
    conf: float,
    detection_method: str = "clustering"
):
    """
    Step 1: Detect shelf rows and return information about them.

    Returns:
        Tuple of (info_text, dropdown_choices)
    """
    if not SHELF_ANALYSIS_AVAILABLE:
        return "⚠️ Shelf analysis models are not available.", gr.Dropdown(choices=[("All Shelves", "all")], value="all")

    if image is None:
        return "⚠️ Please upload an image first.", gr.Dropdown(choices=[("All Shelves", "all")], value="all")

    if detection_method == "none":
        return "ℹ️ Using full image (no shelf detection)", gr.Dropdown(choices=[("Full Image", "all")], value="all")

    # Product clustering method
    if detection_method == "clustering":
        print(f"🔍 Stage 0: Clustering products into shelf rows...")

        if not PRODUCT_CLUSTERING_AVAILABLE:
            return "⚠️ Product clustering not available", gr.Dropdown(choices=[("All Shelves", "all")], value="all")

        # Detect all products using SKU110K
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        image.save(temp_path, quality=95)

        try:
            results_all = SKU110K_MODEL(temp_path, conf=conf, verbose=False)[0]
            all_products_boxes = results_all.boxes.xyxy.cpu().numpy() if results_all.boxes is not None else np.array([])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if len(all_products_boxes) == 0:
            return "⚠️ No products detected in image", gr.Dropdown(choices=[("All Shelves", "all")], value="all")

        # Cluster products into shelf rows
        # Use moderate clustering distance for regional grouping
        clustering_distance = max(40, image.height * 0.04)
        print(f"   Using clustering distance: {clustering_distance:.0f}px")

        try:
            shelf_crops = detect_shelves_from_products(
                image,
                all_products_boxes,
                clustering_eps=clustering_distance,
                min_products_per_row=3,
                expand_margin=0.15
            )
            num_shelves = len(shelf_crops)
            print(f"   Found {num_shelves} shelf row(s) from {len(all_products_boxes)} products")

            # Build info text
            info_lines = [f"✅ **Detected {num_shelves} shelf row(s)** from {len(all_products_boxes)} products\n"]
            for idx, (_, meta) in enumerate(shelf_crops):
                product_count = meta.get('product_count', 0)
                info_lines.append(f"- **Shelf {idx}**: {product_count} products")

            info_text = "\n".join(info_lines)

            # Build dropdown choices
            choices = [("All Shelves (Combined)", "all")]
            for idx in range(num_shelves):
                choices.append((f"Shelf {idx} ({'Top' if idx == 0 else 'Bottom' if idx == num_shelves-1 else 'Middle'})", str(idx)))

            return info_text, gr.Dropdown(choices=choices, value="all")

        except Exception as e:
            return f"⚠️ Shelf detection failed: {e}", gr.Dropdown(choices=[("All Shelves", "all")], value="all")

    return "⚠️ Unknown detection method", gr.Dropdown(choices=[("All Shelves", "all")], value="all")


def run_shelf_analysis_with_preprocessing(
    image: Image.Image,
    conf: float,
    iou_threshold: float,
    detection_method: str = "none",
    specific_shelf: str = "all"
):
    """
    Enhanced shelf analysis with product clustering.

    Detection methods:
        - "none": Use full image (original behavior)
        - "clustering": Cluster detected products into rows (recommended)

    Args:
        image: Input PIL Image
        conf: YOLO confidence threshold
        iou_threshold: IoU threshold for matching detections
        detection_method: Shelf detection method ("none" or "clustering")
        specific_shelf: Which shelf to analyze ("all" or shelf index "0", "1", etc.)

    Returns:
        Tuple of (annotated_image, metrics_json_text)
    """
    if not SHELF_ANALYSIS_AVAILABLE:
        raise gr.Error("Shelf analysis models are not available.")

    if image is None:
        raise gr.Error("Please upload an image.")

    # If no preprocessing, use original function
    if detection_method == "none":
        return run_shelf_analysis(image, conf, iou_threshold)

    # Product clustering method (RECOMMENDED)
    if detection_method == "clustering":
        print(f"🔍 Stage 0: Clustering products into shelf rows...")

        if not PRODUCT_CLUSTERING_AVAILABLE:
            print("⚠️ Product clustering not available, falling back to full image")
            return run_shelf_analysis(image, conf, iou_threshold)

        # First, detect ALL products using SKU110K
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        image.save(temp_path, quality=95)

        try:
            results_all = SKU110K_MODEL(temp_path, conf=conf, verbose=False)[0]
            all_products_boxes = results_all.boxes.xyxy.cpu().numpy() if results_all.boxes is not None else np.array([])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if len(all_products_boxes) == 0:
            print("⚠️ No products detected, using full image")
            return run_shelf_analysis(image, conf, iou_threshold)

        # Cluster products into shelf rows
        # Use moderate clustering distance for regional grouping
        clustering_distance = max(40, image.height * 0.04)
        print(f"   Using clustering distance: {clustering_distance:.0f}px")

        try:
            shelf_crops = detect_shelves_from_products(
                image,
                all_products_boxes,
                clustering_eps=clustering_distance,
                min_products_per_row=3,  # Require at least 3 products per row
                expand_margin=0.15  # 15% padding
            )
            print(f"   Found {len(shelf_crops)} shelf row(s) from {len(all_products_boxes)} products")
        except Exception as e:
            print(f"⚠️ Product clustering failed: {e}, using full image")
            return run_shelf_analysis(image, conf, iou_threshold)

    else:
        raise gr.Error(f"Unknown detection method: {detection_method}")

    # Filter to specific shelf if requested
    if specific_shelf != "all":
        try:
            selected_idx = int(specific_shelf)
            if 0 <= selected_idx < len(shelf_crops):
                shelf_crops = [shelf_crops[selected_idx]]
                print(f"📌 Analyzing only Shelf {selected_idx} (of {len(shelf_crops)} total)")
            else:
                print(f"⚠️ Invalid shelf index {selected_idx}, analyzing all shelves")
        except ValueError:
            print(f"⚠️ Invalid shelf selection '{specific_shelf}', analyzing all shelves")

    # Aggregate results from all shelf regions
    all_products_boxes_global = []
    ubl_boxes_global = []
    shelf_regions_info = []

    for shelf_idx, (shelf_img, shelf_meta) in enumerate(shelf_crops):
        x_offset, y_offset, _, _ = shelf_meta['box']
        print(f"   Shelf {shelf_idx}/{len(shelf_crops)-1}: {shelf_img.size}, region {shelf_meta['box']}")

        # Save shelf crop to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        shelf_img.save(temp_path, quality=95)

        try:
            # Stage 1: Detect all products in this shelf
            results_all = SKU110K_MODEL(temp_path, conf=conf, verbose=False)[0]
            shelf_all_boxes = results_all.boxes.xyxy.cpu().numpy() if results_all.boxes is not None else np.array([])

            # Stage 2: Detect UBL products in this shelf
            results_ubl = UBL_MODEL(temp_path, conf=conf, verbose=False)[0]
            shelf_ubl_boxes = results_ubl.boxes.xyxy.cpu().numpy() if results_ubl.boxes is not None else np.array([])

            # Transform coordinates from shelf-local to global image coordinates
            if len(shelf_all_boxes) > 0:
                shelf_all_boxes[:, [0, 2]] += x_offset  # x1, x2
                shelf_all_boxes[:, [1, 3]] += y_offset  # y1, y2
                all_products_boxes_global.append(shelf_all_boxes)

            if len(shelf_ubl_boxes) > 0:
                shelf_ubl_boxes[:, [0, 2]] += x_offset  # x1, x2
                shelf_ubl_boxes[:, [1, 3]] += y_offset  # y1, y2
                ubl_boxes_global.append(shelf_ubl_boxes)

            shelf_regions_info.append({
                'index': shelf_idx,
                'box': shelf_meta['box'],
                'products': len(shelf_all_boxes),
                'ubl_products': len(shelf_ubl_boxes)
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Combine all detections
    all_products_boxes = np.vstack(all_products_boxes_global) if all_products_boxes_global else np.array([])
    ubl_boxes = np.vstack(ubl_boxes_global) if ubl_boxes_global else np.array([])

    print(f"✅ Combined: {len(all_products_boxes)} total products, {len(ubl_boxes)} UBL detections")

    # Match detections (same logic as original)
    if len(all_products_boxes) > 0 and len(ubl_boxes) > 0:
        ubl_indices, non_ubl_indices = match_detections(all_products_boxes, ubl_boxes, iou_threshold)
    elif len(all_products_boxes) > 0:
        ubl_indices = set()
        non_ubl_indices = set(range(len(all_products_boxes)))
    else:
        ubl_indices = set()
        non_ubl_indices = set()

    # Calculate metrics
    total_products = len(all_products_boxes)
    ubl_count = len(ubl_indices)
    non_ubl_count = len(non_ubl_indices)
    share_of_shelf = (ubl_count / total_products * 100) if total_products > 0 else 0

    # Create visualization
    result_image = draw_shelf_analysis_results(
        image, all_products_boxes, ubl_indices, non_ubl_indices,
        total_products, ubl_count, non_ubl_count, share_of_shelf
    )

    # Create enhanced metrics with shelf information
    metrics = {
        "shelf_detection_enabled": True,
        "detection_method": detection_method,
        "analyzed_shelf": specific_shelf if specific_shelf != "all" else "all_combined",
        "shelves_detected": len(shelf_crops),
        "shelf_regions": shelf_regions_info,
        "total_products": total_products,
        "unilever_products": ubl_count,
        "non_unilever_products": non_ubl_count,
        "unilever_share_percentage": round(share_of_shelf, 2)
    }

    metrics_text = json.dumps(metrics, indent=2)

    return result_image, metrics_text


# ============================================================================
# Fixed Shelf Analysis Functions
# ============================================================================

def detect_shelftalker_rois(
    image: Image.Image,
    conf: float,
    expand_margin: float = 0.1,
    combine_rois: bool = True
) -> Tuple[List[Tuple[List[float], dict]], List[dict]]:
    """
    Detect shelftalker ROIs in the image and optionally combine them into one planogram area.

    Args:
        image: Input PIL Image
        conf: Confidence threshold for shelftalker detection
        expand_margin: Margin to expand ROI boxes (fraction of combined box dimensions)
        combine_rois: If True, combine all shelftalkers into one large ROI

    Returns:
        Tuple of (combined_rois, individual_shelftalkers)
        - combined_rois: List with single (roi_box, metadata) for the entire planogram
        - individual_shelftalkers: List of individual shelftalker detections for visualization
    """
    if not FIXED_SHELF_ANALYSIS_AVAILABLE:
        return [], []

    # Save PIL image to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    image.save(temp_path, quality=95)

    try:
        # Detect shelftalkers
        results = SHELFTALKER_MODEL(temp_path, conf=conf, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
        scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([])
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.array([])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if len(boxes) == 0:
        return [], []

    # Store individual shelftalker detections
    individual_shelftalkers = []
    for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        shelftalker = {
            'index': idx,
            'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            'score': float(score),
            'class_id': int(class_id),
            'class_name': SHELFTALKER_MODEL.names.get(class_id, str(class_id))
        }
        individual_shelftalkers.append(shelftalker)

    if not combine_rois:
        # Return individual ROIs - each shelftalker is a separate fixed shelf
        rois = []
        for shelf in individual_shelftalkers:
            # Expand each individual shelftalker box
            x1, y1, x2, y2 = shelf['box']
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, x1 - width * expand_margin)
            y1 = max(0, y1 - height * expand_margin)
            x2 = min(image.width, x2 + width * expand_margin)
            y2 = min(image.height, y2 + height * expand_margin)

            roi_box = [float(x1), float(y1), float(x2), float(y2)]
            # Update shelf metadata with expanded box
            shelf_meta = shelf.copy()
            shelf_meta['expanded_box'] = roi_box
            rois.append((roi_box, shelf_meta))
        return rois, individual_shelftalkers

    # Combine all shelftalker boxes into one large ROI
    # Use INNER boundaries to create a tighter ROI
    x1_min = np.min(boxes[:, 0])
    y1_min = np.min(boxes[:, 1])
    x2_max = np.max(boxes[:, 2])
    y2_max = np.max(boxes[:, 3])

    # Calculate dimensions for margin adjustment
    width = x2_max - x1_min
    height = y2_max - y1_min

    # Use NEGATIVE margin to shrink inward (move boundaries inside the shelftalkers)
    # This ensures we only detect products truly within the planogram area
    inward_margin = expand_margin * 0.5  # Use half the expand_margin to shrink inward

    x1_min = max(0, x1_min + width * inward_margin)
    y1_min = max(0, y1_min + height * inward_margin)
    x2_max = min(image.width, x2_max - width * inward_margin)
    y2_max = min(image.height, y2_max - height * inward_margin)

    combined_box = [float(x1_min), float(y1_min), float(x2_max), float(y2_max)]

    # Create metadata for combined ROI
    combined_metadata = {
        'index': 0,
        'score': float(np.mean(scores)),  # Average score of all shelftalkers
        'num_shelftalkers': len(boxes),
        'class_name': 'planogram',
        'individual_shelftalkers': individual_shelftalkers
    }

    return [(combined_box, combined_metadata)], individual_shelftalkers


def detect_ubl_in_rois(
    image: Image.Image,
    rois: List[Tuple[List[float], dict]],
    conf: float
) -> Tuple[List[dict], List[dict]]:
    """
    Detect UBL products within shelftalker ROIs.

    Args:
        image: Input PIL Image
        rois: List of (roi_box, metadata) from detect_shelftalker_rois
        conf: Confidence threshold for UBL detection

    Returns:
        Tuple of (all_detections, roi_summaries)
        - all_detections: List of detection dicts with global coordinates
        - roi_summaries: List of summary dicts per ROI
    """
    if not FIXED_SHELF_ANALYSIS_AVAILABLE or len(rois) == 0:
        return [], []

    all_detections = []
    roi_summaries = []

    for roi_box, roi_meta in rois:
        x_offset, y_offset, x_max, y_max = roi_box
        roi_idx = roi_meta['index']

        # Crop ROI from image
        roi_image = image.crop((x_offset, y_offset, x_max, y_max))

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        roi_image.save(temp_path, quality=95)

        try:
            # Detect UBL products in this ROI
            results = UBL_MODEL(temp_path, conf=conf, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
            scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([])
            class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.array([])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Transform local coordinates to global coordinates
        roi_detections = []
        product_classes = defaultdict(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            # Convert to global coordinates
            x1_global = box[0] + x_offset
            y1_global = box[1] + y_offset
            x2_global = box[2] + x_offset
            y2_global = box[3] + y_offset

            class_name = UBL_MODEL.names.get(class_id, str(class_id))
            product_classes[class_name] += 1

            detection = {
                'box': [float(x1_global), float(y1_global), float(x2_global), float(y2_global)],
                'score': float(score),
                'class_id': int(class_id),
                'class_name': class_name,
                'roi_index': roi_idx
            }

            roi_detections.append(detection)
            all_detections.append(detection)

        # Create summary for this ROI
        summary = {
            'roi_index': roi_idx,
            'roi_box': roi_box,
            'shelftalker_name': roi_meta['class_name'],
            'total_products': len(roi_detections),
            'product_breakdown': dict(product_classes),
            'detections': roi_detections
        }

        roi_summaries.append(summary)

    return all_detections, roi_summaries


def calculate_planogram_adherence(roi_summaries: List[dict]) -> dict:
    """
    Calculate planogram adherence metrics.

    This is a placeholder that can be extended with actual planogram rules.
    For now, it calculates:
    - Number of shelftalkers detected
    - Total products in planogram area
    - Product diversity

    Args:
        roi_summaries: List of ROI summary dicts

    Returns:
        Dict with planogram metrics
    """
    total_fixed_shelves = len(roi_summaries)
    total_products = sum(s['total_products'] for s in roi_summaries)

    # Calculate metrics per shelf
    shelves_detail = []
    for summary in roi_summaries:
        shelf_detail = {
            'shelf_index': summary['roi_index'],
            'shelftalker': summary['shelftalker_name'],
            'product_count': summary['total_products'],
            'unique_products': len(summary['product_breakdown']),
            'product_breakdown': summary['product_breakdown']
        }
        shelves_detail.append(shelf_detail)

    # Placeholder for adherence score (can be refined based on actual planogram rules)
    # For now, we assume shelves with products have good adherence
    adherence_score = 100.0 if total_products > 0 else 0.0

    return {
        'total_fixed_shelves': total_fixed_shelves,
        'total_products': total_products,
        'adherence_score': adherence_score,
        'shelves_detail': shelves_detail
    }


def draw_fixed_shelf_results(
    image: Image.Image,
    rois: List[Tuple[List[float], dict]],
    individual_shelftalkers: List[dict],
    detections: List[dict],
    metrics: dict,
    show_shelftalkers: bool = True,
    show_roi: bool = True,
    show_products: bool = True
) -> Image.Image:
    """
    Draw visualization for fixed shelf analysis.

    Args:
        image: Input PIL Image
        rois: List of (roi_box, metadata) tuples (combined planogram ROI)
        individual_shelftalkers: List of individual shelftalker detections
        detections: List of UBL detection dicts
        metrics: Planogram metrics dict
        show_shelftalkers: Whether to show shelftalker boxes
        show_roi: Whether to show ROI box
        show_products: Whether to show product boxes

    Returns:
        Annotated PIL Image
    """
    # Convert PIL to OpenCV format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_draw = image_bgr.copy()

    # Draw individual shelftalkers in blue FIRST (so they're behind the ROI)
    if show_shelftalkers:
        for shelf in individual_shelftalkers:
            x1, y1, x2, y2 = map(int, shelf['box'])
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 200, 0), 2)  # Light blue for shelftalker stickers

    # Draw the combined planogram ROI in orange (on top)
    if show_roi:
        for roi_box, roi_meta in rois:
            x1, y1, x2, y2 = map(int, roi_box)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 165, 255), 4)  # Orange for planogram ROI
            label = f"Planogram ROI ({roi_meta.get('num_shelftalkers', 0)} shelftalkers)"
            cv2.putText(img_draw, label, (x1, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Draw UBL product detections in green
    if show_products:
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}"
            cv2.putText(img_draw, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Add metrics overlay
    overlay = img_draw.copy()
    height, width = img_draw.shape[:2]

    # Draw semi-transparent background for metrics
    cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
    img_draw = cv2.addWeighted(img_draw, 0.7, overlay, 0.3, 0)

    # Add metrics text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_draw, f"No of Shelftalker: {metrics['total_fixed_shelves']}",
               (20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img_draw, f"Total Products: {metrics['total_products']}",
               (20, 75), font, 0.8, (0, 255, 0), 2)

    # Add adherence score prominently
    adherence_text = f"Adherence: {metrics['adherence_score']:.0f}%"
    text_size = cv2.getTextSize(adherence_text, font, 1.2, 3)[0]
    text_x = width - text_size[0] - 20
    text_y = 50
    cv2.rectangle(img_draw, (text_x-10, text_y-40),
                 (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
    cv2.putText(img_draw, adherence_text, (text_x, text_y),
               font, 1.2, (0, 255, 255), 3)

    # Convert back to RGB for PIL
    result_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def run_fixed_shelf_analysis(
    image: Image.Image,
    shelftalker_conf: float,
    ubl_conf: float,
    expand_margin: float = 0.1,
    show_shelftalkers: bool = True,
    show_roi: bool = True,
    show_products: bool = True
):
    """
    Run complete fixed shelf analysis pipeline.

    Pipeline:
    1. Detect all shelftalkers (top, bottom, sides)
    2. Use outer boundaries to create ONE combined planogram ROI
    3. Detect UBL products within the combined planogram ROI
    4. Calculate planogram adherence metrics

    Args:
        image: Input PIL Image
        shelftalker_conf: Confidence threshold for shelftalker detection
        ubl_conf: Confidence threshold for UBL detection
        expand_margin: Margin to expand the combined ROI box
        show_shelftalkers: Whether to show shelftalker boxes
        show_roi: Whether to show ROI box
        show_products: Whether to show product boxes

    Returns:
        Tuple of (annotated_image, metrics_json_text)
    """
    if not FIXED_SHELF_ANALYSIS_AVAILABLE:
        raise gr.Error("Fixed Shelf Analysis requires the Shelftalker model.")

    if image is None:
        raise gr.Error("Please upload an image.")

    # Step 1: Detect all shelftalkers and combine into one planogram ROI
    print("🔍 Step 1: Detecting shelftalkers and creating planogram ROI...")
    rois, individual_shelftalkers = detect_shelftalker_rois(image, shelftalker_conf, expand_margin, combine_rois=True)

    if len(rois) == 0:
        print("⚠️ No shelftalkers detected")
        return image, json.dumps({
            "message": "No shelftalkers detected in the image",
            "total_fixed_shelves": 0
        }, indent=2)

    print(f"   Found {len(individual_shelftalkers)} shelftalker(s), combined into planogram ROI")

    # Step 2: Detect UBL products within the combined planogram ROI
    print("📦 Step 2: Detecting UBL products within planogram ROI...")
    all_detections, roi_summaries = detect_ubl_in_rois(image, rois, ubl_conf)
    print(f"   Found {len(all_detections)} UBL product(s) in planogram area")

    # Step 3: Calculate planogram metrics
    print("📊 Step 3: Calculating planogram adherence...")
    metrics = calculate_planogram_adherence(roi_summaries)

    # Update to reflect number of shelftalkers
    metrics['total_fixed_shelves'] = len(individual_shelftalkers)

    # Step 4: Create visualization
    print("🎨 Step 4: Creating visualization...")
    result_image = draw_fixed_shelf_results(
        image, rois, individual_shelftalkers, all_detections, metrics,
        show_shelftalkers=show_shelftalkers,
        show_roi=show_roi,
        show_products=show_products
    )

    # Create detailed JSON output
    output = {
        "no_of_shelftalker": metrics['total_fixed_shelves'],
        "total_products": metrics['total_products'],
        "planogram_adherence_score": metrics['adherence_score'],
        "shelftalkers_detected": [
            {
                'position': f"shelftalker_{s['index']}",
                'class_name': s['class_name'],
                'confidence': s['score']
            } for s in individual_shelftalkers
        ],
        "product_breakdown": roi_summaries[0]['product_breakdown'] if roi_summaries else {}
    }

    metrics_text = json.dumps(output, indent=2)

    print("✅ Fixed shelf analysis complete!")

    return result_image, metrics_text


# ============================================================================
# Category Analysis Functions
# ============================================================================

def run_category_analysis(image: Image.Image, conf: float):
    """
    Run detection and categorize products into broad categories.

    Uses the UBL model to detect products and then categorizes them
    based on their class names.
    """
    if not SHELF_ANALYSIS_AVAILABLE:
        raise gr.Error("Category analysis requires the UBL model.")

    if image is None:
        raise gr.Error("Please upload an image.")

    # Save PIL image to temporary file for consistent preprocessing
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    image.save(temp_path, quality=95)

    try:
        # Detect UBL products
        results = UBL_MODEL(temp_path, conf=conf, verbose=False)[0]
        total_detections = len(results.boxes) if results.boxes is not None else 0
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if total_detections == 0:
        # Return original image with message
        return image, json.dumps({"message": "No products detected", "total_products": 0}, indent=2)

    # Categorize detections
    category_counts, category_detections = categorize_detections(results, UBL_MODEL)

    # Create visualization with bounding boxes
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    result_image = draw_category_results(image_bgr, category_detections)

    # Add title overlay
    overlay = result_image.copy()
    height, width = result_image.shape[:2]

    # Draw semi-transparent background for title
    cv2.rectangle(overlay, (10, 10), (400, 60), (0, 0, 0), -1)
    result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)

    # Add title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_image, f'Total Products: {total_detections}',
               (20, 45), font, 1.0, (255, 255, 255), 2)

    # Add category breakdown chart
    result_image = draw_category_pie_chart(result_image, category_counts, total_detections)

    # Convert BGR back to RGB for PIL
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    # Create metrics JSON
    metrics = {
        "total_products": total_detections,
        "categories": {}
    }

    # Add category breakdown with percentages
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        metrics["categories"][display_name] = {
            "count": count,
            "percentage": round(percentage, 2)
        }

    metrics_text = json.dumps(metrics, indent=2)

    return result_pil, metrics_text


# Run YOLO inference on a PIL image and return annotated image, detections, and optional YOLO TXT lines
def _run_yolo_on_pil(image: Image.Image, conf: float, iou: float, imgsz: int, selected_classes: Optional[List[str]] = None, want_yolo_boxes: bool = False,
                     include_conf: bool = False, show_labels: bool = True, show_conf: bool = True) -> Tuple[Image.Image, dict, Optional[List[str]]]:

    # Save PIL image to temporary file to match annotate.py preprocessing
    # This ensures identical results between app.py and annotate.py
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    image.save(temp_path, quality=95)

    try:
        # Run YOLO model prediction from file path (same as annotate.py)
        results = MODEL.predict(source=temp_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        r = results[0]           # first (and only) prediction result
        names = r.names          # class name dictionary
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    dets = []                # store detections for JSON output
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        xyxy = boxes.xyxy    # bounding boxes (x1, y1, x2, y2)
        confs = boxes.conf   # confidence scores
        clss  = boxes.cls    # class IDs

        # Filter detections by selected classes if provided
        if selected_classes:
            allowed_ids = {cid for cid, n in names.items() if n in selected_classes}
            mask = torch.tensor([int(c.item()) in allowed_ids for c in clss], dtype=torch.bool, device=clss.device)
            r.boxes = boxes[mask]
            xyxy = r.boxes.xyxy
            confs = r.boxes.conf
            clss  = r.boxes.cls

        # Convert boxes and metadata into JSON-friendly dicts
        if len(r.boxes) > 0:
            xyxy_np = xyxy.detach().cpu().numpy()
            conf_np = confs.detach().cpu().numpy()
            cls_np  = clss.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy_np, conf_np, cls_np):
                dets.append({
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(c),
                    "class_id": int(k),
                    "class_name": names.get(int(k), str(int(k)))
                })

    # Optionally create YOLO-format TXT lines
    yolo_lines = _yolo_box_txt(r, include_conf=include_conf) if want_yolo_boxes else None

    # Create annotated image for visualization
    # Note: r.plot() returns BGR numpy array (OpenCV format)
    annotated_bgr = r.plot(labels=show_labels, conf=show_conf, line_width=2)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Return annotated image, detections JSON, and YOLO TXT lines
    return annotated_pil, {"detections": dets}, yolo_lines





# Save annotated image and detections (JSON OR YOLO TXT) into a temporary ZIP file
def _save_results_zip(annotated_img: Image.Image, dets: dict, yolo_lines: Optional[List[str]] = None, base_name: str = "result") -> str:
    run_id = uuid.uuid4().hex[:8]  # unique ID for each export
    out_dir = os.path.join(tempfile.gettempdir(), f"yolo_{run_id}")
    os.makedirs(out_dir, exist_ok=True)  # create temp folder

    img_path = os.path.join(out_dir, f"{base_name}_annotated.png")  # save annotated image
    annotated_img.save(img_path)

    if yolo_lines is not None:  # export YOLO TXT labels
        txt_path = os.path.join(out_dir, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    else:  # export JSON detections
        json_path = os.path.join(out_dir, f"{base_name}_detections.json")
        with open(json_path, "w") as f:
            json.dump(dets, f, indent=2)

    # compress results into a single ZIP archive
    zip_base = os.path.join(tempfile.gettempdir(), f"yolo_results_{run_id}")
    zip_path = shutil.make_archive(zip_base, "zip", out_dir)
    return zip_path  # return ZIP path for download





# Run detection on a single uploaded image and return annotated preview, JSON, and ZIP
def ui_predict_single(image: Image.Image, conf: float, iou: float, imgsz: int, class_filter: List[str], export_fmt: str, include_conf: bool, show_labels: bool, show_conf: bool):
    if image is None:
        raise gr.Error("🖼️ Please upload an image.")  # handle empty input

    want_yolo = (export_fmt == "YOLO TXT")  # check export format choice
    # Run YOLO inference and get annotated image + detections + optional YOLO TXT lines
    annotated, dets, yolo_lines = _run_yolo_on_pil(image, conf, iou, imgsz, class_filter or None, want_yolo_boxes=want_yolo, include_conf=include_conf, show_labels=show_labels, show_conf=show_conf)

    # Save results (image + JSON or YOLO TXT) into a ZIP for download
    zip_path = _save_results_zip(annotated, dets, yolo_lines=yolo_lines if want_yolo else None, base_name="image")

    # Return annotated image preview, JSON detections text, and ZIP download path
    return annotated, json.dumps(dets, indent=2), zip_path





# Run YOLO inference on multiple uploaded images and return gallery, combined JSON, and ZIP
def ui_predict_batch(files: List[gr.File], conf: float, iou: float, imgsz: int, class_filter: List[str], export_fmt: str, include_conf: bool, show_labels: bool, show_conf: bool):
    if not files:
        raise gr.Error("🗂️ Please upload one or more images.")  # handle empty batch

    want_yolo = (export_fmt == "YOLO TXT")  # check chosen export format

    gallery_outputs = []  # store annotated images for gallery preview
    all_json = []         # store combined detection data

    run_id = uuid.uuid4().hex[:8]  # unique ID for batch run
    out_dir = os.path.join(tempfile.gettempdir(), f"yolo_batch_{run_id}")
    os.makedirs(out_dir, exist_ok=True)  # create temp folder for results

    # Process each uploaded image
    for f in files:
        fpath = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
        try:
            img = Image.open(fpath).convert("RGB")  # open image safely
        except Exception as e:
            raise gr.Error(f"❌ Failed to open image: {fpath}\n{e}")
        fname = os.path.basename(fpath)

        # Run YOLO model and get detections
        annotated, dets, yolo_lines = _run_yolo_on_pil(img, conf, iou, imgsz, class_filter or None, want_yolo_boxes=want_yolo, include_conf=include_conf, show_labels=show_labels, show_conf=show_conf)

        # Show directly in the Gallery
        gallery_outputs.append((annotated, fname))

        # Append detections to combined list
        all_json.append({"file": fname, **dets})

        # Save individual outputs for ZIP
        base = os.path.splitext(fname)[0]
        annotated_path = os.path.join(out_dir, f"{base}_annotated.png")
        annotated.save(annotated_path)

        if want_yolo:  # save YOLO TXT format
            txt_path = os.path.join(out_dir, f"{base}.txt")
            with open(txt_path, "w") as tf:
                tf.write("\n".join(yolo_lines or []) + "\n")
        else:  # save JSON format
            json_path = os.path.join(out_dir, f"{base}_detections.json")
            with open(json_path, "w") as jf:
                json.dump(dets, jf, indent=2)

    # Compress all outputs into one ZIP file
    zip_base = os.path.join(tempfile.gettempdir(), f"yolo_batch_{run_id}")
    zip_path = shutil.make_archive(zip_base, "zip", out_dir)

    # Return gallery preview, combined JSON text, and ZIP path for download
    return gallery_outputs, json.dumps(all_json, indent=2), zip_path

def ui_load_classes():
    return gr.update(choices=_class_names(), value=[])

def update_ui_visibility_for_shelf():
    """Show only shelf share settings"""
    return (
        gr.update(visible=False),  # model_accordion
        gr.update(visible=False),  # controls_accordion
        gr.update(visible=False),  # class_filter_accordion
        gr.update(visible=False),  # export_accordion
        gr.update(visible=True),   # shelf_settings_accordion
        gr.update(visible=False),  # category_settings_accordion
        gr.update(visible=False),  # fixed_settings_accordion
        gr.update(visible=False),  # examples_accordion
        gr.update(visible=True),   # shelf_examples_accordion
        gr.update(visible=False),  # category_examples_accordion
        gr.update(visible=False),  # fixed_examples_accordion
    )

def update_ui_visibility_for_category():
    """Show only category analysis settings"""
    return (
        gr.update(visible=False),  # model_accordion
        gr.update(visible=False),  # controls_accordion
        gr.update(visible=False),  # class_filter_accordion
        gr.update(visible=False),  # export_accordion
        gr.update(visible=False),  # shelf_settings_accordion
        gr.update(visible=True),   # category_settings_accordion
        gr.update(visible=False),  # fixed_settings_accordion
        gr.update(visible=False),  # examples_accordion
        gr.update(visible=False),  # shelf_examples_accordion
        gr.update(visible=True),   # category_examples_accordion
        gr.update(visible=False),  # fixed_examples_accordion
    )

def update_ui_visibility_for_fixed():
    """Show only fixed shelf settings"""
    return (
        gr.update(visible=False),  # model_accordion
        gr.update(visible=False),  # controls_accordion
        gr.update(visible=False),  # class_filter_accordion
        gr.update(visible=False),  # export_accordion
        gr.update(visible=False),  # shelf_settings_accordion
        gr.update(visible=False),  # category_settings_accordion
        gr.update(visible=True),   # fixed_settings_accordion
        gr.update(visible=False),  # examples_accordion
        gr.update(visible=False),  # shelf_examples_accordion
        gr.update(visible=False),  # category_examples_accordion
        gr.update(visible=True),   # fixed_examples_accordion
    )

def update_ui_visibility_for_regular():
    """Show regular detection settings"""
    return (
        gr.update(visible=True),   # model_accordion
        gr.update(visible=True),   # controls_accordion
        gr.update(visible=True),   # class_filter_accordion
        gr.update(visible=True),   # export_accordion
        gr.update(visible=False),  # shelf_settings_accordion
        gr.update(visible=False),  # category_settings_accordion
        gr.update(visible=False),  # fixed_settings_accordion
        gr.update(visible=True),   # examples_accordion
        gr.update(visible=False),  # shelf_examples_accordion
        gr.update(visible=False),  # category_examples_accordion
        gr.update(visible=False),  # fixed_examples_accordion
    )





with gr.Blocks(
    title="UBL-YOLO Product Detection",
    theme=theme,
    css="""
        .gradio-container {
            max-width: 100% !important;
        }
        .header-section {
            padding: 2rem 0;
            border-bottom: 2px solid rgba(0, 87, 184, 0.1);
            margin-bottom: 2rem;
        }
        .logo-img {
            border-radius: 8px;
        }
        .app-title {
            font-size: 2.25rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.2;
        }
        .app-subtitle {
            font-size: 1rem;
            margin: 0.75rem 0 0 0;
            opacity: 0.65;
            line-height: 1.4;
        }
        .info-box {
            background: rgba(0, 87, 184, 0.05);
            border-left: 3px solid #0057B8;
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .info-box p {
            margin: 0.25rem 0;
            line-height: 1.6;
        }
    """
) as demo:
    with gr.Row(elem_classes=["header-section"]):
        with gr.Column(scale=0, min_width=140):
            gr.Image(
                value="./public/unilever.jpg",
                show_label=False,
                show_download_button=False,
                container=False,
                height=120,
                width=120,
                elem_classes=["logo-img"]
            )
        with gr.Column(scale=1):
            gr.HTML('<h1 class="app-title">Unilever Product Detector</h1>')
            gr.HTML('<p class="app-subtitle">AI-powered product recognition and shelf share analysis</p>')

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                with gr.Tab("Single Image", id="single") as tab_single:
                    in_img  = gr.Image(type="pil", label="Input Image", height=720)
                    with gr.Row():
                        run_btn = gr.Button("Run Detection", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear")
                    out_img = gr.Image(type="pil", label="Detection Results", height=720)
                    out_zip = gr.File(label="Download Results (ZIP)")
                    out_json = gr.Code(label="Detections (JSON)", language="json", lines=18, max_lines=18, interactive=False)

                with gr.Tab("Batch Processing", id="batch") as tab_batch:
                    in_files = gr.File(label="Upload Multiple Images", file_types=["image"], file_count="multiple")
                    with gr.Row():
                        run_batch_btn = gr.Button("Run Batch Detection", variant="primary", size="lg")
                        clear_batch_btn = gr.Button("Clear")
                    out_gallery = gr.Gallery(label="Detection Results", columns=2, height=720)
                    out_batch_zip = gr.File(label="Download Batch Results (ZIP)")
                    out_batch_json = gr.Code(label="Combined Detections (JSON)", language="json", lines=18, max_lines=18, interactive=False)

                with gr.Tab("Shelf Share Analysis", id="shelf_share") as tab_shelf:
                    shelf_in_img = gr.Image(type="pil", label="Input Shelf Image", height=720)

                    gr.Markdown("### Step 1: Detect Shelves")
                    with gr.Row():
                        detect_shelves_btn = gr.Button("🔍 Detect Shelves", variant="secondary", size="lg")
                        shelf_clear_btn = gr.Button("Clear")
                    shelf_detection_info = gr.Markdown("Upload an image and click 'Detect Shelves' to begin")

                    gr.Markdown("### Step 2: Select & Analyze")
                    shelf_selector = gr.Dropdown(
                        label="Select Shelf to Analyze",
                        choices=[("All Shelves (Combined)", "all")],
                        value="all",
                        interactive=True
                    )
                    shelf_run_btn = gr.Button("📊 Analyze Shelf Share", variant="primary", size="lg")

                    shelf_out_img = gr.Image(type="pil", label="Analysis Results", height=720)
                    shelf_metrics = gr.Code(label="Share Metrics (JSON)", language="json", lines=10, max_lines=10, interactive=False)

                    gr.HTML("""
                        <div class="info-box">
                            <p><strong>How it works:</strong></p>
                            <p>1. Detect all products using SKU-110K model</p>
                            <p>2. Group products into shelf rows automatically</p>
                            <p>3. Select which shelf to analyze</p>
                            <p>4. Calculate Unilever share of shelf</p>
                            <p><strong>Result:</strong> Green = Unilever, Red = Non-Unilever</p>
                        </div>
                    """)

                with gr.Tab("Category Analysis", id="category_analysis") as tab_category:
                    category_in_img = gr.Image(type="pil", label="Input Image", height=720)
                    with gr.Row():
                        category_run_btn = gr.Button("Analyze Categories", variant="primary", size="lg")
                        category_clear_btn = gr.Button("Clear")
                    category_out_img = gr.Image(type="pil", label="Category Results", height=720)
                    category_metrics = gr.Code(label="Category Metrics (JSON)", language="json", lines=10, max_lines=10, interactive=False)
                    gr.HTML("""
                        <div class="info-box">
                            <p><strong>Product Categories:</strong></p>
                            <p>Haircare • Oralcare • Skincare • Home & Hygiene • Food & Nutrition</p>
                        </div>
                    """)

                with gr.Tab("Fixed Shelf Analysis", id="fixed_shelf") as tab_fixed:
                    fixed_in_img = gr.Image(type="pil", label="Input Image", height=720)
                    with gr.Row():
                        fixed_run_btn = gr.Button("Analyze Fixed Shelves", variant="primary", size="lg")
                        fixed_clear_btn = gr.Button("Clear")
                    fixed_out_img = gr.Image(type="pil", label="Fixed Shelf Results", height=720)
                    fixed_metrics = gr.Code(label="Planogram Metrics (JSON)", language="json", lines=10, max_lines=10, interactive=False)
                    gr.HTML("""
                        <div class="info-box">
                            <p><strong>How it works:</strong></p>
                            <p>1. Detect all UBL shelftalker stickers (top, bottom, sides) using Shelftalker model</p>
                            <p>2. Use INNER boundaries of shelftalkers to create ONE tight planogram ROI</p>
                            <p>3. Detect ONLY UBL products INSIDE the planogram ROI using DA_YOLO11X model</p>
                            <p>4. Calculate planogram adherence metrics (entire area = one shelf)</p>
                            <p><strong>Result:</strong> Orange = Planogram ROI, Blue = Shelftalker Stickers, Green = UBL Products</p>
                        </div>
                    """)

        with gr.Column(scale=1):

            with gr.Accordion("Model Selection", open=True, visible=True) as model_accordion:
                model_selector = gr.Dropdown(label="YOLO Model", choices=[os.path.basename(m) for m in model_files], value=os.path.basename(current_model_path),
                                             filterable=False, allow_custom_value=False)
                model_status = gr.Markdown(f"**Active:** {os.path.basename(current_model_path)}")

            with gr.Accordion("Detection Settings", open=True, visible=True) as controls_accordion:
                conf = gr.Slider(0.01, 0.99, value=DEFAULT_CONF, step=0.01, label="Confidence Threshold")
                iou = gr.Slider(0.01, 0.99, value=DEFAULT_IOU, step=0.01, label="IoU Threshold")
                imgsz= gr.Slider(256, 1024, value=DEFAULT_IMGSZ, step=32, label="Image Size")

                gr.Markdown("**Display Options**")
                with gr.Row():
                    show_labels = gr.Checkbox(label="Show Labels", value=True)
                    show_conf = gr.Checkbox(label="Show Confidence", value=True)

            with gr.Accordion("Class Filter", open=False, visible=True) as class_filter_accordion:
                class_filter = gr.CheckboxGroup(label="Filter Classes (Optional)", choices=[], value=[])

            with gr.Accordion("Export Options", open=True, visible=True) as export_accordion:
                export_fmt = gr.Radio(label="Export Format", choices=["YOLO TXT", "JSON"], value="YOLO TXT")
                include_conf = gr.Checkbox(label="Include confidence scores", value=False)

            with gr.Accordion("Shelf Analysis Settings", open=True, visible=False) as shelf_settings_accordion:
                shelf_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="Confidence Threshold")
                shelf_iou = gr.Slider(0.01, 0.99, value=0.50, step=0.01, label="IoU Matching Threshold")

                gr.Markdown("**Shelf Detection Method**")
                shelf_detection_method = gr.Radio(
                    label="Detection Strategy",
                    choices=[
                        ("None - Full Image", "none"),
                        ("Product Clustering", "clustering")
                    ],
                    value="clustering",
                    info="Product Clustering: Automatically groups detected products into shelf rows"
                )

            with gr.Accordion("Category Settings", open=True, visible=False) as category_settings_accordion:
                category_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="Confidence Threshold")

            with gr.Accordion("Fixed Shelf Settings", open=True, visible=False) as fixed_settings_accordion:
                fixed_shelftalker_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="Shelftalker Confidence")
                fixed_ubl_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="UBL Product Confidence")
                fixed_expand_margin = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="ROI Tightness (lower = tighter)")

                gr.Markdown("**Display Options**")
                with gr.Row():
                    fixed_show_shelftalkers = gr.Checkbox(label="Show Shelftalker Boxes", value=True)
                    fixed_show_roi = gr.Checkbox(label="Show ROI Box", value=True)
                    fixed_show_products = gr.Checkbox(label="Show Product Boxes", value=True)

            # Collect examples by subfolder
            da_files     = sorted(glob.glob("examples/DA/*.*"))
            posm_files   = sorted(glob.glob("examples/POSM/*.*"))
            sachet_files = sorted(glob.glob("examples/SACHET/*.*"))
            sos_files    = sorted(glob.glob("examples/ShareOfShelf/**/*.*", recursive=True))

            with gr.Accordion("Examples", open=False) as examples_accordion:
                with gr.Tabs() as examples_tabs:
                    with gr.Tab("DA") as examples_da_tab:
                        examples_da = gr.Examples(examples=[[f] for f in da_files], inputs=[in_img], examples_per_page=len(da_files) or 1)
                    with gr.Tab("POSM") as examples_posm_tab:
                        examples_posm = gr.Examples(examples=[[f] for f in posm_files], inputs=[in_img], examples_per_page=len(posm_files) or 1)
                    with gr.Tab("SACHET") as examples_sachet_tab:
                        examples_sachet = gr.Examples(examples=[[f] for f in sachet_files], inputs=[in_img], examples_per_page=len(sachet_files) or 1)
                    with gr.Tab("ShareOfShelf") as examples_sos_tab:
                        examples_sos = gr.Examples(examples=[[f] for f in sos_files], inputs=[in_img], examples_per_page=len(sos_files) or 1)

            with gr.Accordion("Examples", open=False, visible=False) as shelf_examples_accordion:
                shelf_examples = gr.Examples(examples=[[f] for f in da_files], inputs=[shelf_in_img], examples_per_page=len(da_files) or 1)

            with gr.Accordion("Examples", open=False, visible=False) as category_examples_accordion:
                category_examples = gr.Examples(examples=[[f] for f in da_files], inputs=[category_in_img], examples_per_page=len(da_files) or 1)

            with gr.Accordion("Examples", open=False, visible=False) as fixed_examples_accordion:
                fixed_examples = gr.Examples(examples=[[f] for f in da_files], inputs=[fixed_in_img], examples_per_page=len(da_files) or 1)

    # Wire functions
    model_selector.change(fn=select_model, inputs=model_selector, outputs=[model_status, class_filter])
    run_btn.click(fn=ui_predict_single, inputs=[in_img, conf, iou, imgsz, class_filter, export_fmt, include_conf, show_labels, show_conf], outputs=[out_img, out_json, out_zip])
    clear_btn.click(fn=lambda: (gr.update(value=None), None, "", None), inputs=None, outputs=[in_img, out_img, out_json, out_zip])

    run_batch_btn.click(fn=ui_predict_batch, inputs=[in_files, conf, iou, imgsz, class_filter, export_fmt, include_conf, show_labels, show_conf],
                        outputs=[out_gallery, out_batch_json, out_batch_zip])
    clear_batch_btn.click(fn=lambda: (gr.update(value=None), None, "", None), inputs=None, outputs=[in_files, out_gallery, out_batch_json, out_batch_zip])

    # Shelf share analysis handlers
    if SHELF_ANALYSIS_AVAILABLE:
        # Step 1: Detect shelves
        detect_shelves_btn.click(
            fn=detect_shelf_rows,
            inputs=[shelf_in_img, shelf_conf, shelf_detection_method],
            outputs=[shelf_detection_info, shelf_selector]
        )

        # Step 2: Analyze selected shelf
        shelf_run_btn.click(
            fn=run_shelf_analysis_with_preprocessing,
            inputs=[shelf_in_img, shelf_conf, shelf_iou, shelf_detection_method, shelf_selector],
            outputs=[shelf_out_img, shelf_metrics]
        )

        # Clear button
        shelf_clear_btn.click(
            fn=lambda: (
                gr.update(value=None),  # Clear image
                "Upload an image and click 'Detect Shelves' to begin",  # Reset info
                gr.Dropdown(choices=[("All Shelves (Combined)", "all")], value="all"),  # Reset dropdown
                None,  # Clear output image
                ""  # Clear metrics
            ),
            inputs=None,
            outputs=[shelf_in_img, shelf_detection_info, shelf_selector, shelf_out_img, shelf_metrics]
        )
    else:
        detect_shelves_btn.click(fn=lambda: ("⚠️ Shelf analysis models not available", gr.Dropdown(choices=[("All Shelves", "all")], value="all")), inputs=None, outputs=[shelf_detection_info, shelf_selector])
        shelf_run_btn.click(fn=lambda: (None, "Shelf analysis models not available"), inputs=None, outputs=[shelf_out_img, shelf_metrics])

    # Category analysis handlers
    if SHELF_ANALYSIS_AVAILABLE:
        category_run_btn.click(fn=run_category_analysis, inputs=[category_in_img, category_conf], outputs=[category_out_img, category_metrics])
        category_clear_btn.click(fn=lambda: (gr.update(value=None), None, ""), inputs=None, outputs=[category_in_img, category_out_img, category_metrics])
    else:
        category_run_btn.click(fn=lambda: (None, "Category analysis requires UBL model"), inputs=None, outputs=[category_out_img, category_metrics])

    # Fixed Shelf analysis handlers
    if FIXED_SHELF_ANALYSIS_AVAILABLE:
        fixed_run_btn.click(
            fn=run_fixed_shelf_analysis,
            inputs=[fixed_in_img, fixed_shelftalker_conf, fixed_ubl_conf, fixed_expand_margin,
                    fixed_show_shelftalkers, fixed_show_roi, fixed_show_products],
            outputs=[fixed_out_img, fixed_metrics]
        )
        fixed_clear_btn.click(fn=lambda: (gr.update(value=None), None, ""), inputs=None, outputs=[fixed_in_img, fixed_out_img, fixed_metrics])
    else:
        fixed_run_btn.click(fn=lambda: (None, "Fixed Shelf Analysis requires Shelftalker model"), inputs=None, outputs=[fixed_out_img, fixed_metrics])

    # Tab change handlers to update UI visibility
    tab_single.select(
        fn=update_ui_visibility_for_regular,
        inputs=None,
        outputs=[model_accordion, controls_accordion, class_filter_accordion, export_accordion, shelf_settings_accordion, category_settings_accordion, fixed_settings_accordion, examples_accordion, shelf_examples_accordion, category_examples_accordion, fixed_examples_accordion]
    )
    tab_batch.select(
        fn=update_ui_visibility_for_regular,
        inputs=None,
        outputs=[model_accordion, controls_accordion, class_filter_accordion, export_accordion, shelf_settings_accordion, category_settings_accordion, fixed_settings_accordion, examples_accordion, shelf_examples_accordion, category_examples_accordion, fixed_examples_accordion]
    )
    tab_shelf.select(
        fn=update_ui_visibility_for_shelf,
        inputs=None,
        outputs=[model_accordion, controls_accordion, class_filter_accordion, export_accordion, shelf_settings_accordion, category_settings_accordion, fixed_settings_accordion, examples_accordion, shelf_examples_accordion, category_examples_accordion, fixed_examples_accordion]
    )
    tab_category.select(
        fn=update_ui_visibility_for_category,
        inputs=None,
        outputs=[model_accordion, controls_accordion, class_filter_accordion, export_accordion, shelf_settings_accordion, category_settings_accordion, fixed_settings_accordion, examples_accordion, shelf_examples_accordion, category_examples_accordion, fixed_examples_accordion]
    )
    tab_fixed.select(
        fn=update_ui_visibility_for_fixed,
        inputs=None,
        outputs=[model_accordion, controls_accordion, class_filter_accordion, export_accordion, shelf_settings_accordion, category_settings_accordion, fixed_settings_accordion, examples_accordion, shelf_examples_accordion, category_examples_accordion, fixed_examples_accordion]
    )

    demo.load(fn=ui_load_classes, inputs=None, outputs=[class_filter])

if __name__ == "__main__":
    demo.launch()
    # demo.launch(share=True)