#!/usr/bin/env python3
"""
Two-stage detection pipeline for calculating UBL share of shelf.

Stage 1: Detect all products using SKU-110K trained model
Stage 2: Detect UBL-specific products using UBL shelf detector
Result: Calculate share of shelf (UBL products / Total products)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


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


def draw_results(image, all_products_boxes, ubl_indices, non_ubl_indices):
    """
    Draw bounding boxes on image with different colors for UBL and Non-UBL products.

    Green: UBL products
    Red: Non-UBL products
    """
    img_draw = image.copy()

    # Draw Non-UBL products in red
    for idx in non_ubl_indices:
        box = all_products_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_draw, 'Non-UBL', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw UBL products in green
    for idx in ubl_indices:
        box = all_products_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, 'UBL', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_draw


def analyze_shelf_share(image_path, sku110k_model_path, ubl_model_path,
                       confidence=0.25, iou_threshold=0.5, output_dir='results'):
    """
    Run two-stage detection pipeline and calculate share of shelf.
    """
    print("="*60)
    print("UBL Share of Shelf Analysis")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"SKU-110K Model: {sku110k_model_path}")
    print(f"UBL Model: {ubl_model_path}")
    print(f"Confidence threshold: {confidence}")
    print(f"IoU threshold: {iou_threshold}")
    print("="*60)
    print()

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Load models
    print("Loading models...")
    sku110k_model = YOLO(sku110k_model_path)
    ubl_model = YOLO(ubl_model_path)
    print("Models loaded successfully!")
    print()

    # Stage 1: Detect all products
    print("Stage 1: Detecting all products...")
    results_all = sku110k_model(image, conf=confidence, verbose=False)[0]
    all_products_boxes = results_all.boxes.xyxy.cpu().numpy()
    print(f"Detected {len(all_products_boxes)} total products")
    print()

    # Stage 2: Detect UBL products
    print("Stage 2: Detecting UBL products...")
    results_ubl = ubl_model(image, conf=confidence, verbose=False)[0]
    ubl_boxes = results_ubl.boxes.xyxy.cpu().numpy()
    print(f"Detected {len(ubl_boxes)} UBL products")
    print()

    # Match detections
    print("Matching detections...")
    ubl_indices, non_ubl_indices = match_detections(
        all_products_boxes, ubl_boxes, iou_threshold
    )

    # Calculate metrics
    total_products = len(all_products_boxes)
    ubl_count = len(ubl_indices)
    non_ubl_count = len(non_ubl_indices)
    share_of_shelf = (ubl_count / total_products * 100) if total_products > 0 else 0

    # Print results
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Products Detected: {total_products}")
    print(f"UBL Products: {ubl_count}")
    print(f"Non-UBL Products: {non_ubl_count}")
    print(f"UBL Share of Shelf: {share_of_shelf:.2f}%")
    print("="*60)
    print()

    # Visualize results
    print("Creating visualization...")
    result_image = draw_results(image, all_products_boxes, ubl_indices, non_ubl_indices)

    # Add text overlay with metrics
    overlay = result_image.copy()
    height, width = result_image.shape[:2]

    # Draw semi-transparent background for text
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_image, f'Total Products: {total_products}',
               (20, 40), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result_image, f'UBL Products: {ubl_count}',
               (20, 70), font, 0.7, (0, 255, 0), 2)
    cv2.putText(result_image, f'Non-UBL Products: {non_ubl_count}',
               (20, 100), font, 0.7, (0, 0, 255), 2)

    # Add share of shelf prominently
    share_text = f'UBL Share: {share_of_shelf:.1f}%'
    text_size = cv2.getTextSize(share_text, font, 1.5, 3)[0]
    text_x = width - text_size[0] - 20
    text_y = 50
    cv2.rectangle(result_image, (text_x-10, text_y-40),
                 (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
    cv2.putText(result_image, share_text, (text_x, text_y),
               font, 1.5, (0, 255, 255), 3)

    # Save result
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image_name = Path(image_path).stem
    output_file = output_path / f"{image_name}_shelf_share.jpg"
    cv2.imwrite(str(output_file), result_image)

    print(f"Result saved to: {output_file}")
    print()

    return {
        'total_products': total_products,
        'ubl_count': ubl_count,
        'non_ubl_count': non_ubl_count,
        'share_of_shelf': share_of_shelf,
        'output_file': str(output_file),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Two-stage detection pipeline for UBL share of shelf analysis'
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--sku110k-model', type=str,
                       default='models/best_sku110k.pt',
                       help='Path to SKU-110K trained model')
    parser.add_argument('--ubl-model', type=str,
                       default='models/best_ubl_shelf.pt',
                       help='Path to UBL shelf detector model')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching detections')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if files exist
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return

    if not Path(args.sku110k_model).exists():
        print(f"Error: SKU-110K model not found: {args.sku110k_model}")
        return

    if not Path(args.ubl_model).exists():
        print(f"Error: UBL model not found: {args.ubl_model}")
        return

    # Run analysis
    analyze_shelf_share(
        args.image,
        args.sku110k_model,
        args.ubl_model,
        args.confidence,
        args.iou_threshold,
        args.output_dir
    )


if __name__ == '__main__':
    main()
