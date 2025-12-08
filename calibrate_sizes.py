#!/usr/bin/env python3
"""
Size Variant Calibration Tool

This script helps you calibrate the absolute height thresholds for size variant detection.

Usage:
1. Prepare labeled test images (know which size each product is)
2. Run detection on them and save JSON outputs
3. Use this script to analyze the heights and generate threshold recommendations
"""

import json
import sys
from collections import defaultdict
import numpy as np


def calculate_height(bbox):
    """Calculate height from bbox_xyxy."""
    return bbox[3] - bbox[1]


def analyze_detection_file(json_file, size_labels):
    """
    Analyze a detection JSON file and extract heights.

    Args:
        json_file: Path to detection JSON
        size_labels: Dict mapping class_name -> expected_size (e.g., {"horlicks_std": "1kg"})

    Returns:
        Dict with heights for each (class, size) combination
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    heights = defaultdict(list)

    for det in data.get("detections", []):
        class_name = det.get("class_name")
        if class_name in size_labels:
            expected_size = size_labels[class_name]
            height = calculate_height(det["bbox_xyxy"])
            heights[(class_name, expected_size)].append(height)

    return heights


def generate_threshold_recommendations(all_heights):
    """
    Generate threshold recommendations based on collected heights.

    Args:
        all_heights: Dict mapping (class_name, size) -> list of heights

    Returns:
        Recommendations as dict
    """
    # Group by class
    by_class = defaultdict(dict)
    for (class_name, size), heights in all_heights.items():
        by_class[class_name][size] = heights

    recommendations = {}

    for class_name, size_dict in by_class.items():
        recommendations[class_name] = []

        # Sort sizes by mean height to assign relative_factor
        sizes_sorted = sorted(size_dict.items(), key=lambda x: np.mean(x[1]))
        base_mean = np.mean(sizes_sorted[0][1]) if sizes_sorted else 1.0

        for size, heights in sizes_sorted:
            heights_arr = np.array(heights)
            mean_height = float(np.mean(heights_arr))
            min_height = float(np.min(heights_arr))
            max_height = float(np.max(heights_arr))
            std_height = float(np.std(heights_arr))

            # Add 15% margin for safety
            suggested_min = max(1, int(min_height * 0.85))
            suggested_max = int(max_height * 1.15)

            # Calculate relative factor
            relative_factor = round(mean_height / base_mean, 2)

            recommendations[class_name].append({
                "size": size,
                "relative_factor": relative_factor,
                "sample_count": len(heights),
                "height_range": f"{min_height:.1f} - {max_height:.1f}",
                "mean_height": f"{mean_height:.1f}",
                "std_height": f"{std_height:.1f}",
                "suggested_min": suggested_min,
                "suggested_max": suggested_max,
                "config": {
                    "size": size,
                    "relative_factor": relative_factor,
                    "min_height": suggested_min,
                    "max_height": suggested_max,
                    "aspect_ratio": None
                }
            })

    return recommendations


def print_recommendations(recommendations):
    """Print recommendations in a readable format."""
    print("=" * 80)
    print("SIZE VARIANT CALIBRATION RECOMMENDATIONS")
    print("=" * 80)
    print()

    for class_name, variants in recommendations.items():
        print(f"\n{class_name}:")
        print("-" * 80)

        for variant in variants:
            print(f"\n  {variant['size']}:")
            print(f"    Sample count: {variant['sample_count']}")
            print(f"    Height range: {variant['height_range']} px")
            print(f"    Mean height: {variant['mean_height']} px")
            print(f"    Std dev: {variant['std_height']} px")
            print(f"    Relative factor: {variant['relative_factor']}x")
            print(f"    → Suggested min_height: {variant['suggested_min']}")
            print(f"    → Suggested max_height: {variant['suggested_max']}")

    print("\n" + "=" * 80)
    print("CONFIGURATION CODE (Copy to size_variant_detector.py)")
    print("=" * 80)
    print()
    print("PRODUCT_SIZE_VARIANTS = {")

    for class_name, variants in recommendations.items():
        print(f'    "{class_name}": [')
        for variant in variants:
            config = variant['config']
            print(f"        {{")
            print(f'            "size": "{config["size"]}",')
            print(f'            "relative_factor": {config["relative_factor"]},')
            print(f'            "min_height": {config["min_height"]},')
            print(f'            "max_height": {config["max_height"]},')
            print(f'            "aspect_ratio": {config["aspect_ratio"]}')
            print(f"        }},")
        print(f"    ],")

    print("}")
    print()


def main():
    """
    Interactive calibration tool.

    You'll be prompted to provide:
    1. Detection JSON file paths
    2. Labels for what size each image contains
    """
    print("=" * 80)
    print("Size Variant Calibration Tool")
    print("=" * 80)
    print()
    print("This tool will analyze your detection results and recommend threshold values.")
    print()

    # Example usage instructions
    print("QUICK START:")
    print("-" * 80)
    print("1. Run detection on images with KNOWN product sizes")
    print("2. For each image, note the actual size (e.g., '500g', '1kg')")
    print("3. Run this script and provide the JSON file paths and size labels")
    print()
    print("Example:")
    print('  Image 1: horlicks_std_1kg.jpg → JSON output → Label as {"horlicks_std": "1kg"}')
    print('  Image 2: horlicks_std_500g.jpg → JSON output → Label as {"horlicks_std": "500g"}')
    print()

    # Manual input mode
    print("=" * 80)
    print("Enter detection data:")
    print("-" * 80)

    all_heights = defaultdict(list)

    while True:
        print("\nAdd a calibration sample (or press Enter to finish):")
        json_path = input("  JSON file path: ").strip()

        if not json_path:
            break

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue

        # Show detected classes
        classes = set(det.get("class_name") for det in data.get("detections", []))
        print(f"  Detected classes: {', '.join(classes)}")

        # Get labels
        print("  Enter size labels for each class (format: class_name:size)")
        print("  Example: horlicks_std:1kg")

        labels = {}
        for class_name in classes:
            size = input(f"    {class_name} size: ").strip()
            if size:
                labels[class_name] = size

        # Analyze this file
        heights = analyze_detection_file(json_path, labels)

        # Add to collection
        for key, height_list in heights.items():
            all_heights[key].extend(height_list)

        print(f"  ✓ Added {sum(len(h) for h in heights.values())} samples")

    if not all_heights:
        print("\nNo calibration data provided. Exiting.")
        return

    # Generate recommendations
    recommendations = generate_threshold_recommendations(all_heights)

    # Print results
    print_recommendations(recommendations)

    # Save to file
    output_file = "calibration_recommendations.txt"
    with open(output_file, 'w') as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_recommendations(recommendations)
        sys.stdout = old_stdout

    print(f"Recommendations saved to: {output_file}")


if __name__ == "__main__":
    main()
