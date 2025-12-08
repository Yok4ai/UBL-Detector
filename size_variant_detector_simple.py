"""
Simple Size Variant Detection - Ratio-Based Approach

Uses relative height ratios within the same image to determine size variants.
No complex thresholds or aspect ratios - just simple clustering.
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict


# Simple product size mapping - just names and relative sizes
PRODUCT_SIZE_VARIANTS = {
    "horlicks_std": ["500g", "1kg"],  # Smaller to larger
    "horlicks_mother": ["350g"],
    "horlicks_women": ["400g"],
    "horlicks_choco": ["500g"],
    "horlicks_junior": ["500g"],
}


def calculate_bbox_height(bbox_xyxy: List[float]) -> float:
    """Calculate the height of a bounding box."""
    return bbox_xyxy[3] - bbox_xyxy[1]


def assign_size_variants_simple(detections: List[Dict]) -> List[Dict]:
    """
    Assign size variants using simple ratio-based clustering.

    Logic:
    1. Group detections by class
    2. Sort by height within each class
    3. If 2+ variants exist, split into groups (smaller = smaller size)
    4. If 1 variant exists, assign that variant to all

    Args:
        detections: List of detections with bbox_xyxy and class_name

    Returns:
        Enriched detections with "size_variant" field
    """
    # Group by class
    by_class = defaultdict(list)
    for idx, det in enumerate(detections):
        by_class[det["class_name"]].append((idx, det))

    # Create output
    enriched = [det.copy() for det in detections]

    for class_name, items in by_class.items():
        # Get size variants for this class
        variants = PRODUCT_SIZE_VARIANTS.get(class_name, None)

        if variants is None or len(variants) == 0:
            # No variants configured - skip size detection
            for idx, _ in items:
                enriched[idx]["size_variant"] = "N/A"
            continue

        if len(variants) == 1:
            # Only one size exists - assign to all
            for idx, _ in items:
                enriched[idx]["size_variant"] = variants[0]
            continue

        # Multiple variants - cluster by height
        heights = [(idx, calculate_bbox_height(det["bbox_xyxy"])) for idx, det in items]
        heights_sorted = sorted(heights, key=lambda x: x[1])

        if len(heights) == 1:
            # Single detection - assume it's the larger variant
            idx = heights[0][0]
            enriched[idx]["size_variant"] = variants[-1]  # Largest size
            continue

        # Multiple detections - split into groups
        # Use k-means-like clustering based on height
        height_values = [h for _, h in heights_sorted]

        # Simple threshold: if gap > 15% of mean, it's a different cluster
        mean_height = np.mean(height_values)
        threshold = mean_height * 0.15

        clusters = []
        current_cluster = [heights_sorted[0]]

        for i in range(1, len(heights_sorted)):
            prev_height = heights_sorted[i-1][1]
            curr_height = heights_sorted[i][1]

            if curr_height - prev_height > threshold:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [heights_sorted[i]]
            else:
                current_cluster.append(heights_sorted[i])

        clusters.append(current_cluster)

        # Assign sizes: smallest cluster = smallest variant, etc.
        num_clusters = len(clusters)

        if num_clusters >= len(variants):
            # More clusters than variants - merge smallest clusters
            # Use only the largest N clusters where N = num variants
            clusters = clusters[-len(variants):]

        # Map clusters to variants
        for cluster_idx, cluster in enumerate(clusters):
            variant_idx = min(cluster_idx, len(variants) - 1)
            size = variants[variant_idx]
            for idx, _ in cluster:
                enriched[idx]["size_variant"] = size

    return enriched


def get_size_summary(detections: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Create a clean summary of size variant counts.

    Returns:
        Dict: {class_name: {size: count}}
    """
    summary = defaultdict(lambda: defaultdict(int))

    for det in detections:
        class_name = det.get("class_name", "unknown")
        size = det.get("size_variant", "N/A")

        # Skip products without size variants
        if size == "N/A":
            continue

        summary[class_name][size] += 1

    return {k: dict(v) for k, v in summary.items()}


def format_summary_text(summary: Dict[str, Dict[str, int]]) -> List[str]:
    """
    Format summary as readable text lines.

    Returns list of strings like: ["horlicks_std: 5x 1kg, 2x 500g"]
    """
    lines = []
    for class_name, sizes in sorted(summary.items()):
        # Use 'x' instead of '×' to avoid unicode issues
        size_parts = [f"{count}x {size}" for size, count in sorted(sizes.items(), reverse=True)]
        lines.append(f"{class_name}: {', '.join(size_parts)}")

    return lines
