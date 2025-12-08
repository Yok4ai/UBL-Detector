"""
Size Variant Detection Module

This module detects product size variants based on bounding box dimensions.
It uses a combination of relative height ratios and clustering to determine
which size variant each detection represents.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# Product size variant mapping
# Format: class_name -> list of size variant definitions
# Each variant has:
#   - size: Display name (e.g., "500g", "1kg")
#   - relative_factor: Height ratio relative to base size (1.0 = base)
#   - min_height: Minimum pixel height (for absolute thresholding)
#   - max_height: Maximum pixel height (for absolute thresholding)
#   - aspect_ratio: Expected width/height ratio (optional)
#
# CALIBRATION GUIDE:
# To calibrate thresholds, collect sample images and measure bbox heights:
# 1. Run detection on 10-20 images with KNOWN product sizes
# 2. Record min/max heights for each size variant
# 3. Add 10% margin to min/max values for robustness
# 4. Update the thresholds below

PRODUCT_SIZE_VARIANTS = {
    "horlicks_std": [
        {
            "size": "500g",
            "relative_factor": 1.0,
            "min_height": 100,  # NEEDS CALIBRATION: Update based on your images!
            "max_height": 150,  # No overlap with 1kg
            "aspect_ratio": 0.58,  # Width/Height ratio for 500g (typically wider/shorter)
            "aspect_ratio_tolerance": 0.08  # Allow ±0.08 variation
        },
        {
            "size": "1kg",
            "relative_factor": 1.45,  # 1kg is ~45% taller than 500g
            "min_height": 151,  # NEEDS CALIBRATION: Update based on your images!
            "max_height": 300,  # No overlap with 500g
            "aspect_ratio": 0.50,  # Width/Height ratio for 1kg (typically thinner/taller)
            "aspect_ratio_tolerance": 0.06  # Allow ±0.06 variation
        }
    ],
    "horlicks_mother": [
        {
            "size": "350g",
            "relative_factor": 1.0,
            "min_height": 120,
            "max_height": 170,
            "aspect_ratio": None
        }
    ],
    "horlicks_women": [
        {
            "size": "400g",
            "relative_factor": 1.0,
            "min_height": 120,
            "max_height": 170,
            "aspect_ratio": None
        }
    ],
    "horlicks_choco": [
        {
            "size": "500g",
            "relative_factor": 1.0,
            "min_height": 120,
            "max_height": 170,
            "aspect_ratio": None
        }
    ],
    "horlicks_junior": [
        {
            "size": "500g",
            "relative_factor": 1.0,
            "min_height": 120,
            "max_height": 170,
            "aspect_ratio": None
        }
    ],
    # Add more products and their variants here
    # Example for a product with 3 variants:
    # "product_name": [
    #     {"size": "250g", "relative_factor": 0.7, "min_height": 80, "max_height": 120},
    #     {"size": "500g", "relative_factor": 1.0, "min_height": 110, "max_height": 150},
    #     {"size": "1kg", "relative_factor": 1.45, "min_height": 160, "max_height": 220}
    # ]
}


def calculate_bbox_height(bbox_xyxy: List[float]) -> float:
    """Calculate the height of a bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    return y2 - y1


def calculate_bbox_width(bbox_xyxy: List[float]) -> float:
    """Calculate the width of a bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    return x2 - x1


def calculate_aspect_ratio(bbox_xyxy: List[float]) -> float:
    """Calculate the aspect ratio (width/height) of a bounding box."""
    width = calculate_bbox_width(bbox_xyxy)
    height = calculate_bbox_height(bbox_xyxy)
    return width / height if height > 0 else 0


def calculate_bbox_area(bbox_xyxy: List[float]) -> float:
    """Calculate the area of a bounding box."""
    return calculate_bbox_height(bbox_xyxy) * calculate_bbox_width(bbox_xyxy)


def cluster_heights_by_ratio(heights: List[float], threshold: float = 0.15) -> List[List[int]]:
    """
    Cluster heights into groups based on relative ratios.

    Args:
        heights: List of bounding box heights
        threshold: Relative difference threshold (default 15%)

    Returns:
        List of clusters, where each cluster is a list of indices
    """
    if not heights:
        return []

    # Sort indices by height
    sorted_indices = sorted(range(len(heights)), key=lambda i: heights[i])

    clusters = []
    current_cluster = [sorted_indices[0]]
    current_mean = heights[sorted_indices[0]]

    for idx in sorted_indices[1:]:
        height = heights[idx]
        # Check if this height is within threshold of current cluster mean
        relative_diff = abs(height - current_mean) / current_mean

        if relative_diff <= threshold:
            current_cluster.append(idx)
            # Update mean
            current_mean = np.mean([heights[i] for i in current_cluster])
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [idx]
            current_mean = height

    # Add last cluster
    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def assign_size_by_absolute_threshold(
    bbox: List[float],
    size_variants: List[Dict],
    use_aspect_ratio: bool = True
) -> Tuple[str, float]:
    """
    Assign size variant based on absolute height and aspect ratio thresholds.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        size_variants: List of size variant definitions with min/max heights
        use_aspect_ratio: Whether to use aspect ratio for discrimination

    Returns:
        Tuple of (size_name, confidence)
        confidence: 1.0 if within range, < 1.0 if outside but closest match
    """
    height = calculate_bbox_height(bbox)
    aspect_ratio = calculate_aspect_ratio(bbox)

    # Try aspect ratio first if available (more reliable than pixel height)
    if use_aspect_ratio:
        # Check if any variants have aspect ratio defined
        variants_with_ar = [v for v in size_variants if v.get("aspect_ratio") is not None]

        if variants_with_ar:
            # Find best match by aspect ratio
            best_variant = None
            best_ar_diff = float('inf')

            for variant in variants_with_ar:
                expected_ar = variant.get("aspect_ratio", 0)
                ar_tolerance = variant.get("aspect_ratio_tolerance", 0.1)

                ar_diff = abs(aspect_ratio - expected_ar)

                if ar_diff < best_ar_diff:
                    best_ar_diff = ar_diff
                    best_variant = variant

            # If aspect ratio match is good, use it
            if best_variant and best_ar_diff < best_variant.get("aspect_ratio_tolerance", 0.1):
                confidence = max(0.7, 1.0 - (best_ar_diff * 5))
                return best_variant["size"], confidence

    # Fall back to height-based detection
    # Find exact matches within threshold range
    for variant in size_variants:
        min_h = variant.get("min_height", 0)
        max_h = variant.get("max_height", float('inf'))

        if min_h <= height <= max_h:
            return variant["size"], 1.0

    # No exact match, find closest variant by relative factor
    # This helps when actual heights don't match calibrated ranges
    mean_height = np.mean([v.get("min_height", 0) + v.get("max_height", 1000) for v in size_variants]) / 2

    best_variant = size_variants[0]
    best_score = float('inf')

    for variant in size_variants:
        expected_range_mid = (variant.get("min_height", 0) + variant.get("max_height", 1000)) / 2
        relative_diff = abs(height / expected_range_mid - 1.0) if expected_range_mid > 0 else float('inf')

        if relative_diff < best_score:
            best_score = relative_diff
            best_variant = variant

    # Calculate confidence based on relative difference
    confidence = max(0.3, 1.0 - best_score)

    return best_variant["size"], confidence


def assign_size_variants_by_clustering(
    detections_of_class: List[Dict],
    size_variants: List[Dict],
    clustering_threshold: float = 0.15
) -> Tuple[List[str], List[float]]:
    """
    Assign size variants to detections based on height clustering.

    Args:
        detections_of_class: List of detections for a single product class
        size_variants: List of size variant definitions from PRODUCT_SIZE_VARIANTS
        clustering_threshold: Threshold for grouping similar heights (default 15%)

    Returns:
        Tuple of (size_assignments, confidences)
        - size_assignments: List of size variant names
        - confidences: List of confidence scores (0.0 to 1.0)
    """
    if not detections_of_class:
        return [], []

    # Extract heights
    heights = [calculate_bbox_height(det["bbox_xyxy"]) for det in detections_of_class]

    # If only one size variant is defined, return that for all with high confidence
    if len(size_variants) == 1:
        return [size_variants[0]["size"]] * len(detections_of_class), [1.0] * len(detections_of_class)

    # Cluster heights
    clusters = cluster_heights_by_ratio(heights, threshold=clustering_threshold)

    # If we found the same number of clusters as variants, match them
    if len(clusters) == len(size_variants):
        # Sort variants by relative_factor
        sorted_variants = sorted(size_variants, key=lambda v: v["relative_factor"])

        # Sort clusters by mean height
        clusters_with_mean = [(cluster, np.mean([heights[i] for i in cluster]))
                               for cluster in clusters]
        sorted_clusters = sorted(clusters_with_mean, key=lambda x: x[1])

        # Create mapping from cluster to size
        size_assignments = [""] * len(detections_of_class)
        confidences = [0.0] * len(detections_of_class)

        for (cluster, _), variant in zip(sorted_clusters, sorted_variants):
            for idx in cluster:
                size_assignments[idx] = variant["size"]
                confidences[idx] = 0.9  # High confidence when clustering matches

        return size_assignments, confidences

    # If clusters don't match variants, use ratio-based assignment
    # Calculate mean height and use it as reference
    mean_height = np.mean(heights)

    size_assignments = []
    confidences = []

    for height in heights:
        # Find closest variant based on height ratio
        height_ratio = height / mean_height

        best_variant = size_variants[0]
        best_diff = float('inf')

        for variant in size_variants:
            # Compare actual ratio to expected ratio
            expected_ratio = variant["relative_factor"]
            diff = abs(height_ratio - expected_ratio)

            if diff < best_diff:
                best_diff = diff
                best_variant = variant

        size_assignments.append(best_variant["size"])

        # Confidence based on how well the ratio matches
        # Perfect match = 1.0, 20% deviation = 0.5
        confidence = max(0.5, 1.0 - (best_diff * 2.0))
        confidences.append(confidence)

    return size_assignments, confidences


def assign_size_variants_hybrid(
    detections_of_class: List[Dict],
    size_variants: List[Dict],
    use_absolute_threshold: bool = True,
    clustering_threshold: float = 0.15
) -> Tuple[List[str], List[float], List[str]]:
    """
    Hybrid approach: Use both clustering and absolute thresholds.

    Strategy:
    1. Try absolute thresholds first (if available and enabled)
    2. If all detections match with high confidence, use absolute method
    3. Otherwise, use clustering for relative comparison
    4. Return method used for transparency

    Args:
        detections_of_class: List of detections for a single product class
        size_variants: List of size variant definitions
        use_absolute_threshold: Whether to use absolute height thresholds
        clustering_threshold: Threshold for clustering (default 15%)

    Returns:
        Tuple of (size_assignments, confidences, methods)
        - size_assignments: List of size variant names
        - confidences: List of confidence scores (0.0 to 1.0)
        - methods: List of method used ("absolute", "clustering", "ratio")
    """
    if not detections_of_class:
        return [], [], []

    heights = [calculate_bbox_height(det["bbox_xyxy"]) for det in detections_of_class]

    # Check if absolute thresholds are available for all variants
    has_absolute_thresholds = all(
        "min_height" in v and "max_height" in v
        for v in size_variants
    )

    # Try absolute threshold method first
    if use_absolute_threshold and has_absolute_thresholds and len(size_variants) > 1:
        abs_assignments = []
        abs_confidences = []

        for det in detections_of_class:
            bbox = det["bbox_xyxy"]
            size, conf = assign_size_by_absolute_threshold(bbox, size_variants)
            abs_assignments.append(size)
            abs_confidences.append(conf)

        # If average confidence is high (> 0.85), use absolute method
        avg_confidence = np.mean(abs_confidences)
        if avg_confidence > 0.85:
            methods = ["absolute"] * len(detections_of_class)
            return abs_assignments, abs_confidences, methods

    # Fall back to clustering method
    cluster_assignments, cluster_confidences = assign_size_variants_by_clustering(
        detections_of_class, size_variants, clustering_threshold
    )

    # Determine method used
    if len(set(heights)) == 1:
        methods = ["single_height"] * len(detections_of_class)
    elif len(cluster_heights_by_ratio(heights, clustering_threshold)) == len(size_variants):
        methods = ["clustering"] * len(detections_of_class)
    else:
        methods = ["ratio"] * len(detections_of_class)

    return cluster_assignments, cluster_confidences, methods


def assign_size_variants_single_image(
    detections: List[Dict],
    product_size_mapping: Dict[str, List[Dict]] = PRODUCT_SIZE_VARIANTS,
    use_absolute_threshold: bool = True,
    clustering_threshold: float = 0.15,
    include_confidence: bool = False
) -> List[Dict]:
    """
    Assign size variants to all detections in an image using hybrid approach.

    Args:
        detections: List of detection dictionaries with bbox_xyxy and class_name
        product_size_mapping: Mapping of product classes to size variants
        use_absolute_threshold: Whether to use absolute height thresholds
        clustering_threshold: Threshold for height clustering (default 15%)
        include_confidence: Whether to include confidence and method fields in output

    Returns:
        List of detections with added fields:
        - "size_variant": Size name (e.g., "500g", "1kg")
        - "size_confidence": Confidence score (only if include_confidence=True)
        - "size_method": Method used (only if include_confidence=True)
    """
    # Group detections by class
    detections_by_class = defaultdict(list)
    detection_indices_by_class = defaultdict(list)

    for idx, det in enumerate(detections):
        class_name = det.get("class_name", "")
        detections_by_class[class_name].append(det)
        detection_indices_by_class[class_name].append(idx)

    # Create output list with size variants
    enriched_detections = [det.copy() for det in detections]

    # Process each class separately
    for class_name, class_detections in detections_by_class.items():
        # Get size variants for this class
        size_variants = product_size_mapping.get(class_name, [])

        if not size_variants:
            # No size variants defined, skip
            for idx in detection_indices_by_class[class_name]:
                enriched_detections[idx]["size_variant"] = "unknown"
                if include_confidence:
                    enriched_detections[idx]["size_confidence"] = 0.0
                    enriched_detections[idx]["size_method"] = "none"
            continue

        # Assign sizes using hybrid approach
        size_assignments, confidences, methods = assign_size_variants_hybrid(
            class_detections,
            size_variants,
            use_absolute_threshold,
            clustering_threshold
        )

        # Apply to enriched detections
        for det_idx, size, conf, method in zip(
            detection_indices_by_class[class_name],
            size_assignments,
            confidences,
            methods
        ):
            enriched_detections[det_idx]["size_variant"] = size
            if include_confidence:
                enriched_detections[det_idx]["size_confidence"] = round(conf, 2)
                enriched_detections[det_idx]["size_method"] = method

    return enriched_detections


def get_size_variant_summary(detections: List[Dict]) -> Dict:
    """
    Generate a clean, readable summary of size variants per product class.

    Args:
        detections: List of detections with size_variant field

    Returns:
        Dictionary mapping class_name -> size_variant -> summary object
        Each summary object contains: count, confidence, method
    """
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "confidences": [], "methods": set()}))

    for det in detections:
        class_name = det.get("class_name", "unknown")
        size_variant = det.get("size_variant", "unknown")
        confidence = det.get("size_confidence", 0.0)
        method = det.get("size_method", "unknown")

        stats[class_name][size_variant]["count"] += 1
        stats[class_name][size_variant]["confidences"].append(confidence)
        stats[class_name][size_variant]["methods"].add(method)

    # Build clean summary
    summary = {}
    for class_name, size_dict in stats.items():
        summary[class_name] = {}
        for size, data in size_dict.items():
            avg_conf = round(np.mean(data["confidences"]), 2)
            # Only show method if it's not "clustering" or if confidence is low
            method_info = list(data["methods"])[0] if len(data["methods"]) == 1 else "mixed"

            # Simple format: just count if confidence is perfect
            if avg_conf >= 0.99 and method_info == "absolute":
                summary[class_name][size] = data["count"]
            else:
                # Show details when there's uncertainty
                summary[class_name][size] = {
                    "count": data["count"],
                    "confidence": avg_conf,
                    "method": method_info
                }

    return summary


def generate_calibration_report(detections: List[Dict]) -> Dict:
    """
    Generate a calibration report to help set absolute thresholds.

    Use this function to analyze your detection results and determine
    appropriate min/max height values for each size variant.

    Args:
        detections: List of detections with bbox_xyxy and size_variant

    Returns:
        Dictionary with height statistics per product class and size variant
    """
    stats = defaultdict(lambda: defaultdict(list))

    for det in detections:
        class_name = det.get("class_name", "unknown")
        size_variant = det.get("size_variant", "unknown")
        height = calculate_bbox_height(det["bbox_xyxy"])
        stats[class_name][size_variant].append(height)

    # Calculate statistics
    report = {}
    for class_name, size_dict in stats.items():
        report[class_name] = {}
        for size, heights in size_dict.items():
            heights_array = np.array(heights)
            report[class_name][size] = {
                "count": len(heights),
                "min_height": round(float(np.min(heights_array)), 2),
                "max_height": round(float(np.max(heights_array)), 2),
                "mean_height": round(float(np.mean(heights_array)), 2),
                "std_height": round(float(np.std(heights_array)), 2),
                "suggested_min": round(float(np.min(heights_array) * 0.9), 2),  # 10% margin
                "suggested_max": round(float(np.max(heights_array) * 1.1), 2),  # 10% margin
            }

    return report


def add_size_variants_to_detection_json(detection_json: Dict) -> Dict:
    """
    Add size variant information to a detection JSON output.

    Args:
        detection_json: Dictionary with "detections" key containing list of detections

    Returns:
        Enhanced detection JSON with size variants and summary
    """
    detections = detection_json.get("detections", [])

    if not detections:
        return detection_json

    # Assign size variants
    enriched_detections = assign_size_variants_single_image(detections)

    # Generate summary
    size_summary = get_size_variant_summary(enriched_detections)

    # Create enhanced output
    enhanced_json = {
        "detections": enriched_detections,
        "size_variant_summary": size_summary,
        "total_detections": len(enriched_detections)
    }

    return enhanced_json


# Configuration for absolute thresholds (optional, for future enhancement)
# This can be used when you have calibration data for specific camera distances
ABSOLUTE_SIZE_THRESHOLDS = {
    # Example:
    # "horlicks_std": {
    #     "500g": {"min_height": 100, "max_height": 150},
    #     "1kg": {"min_height": 140, "max_height": 200}
    # }
}


def estimate_size_from_absolute_thresholds(
    class_name: str,
    bbox_height: float,
    thresholds: Dict = ABSOLUTE_SIZE_THRESHOLDS
) -> Optional[str]:
    """
    Estimate size variant using absolute pixel height thresholds.

    This is useful when you have only one variant in an image and have
    calibration data for expected pixel heights.

    Args:
        class_name: Product class name
        bbox_height: Bounding box height in pixels
        thresholds: Absolute threshold configuration

    Returns:
        Size variant name or None if no match
    """
    if class_name not in thresholds:
        return None

    class_thresholds = thresholds[class_name]

    for size, bounds in class_thresholds.items():
        if bounds["min_height"] <= bbox_height <= bounds["max_height"]:
            return size

    return None
