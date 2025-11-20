"""
Shelf Detection Module using llmdet
This module provides functionality to detect and crop shelf regions from retail images
before applying product detection models.
"""

from typing import List, Tuple, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ShelfDetector:
    """
    Shelf detector using llmdet (iSEE-Laboratory/llmdet_large) for zero-shot shelf detection.
    """

    def __init__(self, model_name: str = "iSEE-Laboratory/llmdet_large", device: Optional[str] = None):
        """
        Initialize the shelf detector with llmdet model.

        Args:
            model_name: HuggingFace model identifier for llmdet
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading llmdet model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("llmdet model loaded successfully!")

    def detect_shelves(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.3,
        text_queries: List[str] = None
    ) -> List[dict]:
        """
        Detect shelves in an image using zero-shot object detection.

        Args:
            image: PIL Image to detect shelves in
            confidence_threshold: Minimum confidence score for detections
            text_queries: Text descriptions for shelf detection.
                         Defaults to ["shelf", "store shelf", "retail shelf", "product shelf"]

        Returns:
            List of detected shelf regions, each containing:
                - 'box': [x1, y1, x2, y2] coordinates
                - 'score': confidence score
                - 'label': detected label
        """
        if text_queries is None:
            text_queries = [
                "shelf row",
                "product shelf row",
                "horizontal shelf",
                "shelf level",
                "individual shelf"
            ]

        # Prepare inputs
        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # [height, width]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]

        # Convert to list of dicts
        detections = []

        # Handle both old and new transformers API
        # New API (current): labels are strings in results["labels"]
        # Future API (v4.51.0+): labels are indices, text_labels are strings
        use_text_labels = "text_labels" in results
        labels_key = "text_labels" if use_text_labels else "labels"

        for box, score, label in zip(results["boxes"], results["scores"], results[labels_key]):
            # Handle label - could be string (current) or index (future)
            if isinstance(label, str):
                label_text = label
            else:
                # It's a tensor index
                label_idx = label.cpu().item() if hasattr(label, 'cpu') else int(label)
                label_text = text_queries[label_idx]

            detections.append({
                'box': box.cpu().tolist(),  # [x1, y1, x2, y2]
                'score': score.cpu().item(),
                'label': label_text
            })

        # Sort by confidence score (highest first)
        detections.sort(key=lambda x: x['score'], reverse=True)

        return detections

    def crop_shelves(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.3,
        min_shelf_area: float = 0.05,  # Minimum 5% of image area
        text_queries: List[str] = None
    ) -> List[Tuple[Image.Image, dict]]:
        """
        Detect and crop shelf regions from an image.

        Args:
            image: PIL Image to process
            confidence_threshold: Minimum confidence for shelf detection
            min_shelf_area: Minimum shelf area as fraction of total image area
            text_queries: Text descriptions for shelf detection

        Returns:
            List of tuples (cropped_image, metadata) where metadata contains:
                - 'box': [x1, y1, x2, y2] original coordinates
                - 'score': confidence score
                - 'label': detected label
                - 'index': shelf index (0-based)
        """
        detections = self.detect_shelves(image, confidence_threshold, text_queries)

        if not detections:
            print("No shelves detected, returning original image")
            return [(image, {
                'box': [0, 0, image.width, image.height],
                'score': 1.0,
                'label': 'full_image',
                'index': 0
            })]

        # Filter by minimum area
        image_area = image.width * image.height
        min_area = image_area * min_shelf_area

        cropped_shelves = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            shelf_area = (x2 - x1) * (y2 - y1)

            if shelf_area >= min_area:
                # Crop the shelf region
                cropped = image.crop((x1, y1, x2, y2))

                # Add metadata
                metadata = det.copy()
                metadata['index'] = idx

                cropped_shelves.append((cropped, metadata))

        if not cropped_shelves:
            print(f"No shelves met minimum area requirement ({min_shelf_area*100}%), returning original image")
            return [(image, {
                'box': [0, 0, image.width, image.height],
                'score': 1.0,
                'label': 'full_image',
                'index': 0
            })]

        return cropped_shelves

    def split_shelf_into_rows(
        self,
        shelf_box: List[float],
        num_rows: int = 3,
        overlap_ratio: float = 0.1
    ) -> List[List[float]]:
        """
        Split a large shelf bounding box into horizontal rows.

        Args:
            shelf_box: [x1, y1, x2, y2] coordinates of the shelf
            num_rows: Number of rows to split into
            overlap_ratio: Overlap between rows (0.1 = 10% overlap)

        Returns:
            List of [x1, y1, x2, y2] coordinates for each row
        """
        x1, y1, x2, y2 = shelf_box
        height = y2 - y1
        row_height = height / num_rows
        overlap_px = row_height * overlap_ratio

        rows = []
        for i in range(num_rows):
            row_y1 = y1 + (i * row_height) - (overlap_px if i > 0 else 0)
            row_y2 = y1 + ((i + 1) * row_height) + (overlap_px if i < num_rows - 1 else 0)
            rows.append([x1, row_y1, x2, row_y2])

        return rows

    def crop_shelves_with_splitting(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.3,
        min_shelf_area: float = 0.05,
        max_shelf_area: float = 0.7,  # Split shelves larger than 70% of image
        split_into_rows: int = 3,
        text_queries: List[str] = None
    ) -> List[Tuple[Image.Image, dict]]:
        """
        Detect and crop shelf regions, automatically splitting large shelves into rows.

        Args:
            image: PIL Image to process
            confidence_threshold: Minimum confidence for shelf detection
            min_shelf_area: Minimum shelf area as fraction of image area
            max_shelf_area: Maximum shelf area before splitting (fraction of image area)
            split_into_rows: Number of rows to split large shelves into
            text_queries: Text descriptions for shelf detection

        Returns:
            List of (cropped_image, metadata) tuples
        """
        detections = self.detect_shelves(image, confidence_threshold, text_queries)

        if not detections:
            print("No shelves detected, returning original image")
            return [(image, {
                'box': [0, 0, image.width, image.height],
                'score': 1.0,
                'label': 'full_image',
                'index': 0
            })]

        image_area = image.width * image.height
        min_area = image_area * min_shelf_area
        max_area = image_area * max_shelf_area

        cropped_shelves = []
        shelf_idx = 0

        for det in detections:
            x1, y1, x2, y2 = det['box']
            shelf_area = (x2 - x1) * (y2 - y1)

            # Skip shelves that are too small
            if shelf_area < min_area:
                continue

            # If shelf is too large, split it into rows
            if shelf_area > max_area:
                print(f"   Large shelf detected ({shelf_area/image_area*100:.1f}%), splitting into {split_into_rows} rows")
                row_boxes = self.split_shelf_into_rows(det['box'], num_rows=split_into_rows)

                for row_idx, row_box in enumerate(row_boxes):
                    rx1, ry1, rx2, ry2 = row_box
                    # Ensure coordinates are within image bounds
                    rx1, ry1 = max(0, rx1), max(0, ry1)
                    rx2, ry2 = min(image.width, rx2), min(image.height, ry2)

                    cropped = image.crop((rx1, ry1, rx2, ry2))
                    metadata = {
                        'box': [rx1, ry1, rx2, ry2],
                        'score': det['score'],
                        'label': f"{det['label']}_row_{row_idx}",
                        'index': shelf_idx,
                        'is_split': True,
                        'parent_shelf': det['box']
                    }
                    cropped_shelves.append((cropped, metadata))
                    shelf_idx += 1
            else:
                # Normal-sized shelf, crop as-is
                cropped = image.crop((x1, y1, x2, y2))
                metadata = det.copy()
                metadata['index'] = shelf_idx
                metadata['is_split'] = False
                cropped_shelves.append((cropped, metadata))
                shelf_idx += 1

        if not cropped_shelves:
            print(f"No shelves met area requirements, returning original image")
            return [(image, {
                'box': [0, 0, image.width, image.height],
                'score': 1.0,
                'label': 'full_image',
                'index': 0
            })]

        return cropped_shelves

    def visualize_shelves(
        self,
        image: Image.Image,
        detections: List[dict],
        color: str = "blue",
        width: int = 3
    ) -> Image.Image:
        """
        Draw detected shelf bounding boxes on the image.

        Args:
            image: PIL Image
            detections: List of shelf detections from detect_shelves()
            color: Box color
            width: Line width

        Returns:
            Annotated PIL Image
        """
        from PIL import ImageDraw, ImageFont

        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            score = det['score']
            label = det['label']

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            # Draw label with background
            text = f"{label}: {score:.2f}"
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), text, fill="white", font=font)

        return img_copy


# Global instance (lazy loaded)
_shelf_detector_instance = None


def get_shelf_detector(force_reload: bool = False) -> ShelfDetector:
    """
    Get or create the global ShelfDetector instance (singleton pattern).

    Args:
        force_reload: Force reload the model even if already loaded

    Returns:
        ShelfDetector instance
    """
    global _shelf_detector_instance

    if _shelf_detector_instance is None or force_reload:
        _shelf_detector_instance = ShelfDetector()

    return _shelf_detector_instance


def detect_and_crop_shelves(
    image: Image.Image,
    confidence_threshold: float = 0.3,
    min_shelf_area: float = 0.05,
    text_queries: List[str] = None,
    auto_split_large: bool = True,
    max_shelf_area: float = 0.7,
    split_into_rows: int = 3
) -> List[Tuple[Image.Image, dict]]:
    """
    Convenience function to detect and crop shelves using the global detector instance.

    Args:
        image: PIL Image to process
        confidence_threshold: Minimum confidence for shelf detection
        min_shelf_area: Minimum shelf area as fraction of image area
        text_queries: Text descriptions for shelf detection
        auto_split_large: Automatically split large shelves into rows
        max_shelf_area: Maximum shelf area before auto-splitting (fraction of image)
        split_into_rows: Number of rows to split large shelves into

    Returns:
        List of (cropped_image, metadata) tuples
    """
    detector = get_shelf_detector()

    if auto_split_large:
        return detector.crop_shelves_with_splitting(
            image,
            confidence_threshold,
            min_shelf_area,
            max_shelf_area,
            split_into_rows,
            text_queries
        )
    else:
        return detector.crop_shelves(image, confidence_threshold, min_shelf_area, text_queries)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python shelf_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Detecting shelves...")
    detector = ShelfDetector()

    # Detect shelves
    detections = detector.detect_shelves(image, confidence_threshold=0.3)
    print(f"\nFound {len(detections)} shelf regions:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['label']}: {det['score']:.3f} at {det['box']}")

    # Visualize
    viz_image = detector.visualize_shelves(image, detections)
    output_path = image_path.replace('.', '_shelves.')
    viz_image.save(output_path)
    print(f"\nVisualization saved to: {output_path}")

    # Crop shelves
    cropped = detector.crop_shelves(image, confidence_threshold=0.3)
    print(f"\nCropped {len(cropped)} shelf regions:")
    for img, meta in cropped:
        crop_path = image_path.replace('.', f'_shelf_{meta["index"]}.')
        img.save(crop_path)
        print(f"  Shelf {meta['index']}: {img.size} saved to {crop_path}")
