# UBL-YOLO Product Detection

AI-powered product recognition and shelf share analysis for Unilever products using YOLO11X deep learning models.

## 🌟 Features

### 1. **Product Detection**
- Single image and batch processing
- Multiple YOLO models support (DA, POSM, SACHET)
- Export results in JSON or YOLO TXT format
- Configurable confidence and IoU thresholds
- Class filtering and visualization options

### 2. **Shelf Share Analysis**
- Automatic product detection using SKU-110K model
- Product clustering into shelf rows
- Calculate Unilever share of shelf percentage
- Visual differentiation (Green = Unilever, Red = Non-Unilever)

### 3. **Category Analysis**
- Product categorization into:
  - Haircare
  - Oralcare
  - Skincare
  - Home & Hygiene
  - Food & Nutrition
- Category distribution visualization with pie charts

### 4. **Fixed Shelf Analysis (Planogram Adherence)**
- Detect shelftalker stickers (top, bottom, sides)
- Create planogram ROI from shelftalker boundaries
- Detect UBL products within planogram area
- Calculate adherence metrics
- Toggle visibility of different bounding boxes:
  - Shelftalker boxes (blue/light blue)
  - ROI box (orange)
  - Product boxes (green)


## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yok4ai/UBL-Detector.git
cd UBL-Detector
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

```bash
# Using conda
conda create -n ubl-detector python=3.12
conda activate ubl-detector
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<!-- ### 4. Download Model Weights

Place the following YOLO model files in the `models/` directory:

- `DA_YOLO11X.pt` - Main Unilever product detection model (~327MB)
- `SKU110k_YOLO11X.pt` - General product detection model (~110MB)
- `POSM_YOLO11X.pt` - POSM detection model (~110MB)
- `SACHET_YOLO11X.pt` - Sachet detection model (~110MB)
- `Shelftalker.pt` - Shelftalker sticker detection model (~327MB) -->

**Directory structure should look like:**
```
UBL-Detector/
├── models/
│   ├── DA_YOLO11X.pt
│   ├── SKU110k_YOLO11X.pt
│   ├── POSM_YOLO11X.pt
│   ├── SACHET_YOLO11X.pt
│   └── Shelftalker.pt
├── app.py
├── requirements.txt
└── ...
```

## 🎯 Running the Application

### Start Gradio Web Interface

```bash
gradio app.py
```

The application will start and display:
```
Running on local URL:  http://127.0.0.1:7860
```

Open your browser and navigate to the URL to use the application.

### Optional: Public URL with Gradio

To create a public shareable link:

```python
# Modify the last line in app.py:
demo.launch(share=True)
```

## 📖 Usage Guide

### Single Image Detection

1. Navigate to the **"Single Image"** tab
2. Upload an image
3. Configure detection settings:
   - **Confidence Threshold**: Minimum confidence score (0.01-0.99)
   - **IoU Threshold**: Intersection over Union threshold
   - **Image Size**: Processing resolution
4. Click **"Run Detection"**
5. Download results as ZIP (image + JSON/TXT)

### Batch Processing

1. Navigate to the **"Batch Processing"** tab
2. Upload multiple images
3. Configure settings
4. Click **"Run Batch Detection"**
5. View results in gallery and download ZIP

### Shelf Share Analysis

1. Navigate to the **"Shelf Share Analysis"** tab
2. Upload a shelf image
3. **Step 1**: Click **"Detect Shelves"** to identify shelf rows
4. **Step 2**: Select which shelf to analyze (or "All Shelves")
5. Click **"Analyze Shelf Share"**
6. View metrics: total products, Unilever products, share percentage

### Category Analysis

1. Navigate to the **"Category Analysis"** tab
2. Upload an image
3. Click **"Analyze Categories"**
4. View product distribution by category

### Fixed Shelf Analysis

1. Navigate to the **"Fixed Shelf Analysis"** tab
2. Upload an image with visible shelftalker stickers
3. Configure settings:
   - **Shelftalker Confidence**: Detection threshold for stickers
   - **UBL Product Confidence**: Detection threshold for products
   - **ROI Tightness**: Control planogram boundary (lower = tighter)
4. Toggle visualization options:
   - ☑️ Show Shelftalker Boxes
   - ☑️ Show ROI Box
   - ☑️ Show Product Boxes
5. Click **"Analyze Fixed Shelves"**
6. View adherence metrics and visualization


## 📁 Project Structure

```
UBL-Detector/
├── app.py                      # Main Gradio application
├── annotate.py                 # Command-line annotation script
├── category_analysis.py        # Category classification logic
├── product_clustering.py       # Product clustering for shelf detection
├── shelf_detector.py           # Shelf detection utilities
├── shelf_analysis.py           # Shelf analysis utilities
├── test_inference.py           # Testing script
├── requirements.txt            # Python dependencies
├── data.yaml                   # YOLO data configuration
├── models/                     # YOLO model weights
│   ├── DA_YOLO11X.pt
│   ├── SKU110k_YOLO11X.pt
│   ├── POSM_YOLO11X.pt
│   ├── SACHET_YOLO11X.pt
│   └── Shelftalker.pt
├── examples/                   # Example images
│   ├── DA/
│   ├── POSM/
│   ├── SACHET/
│   └── ShareOfShelf/
└── public/                     # Static assets (logo, etc.)
```

## 🔧 Troubleshooting

### GPU Not Detected

If PyTorch doesn't detect your GPU:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce image size in settings
2. Lower batch size for batch processing
3. Use CPU instead of GPU (slower but more stable)

### Model Not Found

Ensure all model files are placed in the `models/` directory with correct filenames.

## 📊 Model Information

### Models Used

| Model | Purpose | Size | Classes |
|-------|---------|------|---------|
| DA_YOLO11X.pt | Unilever product detection | ~327MB | 100+ UBL products |
| SKU110k_YOLO11X.pt | General product detection | ~110MB | Generic products |
| POSM_YOLO11X.pt | POSM material detection | ~110MB | POSM items |
| SACHET_YOLO11X.pt | Sachet detection | ~110MB | Sachets |
| Shelftalker.pt | Shelftalker sticker detection | ~327MB | Stickers |

## 🎨 Output Formats

### JSON Format
```json
{
  "detections": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "product_name"
    }
  ]
}
```

### YOLO TXT Format
```
class_id x_center y_center width height [confidence]
```

## 📝 License

This project is proprietary software developed for Unilever product detection and analysis.

## 🤝 Contributing

This is an internal project. For questions or support, contact the development team.

## 🔗 Links

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- Gradio Documentation: https://gradio.app/docs/

---

**Built with ❤️ using Gradio and YOLO11X**
