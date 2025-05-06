# Rooftop Detection Web Application

This web application allows users to upload images and detect rooftops using two different YOLO models. The application processes large images by splitting them into chunks and then combines the results to create a complete detection.

## Features

- Upload images in JPG or PNG format
- Support for large images
- Automatic image chunking (1024x1024)
- Processing with two different YOLO models
- Side-by-side comparison of results
- Mask and overlay visualization
- Modern, responsive UI with drag-and-drop support

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure your YOLO models are in the `models` folder:
- `models/yolov11_phase3.pt`
- `models/Phase_3_YOLOv8-seg.pt`

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the "Choose File" button or drag and drop an image onto the upload area
2. Wait for the processing to complete
3. View the results:
   - Left side: Model 1 results (mask and overlay)
   - Right side: Model 2 results (mask and overlay)

## Technical Details

- The application splits large images into 1024x1024 chunks
- Each chunk is processed by both YOLO models
- Results are combined to create complete masks
- Overlays are created by applying the masks to the original image
- Model 1 results are shown in green
- Model 2 results are shown in red

