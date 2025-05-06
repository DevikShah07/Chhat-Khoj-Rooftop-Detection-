from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import io
import traceback
import logging
import math

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load models with error handling
try:
    logger.info("Loading Model 1...")
    model1 = YOLO('models/yolov11_phase3.pt')
    logger.info("Model 1 loaded successfully")
except Exception as e:
    logger.error(f"Error loading Model 1: {str(e)}")
    model1 = None

try:
    logger.info("Loading Model 2...")
    model2 = YOLO('models/Phase_3_YOLOv8-seg.pt')
    logger.info("Model 2 loaded successfully")
except Exception as e:
    logger.error(f"Error loading Model 2: {str(e)}")
    model2 = None

def split_image(image, chunk_size=1024):
    """
    Split a large image into fixed-size chunks of 1024x1024 pixels.
    """
    try:
        H, W = image.shape[:2]
        chunks = []
        positions = []
        
        logger.info(f"Splitting image of size {H}x{W} into {chunk_size}x{chunk_size} chunks")
        
        for y in range(0, H, chunk_size):
            for x in range(0, W, chunk_size):
                chunk = image[y:min(y + chunk_size, H), x:min(x + chunk_size, W)]
                if chunk.shape[:2] != (chunk_size, chunk_size):  # Handle edge cases
                    logger.info(f"Padding chunk at position ({x},{y}) from size {chunk.shape[:2]} to {chunk_size}x{chunk_size}")
                    padded_chunk = np.zeros((chunk_size, chunk_size, 3), dtype=np.uint8)
                    padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                    chunk = padded_chunk
                chunks.append(chunk)
                positions.append((x, y))
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks, positions, (H, W)
    except Exception as e:
        logger.error(f"Error in split_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def predict_chunks(chunks, model_instance):
    """
    Run prediction on each chunk using the specified YOLO model.
    """
    try:
        segmented_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Predicting chunk {i+1}/{len(chunks)}")
            results = model_instance(chunk, conf=0.25)
            result = results[0]  # Get first result
            
            # Create an empty mask
            mask = np.zeros(chunk.shape[:2], dtype=np.uint8)
            
            if result.masks is not None:
                for mask_data in result.masks.data:
                    mask_points = (mask_data.cpu().numpy() * 255).astype(np.uint8)
                    mask = np.maximum(mask, mask_points)
            
            segmented_chunks.append(mask)
        
        return segmented_chunks
    except Exception as e:
        logger.error(f"Error in predict_chunks: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def merge_chunks(segmented_chunks, positions, original_size, chunk_size=1024):
    """
    Merge predicted chunk masks into a single mask matching the original image dimensions.
    """
    try:
        H, W = original_size
        logger.info(f"Merging {len(segmented_chunks)} chunks into final mask of size {H}x{W}")
        final_mask = np.zeros((H, W), dtype=np.uint8)
        
        for i, (mask, (x, y)) in enumerate(zip(segmented_chunks, positions)):
            logger.info(f"Processing chunk {i+1}/{len(segmented_chunks)} at position ({x},{y})")
            h, w = min(chunk_size, H - y), min(chunk_size, W - x)
            final_mask[y:y+h, x:x+w] = np.maximum(final_mask[y:y+h, x:x+w], mask[:h, :w])
        
        logger.info("Successfully merged all chunks")
        return final_mask
    except Exception as e:
        logger.error(f"Error in merge_chunks: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_overlay(image, mask, color):
    """
    Create an overlay by blending the original image with a colored mask.
    """
    try:
        logger.info(f"Creating overlay with color {color}")
        
        # Create a copy of the original image
        overlay = image.copy()
        
        # Convert binary mask to colored mask
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        
        if color == "green":
            colored_mask[:, :, 1] = mask  # Green channel
        elif color == "red":
            colored_mask[:, :, 2] = mask  # Red channel
        else:
            colored_mask[mask > 0] = color
        
        # Blend the colored mask with the original image
        alpha = 0.5
        result = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        logger.info("Overlay created successfully")
        return result
    except Exception as e:
        logger.error(f"Error in create_overlay: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def process_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        original_shape = image.shape
        logger.info(f"Image shape: {original_shape}")
        
        # Split image into 1024x1024 chunks
        chunks, positions, original_size = split_image(image, chunk_size=1024)
        logger.info(f"Image split into {len(chunks)} chunks")
        
        # Process chunks with both models
        logger.info("Processing chunks with Model 1...")
        if model1 is None:
            raise ValueError("Model 1 not loaded")
        results1 = predict_chunks(chunks, model1)
        
        logger.info("Processing chunks with Model 2...")
        if model2 is None:
            raise ValueError("Model 2 not loaded")
        results2 = predict_chunks(chunks, model2)
        
        # Merge results
        logger.info("Starting to merge results...")
        combined_mask1 = merge_chunks(results1, positions, original_size)
        logger.info("Successfully merged mask 1")
        combined_mask2 = merge_chunks(results2, positions, original_size)
        logger.info("Successfully merged mask 2")
        
        # Create overlays
        logger.info("Creating overlays...")
        overlay1 = create_overlay(image, combined_mask1, "green")  # Green for model 1
        logger.info("Created overlay 1 (green)")
        overlay2 = create_overlay(image, combined_mask2, "red")  # Red for model 2
        logger.info("Created overlay 2 (red)")
        
        # Save results
        timestamp = os.path.basename(image_path).split('.')[0]
        logger.info("Saving results...")
        
        try:
            # Save masks first
            mask1_path = os.path.join(RESULT_FOLDER, f'{timestamp}_mask1.jpg')
            mask2_path = os.path.join(RESULT_FOLDER, f'{timestamp}_mask2.jpg')
            
            logger.info(f"Saving mask1 to: {mask1_path}")
            cv2.imwrite(mask1_path, combined_mask1)
            
            logger.info(f"Saving mask2 to: {mask2_path}")
            cv2.imwrite(mask2_path, combined_mask2)
            
            # Verify masks were saved
            if not os.path.exists(mask1_path) or not os.path.exists(mask2_path):
                raise FileNotFoundError("Failed to save mask files")
            
            # Save overlays
            overlay1_path = os.path.join(RESULT_FOLDER, f'{timestamp}_overlay1.jpg')
            overlay2_path = os.path.join(RESULT_FOLDER, f'{timestamp}_overlay2.jpg')
            
            logger.info(f"Saving overlay1 to: {overlay1_path}")
            cv2.imwrite(overlay1_path, overlay1)
            
            logger.info(f"Saving overlay2 to: {overlay2_path}")
            cv2.imwrite(overlay2_path, overlay2)
            
            # Verify overlays were saved
            if not os.path.exists(overlay1_path) or not os.path.exists(overlay2_path):
                raise FileNotFoundError("Failed to save overlay files")
            
            logger.info("Successfully saved all result images")
        except Exception as save_error:
            logger.error(f"Error saving results: {str(save_error)}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info("Processing completed successfully")
        return {
            'mask1': f'{timestamp}_mask1.jpg',
            'mask2': f'{timestamp}_mask2.jpg',
            'overlay1': f'{timestamp}_overlay1.jpg',
            'overlay2': f'{timestamp}_overlay2.jpg'
        }
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Received upload request")
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Save uploaded file
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            logger.info(f"Saving file to: {filename}")
            file.save(filename)
            
            # Process image
            logger.info("Starting image processing")
            results = process_image(filename)
            logger.info("Image processing completed")
            
            return jsonify({
                'success': True,
                'results': results
            })
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 