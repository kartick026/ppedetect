#!/usr/bin/env python3
"""
Enhanced PPE Detection FastAPI Backend
High-accuracy detection with YOLOv8m and improved inference
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Any
import shutil
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced PPE Detection API",
    description="High-accuracy PPE detection using YOLOv8m",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/detections", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model
MODEL_PATH = "runs/detect/ppe_detector_v2/weights/best.pt"
FALLBACK_MODEL = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"[INFO] Loaded enhanced model: {MODEL_PATH}")
    else:
        model = YOLO(FALLBACK_MODEL)
        print(f"[INFO] Using fallback model: {FALLBACK_MODEL}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# Class names mapping
CLASS_NAMES = {
    0: 'helmet',
    1: 'vest', 
    2: 'gloves',
    3: 'glasses'
}

def enhance_image_for_detection(image: np.ndarray) -> List[np.ndarray]:
    """Create multiple enhanced versions of the image for better detection"""
    enhanced_images = []
    
    # Original image
    enhanced_images.append(image)
    
    # Brightness and contrast enhancement
    alpha = 1.2  # Contrast
    beta = 30    # Brightness
    enhanced1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_images.append(enhanced1)
    
    # Histogram equalization
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced_images.append(enhanced2)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    enhanced3 = cv2.merge(lab_planes)
    enhanced3 = cv2.cvtColor(enhanced3, cv2.COLOR_LAB2BGR)
    enhanced_images.append(enhanced3)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced4 = cv2.filter2D(image, -1, kernel)
    enhanced_images.append(enhanced4)
    
    return enhanced_images

def smart_duplicate_removal(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """Remove duplicate detections using IoU-based merging"""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    for i, det1 in enumerate(detections):
        is_duplicate = False
        
        for j, det2 in enumerate(filtered_detections):
            if det1['class'] == det2['class']:  # Same class
                # Calculate IoU
                iou = calculate_iou(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_detections.append(det1)
    
    return filtered_detections

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def comprehensive_detection(image: np.ndarray) -> List[Dict]:
    """Perform comprehensive PPE detection with multiple strategies"""
    all_detections = []
    
    # Strategy 1: Standard detection with optimized parameters
    if model:
        results = model.predict(
            image,
            conf=0.25,      # Confidence threshold
            iou=0.45,       # IoU threshold
            imgsz=960,      # Image size
            max_det=300,    # Maximum detections
            half=True,      # Half precision
            device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
            verbose=False
        )
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    
                    all_detections.append({
                        'class': CLASS_NAMES.get(cls, f'class_{cls}'),
                        'confidence': round(conf, 3),
                        'bbox': bbox,
                        'strategy': 'standard'
                    })
    
    # Strategy 2: Enhanced image detection
    enhanced_images = enhance_image_for_detection(image)
    
    for i, enhanced_img in enumerate(enhanced_images[1:], 1):  # Skip original
        if model:
            results = model.predict(
                enhanced_img,
                conf=0.15,      # Lower confidence for enhanced images
                iou=0.3,
                imgsz=960,
                max_det=300,
                half=True,
                verbose=False
            )
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        # Check if already detected
                        already_detected = any(
                            d['class'] == CLASS_NAMES.get(cls, f'class_{cls}') 
                            for d in all_detections
                        )
                        
                        if not already_detected and conf >= 0.1:
                            all_detections.append({
                                'class': CLASS_NAMES.get(cls, f'class_{cls}'),
                                'confidence': round(conf, 3),
                                'bbox': bbox,
                                'strategy': f'enhanced_{i}'
                            })
    
    # Remove duplicates
    final_detections = smart_duplicate_removal(all_detections, iou_threshold=0.3)
    
    return final_detections

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced PPE Detection API",
        "version": "2.0.0",
        "model": "YOLOv8m",
        "classes": list(CLASS_NAMES.values()),
        "endpoints": {
            "/detect": "POST - Upload image for PPE detection",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "YOLOv8m",
        "classes": list(CLASS_NAMES.values()),
        "image_size": 960,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300
    }

@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...)):
    """
    Enhanced PPE detection endpoint
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Read image
        image = cv2.imread(tmp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Perform comprehensive detection
        detections = comprehensive_detection(image)
        
        # Create annotated image
        annotated_image = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_name = det['class']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save annotated image
        result_filename = f"detection_{file.filename}"
        result_path = f"static/detections/{result_filename}"
        cv2.imwrite(result_path, annotated_image)
        
        # Calculate statistics
        ppe_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}
        for det in detections:
            ppe_counts[det['class']] += 1
        
        # Calculate compliance
        num_people = max(ppe_counts.values()) if any(ppe_counts.values()) else 1
        required_ppe = list(CLASS_NAMES.values())
        missing_ppe = [ppe for ppe in required_ppe if ppe_counts[ppe] < num_people]
        compliance_status = 'COMPLIANT' if not missing_ppe else 'NON-COMPLIANT'
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "ppe_counts": ppe_counts,
            "num_people": num_people,
            "compliance_status": compliance_status,
            "missing_ppe": missing_ppe,
            "annotated_image": f"/static/detections/{result_filename}",
            "total_detections": len(detections),
            "model_info": {
                "model": "YOLOv8m",
                "image_size": 960,
                "confidence_threshold": 0.25
            }
        })
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/batch-detect")
async def batch_detect_ppe(files: List[UploadFile] = File(...)):
    """
    Batch PPE detection for multiple images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Process each file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            image = cv2.imread(tmp_path)
            if image is not None:
                detections = comprehensive_detection(image)
                
                ppe_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}
                for det in detections:
                    ppe_counts[det['class']] += 1
                
                results.append({
                    "filename": file.filename,
                    "detections": detections,
                    "ppe_counts": ppe_counts,
                    "total_detections": len(detections)
                })
            
            os.unlink(tmp_path)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "success": True,
        "batch_results": results,
        "total_files": len(files),
        "successful_detections": len([r for r in results if 'error' not in r])
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced PPE Detection API")
    print("="*50)
    print(f"Model: YOLOv8m")
    print(f"Image Size: 960")
    print(f"Classes: {list(CLASS_NAMES.values())}")
    print("="*50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
