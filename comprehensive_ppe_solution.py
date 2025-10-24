#!/usr/bin/env python3
"""
Comprehensive PPE Detection Solution
Works with existing model and provides multiple detection strategies
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the existing trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

print("="*70)
print("COMPREHENSIVE PPE DETECTION SOLUTION")
print("="*70)
print(f"[INFO] Model loaded: {model_path}")
print(f"[INFO] Classes: {model.names}")
print("[INFO] Advanced detection with multiple strategies")
print("[INFO] Smart filtering and validation")
print("[INFO] Starting web server...")
print("[INFO] Open: http://localhost:5000")
print("="*70)

def enhance_image_for_detection(image):
    """Create multiple enhanced versions of the image for better detection"""
    enhanced_versions = []
    
    # Original image
    enhanced_versions.append(("original", image))
    
    # Strategy 1: Brightness and contrast enhancement
    alpha = 1.3  # Contrast
    beta = 40    # Brightness
    enhanced1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_versions.append(("brightness", enhanced1))
    
    # Strategy 2: Histogram equalization
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced_versions.append(("histogram", enhanced2))
    
    # Strategy 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))  # Convert tuple to list
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    enhanced3 = cv2.merge(lab_planes)
    enhanced3 = cv2.cvtColor(enhanced3, cv2.COLOR_LAB2BGR)
    enhanced_versions.append(("clahe", enhanced3))
    
    # Strategy 4: Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced4 = cv2.filter2D(image, -1, kernel)
    enhanced_versions.append(("sharpening", enhanced4))
    
    # Strategy 5: Color space enhancement for PPE
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)  # Increase saturation
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)   # Increase brightness
    enhanced5 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced_versions.append(("color_enhance", enhanced5))
    
    return enhanced_versions

def smart_duplicate_removal(detections, iou_threshold=0.3):
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

def calculate_iou(box1, box2):
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

def comprehensive_ppe_detection(image):
    """Perform comprehensive PPE detection with multiple strategies"""
    print("[INFO] Starting comprehensive PPE detection...")
    
    all_detections = []
    
    # Get enhanced versions
    enhanced_versions = enhance_image_for_detection(image)
    
    # Strategy 1: Standard detection with optimized parameters
    results = model.predict(
        image,
        conf=0.25,      # Confidence threshold
        iou=0.45,       # IoU threshold
        imgsz=640,      # Image size
        max_det=300,    # Maximum detections
        half=True,      # Half precision
        verbose=False
    )
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                all_detections.append({
                    'class': model.names[cls],
                    'confidence': round(conf, 3),
                    'bbox': bbox,
                    'strategy': 'standard'
                })
    
    # Strategy 2: Enhanced image detection
    for strategy_name, enhanced_img in enhanced_versions[1:]:  # Skip original
        results = model.predict(
            enhanced_img,
            conf=0.15,      # Lower confidence for enhanced images
            iou=0.3,
            imgsz=640,
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
                        d['class'] == model.names[cls] 
                        for d in all_detections
                    )
                    
                    if not already_detected and conf >= 0.1:
                        all_detections.append({
                            'class': model.names[cls],
                            'confidence': round(conf, 3),
                            'bbox': bbox,
                            'strategy': strategy_name
                        })
    
    # Remove duplicates
    final_detections = smart_duplicate_removal(all_detections, iou_threshold=0.3)
    
    print(f"[INFO] Total detections before filtering: {len(all_detections)}")
    print(f"[INFO] Total detections after filtering: {len(final_detections)}")
    
    return final_detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Perform comprehensive detection
        detections = comprehensive_ppe_detection(image)
        
        # Process results
        ppe_counts = {'helmet': 0, 'safety_vest': 0, 'goggles': 0, 'gloves': 0}
        detection_details = []
        
        for det in detections:
            class_name = det['class']
            if class_name in ppe_counts:
                ppe_counts[class_name] += 1
            detection_details.append({
                'class': class_name,
                'confidence': det['confidence'],
                'strategy': det['strategy']
            })
        
        # Calculate compliance
        num_people = max(ppe_counts.values()) if any(ppe_counts.values()) else 1
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        missing_ppe = []
        
        for ppe in required_ppe:
            if ppe_counts[ppe] < num_people:
                missing_ppe.append(ppe)
        
        compliance_status = 'COMPLIANT' if not missing_ppe else 'NON-COMPLIANT'
        
        # Log detection details
        print(f"[INFO] Detection Results:")
        print(f"  People: {num_people}")
        print(f"  PPE Counts: {ppe_counts}")
        print(f"  Missing: {missing_ppe}")
        print(f"  Status: {compliance_status}")
        
        return jsonify({
            'success': True,
            'compliance_status': compliance_status,
            'num_people': num_people,
            'ppe_counts': ppe_counts,
            'missing_ppe': missing_ppe,
            'detection_details': detection_details,
            'total_detections': len(detections)
        })
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def stats():
    return jsonify({
        'total_detections': 0,
        'compliance_rate': 0,
        'ppe_counts': {'helmet': 0, 'safety_vest': 0, 'goggles': 0, 'gloves': 0}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
