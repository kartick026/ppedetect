#!/usr/bin/env python3
"""
Fix Duplicate Safety Vests and Missing Goggles Detection
Specialized solution for the exact issues reported
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

print("="*70)
print("FIX DUPLICATE VESTS & MISSING GOGGLES")
print("="*70)
print(f"[INFO] Model loaded: {model_path}")
print(f"[INFO] Classes: {model.names}")
print("[INFO] Fixing duplicate safety vest detection")
print("[INFO] Enhanced goggles detection")
print("[INFO] Smart merging to eliminate duplicates")
print("[INFO] Starting web server...")
print("[INFO] Open: http://localhost:5000")
print("="*70)

def enhance_for_goggles_detection(image):
    """Specialized enhancement specifically for goggles detection"""
    enhanced_versions = []
    
    # Strategy 1: High contrast for transparent goggles
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    enhanced1 = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_versions.append(enhanced1)
    
    # Strategy 2: Edge detection for goggles frames
    edges = cv2.Canny(gray, 30, 100)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced2 = cv2.addWeighted(image, 0.6, edges_colored, 0.4, 0)
    enhanced_versions.append(enhanced2)
    
    # Strategy 3: Brightness boost for small objects
    alpha = 1.8  # High contrast
    beta = 50    # High brightness
    enhanced3 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_versions.append(enhanced3)
    
    # Strategy 4: Sharpening for small details
    kernel = np.array([[-1,-1,-1], [-1,12,-1], [-1,-1,-1]])
    enhanced4 = cv2.filter2D(image, -1, kernel)
    enhanced_versions.append(enhanced4)
    
    # Strategy 5: Multi-scale for small objects
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        enhanced_versions.append((scaled, scale))
    
    return enhanced_versions

def smart_duplicate_removal(detections, iou_threshold=0.3):
    """Remove duplicate detections using smart IoU-based merging"""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
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

def detect_goggles_with_ultra_sensitivity(image):
    """Ultra-sensitive goggles detection with multiple strategies"""
    all_detections = []
    
    # Strategy 1: Ultra-low confidence detection
    conf_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    for conf_thresh in conf_levels:
        results = model(image, conf=conf_thresh, iou=0.1, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 2:  # goggles
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': f'ultra_low_conf_{conf_thresh}'
                        })
    
    # Strategy 2: Enhanced preprocessing
    enhanced_images = enhance_for_goggles_detection(image)
    for i, enhanced_data in enumerate(enhanced_images):
        if isinstance(enhanced_data, tuple):
            enhanced_img, scale = enhanced_data
        else:
            enhanced_img = enhanced_data
            scale = 1.0
        
        results = model(enhanced_img, conf=0.001, iou=0.1, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 2:  # goggles
                        bbox = box.xyxy[0].cpu().numpy()
                        if scale != 1.0:
                            bbox = bbox / scale
                        
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': bbox,
                            'strategy': f'enhanced_{i+1}'
                        })
    
    # Smart merging to get best detection
    if all_detections:
        # Remove duplicates
        filtered_detections = smart_duplicate_removal(all_detections, iou_threshold=0.2)
        # Return the best detection
        if filtered_detections:
            return [filtered_detections[0]]
    
    return []

def comprehensive_ppe_detection_fixed(image):
    """Comprehensive PPE detection with duplicate removal and enhanced goggles detection"""
    print("[INFO] Starting comprehensive PPE detection with fixes...")
    
    # Standard detection for helmets and safety vests
    results = model(image, conf=0.3, iou=0.5, verbose=False)
    standard_detections = []
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls in [0, 1]:  # helmet and safety_vest
                    standard_detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'strategy': 'standard'
                    })
    
    # Remove duplicate safety vests
    if standard_detections:
        standard_detections = smart_duplicate_removal(standard_detections, iou_threshold=0.3)
    
    # Enhanced goggles detection
    goggles_detections = detect_goggles_with_ultra_sensitivity(image)
    
    # Standard gloves detection with low confidence
    gloves_results = model(image, conf=0.01, iou=0.3, verbose=False)
    gloves_detections = []
    for r in gloves_results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == 3:  # gloves
                    gloves_detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'strategy': 'low_conf'
                    })
    
    # Combine all detections
    all_detections = standard_detections + goggles_detections + gloves_detections
    
    print(f"[INFO] Standard detections: {len(standard_detections)}")
    print(f"[INFO] Goggles detections: {len(goggles_detections)}")
    print(f"[INFO] Gloves detections: {len(gloves_detections)}")
    print(f"[INFO] Total detections: {len(all_detections)}")
    
    return all_detections

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
        
        # Perform comprehensive detection with fixes
        detections = comprehensive_ppe_detection_fixed(image)
        
        # Process results
        ppe_counts = {'helmet': 0, 'safety_vest': 0, 'goggles': 0, 'gloves': 0}
        detection_details = []
        
        for det in detections:
            class_name = model.names[det['class']]
            ppe_counts[class_name] += 1
            detection_details.append({
                'class': class_name,
                'confidence': round(det['confidence'], 3),
                'strategy': det['strategy']
            })
        
        # Calculate compliance
        num_people = max(ppe_counts['helmet'], ppe_counts['safety_vest'], 1)
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
            'detection_details': detection_details
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
