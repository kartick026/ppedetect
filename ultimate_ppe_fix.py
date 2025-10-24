#!/usr/bin/env python3
"""
Ultimate PPE Detection Fix
Comprehensive solution for all detection issues
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
print("ULTIMATE PPE DETECTION FIX")
print("="*70)
print(f"[INFO] Model loaded: {model_path}")
print(f"[INFO] Classes: {model.names}")
print("[INFO] Comprehensive fix for all detection issues")
print("[INFO] Eliminating false positives")
print("[INFO] Enhanced detection for small objects")
print("[INFO] Smart filtering and validation")
print("[INFO] Starting web server...")
print("[INFO] Open: http://localhost:5000")
print("="*70)

def validate_helmet_detection(bbox, image_shape, confidence):
    """Strict validation for helmet detection to eliminate false positives"""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calculate properties
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    image_area = w * h
    area_ratio = bbox_area / image_area
    
    # Strict validation criteria
    if confidence < 0.7:  # Very high confidence required
        return False
    
    if area_ratio < 0.005 or area_ratio > 0.2:  # Size filtering
        return False
    
    # Position check - helmet should be in upper portion
    center_y = (y1 + y2) / 2
    if center_y > h * 0.5:  # Must be in upper 50%
        return False
    
    # Aspect ratio check
    aspect_ratio = bbox_width / bbox_height
    if aspect_ratio < 0.6 or aspect_ratio > 1.8:  # Roughly circular/oval
        return False
    
    return True

def enhance_for_goggles_detection(image):
    """Specialized enhancement for goggles detection"""
    enhanced_versions = []
    
    # Strategy 1: High contrast for transparent goggles
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    enhanced1 = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_versions.append(enhanced1)
    
    # Strategy 2: Edge detection for goggles frames
    edges = cv2.Canny(gray, 20, 80)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced2 = cv2.addWeighted(image, 0.5, edges_colored, 0.5, 0)
    enhanced_versions.append(enhanced2)
    
    # Strategy 3: Brightness and contrast boost
    alpha = 2.0  # High contrast
    beta = 50    # High brightness
    enhanced3 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_versions.append(enhanced3)
    
    # Strategy 4: Multi-scale detection
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        enhanced_versions.append((scaled, scale))
    
    return enhanced_versions

def enhance_for_gloves_detection(image):
    """Specialized enhancement for gloves detection"""
    enhanced_versions = []
    
    # Strategy 1: Color space enhancement for skin tones
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 2.0)  # High saturation
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.5)  # High brightness
    enhanced1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced_versions.append(enhanced1)
    
    # Strategy 2: Edge detection for glove contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 60)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced2 = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    enhanced_versions.append(enhanced2)
    
    # Strategy 3: Multi-scale detection
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        enhanced_versions.append((scaled, scale))
    
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
    """Comprehensive PPE detection with all fixes applied"""
    print("[INFO] Starting comprehensive PPE detection with all fixes...")
    
    all_detections = []
    
    # Strategy 1: Standard detection with class-specific confidence
    results = model(image, conf=0.3, iou=0.5, verbose=False)
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Class-specific confidence thresholds
                if cls == 0:  # helmet - very strict
                    if conf >= 0.7:
                        bbox = box.xyxy[0].cpu().numpy()
                        if validate_helmet_detection(bbox, image.shape, conf):
                            all_detections.append({
                                'class': cls,
                                'confidence': conf,
                                'bbox': bbox,
                                'strategy': 'strict_helmet'
                            })
                elif cls == 1:  # safety_vest - medium strict
                    if conf >= 0.4:
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': 'standard'
                        })
                elif cls in [2, 3]:  # goggles, gloves - sensitive
                    if conf >= 0.2:
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': 'sensitive'
                        })
    
    # Strategy 2: Enhanced goggles detection
    goggles_enhanced = enhance_for_goggles_detection(image)
    for i, enhanced_data in enumerate(goggles_enhanced):
        if isinstance(enhanced_data, tuple):
            enhanced_img, scale = enhanced_data
        else:
            enhanced_img = enhanced_data
            scale = 1.0
        
        results = model(enhanced_img, conf=0.01, iou=0.3, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 2:  # goggles
                        bbox = box.xyxy[0].cpu().numpy()
                        if scale != 1.0:
                            bbox = bbox / scale
                        
                        # Check if already detected
                        already_detected = any(d['class'] == 2 for d in all_detections)
                        if not already_detected and conf >= 0.05:
                            all_detections.append({
                                'class': cls,
                                'confidence': conf,
                                'bbox': bbox,
                                'strategy': f'goggles_enhanced_{i+1}'
                            })
    
    # Strategy 3: Enhanced gloves detection
    gloves_enhanced = enhance_for_gloves_detection(image)
    for i, enhanced_data in enumerate(gloves_enhanced):
        if isinstance(enhanced_data, tuple):
            enhanced_img, scale = enhanced_data
        else:
            enhanced_img = enhanced_data
            scale = 1.0
        
        results = model(enhanced_img, conf=0.01, iou=0.3, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 3:  # gloves
                        bbox = box.xyxy[0].cpu().numpy()
                        if scale != 1.0:
                            bbox = bbox / scale
                        
                        # Check if already detected
                        already_detected = any(d['class'] == 3 for d in all_detections)
                        if not already_detected and conf >= 0.05:
                            all_detections.append({
                                'class': cls,
                                'confidence': conf,
                                'bbox': bbox,
                                'strategy': f'gloves_enhanced_{i+1}'
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
