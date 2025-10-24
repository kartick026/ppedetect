#!/usr/bin/env python3
"""
Fix False Positive Helmet Detection
Specialized solution to eliminate false helmet detections
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
print("FIX FALSE POSITIVE HELMET DETECTION")
print("="*70)
print(f"[INFO] Model loaded: {model_path}")
print(f"[INFO] Classes: {model.names}")
print("[INFO] Eliminating false positive helmet detections")
print("[INFO] Higher confidence thresholds for helmets")
print("[INFO] Smart filtering based on size and position")
print("[INFO] Starting web server...")
print("[INFO] Open: http://localhost:5000")
print("="*70)

def is_valid_helmet_detection(bbox, image_shape, confidence):
    """Validate if a helmet detection is actually a helmet"""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calculate bounding box properties
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    image_area = w * h
    
    # Size filtering - helmet should be reasonable size
    area_ratio = bbox_area / image_area
    if area_ratio < 0.001 or area_ratio > 0.3:  # Too small or too large
        return False
    
    # Position filtering - helmet should be in upper portion of image
    center_y = (y1 + y2) / 2
    if center_y > h * 0.6:  # Not in upper 60% of image
        return False
    
    # Aspect ratio filtering - helmet should be roughly circular/oval
    aspect_ratio = bbox_width / bbox_height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Too elongated
        return False
    
    # Confidence filtering - require higher confidence for helmets
    if confidence < 0.6:  # Higher threshold for helmets
        return False
    
    return True

def smart_helmet_filtering(detections, image_shape):
    """Smart filtering to remove false positive helmet detections"""
    valid_detections = []
    
    for det in detections:
        if det['class'] == 0:  # helmet
            if is_valid_helmet_detection(det['bbox'], image_shape, det['confidence']):
                valid_detections.append(det)
            else:
                print(f"[INFO] Filtered out false positive helmet: conf={det['confidence']:.3f}")
        else:
            valid_detections.append(det)
    
    return valid_detections

def enhanced_ppe_detection_with_helmet_filtering(image):
    """Enhanced PPE detection with strict helmet filtering"""
    print("[INFO] Starting enhanced PPE detection with helmet filtering...")
    
    # Standard detection with higher confidence for helmets
    results = model(image, conf=0.4, iou=0.5, verbose=False)
    all_detections = []
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Use different confidence thresholds for different classes
                if cls == 0:  # helmet - higher confidence required
                    if conf >= 0.6:
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': 'high_conf_helmet'
                        })
                elif cls == 1:  # safety_vest - medium confidence
                    if conf >= 0.3:
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': 'standard'
                        })
                elif cls in [2, 3]:  # goggles, gloves - lower confidence
                    if conf >= 0.1:
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': 'low_conf_small'
                        })
    
    # Apply smart helmet filtering
    filtered_detections = smart_helmet_filtering(all_detections, image.shape)
    
    # Additional detection for small objects with ultra-low confidence
    small_object_results = model(image, conf=0.01, iou=0.3, verbose=False)
    for r in small_object_results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Only add if not already detected
                if cls in [2, 3]:  # goggles, gloves
                    # Check if already detected
                    already_detected = any(d['class'] == cls for d in filtered_detections)
                    if not already_detected and conf >= 0.05:
                        filtered_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': 'ultra_low_conf'
                        })
    
    print(f"[INFO] Total detections before filtering: {len(all_detections)}")
    print(f"[INFO] Total detections after filtering: {len(filtered_detections)}")
    
    return filtered_detections

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
        
        # Perform enhanced detection with helmet filtering
        detections = enhanced_ppe_detection_with_helmet_filtering(image)
        
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
