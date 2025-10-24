#!/usr/bin/env python3
"""
Advanced PPE Detection System
Specialized for detecting small, subtle PPE items like goggles and gloves
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
print("ADVANCED PPE DETECTION SYSTEM")
print("="*70)
print(f"[INFO] Model loaded: {model_path}")
print(f"[INFO] Classes: {model.names}")
print("[INFO] Specialized for small object detection")
print("[INFO] Advanced preprocessing for goggles and gloves")
print("[INFO] Multi-scale and multi-strategy detection")
print("[INFO] Starting web server...")
print("[INFO] Open: http://localhost:5000")
print("="*70)

def advanced_image_preprocessing(image):
    """Advanced preprocessing specifically for small PPE detection"""
    enhanced_versions = []
    
    # Version 1: High contrast for goggles (transparent objects)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    enhanced1 = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_versions.append(enhanced1)
    
    # Version 2: Edge enhancement for small objects
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced2 = cv2.filter2D(image, -1, kernel)
    enhanced_versions.append(enhanced2)
    
    # Version 3: Color space enhancement for gloves
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Enhance skin tones and glove colors
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)  # Increase saturation
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)  # Increase brightness
    enhanced3 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced_versions.append(enhanced3)
    
    # Version 4: Multi-scale pyramid
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
            enhanced_versions.append(scaled)
    
    # Version 5: Noise reduction and sharpening
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
    enhanced_versions.append(sharpened)
    
    return enhanced_versions

def detect_small_ppe_objects(image):
    """Specialized detection for small PPE objects"""
    all_detections = []
    
    # Strategy 1: Ultra-low confidence detection
    results = model(image, conf=0.001, iou=0.1, verbose=False)
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls in [2, 3]:  # goggles and gloves
                    all_detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'strategy': 'ultra_low_conf'
                    })
    
    # Strategy 2: Enhanced preprocessing
    enhanced_images = advanced_image_preprocessing(image)
    for i, enhanced_img in enumerate(enhanced_images):
        results = model(enhanced_img, conf=0.01, iou=0.3, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls in [2, 3]:  # goggles and gloves
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': f'enhanced_{i+1}'
                        })
    
    # Strategy 3: Multi-scale detection
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_img = cv2.resize(image, (new_w, new_h))
        
        results = model(scaled_img, conf=0.005, iou=0.2, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls in [2, 3]:  # goggles and gloves
                        # Scale back bbox coordinates
                        bbox = box.xyxy[0].cpu().numpy()
                        bbox = bbox / scale
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': bbox,
                            'strategy': f'scale_{scale}'
                        })
    
    return all_detections

def smart_detection_merging(detections):
    """Intelligent merging of detections from different strategies"""
    if not detections:
        return []
    
    # Group by class
    goggles_detections = [d for d in detections if d['class'] == 2]
    gloves_detections = [d for d in detections if d['class'] == 3]
    
    merged = []
    
    # Merge goggles detections
    if goggles_detections:
        # Sort by confidence
        goggles_detections.sort(key=lambda x: x['confidence'], reverse=True)
        # Take the best detection
        merged.append(goggles_detections[0])
    
    # Merge gloves detections
    if gloves_detections:
        # Sort by confidence
        gloves_detections.sort(key=lambda x: x['confidence'], reverse=True)
        # Take the best detection
        merged.append(gloves_detections[0])
    
    return merged

def comprehensive_ppe_detection(image):
    """Comprehensive PPE detection with all strategies"""
    print("[INFO] Starting comprehensive PPE detection...")
    
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
    
    # Specialized detection for small objects
    small_object_detections = detect_small_ppe_objects(image)
    
    # Merge all detections
    all_detections = standard_detections + small_object_detections
    
    # Smart merging for small objects
    merged_small = smart_detection_merging(small_object_detections)
    
    # Final detections
    final_detections = standard_detections + merged_small
    
    print(f"[INFO] Standard detections: {len(standard_detections)}")
    print(f"[INFO] Small object detections: {len(small_object_detections)}")
    print(f"[INFO] Merged small objects: {len(merged_small)}")
    print(f"[INFO] Final detections: {len(final_detections)}")
    
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
