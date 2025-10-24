#!/usr/bin/env python3
"""
Specialized Small PPE Detector
Ultra-advanced detection for goggles and gloves
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
print("SPECIALIZED SMALL PPE DETECTOR")
print("="*70)
print(f"[INFO] Model loaded: {model_path}")
print(f"[INFO] Classes: {model.names}")
print("[INFO] Ultra-advanced detection for goggles and gloves")
print("[INFO] Multiple enhancement strategies")
print("[INFO] Multi-scale detection")
print("[INFO] Smart merging algorithms")
print("[INFO] Starting web server...")
print("[INFO] Open: http://localhost:5000")
print("="*70)

def enhance_for_goggles(image):
    """Specialized enhancement for goggles detection"""
    enhanced_versions = []
    
    # Strategy 1: High contrast for transparent objects
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    enhanced1 = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_versions.append(enhanced1)
    
    # Strategy 2: Edge detection for goggles frames
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced2 = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    enhanced_versions.append(enhanced2)
    
    # Strategy 3: Brightness and contrast boost
    alpha = 1.5  # Contrast
    beta = 30    # Brightness
    enhanced3 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_versions.append(enhanced3)
    
    # Strategy 4: Sharpening for small details
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced4 = cv2.filter2D(image, -1, kernel)
    enhanced_versions.append(enhanced4)
    
    # Strategy 5: Multi-scale pyramid
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        enhanced_versions.append(scaled)
    
    return enhanced_versions

def enhance_for_gloves(image):
    """Specialized enhancement for gloves detection"""
    enhanced_versions = []
    
    # Strategy 1: Color space enhancement for skin tones
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Enhance saturation and brightness
    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.8)  # Increase saturation
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.3)  # Increase brightness
    enhanced1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced_versions.append(enhanced1)
    
    # Strategy 2: Edge detection for glove contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced2 = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    enhanced_versions.append(enhanced2)
    
    # Strategy 3: Histogram equalization
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced3 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced_versions.append(enhanced3)
    
    # Strategy 4: Noise reduction and sharpening
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced4 = cv2.filter2D(denoised, -1, kernel_sharp)
    enhanced_versions.append(enhanced4)
    
    # Strategy 5: Multi-scale detection
    scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for scale in scales:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        enhanced_versions.append(scaled)
    
    return enhanced_versions

def ultra_low_confidence_detection(image, target_classes=[2, 3]):
    """Ultra-low confidence detection for small objects"""
    detections = []
    
    # Strategy 1: Original image with ultra-low confidence
    results = model(image, conf=0.001, iou=0.1, verbose=False)
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls in target_classes:
                    detections.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'strategy': 'ultra_low_conf'
                    })
    
    # Strategy 2: Multiple confidence levels
    conf_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    for conf_thresh in conf_levels:
        results = model(image, conf=conf_thresh, iou=0.1, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls in target_classes:
                        detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy(),
                            'strategy': f'conf_{conf_thresh}'
                        })
    
    return detections

def specialized_goggles_detection(image):
    """Specialized detection for goggles"""
    print("[INFO] Starting specialized goggles detection...")
    
    all_detections = []
    
    # Ultra-low confidence detection
    ultra_detections = ultra_low_confidence_detection(image, [2])
    all_detections.extend(ultra_detections)
    
    # Enhanced preprocessing for goggles
    enhanced_images = enhance_for_goggles(image)
    for i, enhanced_img in enumerate(enhanced_images):
        results = model(enhanced_img, conf=0.001, iou=0.1, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 2:  # goggles
                        bbox = box.xyxy[0].cpu().numpy()
                        # Scale back if image was resized
                        if i >= 5:  # Multi-scale images
                            scale = [0.5, 0.75, 1.25, 1.5, 2.0][i-5]
                            bbox = bbox / scale
                        
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': bbox,
                            'strategy': f'goggles_enhanced_{i+1}'
                        })
    
    # Smart merging for goggles
    if all_detections:
        # Sort by confidence and take the best
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        return [all_detections[0]]  # Return only the best detection
    
    return []

def specialized_gloves_detection(image):
    """Specialized detection for gloves"""
    print("[INFO] Starting specialized gloves detection...")
    
    all_detections = []
    
    # Ultra-low confidence detection
    ultra_detections = ultra_low_confidence_detection(image, [3])
    all_detections.extend(ultra_detections)
    
    # Enhanced preprocessing for gloves
    enhanced_images = enhance_for_gloves(image)
    for i, enhanced_img in enumerate(enhanced_images):
        results = model(enhanced_img, conf=0.001, iou=0.1, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 3:  # gloves
                        bbox = box.xyxy[0].cpu().numpy()
                        # Scale back if image was resized
                        if i >= 5:  # Multi-scale images
                            scale = [0.5, 0.75, 1.25, 1.5, 2.0][i-5]
                            bbox = bbox / scale
                        
                        all_detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': bbox,
                            'strategy': f'gloves_enhanced_{i+1}'
                        })
    
    # Smart merging for gloves
    if all_detections:
        # Sort by confidence and take the best
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        return [all_detections[0]]  # Return only the best detection
    
    return []

def comprehensive_small_ppe_detection(image):
    """Comprehensive detection for small PPE items"""
    print("[INFO] Starting comprehensive small PPE detection...")
    
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
    
    # Specialized detection for goggles
    goggles_detections = specialized_goggles_detection(image)
    
    # Specialized detection for gloves
    gloves_detections = specialized_gloves_detection(image)
    
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
        
        # Perform comprehensive detection
        detections = comprehensive_small_ppe_detection(image)
        
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
