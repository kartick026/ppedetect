#!/usr/bin/env python3
"""
Ultimate PPE Detection Solution
Addresses all detection issues with comprehensive approach
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

# Store detection history
detection_history = []

def ultimate_image_enhancement(image):
    """Ultimate image enhancement for maximum detection"""
    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create multiple enhanced versions
    enhanced_versions = []
    
    # Version 1: Brightness and contrast enhancement
    alpha = 1.5  # Contrast control
    beta = 30    # Brightness control
    enhanced1 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_versions.append(enhanced1)
    
    # Version 2: Histogram equalization
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced_versions.append(enhanced2)
    
    # Version 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab_planes = list(cv2.split(lab))  # Convert tuple to list
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    enhanced3 = cv2.merge(lab_planes)
    enhanced3 = cv2.cvtColor(enhanced3, cv2.COLOR_LAB2BGR)
    enhanced_versions.append(enhanced3)
    
    # Version 4: Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced4 = cv2.filter2D(image, -1, kernel)
    enhanced_versions.append(enhanced4)
    
    # Version 5: Color space enhancement for PPE
    # Enhance orange/yellow for safety vests
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Enhance saturation and brightness for PPE colors
    hsv_enhanced = hsv.copy()
    hsv_enhanced[:,:,1] = np.where(orange_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
    hsv_enhanced[:,:,2] = np.where(orange_mask > 0, np.minimum(hsv[:,:,2] * 1.5, 255), hsv[:,:,2])
    hsv_enhanced[:,:,1] = np.where(yellow_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
    hsv_enhanced[:,:,2] = np.where(yellow_mask > 0, np.minimum(hsv[:,:,2] * 1.5, 255), hsv[:,:,2])
    
    enhanced5 = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    enhanced_versions.append(enhanced5)
    
    return enhanced_versions

def detect_with_multiple_strategies(image):
    """Ultimate detection with multiple strategies"""
    all_detections = []
    
    # Strategy 1: Original image with ultra-low confidence
    print("[INFO] Strategy 1: Original image (conf=0.001)")
    results1 = model(image, conf=0.001, verbose=False)
    for r in results1:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                all_detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'strategy': 'original'
                })
    
    # Strategy 2: Multiple enhanced versions
    enhanced_versions = ultimate_image_enhancement(image)
    for i, enhanced_img in enumerate(enhanced_versions):
        print(f"[INFO] Strategy {i+2}: Enhanced version {i+1} (conf=0.001)")
        results = model(enhanced_img, conf=0.001, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in all_detections:
                        if existing['class'] == class_name:
                            ex_bbox = existing['bbox']
                            # More lenient duplicate check
                            if abs(x1 - ex_bbox[0]) < 50 and abs(y1 - ex_bbox[1]) < 50:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        all_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'strategy': f'enhanced_{i+1}'
                        })
    
    # Strategy 3: Multiple image scales
    scales = [0.8, 1.0, 1.2, 1.5]
    for scale in scales:
        print(f"[INFO] Strategy {len(enhanced_versions)+2}: Scale {scale} (conf=0.001)")
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_img = cv2.resize(image, (new_w, new_h))
        
        results = model(scaled_img, conf=0.001, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Scale back coordinates
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in all_detections:
                        if existing['class'] == class_name:
                            ex_bbox = existing['bbox']
                            if abs(x1 - ex_bbox[0]) < 50 and abs(y1 - ex_bbox[1]) < 50:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        all_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'strategy': f'scale_{scale}'
                        })
    
    return all_detections

def smart_merge_detections(all_detections):
    """Smart merging with class-specific logic"""
    if not all_detections:
        return []
    
    # Separate by class
    class_detections = {}
    for det in all_detections:
        class_name = det['class']
        if class_name not in class_detections:
            class_detections[class_name] = []
        class_detections[class_name].append(det)
    
    merged_detections = []
    
    for class_name, detections in class_detections.items():
        if not detections:
            continue
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Class-specific merging logic
        if class_name == 'helmet':
            # Keep top 2-3 helmet detections (for multiple people)
            for det in detections[:3]:
                if det['confidence'] > 0.1:  # Minimum confidence
                    merged_detections.append(det)
        elif class_name == 'safety_vest':
            # Keep top 2-3 safety vest detections
            for det in detections[:3]:
                if det['confidence'] > 0.05:  # Lower threshold for vests
                    merged_detections.append(det)
        elif class_name == 'goggles':
            # Keep top 2-3 goggles detections
            for det in detections[:3]:
                if det['confidence'] > 0.01:  # Very low threshold for goggles
                    merged_detections.append(det)
        elif class_name == 'gloves':
            # Keep top 2-3 gloves detections
            for det in detections[:3]:
                if det['confidence'] > 0.01:  # Very low threshold for gloves
                    merged_detections.append(det)
        else:
            # For other classes, keep best detection
            if detections[0]['confidence'] > 0.1:
                merged_detections.append(detections[0])
    
    return merged_detections

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Ultimate PPE detection"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Ultimate detection with multiple strategies
        all_detections = detect_with_multiple_strategies(image)
        
        # Smart merging
        final_detections = smart_merge_detections(all_detections)
        
        # Update detected classes
        detected_classes = set(det['class'] for det in final_detections)
        
        # Check compliance
        compliance_status = "COMPLIANT"
        missing_ppe = []
        
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        for ppe in required_ppe:
            if ppe not in detected_classes:
                missing_ppe.append(ppe)
                compliance_status = "NON-COMPLIANT"
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'detections': final_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(final_detections)
        }
        detection_history.append(detection_record)
        
        # Save annotated image
        if final_detections:
            annotated_image = image.copy()
            for det in final_detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['class']}: {det['confidence']:.2f}"
                color = (0, 255, 0)  # Green
                if det['class'] == 'helmet': color = (0, 255, 255)  # Yellow
                elif det['class'] == 'safety_vest': color = (0, 165, 255)  # Orange
                elif det['class'] == 'goggles': color = (255, 0, 255)  # Magenta
                elif det['class'] == 'gloves': color = (255, 255, 0)  # Cyan
                
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            output_path = f"static/detections/detection_{len(detection_history)}.jpg"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            detection_record['annotated_image'] = output_path
        
        print(f"[INFO] Ultimate detections: {len(final_detections)}")
        for det in final_detections:
            print(f"  {det['class']}: {det['confidence']:.3f} ({det['strategy']})")
        
        return jsonify({
            'success': True,
            'detections': final_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(final_detections),
            'annotated_image': detection_record.get('annotated_image', '')
        })
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get detection history"""
    return jsonify(detection_history)

@app.route('/stats')
def get_stats():
    """Get compliance statistics"""
    if not detection_history:
        return jsonify({'message': 'No detections yet'})
    
    total_detections = len(detection_history)
    compliant = sum(1 for d in detection_history if d['compliance_status'] == 'COMPLIANT')
    non_compliant = total_detections - compliant
    
    # PPE detection counts
    ppe_counts = {}
    for detection in detection_history:
        for det in detection['detections']:
            ppe_type = det['class']
            ppe_counts[ppe_type] = ppe_counts.get(ppe_type, 0) + 1
    
    return jsonify({
        'total_detections': total_detections,
        'compliant': compliant,
        'non_compliant': non_compliant,
        'compliance_rate': (compliant / total_detections * 100) if total_detections > 0 else 0,
        'ppe_counts': ppe_counts
    })

if __name__ == '__main__':
    print("="*70)
    print("ULTIMATE PPE DETECTION SOLUTION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Ultimate detection with multiple strategies")
    print(f"[INFO] Ultra-low confidence (0.001)")
    print(f"[INFO] Multiple image enhancements")
    print(f"[INFO] Multiple scales")
    print(f"[INFO] Smart merging")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
