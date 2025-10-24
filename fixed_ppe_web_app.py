#!/usr/bin/env python3
"""
Fixed PPE Detection Web Application
Specialized for obscured safety vests
"""

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

# Store detection history
detection_history = []

def enhance_reflective_strips(image):
    """Enhance image to highlight reflective strips on safety vests"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for bright/reflective areas
    lower_bright = np.array([0, 0, 200])
    upper_bright = np.array([180, 30, 255])
    bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Enhance brightness in bright areas
    hsv[:,:,2] = np.where(bright_mask > 0, np.minimum(hsv[:,:,2] * 1.3, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def enhance_safety_colors(image):
    """Enhance orange and yellow colors typical of safety vests"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Orange color range
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Yellow color range
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    safety_mask = cv2.bitwise_or(orange_mask, yellow_mask)
    
    # Enhance saturation and brightness for safety colors
    hsv[:,:,1] = np.where(safety_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
    hsv[:,:,2] = np.where(safety_mask > 0, np.minimum(hsv[:,:,2] * 1.2, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def boost_brightness(image, factor):
    """Boost image brightness"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=30)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Fixed PPE detection with specialized obscured vest detection"""
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
        
        # FIXED DETECTION: Multiple specialized strategies
        all_detections = []
        detected_classes = set()
        
        # Strategy 1: Very low confidence for all PPE
        results1 = model(image, conf=0.01, verbose=False)
        for r in results1:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    detected_classes.add(class_name)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    all_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Strategy 2: Enhanced reflective strips for safety vests
        enhanced_strips = enhance_reflective_strips(image)
        results2 = model(enhanced_strips, conf=0.01, verbose=False)
        for r in results2:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'safety_vest':  # Only add safety vests from this pass
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'safety_vest':
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 100 and abs(y1 - ex_bbox[1]) < 100:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
        # Strategy 3: Enhanced safety colors for vests
        color_enhanced = enhance_safety_colors(image)
        results3 = model(color_enhanced, conf=0.01, verbose=False)
        for r in results3:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'safety_vest':
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'safety_vest':
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 100 and abs(y1 - ex_bbox[1]) < 100:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
        # Strategy 4: Brightness boost for hidden vests
        bright = boost_brightness(image, 1.5)
        results4 = model(bright, conf=0.01, verbose=False)
        for r in results4:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'safety_vest':
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'safety_vest':
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 100 and abs(y1 - ex_bbox[1]) < 100:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
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
            'detections': all_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(all_detections)
        }
        detection_history.append(detection_record)
        
        # Save annotated image
        if results1:
            annotated_image = results1[0].plot()
            output_path = f"static/detections/detection_{len(detection_history)}.jpg"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            detection_record['annotated_image'] = output_path
        
        return jsonify({
            'success': True,
            'detections': all_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(all_detections),
            'annotated_image': detection_record.get('annotated_image', '')
        })
        
    except Exception as e:
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
    print("FIXED PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Specialized for obscured safety vests")
    print(f"[INFO] Very low confidence thresholds")
    print(f"[INFO] Multiple enhancement strategies")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
