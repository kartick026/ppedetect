#!/usr/bin/env python3
"""
PPE Compliance Checker
Simple YES/NO detection for each person wearing all 4 PPE items
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

def enhance_for_ppe_detection(image):
    """Enhanced image processing for better PPE detection"""
    # Convert to HSV for color enhancement
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Enhance brightness and contrast
    alpha = 1.3  # Contrast
    beta = 20    # Brightness
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    enhanced = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_ppe_compliance(image):
    """Detect PPE compliance with simple YES/NO logic"""
    all_detections = []
    
    # Strategy 1: Original image with low confidence
    print("[INFO] Strategy 1: Original image (conf=0.05)")
    results1 = model(image, conf=0.05, verbose=False)
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
                    'bbox': [x1, y1, x2, y2]
                })
    
    # Strategy 2: Enhanced image with very low confidence
    print("[INFO] Strategy 2: Enhanced image (conf=0.01)")
    enhanced = enhance_for_ppe_detection(image)
    results2 = model(enhanced, conf=0.01, verbose=False)
    for r in results2:
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
                        if abs(x1 - ex_bbox[0]) < 100 and abs(y1 - ex_bbox[1]) < 100:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    all_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
    
    return all_detections

def check_person_compliance(detections):
    """Check if each person has all required PPE"""
    required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
    
    # Count detections by class
    ppe_counts = {}
    for det in detections:
        class_name = det['class']
        ppe_counts[class_name] = ppe_counts.get(class_name, 0) + 1
    
    # Simple compliance logic
    compliance_status = "COMPLIANT"
    missing_ppe = []
    
    for ppe in required_ppe:
        if ppe not in ppe_counts or ppe_counts[ppe] == 0:
            missing_ppe.append(ppe)
            compliance_status = "NON-COMPLIANT"
    
    # Determine number of people (based on helmet count)
    num_people = ppe_counts.get('helmet', 0)
    
    # Check if we have enough PPE for all people
    for ppe in required_ppe:
        if ppe in ppe_counts:
            if ppe_counts[ppe] < num_people:
                if ppe not in missing_ppe:
                    missing_ppe.append(ppe)
                compliance_status = "NON-COMPLIANT"
    
    return {
        'compliance_status': compliance_status,
        'missing_ppe': missing_ppe,
        'num_people': num_people,
        'ppe_counts': ppe_counts,
        'detections': detections
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """PPE compliance detection"""
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
        
        # Detect PPE
        detections = detect_ppe_compliance(image)
        
        # Check compliance
        compliance_result = check_person_compliance(detections)
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'compliance_status': compliance_result['compliance_status'],
            'missing_ppe': compliance_result['missing_ppe'],
            'num_people': compliance_result['num_people'],
            'ppe_counts': compliance_result['ppe_counts'],
            'total_detections': len(detections)
        }
        detection_history.append(detection_record)
        
        # Save annotated image
        if detections:
            annotated_image = image.copy()
            for det in detections:
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
        
        print(f"[INFO] Compliance Check Results:")
        print(f"  Status: {compliance_result['compliance_status']}")
        print(f"  People: {compliance_result['num_people']}")
        print(f"  PPE Counts: {compliance_result['ppe_counts']}")
        if compliance_result['missing_ppe']:
            print(f"  Missing: {compliance_result['missing_ppe']}")
        
        return jsonify({
            'success': True,
            'compliance_status': compliance_result['compliance_status'],
            'missing_ppe': compliance_result['missing_ppe'],
            'num_people': compliance_result['num_people'],
            'ppe_counts': compliance_result['ppe_counts'],
            'total_detections': len(detections),
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
    
    return jsonify({
        'total_detections': total_detections,
        'compliant': compliant,
        'non_compliant': non_compliant,
        'compliance_rate': (compliant / total_detections * 100) if total_detections > 0 else 0
    })

if __name__ == '__main__':
    print("="*70)
    print("PPE COMPLIANCE CHECKER")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Simple YES/NO compliance checking")
    print(f"[INFO] Required PPE: helmet, safety_vest, goggles, gloves")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
