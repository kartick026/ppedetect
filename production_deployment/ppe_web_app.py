#!/usr/bin/env python3
"""
PPE Detection Web Application
Real-time PPE monitoring dashboard
"""

from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

# Store detection history
detection_history = []

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Detect PPE in uploaded image"""
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
        
        # Run PPE detection
        results = model(image, conf=0.5)
        
        # Process results
        detections = []
        compliance_status = "COMPLIANT"
        missing_ppe = []
        
        detected_classes = set()
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    detected_classes.add(class_name)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Check compliance
        required_ppe = ['helmet', 'safety_vest']
        for ppe in required_ppe:
            if ppe not in detected_classes:
                missing_ppe.append(ppe)
                compliance_status = "NON-COMPLIANT"
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'detections': detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(detections)
        }
        detection_history.append(detection_record)
        
        # Save annotated image
        if results:
            annotated_image = results[0].plot()
            output_path = f"static/detections/detection_{len(detection_history)}.jpg"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            detection_record['annotated_image'] = output_path
        
        return jsonify({
            'success': True,
            'detections': detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(detections),
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

@app.route('/live')
def live_monitoring():
    """Live monitoring page"""
    return render_template('live.html')

if __name__ == '__main__':
    print("="*70)
    print("PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
