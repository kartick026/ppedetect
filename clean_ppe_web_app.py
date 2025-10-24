#!/usr/bin/env python3
"""
Clean PPE Detection Web Application
Error-free version with proper error handling
"""

from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Global variables
model = None
detection_history = []

def load_model():
    """Load the YOLO model with error handling"""
    global model
    try:
        model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return False
        
        model = YOLO(model_path)
        print(f"[SUCCESS] Model loaded: {model_path}")
        print(f"[INFO] Classes: {model.names}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def detect_ppe_safe(image):
    """Detect PPE with comprehensive error handling"""
    try:
        if model is None:
            return {
                'detections': [],
                'compliance_status': 'ERROR',
                'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
                'total_detections': 0,
                'people_count': 0,
                'detected_classes': []
            }
        
        # Run YOLO detection
        results = model(image, conf=0.3, verbose=False)
        
        detections = []
        detected_classes = set()
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                        detected_classes.add(class_name)
                    except Exception as e:
                        print(f"[WARNING] Error processing detection: {e}")
                        continue
        
        # Determine compliance
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
        
        compliance_status = "COMPLIANT" if len(missing_ppe) == 0 else "NON-COMPLIANT"
        people_count = max(1, len(detections) // 2) if detections else 1
        
        return {
            'detections': detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(detections),
            'people_count': people_count,
            'detected_classes': list(detected_classes)
        }
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return {
            'detections': [],
            'compliance_status': 'ERROR',
            'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
            'total_detections': 0,
            'people_count': 0,
            'detected_classes': []
        }

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"[ERROR] Index route failed: {e}")
        return f"Error loading page: {e}", 500

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """PPE detection endpoint with error handling"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        try:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image format'}), 400
                
        except Exception as e:
            print(f"[ERROR] Image processing failed: {e}")
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Run detection
        result = detect_ppe_safe(image)
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'detections': result['detections'],
            'compliance_status': result['compliance_status'],
            'missing_ppe': result['missing_ppe'],
            'total_detections': result['total_detections'],
            'people_count': result['people_count']
        }
        detection_history.append(detection_record)
        
        # Save annotated image if model is available
        annotated_image_path = None
        if model is not None:
            try:
                results = model(image, conf=0.3, verbose=False)
                if results and len(results) > 0:
                    annotated_image = results[0].plot()
                    output_path = f"static/detections/detection_{len(detection_history)}.jpg"
                    os.makedirs("static/detections", exist_ok=True)
                    cv2.imwrite(output_path, annotated_image)
                    annotated_image_path = output_path
                    detection_record['annotated_image'] = annotated_image_path
            except Exception as e:
                print(f"[WARNING] Failed to save annotated image: {e}")
        
        print(f"[INFO] Detection completed: {result['total_detections']} objects found")
        for det in result['detections']:
            print(f"  {det['class']}: {det['confidence']:.2f}")
        
        return jsonify({
            'success': True,
            'detections': result['detections'],
            'compliance_status': result['compliance_status'],
            'missing_ppe': result['missing_ppe'],
            'total_detections': result['total_detections'],
            'people_count': result['people_count'],
            'detected_classes': result['detected_classes'],
            'annotated_image': annotated_image_path or ''
        })
        
    except Exception as e:
        print(f"[ERROR] Detection endpoint failed: {e}")
        return jsonify({'error': 'Detection failed: ' + str(e)}), 500

@app.route('/history')
def get_history():
    """Get detection history"""
    try:
        return jsonify(detection_history)
    except Exception as e:
        print(f"[ERROR] History endpoint failed: {e}")
        return jsonify({'error': 'Failed to get history'}), 500

@app.route('/stats')
def get_stats():
    """Get compliance statistics"""
    try:
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
    except Exception as e:
        print(f"[ERROR] Stats endpoint failed: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*70)
    print("CLEAN PPE DETECTION WEB APPLICATION")
    print("="*70)
    
    # Load model
    if load_model():
        print("[SUCCESS] Model loaded successfully")
    else:
        print("[WARNING] Model loading failed - detection will not work")
    
    print("[INFO] Starting web server...")
    print("[INFO] Open: http://localhost:5000")
    print("="*70)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")