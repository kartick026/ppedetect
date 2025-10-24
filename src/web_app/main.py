#!/usr/bin/env python3
"""
PPE Detection System - Main Web Application
Clean, organized main application entry point
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time
import json
from datetime import datetime

# Import configuration
from config.app_config import WEB_CONFIG, MODEL_CONFIG, CAMERA_CONFIG, DETECTION_CONFIG

app = Flask(__name__)

# Global variables
model = None
camera = None
camera_active = False
current_frame = None
frame_lock = threading.Lock()
latest_detection_results = {
    'compliance_status': 'UNKNOWN',
    'people_count': 0,
    'detected_classes': [],
    'missing_ppe': []
}

def load_model():
    """Load the YOLO model with error handling"""
    global model
    try:
        model_path = PROJECT_ROOT / MODEL_CONFIG["ppe_model_path"]
        if not model_path.exists():
            print(f"[ERROR] Model file not found: {model_path}")
            return False
        
        model = YOLO(str(model_path))
        print(f"[SUCCESS] Model loaded: {model_path}")
        print(f"[INFO] Classes: {model.names}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def detect_ppe(image):
    """Detect PPE in the given image"""
    if model is None:
        return None
    
    try:
        results = model(image, conf=MODEL_CONFIG["confidence_threshold"])
        return results[0] if results else None
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return None

def analyze_compliance(detections):
    """Analyze PPE compliance based on detections - More practical approach"""
    if not detections:
        return {
            'compliance_status': 'UNKNOWN',
            'people_count': 0,
            'detected_classes': [],
            'missing_ppe': []
        }
    
    # Extract detected classes
    detected_classes = []
    if hasattr(detections, 'boxes') and detections.boxes is not None:
        for box in detections.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_classes.append({
                'class': class_name,
                'confidence': confidence
            })
    
    # More practical compliance logic
    detected_class_names = [d['class'] for d in detected_classes]
    
    # If ANY PPE is detected, consider it compliant
    if detected_class_names:
        compliance_status = 'COMPLIANT'
        missing_ppe = []
    else:
        compliance_status = 'NON-COMPLIANT'
        missing_ppe = ['No PPE detected']
    
    people_count = max(1, len(detected_classes))  # Estimate people count
    
    return {
        'compliance_status': compliance_status,
        'people_count': people_count,
        'detected_classes': detected_classes,
        'missing_ppe': missing_ppe
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('main/index.html')

@app.route('/live')
def live_monitoring():
    """Live monitoring page"""
    return render_template('live/live.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect PPE in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Detect PPE
        detections = detect_ppe(image)
        results = analyze_compliance(detections)
        
        return jsonify({
            'success': True,
            'detections': results['detected_classes'],
            'compliance_status': results['compliance_status'],
            'people_count': results['people_count'],
            'missing_ppe': results['missing_ppe']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera for live monitoring"""
    global camera, camera_active
    
    try:
        if camera_active:
            return jsonify({'status': 'success', 'message': 'Camera already active'})
        
        camera = cv2.VideoCapture(CAMERA_CONFIG["default_camera"])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["resolution"][0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["resolution"][1])
        camera.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        
        if not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Failed to open camera'})
        
        camera_active = True
        return jsonify({'status': 'success', 'message': 'Camera started'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera"""
    global camera, camera_active
    
    try:
        if camera:
            camera.release()
        camera = None
        camera_active = False
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/camera/status')
def camera_status():
    """Get camera status and latest detection results"""
    return jsonify({
        'active': camera_active,
        'compliance_status': latest_detection_results['compliance_status'],
        'people_count': latest_detection_results['people_count'],
        'detected_classes': latest_detection_results['detected_classes'],
        'missing_ppe': latest_detection_results['missing_ppe']
    })

@app.route('/video_feed')
def video_feed():
    """Video feed for live monitoring"""
    def generate_frames():
        global current_frame, latest_detection_results
        
        while camera_active and camera:
            success, frame = camera.read()
            if not success:
                break
            
            # Perform detection
            detections = detect_ppe(frame)
            results = analyze_compliance(detections)
            latest_detection_results = results
            
            # Draw detections on frame
            if detections and hasattr(detections, 'boxes') and detections.boxes is not None:
                for box in detections.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                              (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Load model
    if load_model():
        print("[INFO] Starting PPE Detection System...")
        app.run(
            host=WEB_CONFIG["host"],
            port=WEB_CONFIG["port"],
            debug=WEB_CONFIG["debug"],
            threaded=WEB_CONFIG["threaded"]
        )
    else:
        print("[ERROR] Failed to load model. Exiting...")
        sys.exit(1)
