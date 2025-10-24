#!/usr/bin/env python3
"""
PPE Detection Web Application
Real-time PPE monitoring dashboard
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import json
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models for improved detection
person_model = YOLO('yolov8n.pt')  # For person detection
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
ppe_model = None

# Try to load custom PPE model
if os.path.exists(model_path):
    try:
        ppe_model = YOLO(model_path)
        print(f"[INFO] PPE model classes: {ppe_model.names}")
    except Exception as e:
        print(f"[WARNING] Could not load PPE model: {e}")
else:
    print(f"[WARNING] PPE model not found: {model_path}")

print(f"[INFO] Person model classes: {person_model.names}")

# Store detection history
detection_history = []

# Global variables for camera
camera = None
camera_lock = threading.Lock()
camera_active = False

def init_camera():
    """Initialize camera for real-time detection"""
    global camera
    try:
        camera = cv2.VideoCapture(0)  # Use default camera
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        return False

def detect_ppe_realtime(frame):
    """Detect PPE in real-time frame"""
    try:
        # Step 1: Detect people using pretrained model
        person_results = person_model(frame, conf=0.5, classes=[0])  # Higher confidence for people
        people_count = 0
        for r in person_results:
            if r.boxes is not None:
                people_count = len(r.boxes)
                break
        
        # If no people detected, assume 1 person but be more conservative with PPE detection
        if people_count == 0:
            people_count = 1
            print(f"[DEBUG] No people detected, assuming 1 person but being conservative")
        
        # Step 2: Try PPE detection with custom model
        detections = []
        detected_classes = set()
        
        if ppe_model is not None:
            ppe_results = ppe_model(frame, conf=0.3, verbose=False)
            for r in ppe_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = ppe_model.names[cls]
                        
                        if conf >= 0.3:
                            detected_classes.add(class_name)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
        
        # Step 3: Strict color-based PPE detection (fallback)
        if len(detections) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for safety vests (yellow/orange)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([20, 255, 255])
            
            # Create masks
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for safety vests
                        valid_contours.append(contour)
            
            # Check if we have enough valid regions
            if len(valid_contours) >= 2:
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                image_area = frame.shape[0] * frame.shape[1]
                coverage_ratio = total_area / image_area
                
                if coverage_ratio > 0.05:  # At least 5% of image
                    detected_classes.add('safety_vest')
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio")
                else:
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio - REJECTED")
            else:
                print(f"[DEBUG] Color detection: {len(valid_contours)} regions, 0.000 ratio - REJECTED")
        
        # Determine compliance status
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        # Require both helmet and safety vest for "PPE WORN"
        if 'helmet' in detected_classes and 'safety_vest' in detected_classes:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        
        return {
            'people_count': people_count,
            'compliance_status': compliance_status,
            'detected_classes': list(detected_classes),
            'total_detections': len(detections),
            'missing_ppe': missing_ppe,
            'detections': detections
        }
        
    except Exception as e:
        print(f"[ERROR] Real-time detection failed: {e}")
        return {
            'people_count': 1,
            'compliance_status': "PPE NOT WORN",
            'detected_classes': [],
            'total_detections': 0,
            'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
            'detections': []
        }

def detect_ppe(image):
    """Enhanced PPE detection with improved accuracy"""
    try:
        # Step 1: Detect people using pretrained model
        person_results = person_model(image, conf=0.5, classes=[0])  # Higher confidence for people
        people_count = 0
        for r in person_results:
            if r.boxes is not None:
                people_count = len(r.boxes)
                break
        
        # If no people detected, assume 1 person but be more conservative with PPE detection
        if people_count == 0:
            people_count = 1
            print(f"[DEBUG] No people detected, assuming 1 person but being conservative")
        
        # Step 2: Try PPE detection with custom model
        detections = []
        detected_classes = set()
        
        if ppe_model is not None:
            ppe_results = ppe_model(image, conf=0.1)  # Very low confidence
            for r in ppe_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = ppe_model.names[cls]
                        
                        if conf >= 0.1:
                            detected_classes.add(class_name)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
        
        # Step 3: Strict color-based PPE detection (fallback)
        if len(detections) == 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for safety vests (yellow/orange)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([20, 255, 255])
            
            # Create masks
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for safety vests
                        valid_contours.append(contour)
            
            # Check if we have enough valid regions
            if len(valid_contours) >= 2:
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                image_area = image.shape[0] * image.shape[1]
                coverage_ratio = total_area / image_area
                
                if coverage_ratio > 0.05:  # At least 5% of image
                    detected_classes.add('safety_vest')
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio")
                else:
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio - REJECTED")
            else:
                print(f"[DEBUG] Color detection: {len(valid_contours)} regions, 0.000 ratio - REJECTED")
        
        # Determine compliance status
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        # Require both helmet and safety vest for "PPE WORN"
        if 'helmet' in detected_classes and 'safety_vest' in detected_classes:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        
        print(f"[DEBUG] Detected classes: {detected_classes}")
        print(f"[DEBUG] Total detections: {len(detections)}")
        print(f"[DEBUG] People count: {people_count}")
        print(f"[DEBUG] Compliance: {compliance_status}")
        
        return {
            'people_count': people_count,
            'compliance_status': compliance_status,
            'detected_classes': list(detected_classes),
            'total_detections': len(detections),
            'missing_ppe': missing_ppe,
            'detections': detections
        }
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return {
            'people_count': 1,
            'compliance_status': "PPE NOT WORN",
            'detected_classes': [],
            'total_detections': 0,
            'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
            'detections': []
        }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/live')
def live():
    """Live monitoring page"""
    return render_template('live.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect PPE in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Run detection
        result = detect_ppe(image)
        
        # Store in history
        detection_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        # Keep only last 100 detections
        if len(detection_history) > 100:
            detection_history.pop(0)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Detection endpoint failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start real-time camera feed"""
    global camera, camera_active
    
    try:
        if not camera_active:
            if init_camera():
                camera_active = True
                return jsonify({'status': 'success', 'message': 'Camera started'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to initialize camera'})
        else:
            return jsonify({'status': 'success', 'message': 'Camera already active'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop real-time camera feed"""
    global camera, camera_active
    
    try:
        if camera_active and camera is not None:
            camera.release()
            camera = None
            camera_active = False
            return jsonify({'status': 'success', 'message': 'Camera stopped'})
        else:
            return jsonify({'status': 'success', 'message': 'Camera was not active'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/camera/feed')
def camera_feed():
    """Generate camera feed"""
    def generate_frames():
        global camera, camera_active
        
        while camera_active and camera is not None:
            try:
                with camera_lock:
                    success, frame = camera.read()
                    if not success:
                        break
                
                # Run detection on frame
                result = detect_ppe_realtime(frame)
                
                # Draw bounding boxes and labels
                for detection in result['detections']:
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detection['class']}: {detection['confidence']:.2f}", 
                              (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add status text
                status_text = f"People: {result['people_count']} | Status: {result['compliance_status']}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"[ERROR] Camera feed error: {e}")
                break
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    """Get detection history"""
    return jsonify(detection_history)

if __name__ == '__main__':
    print("="*70)
    print("PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Person model loaded: yolov8n.pt")
    if ppe_model:
        print(f"[INFO] PPE model loaded: {model_path}")
    else:
        print(f"[INFO] PPE model: Using color-based detection")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
