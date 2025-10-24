#!/usr/bin/env python3
"""
Clean PPE Detection Web Application
Separates live camera functionality from image upload detection
"""

import os
import cv2
import numpy as np
import time
import threading
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models
person_model = None
ppe_model = None

# Global variables for camera
camera = None
camera_lock = threading.Lock()
camera_active = False
latest_detection_result = None

def init_models():
    """Initialize YOLO models"""
    global person_model, ppe_model
    
    try:
        # Load person detection model
        person_model = YOLO('yolov8n.pt')
        print("[INFO] Person model loaded: yolov8n.pt")
        
        # Load PPE detection model
        model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
        if os.path.exists(model_path):
            ppe_model = YOLO(model_path)
            print(f"[INFO] PPE model loaded: {model_path}")
            print(f"[INFO] PPE model classes: {ppe_model.names}")
        else:
            print(f"[WARNING] PPE model not found at {model_path}")
            ppe_model = None
            
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

def detect_ppe_from_image(image):
    """Detect PPE in uploaded image"""
    try:
        # Step 1: Detect people
        person_results = person_model(image, conf=0.5, classes=[0])
        people_count = 0
        for r in person_results:
            if r.boxes is not None:
                people_count = len(r.boxes)
                break
        
        if people_count == 0:
            people_count = 1
            print("[DEBUG] No people detected, assuming 1 person")
        
        # Step 2: Detect PPE
        detections = []
        detected_classes = set()
        
        if ppe_model is not None:
            ppe_results = ppe_model(image, conf=0.7)
            for r in ppe_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = ppe_model.names[cls]
                        
                        if conf >= 0.7:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Basic size filtering
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            bbox_area = bbox_width * bbox_height
                            frame_area = image.shape[0] * image.shape[1]
                            bbox_ratio = bbox_area / frame_area
                            
                            if 0.01 < bbox_ratio < 0.5:
                                detected_classes.add(class_name)
                                detections.append({
                                    'class': class_name,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
        
        # Step 3: Color-based detection (fallback)
        if len(detections) == 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Yellow/orange safety vest detection
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([20, 255, 255])
            
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:
                        valid_contours.append(contour)
            
            if len(valid_contours) >= 2:
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                image_area = image.shape[0] * image.shape[1]
                coverage_ratio = total_area / image_area
                
                if coverage_ratio > 0.05:
                    detected_classes.add('safety_vest')
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio")
        
        # Determine compliance
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        if len(detected_main_items) >= 2:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        
        print(f"[DEBUG] Image - People: {people_count}, Classes: {detected_classes}, Compliance: {compliance_status}")
        
        return {
            'people_count': people_count,
            'compliance_status': compliance_status,
            'detected_classes': list(detected_classes),
            'total_detections': len(detections),
            'missing_ppe': missing_ppe,
            'detections': detections
        }
        
    except Exception as e:
        print(f"[ERROR] Image detection failed: {e}")
        return {
            'people_count': 1,
            'compliance_status': "PPE NOT WORN",
            'detected_classes': [],
            'total_detections': 0,
            'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
            'detections': []
        }

def detect_ppe_realtime(frame):
    """Detect PPE in real-time camera frame"""
    try:
        # Step 1: Detect people
        person_results = person_model(frame, conf=0.6, classes=[0])
        people_count = 0
        for r in person_results:
            if r.boxes is not None:
                people_count = len(r.boxes)
                break
        
        if people_count == 0:
            people_count = 1
            print("[DEBUG] No people detected in real-time, assuming 1 person")
        
        # Step 2: Detect PPE with higher confidence for real-time
        detections = []
        detected_classes = set()
        
        if ppe_model is not None:
            ppe_results = ppe_model(frame, conf=0.8)
            for r in ppe_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = ppe_model.names[cls]
                        
                        if conf >= 0.8:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Stricter filtering for real-time
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            bbox_area = bbox_width * bbox_height
                            frame_area = frame.shape[0] * frame.shape[1]
                            bbox_ratio = bbox_area / frame_area
                            
                            if 0.01 < bbox_ratio < 0.3:
                                detected_classes.add(class_name)
                                detections.append({
                                    'class': class_name,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
        
        # Step 3: Color-based detection (more conservative for real-time)
        if len(detections) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_yellow = np.array([20, 120, 120])
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([10, 120, 120])
            upper_orange = np.array([20, 255, 255])
            
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.8 < aspect_ratio < 2.5:
                        valid_contours.append(contour)
            
            if len(valid_contours) >= 3:
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                image_area = frame.shape[0] * frame.shape[1]
                coverage_ratio = total_area / image_area
                
                if coverage_ratio > 0.08:
                    detected_classes.add('safety_vest')
                    print(f"[DEBUG] Real-time color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio")
        
        # Determine compliance
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        if len(detected_main_items) >= 2:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        
        print(f"[DEBUG] Real-time - People: {people_count}, Classes: {detected_classes}, Compliance: {compliance_status}")
        
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

# Routes for image upload detection
@app.route('/')
def index():
    """Main page for image upload detection"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect PPE in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect PPE
        result = detect_ppe_from_image(image)
        
        return jsonify({
            'success': True,
            'people_count': result['people_count'],
            'compliance_status': result['compliance_status'],
            'detected_classes': result['detected_classes'],
            'total_detections': result['total_detections'],
            'missing_ppe': result['missing_ppe'],
            'detections': result['detections']
        })
        
    except Exception as e:
        print(f"[ERROR] Detection endpoint failed: {e}")
        return jsonify({'error': 'Detection failed: ' + str(e)}), 500

# Routes for live camera functionality
@app.route('/live')
def live():
    """Live camera monitoring page"""
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Generate camera feed"""
    def generate_frames():
        global camera, camera_active, latest_detection_result
        
        while camera_active and camera is not None:
            try:
                with camera_lock:
                    success, frame = camera.read()
                    if not success:
                        break
                
                # Run detection on frame
                result = detect_ppe_realtime(frame)
                latest_detection_result = result
                
                # Draw bounding boxes and labels
                for detection in result['detections']:
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detection['class']}: {detection['confidence']:.2f}", 
                              (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"[ERROR] Frame generation failed: {e}")
                break
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start real-time camera feed"""
    global camera, camera_active
    
    try:
        if not camera_active:
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                camera_active = True
                return jsonify({'status': 'success', 'message': 'Camera started'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to open camera'}), 500
        else:
            return jsonify({'status': 'success', 'message': 'Camera already active'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/camera/status')
def camera_status():
    """Get camera status and latest detection results"""
    global camera_active, camera, latest_detection_result
    
    if latest_detection_result is not None:
        return jsonify({
            'active': camera_active,
            'camera_available': camera is not None,
            'people_count': latest_detection_result.get('people_count', 0),
            'compliance_status': latest_detection_result.get('compliance_status', 'PPE NOT WORN'),
            'detected_classes': latest_detection_result.get('detected_classes', []),
            'total_detections': latest_detection_result.get('total_detections', 0),
            'missing_ppe': latest_detection_result.get('missing_ppe', [])
        })
    else:
        return jsonify({
            'active': camera_active,
            'camera_available': camera is not None,
            'people_count': 0,
            'compliance_status': 'PPE NOT WORN',
            'detected_classes': [],
            'total_detections': 0,
            'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves']
        })

if __name__ == '__main__':
    print("=" * 70)
    print("PPE DETECTION WEB APPLICATION")
    print("=" * 70)
    
    # Initialize models
    init_models()
    
    print("[INFO] Starting web server...")
    print("[INFO] Open: http://localhost:5000")
    print("=" * 70)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
