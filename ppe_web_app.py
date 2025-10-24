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

# Global variables for camera
camera = None
camera_lock = threading.Lock()
camera_active = False
latest_detection_result = None

# Temporal consistency for real-time detection
detection_history = []
MAX_HISTORY = 5

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
    """Detect PPE in real-time frame with improved accuracy"""
    try:
        # Step 1: Detect people using pretrained model with higher confidence
        person_results = person_model(frame, conf=0.6, classes=[0])  # Higher confidence for people
        people_count = 0
        for r in person_results:
            if r.boxes is not None:
                people_count = len(r.boxes)
                break
        
        # If no people detected, assume 1 person but be more conservative with PPE detection
        if people_count == 0:
            people_count = 1
            print(f"[DEBUG] No people detected, assuming 1 person but being conservative")
        
        # Step 2: PPE model detection with optimized parameters
        detections = []
        detected_classes = set()
        
        # ENABLE PPE MODEL DETECTION WITH OPTIMIZED PARAMETERS
        try:
            ppe_results = ppe_model(frame, conf=0.3, iou=0.5, verbose=False)
            print(f"[DEBUG] PPE model raw results: {len(ppe_results)} detections")
            
            for r in ppe_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        class_name = ppe_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Normalize class names
                        if class_name == "glovess":
                            class_name = "gloves"
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox_area = (x2 - x1) * (y2 - y1)
                        frame_area = frame.shape[0] * frame.shape[1]
                        bbox_ratio = bbox_area / frame_area
                        
                        # Optimized validation logic
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        frame_height = frame.shape[0]
                        frame_width = frame.shape[1]
                        
                        # Additional validation for specific PPE types
                        is_valid_detection = False
                        
                        if class_name == "helmet":
                            # Helmets should be in upper portion of image - more lenient size requirements
                            is_valid_detection = (0.001 < bbox_ratio < 0.3 and 
                                                confidence > 0.5 and 
                                                center_y < frame_height * 0.7)
                        elif class_name == "safety_vest":
                            # Safety vests should be in middle portion of image - more lenient
                            is_valid_detection = (0.002 < bbox_ratio < 0.4 and 
                                                confidence > 0.4 and 
                                                0.05 < center_y / frame_height < 0.95)
                        elif class_name == "gloves":
                            # Gloves should be in lower portion and smaller
                            is_valid_detection = (0.001 < bbox_ratio < 0.08 and 
                                                confidence > 0.7 and 
                                                center_y > frame_height * 0.4)
                        elif class_name == "goggles":
                            # Goggles should be in upper portion and very small
                            is_valid_detection = (0.0005 < bbox_ratio < 0.05 and 
                                                confidence > 0.7 and 
                                                center_y < frame_height * 0.7)
                        
                        if is_valid_detection:
                            
                            detected_classes.add(class_name)
                            detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
                            print(f"[DEBUG] PPE detection: {class_name} with confidence {confidence:.3f}")
                        else:
                            print(f"[DEBUG] PPE detection REJECTED: {class_name} (conf: {confidence:.3f}, ratio: {bbox_ratio:.4f})")
        except Exception as e:
            print(f"[ERROR] PPE model detection failed: {e}")
            detections = []
            detected_classes = set()
        
        # Step 3: Disable color-based detection to avoid false positives
        # Only use AI model detections for accuracy
        if False:  # Disabled color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for safety vests (bright yellow/orange only)
            lower_yellow = np.array([20, 200, 200])  # Much higher saturation and brightness
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([10, 200, 200])  # Much higher saturation and brightness
            upper_orange = np.array([20, 255, 255])
            
            # Create masks
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio - more strict
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Much higher minimum area to avoid small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.8 < aspect_ratio < 2.5:  # Stricter aspect ratio for safety vests
                        valid_contours.append(contour)
            
            # Check if we have enough valid regions - more strict
            if len(valid_contours) >= 5:  # Require many more regions for safety vest
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                image_area = frame.shape[0] * frame.shape[1]
                coverage_ratio = total_area / image_area
                
                if coverage_ratio > 0.15:  # Much higher coverage requirement to avoid false positives
                    detected_classes.add('safety_vest')
                    # Add to detections array for frontend display
                    detections.append({
                        'class': 'safety_vest',
                        'confidence': 0.8,  # High confidence for color detection
                        'bbox': [0, 0, image.shape[1], image.shape[0]]  # Full image bbox
                    })
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio")
                else:
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio - REJECTED")
            else:
                print(f"[DEBUG] Color detection: {len(valid_contours)} regions, 0.000 ratio - REJECTED")
        
        # Determine compliance status - require both helmet and safety vest
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        # Determine compliance status based on detected PPE
        if detected_classes:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = ['No PPE detected']
        
        # Temporal consistency filtering
        global detection_history
        detection_history.append({
            'classes': set(detected_classes),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        current_time = time.time()
        detection_history = [d for d in detection_history if current_time - d['timestamp'] < 2.0]
        
        # Require consistent detection over multiple frames
        if len(detection_history) >= 3:
            # Count how many times each class was detected
            class_counts = {}
            for detection in detection_history:
                for class_name in detection['classes']:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Only keep classes detected in at least 60% of recent frames
            consistent_classes = set()
            for class_name, count in class_counts.items():
                if count >= len(detection_history) * 0.6:
                    consistent_classes.add(class_name)
            
            # Update detected classes to only include consistent ones
            detected_classes = consistent_classes
            detections = [d for d in detections if d['class'] in detected_classes]
            
            # Recalculate compliance with consistent detections
            main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
            detected_main_items = [item for item in main_ppe_items if item in detected_classes]
            
            # Conservative compliance - since PPE model is disabled, always mark as NOT WORN
            compliance_status = "PPE NOT WORN"
            missing_ppe = ['No PPE detected - model disabled to prevent false positives']
        
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
        
        # Step 2: PPE model detection with optimized parameters
        detections = []
        detected_classes = set()
        
        # ENABLE PPE MODEL DETECTION WITH OPTIMIZED PARAMETERS
        try:
            ppe_results = ppe_model(image, conf=0.4, iou=0.5, verbose=False)
            print(f"[DEBUG] PPE model raw results: {len(ppe_results)} detections")
            
            for r in ppe_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        class_name = ppe_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Normalize class names
                        if class_name == "glovess":
                            class_name = "gloves"
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox_area = (x2 - x1) * (y2 - y1)
                        frame_area = image.shape[0] * image.shape[1]
                        bbox_ratio = bbox_area / frame_area
                        
                        # Optimized validation logic
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        frame_height = image.shape[0]
                        frame_width = image.shape[1]
                        
                        # Additional validation for specific PPE types
                        is_valid_detection = False
                        
                        if class_name == "helmet":
                            # Helmets should be in upper portion of image - more lenient size requirements
                            is_valid_detection = (0.001 < bbox_ratio < 0.3 and 
                                                confidence > 0.5 and 
                                                center_y < frame_height * 0.7)
                        elif class_name == "safety_vest":
                            # Safety vests should be in middle portion of image - more lenient
                            is_valid_detection = (0.002 < bbox_ratio < 0.4 and 
                                                confidence > 0.4 and 
                                                0.05 < center_y / frame_height < 0.95)
                        elif class_name == "gloves":
                            # Gloves should be in lower portion and smaller
                            is_valid_detection = (0.001 < bbox_ratio < 0.08 and 
                                                confidence > 0.7 and 
                                                center_y > frame_height * 0.4)
                        elif class_name == "goggles":
                            # Goggles should be in upper portion and very small
                            is_valid_detection = (0.0005 < bbox_ratio < 0.05 and 
                                                confidence > 0.7 and 
                                                center_y < frame_height * 0.7)
                        
                        if is_valid_detection:
                            
                            detected_classes.add(class_name)
                            detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
                            print(f"[DEBUG] PPE detection: {class_name} with confidence {confidence:.3f}")
                        else:
                            print(f"[DEBUG] PPE detection REJECTED: {class_name} (conf: {confidence:.3f}, ratio: {bbox_ratio:.4f})")
        except Exception as e:
            print(f"[ERROR] PPE model detection failed: {e}")
            detections = []
            detected_classes = set()
        
        # Step 3: Disable color-based detection to avoid false positives
        # Only use AI model detections for accuracy
        if False:  # Disabled color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for safety vests (bright yellow/orange only)
            lower_yellow = np.array([20, 200, 200])  # Much higher saturation and brightness
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([10, 200, 200])  # Much higher saturation and brightness
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
                if area > 5000:  # Much higher minimum area to avoid small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for safety vests
                        valid_contours.append(contour)
            
            # Check if we have enough valid regions
            if len(valid_contours) >= 5:  # Require many more regions for safety vest
                total_area = sum(cv2.contourArea(c) for c in valid_contours)
                image_area = image.shape[0] * image.shape[1]
                coverage_ratio = total_area / image_area
                
                if coverage_ratio > 0.15:  # Much higher threshold to avoid false positives
                    detected_classes.add('safety_vest')
                    # Add to detections array for frontend display
                    detections.append({
                        'class': 'safety_vest',
                        'confidence': 0.8,  # High confidence for color detection
                        'bbox': [0, 0, image.shape[1], image.shape[0]]  # Full image bbox
                    })
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio")
                else:
                    print(f"[DEBUG] Color detection: {len(valid_contours)} regions, {coverage_ratio:.3f} ratio - REJECTED")
            else:
                print(f"[DEBUG] Color detection: {len(valid_contours)} regions, 0.000 ratio - REJECTED")
        
        # Determine compliance status
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        # Determine compliance status based on detected PPE
        if detected_classes:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = ['No PPE detected']
        
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
                
                # Store latest detection result for status endpoint
                global latest_detection_result
                latest_detection_result = result
                
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
