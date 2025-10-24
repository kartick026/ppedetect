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
        
        # If no people detected, assume 1 person but be more conservative with PPE detection
        if people_count == 0:
            people_count = 1
            print(f"[DEBUG] No people detected, assuming 1 person but being conservative")
        
        # Step 2: Detect PPE using custom model
        detections = []
        detected_classes = set()
        
        if ppe_model:
            ppe_results = ppe_model(frame, conf=0.3, verbose=False)
            for r in ppe_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = ppe_model.names[cls]
                        
                        if conf >= 0.3:
                            detected_classes.add(class_name)
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Validate bounding box size
                            bbox_area = (x2 - x1) * (y2 - y1)
                            frame_area = frame.shape[0] * frame.shape[1]
                            bbox_ratio = bbox_area / frame_area
                            
                            if 0.0001 < bbox_ratio < 0.8:  # More lenient size validation
                                detections.append({
                                    'class': class_name,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
        
        # Step 3: Color-based PPE detection (fallback)
        if len(detections) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for safety vests (more strict)
            yellow_lower = np.array([20, 150, 150])  # Higher saturation
            yellow_upper = np.array([30, 255, 255])
            orange_lower = np.array([10, 150, 150])   # Higher saturation
            orange_upper = np.array([20, 255, 255])
            green_lower = np.array([40, 150, 150])  # Higher saturation
            green_upper = np.array([80, 255, 255])
            
            # Create masks for each color
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Count bright colored regions
            bright_regions = 0
            total_area = 0
            for mask in [yellow_mask, orange_mask, green_mask]:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 5000:  # Larger area requirement
                        # Check aspect ratio (vest-like shape)
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.3 < aspect_ratio < 3.0:  # Vest-like aspect ratio
                            bright_regions += 1
                            total_area += area
                            print(f"[DEBUG] Found colored region: area={area}, aspect_ratio={aspect_ratio:.2f}")
            
            # Only add safety vest if we have enough regions and coverage
            if bright_regions >= 2 and total_area > 0:
                frame_area = frame.shape[0] * frame.shape[1]
                coverage_ratio = total_area / frame_area
                print(f"[DEBUG] Color detection: {bright_regions} regions, {coverage_ratio:.3f} ratio")
                
                if coverage_ratio > 0.05:  # At least 5% of image coverage
                    detected_classes.add('safety_vest')
                    print(f"[DEBUG] Color detection: ACCEPTED")
                else:
                    print(f"[DEBUG] Color detection: REJECTED")
        
        # Check if person is wearing PPE (strict logic)
        # Consider PPE worn ONLY if helmet AND safety vest are detected
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        # Strict logic: PPE worn ONLY if BOTH helmet AND safety vest detected
        if 'helmet' in detected_classes and 'safety_vest' in detected_classes:
            compliance_status = "PPE WORN"
            missing_ppe = []
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        
        # Draw bounding boxes and labels on frame
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = (0, 255, 0) if compliance_status == "PPE WORN" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{det['class']}: {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add status text
        status_color = (0, 255, 0) if compliance_status == "PPE WORN" else (0, 0, 255)
        cv2.putText(annotated_frame, f"Status: {compliance_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(annotated_frame, f"People: {people_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, compliance_status, people_count, len(detections)
        
    except Exception as e:
        print(f"[ERROR] Real-time detection failed: {e}")
        return frame, "ERROR", 0, 0

def generate_frames():
    """Generate frames for video streaming"""
    global camera, camera_active
    
    while camera_active:
        with camera_lock:
            if camera is None:
                break
                
            success, frame = camera.read()
            if not success:
                break
        
        if success:
            # Detect PPE in the frame
            annotated_frame, status, people_count, detections = detect_ppe_realtime(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

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
        
        # Step 1: Detect people using pretrained model
        person_results = person_model(image, conf=0.5, classes=[0])  # Higher confidence for people
        people_count = 0
        for r in person_results:
            if r.boxes is not None:
                people_count = len(r.boxes)
        
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
            
            # More restrictive color ranges for safety vests
            yellow_lower = np.array([20, 150, 150])  # Higher saturation and value
            yellow_upper = np.array([30, 255, 255])
            orange_lower = np.array([10, 150, 150])  # Higher saturation and value
            orange_upper = np.array([20, 255, 255])
            green_lower = np.array([40, 150, 150])   # Higher saturation and value
            green_upper = np.array([80, 255, 255])
            
            # Create masks for each color
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Count bright colored regions with stricter criteria
            bright_regions = 0
            total_bright_area = 0
            
            for mask, color_name in [(yellow_mask, 'yellow'), (orange_mask, 'orange'), (green_mask, 'green')]:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # Much higher area threshold and aspect ratio check
                    if area > 5000:  # Increased from 1000 to 5000
                        # Check aspect ratio to ensure it's vest-like (not too square, not too thin)
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # Safety vests should be roughly rectangular (not too square, not too thin)
                        if 0.3 < aspect_ratio < 3.0:
                            bright_regions += 1
                            total_bright_area += area
                            print(f"[DEBUG] Found {color_name} region: area={area}, aspect_ratio={aspect_ratio:.2f}")
            
            # Only detect safety vest if we have significant bright regions
            # AND they cover a reasonable portion of the image
            image_area = image.shape[0] * image.shape[1]
            bright_ratio = total_bright_area / image_area if image_area > 0 else 0
            
            if bright_regions >= 2 and bright_ratio > 0.05:  # At least 2 regions and 5% of image
                detected_classes.add('safety_vest')
                detections.append({
                    'class': 'safety_vest',
                    'confidence': 0.6,  # Higher confidence for color detection
                    'bbox': [0, 0, image.shape[1], image.shape[0]]  # Full image
                })
                print(f"[DEBUG] Color detection: {bright_regions} regions, {bright_ratio:.3f} ratio")
            else:
                print(f"[DEBUG] Color detection: {bright_regions} regions, {bright_ratio:.3f} ratio - REJECTED")
        
        # Check if person is wearing PPE (strict logic)
        # Consider PPE worn ONLY if helmet AND safety vest are detected
        main_ppe_items = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_main_items = [item for item in main_ppe_items if item in detected_classes]
        
        # Strict logic: PPE worn ONLY if helmet AND safety vest are detected
        # This prevents false positives from color detection alone
        if 'helmet' in detected_classes and 'safety_vest' in detected_classes:
            compliance_status = "PPE WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        else:
            compliance_status = "PPE NOT WORN"
            missing_ppe = [item for item in main_ppe_items if item not in detected_classes]
        
        # People count is already calculated above
        # No need to recalculate
        
        # Debug logging
        print(f"[DEBUG] Detected classes: {detected_classes}")
        print(f"[DEBUG] Total detections: {len(detections)}")
        print(f"[DEBUG] People count: {people_count}")
        print(f"[DEBUG] Compliance: {compliance_status}")
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'detections': detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(detections),
            'num_people': people_count
        }
        detection_history.append(detection_record)
        
        # Save annotated image (create a simple annotated version)
        try:
            annotated_image = image.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{det['class']}: {det['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            output_path = f"static/detections/detection_{len(detection_history)}.jpg"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            detection_record['annotated_image'] = output_path
        except Exception as e:
            print(f"[WARNING] Could not save annotated image: {e}")
        
        return jsonify({
            'success': True,
            'detections': detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(detections),
            'num_people': people_count,
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
                return jsonify({'status': 'error', 'message': 'Failed to initialize camera'}), 500
        else:
            return jsonify({'status': 'success', 'message': 'Camera already active'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop real-time camera feed"""
    global camera, camera_active
    
    try:
        camera_active = False
        if camera:
            camera.release()
            camera = None
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/camera/feed')
def camera_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/status')
def camera_status():
    """Get camera status"""
    return jsonify({
        'active': camera_active,
        'available': camera is not None
    })

if __name__ == '__main__':
    print("="*70)
    print("PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Person model loaded: yolov8n.pt")
    if ppe_model:
        print(f"[INFO] PPE model loaded: {model_path}")
    else:
        print(f"[INFO] PPE model: Using color-based detection")
    print(f"[INFO] Real-time camera support: ENABLED")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
