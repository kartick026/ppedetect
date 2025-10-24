#!/usr/bin/env python3
"""
Enhanced PPE Detection Web Application with Real-time Camera Feed
Beautiful Rive-style frontend with live monitoring capabilities
"""

from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import time
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

# Store detection history
detection_history = []

# Camera and streaming variables
camera_active = False
camera = None
current_frame = None
frame_lock = threading.Lock()
latest_detection_results = {
    'compliance_status': 'UNKNOWN',
    'people_count': 0,
    'detected_classes': [],
    'missing_ppe': []
}

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def merge_close_detections(detections, iou_threshold=0.5):
    """Merge detections that are too close together (same class and high IoU)"""
    if not detections:
        return detections
    
    # Group by class
    class_groups = {}
    for det in detections:
        class_name = det['class']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(det)
    
    merged_detections = []
    
    for class_name, class_detections in class_groups.items():
        # Sort by confidence (highest first)
        class_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS within each class
        keep = []
        for i, det in enumerate(class_detections):
            should_keep = True
            for kept_idx in keep:
                kept_det = class_detections[kept_idx]
                iou = calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(i)
        
        # Add the kept detections
        for idx in keep:
            merged_detections.append(class_detections[idx])
    
    return merged_detections

def detect_ppe_enhanced(image):
    """
    Enhanced PPE detection with better accuracy
    """
    try:
        # Run YOLO detection
        results = model(image, conf=0.3, verbose=False)
        
        detections = []
        detected_classes = set()
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    detections.append(detection)
                    detected_classes.add(class_name)
        
        # Merge close detections
        detections = merge_close_detections(detections)
        
        # Determine compliance
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
        
        compliance_status = "COMPLIANT" if len(missing_ppe) == 0 else "NON-COMPLIANT"
        
        # Estimate people count (simplified - count unique detection groups)
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
        print(f"[ERROR] Enhanced detection failed: {e}")
        return {
            'detections': [],
            'compliance_status': 'ERROR',
            'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
            'total_detections': 0,
            'people_count': 0,
            'detected_classes': []
        }

def generate_frames():
    """Generate video frames for streaming"""
    global camera_active, camera, current_frame, frame_lock, latest_detection_results
    
    while camera_active:
        try:
            if camera is not None and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    with frame_lock:
                        current_frame = frame.copy()
                    
                    # Run detection on the frame
                    detection_result = detect_ppe_enhanced(frame)
                    
                    # Update latest results
                    latest_detection_results = detection_result
                    
                    # Draw detections on frame
                    annotated_frame = frame.copy()
                    
                    for detection in detection_result['detections']:
                        x1, y1, x2, y2 = detection['bbox']
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add compliance status overlay
                    status_color = (0, 255, 0) if detection_result['compliance_status'] == 'COMPLIANT' else (0, 0, 255)
                    status_text = f"Status: {detection_result['compliance_status']}"
                    cv2.putText(annotated_frame, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("[WARNING] Failed to read frame from camera")
                    time.sleep(0.1)
            else:
                print("[WARNING] Camera not available")
                time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] Frame generation error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Main page with beautiful Rive-style design"""
    return render_template('enhanced_index.html')

@app.route('/live')
def live_monitoring():
    """Live monitoring page"""
    return render_template('enhanced_live.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Enhanced PPE detection endpoint"""
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
        
        # Use enhanced detection method
        result = detect_ppe_enhanced(image)
        
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
        
        # Save annotated image
        results = model(image, conf=0.3, verbose=False)
        if results:
            annotated_image = results[0].plot()
            output_path = f"static/detections/detection_{len(detection_history)}.jpg"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            detection_record['annotated_image'] = output_path
        
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
            'annotated_image': detection_record.get('annotated_image', '')
        })
        
    except Exception as e:
        print(f"[ERROR] Detection endpoint failed: {e}")
        return jsonify({'error': 'Detection failed: ' + str(e)}), 500

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera for live monitoring"""
    global camera_active, camera
    
    try:
        if not camera_active:
            camera = cv2.VideoCapture(0)  # Use default camera
            if camera.isOpened():
                camera_active = True
                print("[INFO] Camera started successfully")
                return jsonify({'status': 'success', 'message': 'Camera started'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to open camera'})
        else:
            return jsonify({'status': 'success', 'message': 'Camera already active'})
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera"""
    global camera_active, camera
    
    try:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None
        print("[INFO] Camera stopped")
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    except Exception as e:
        print(f"[ERROR] Failed to stop camera: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/camera/status')
def camera_status():
    """Get camera status and latest detection results"""
    return jsonify({
        'active': camera_active,
        'compliance_status': latest_detection_results.get('compliance_status', 'UNKNOWN'),
        'people_count': latest_detection_results.get('people_count', 0),
        'detected_classes': latest_detection_results.get('detected_classes', []),
        'missing_ppe': latest_detection_results.get('missing_ppe', [])
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

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
    print("ENHANCED PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Real-time camera monitoring enabled")
    print(f"[INFO] Beautiful Rive-style frontend")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Main page: http://localhost:5000")
    print(f"[INFO] Live monitoring: http://localhost:5000/live")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
