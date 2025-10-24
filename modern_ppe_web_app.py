#!/usr/bin/env python3
"""
Modern PPE Detection Web Application
Rive-style frontend with real-time detection and people counting
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

def detect_ppe_modern(image):
    """Modern PPE detection with people counting"""
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
        
        # Smart people counting based on detection patterns
        people_count = estimate_people_count(detections)
        
        # Determine compliance
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
        
        # If we detect people but no PPE, they're non-compliant
        if people_count > 0 and len(detected_classes) == 0:
            compliance_status = "NON-COMPLIANT"
        elif people_count > 0 and len(missing_ppe) > 0:
            compliance_status = "NON-COMPLIANT"
        elif people_count > 0 and len(missing_ppe) == 0:
            compliance_status = "COMPLIANT"
        else:
            compliance_status = "NO_PEOPLE"
        
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

def estimate_people_count(detections):
    """Smart people counting based on detection patterns"""
    if not detections:
        return 0
    
    # Group detections by spatial proximity
    groups = []
    
    for detection in detections:
        added_to_group = False
        
        # Try to add to existing group
        for group in groups:
            group_center = group['center']
            detection_center = {
                'x': (detection['bbox'][0] + detection['bbox'][2]) / 2,
                'y': (detection['bbox'][1] + detection['bbox'][3]) / 2
            }
            
            distance = np.sqrt(
                (group_center['x'] - detection_center['x'])**2 + 
                (group_center['y'] - detection_center['y'])**2
            )
            
            # If close enough, add to this group
            if distance < 150:  # 150 pixel threshold
                group['detections'].append(detection)
                group['center'] = {
                    'x': (group['center']['x'] * len(group['detections']) + detection_center['x']) / (len(group['detections']) + 1),
                    'y': (group['center']['y'] * len(group['detections']) + detection_center['y']) / (len(group['detections']) + 1)
                }
                added_to_group = True
                break
        
        # If not added to any group, create new group
        if not added_to_group:
            groups.append({
                'detections': [detection],
                'center': {
                    'x': (detection['bbox'][0] + detection['bbox'][2]) / 2,
                    'y': (detection['bbox'][1] + detection['bbox'][3]) / 2
                }
            })
    
    return len(groups)

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
                    detection_result = detect_ppe_modern(frame)
                    
                    # Update latest results
                    latest_detection_results = detection_result
                    
                    # Draw detections on frame
                    annotated_frame = frame.copy()
                    
                    for detection in detection_result['detections']:
                        x1, y1, x2, y2 = detection['bbox']
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        # Draw bounding box
                        color = (0, 255, 0) if detection_result['compliance_status'] == 'COMPLIANT' else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add status overlay
                    status_color = (0, 255, 0) if detection_result['compliance_status'] == 'COMPLIANT' else (0, 0, 255)
                    status_text = f"People: {detection_result['people_count']} | Status: {detection_result['compliance_status']}"
                    cv2.putText(annotated_frame, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
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
    """Main page with Rive-style design"""
    return render_template('rive_ppe_frontend.html')

@app.route('/live')
def live_monitoring():
    """Live monitoring page"""
    return render_template('modern_live.html')

@app.route('/logo')
def logo_display():
    """Cool logo display page"""
    return render_template('logo.html')

@app.route('/cool-helmet')
def cool_helmet_logo():
    """Cool helmet logo display page"""
    return render_template('cool_helmet_logo.html')

@app.route('/perfect-logo')
def perfect_logo_placement():
    """Perfect logo placement page"""
    return render_template('perfect_logo_placement.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """PPE detection endpoint"""
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
        result = detect_ppe_modern(image)
        
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
        
        print(f"[INFO] Detection completed: {result['total_detections']} objects found, {result['people_count']} people")
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

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera for live monitoring"""
    global camera_active, camera
    
    try:
        if not camera_active:
            camera = cv2.VideoCapture(0)
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
    print("MODERN PPE DETECTION WEB APPLICATION")
    print("="*70)
    
    # Load model
    if load_model():
        print("[SUCCESS] Model loaded successfully")
    else:
        print("[WARNING] Model loading failed - detection will not work")
    
    print("[INFO] Modern Rive-style frontend enabled")
    print("[INFO] Real-time camera monitoring enabled")
    print("[INFO] Smart people counting enabled")
    print("[INFO] Starting web server...")
    print("[INFO] Main page: http://localhost:5000")
    print("[INFO] Live monitoring: http://localhost:5000/live")
    print("="*70)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
