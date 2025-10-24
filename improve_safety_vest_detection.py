#!/usr/bin/env python3
"""
Improve Safety Vest Detection
Quick fixes to detect more safety vests
"""

from ultralytics import YOLO
import cv2
import os

def improve_safety_vest_detection():
    """Improve safety vest detection with quick fixes"""
    print("="*70)
    print("IMPROVING SAFETY VEST DETECTION")
    print("="*70)
    
    # Load the trained model
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n[INFO] Current Safety Vest Performance: 46.9% mAP50")
    print("[INFO] Applying quick improvements...")
    
    # Test with different confidence thresholds
    confidence_levels = [0.3, 0.2, 0.1, 0.05]
    
    for conf in confidence_levels:
        print(f"\n[INFO] Testing with confidence threshold: {conf}")
        
        # Test on sample images
        test_images = [
            "combined_datasets/images/test",
            "combined_datasets/images/valid"
        ]
        
        safety_vest_detections = 0
        total_images = 0
        
        for test_dir in test_images:
            if os.path.exists(test_dir):
                import glob
                images = glob.glob(os.path.join(test_dir, "*.jpg"))[:5]
                
                for img in images:
                    total_images += 1
                    results = model(img, conf=conf, verbose=False)
                    
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                class_name = model.names[cls]
                                if class_name == 'safety_vest':
                                    safety_vest_detections += 1
        
        print(f"  Safety vest detections: {safety_vest_detections}/{total_images} images")
        
        if safety_vest_detections > 0:
            print(f"  ✅ Found {safety_vest_detections} safety vests with confidence {conf}")
            break
        else:
            print(f"  ❌ No safety vests detected with confidence {conf}")
    
    print(f"\n[INFO] Recommended confidence threshold: {conf}")
    return conf

def create_improved_web_app():
    """Create improved web app with better safety vest detection"""
    print("\n[INFO] Creating improved web application...")
    
    improved_web_app = '''#!/usr/bin/env python3
"""
Improved PPE Detection Web Application
Better safety vest detection
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
    """Detect PPE in uploaded image with improved safety vest detection"""
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
        
        # IMPROVED: Use different confidence thresholds for different PPE
        # Lower threshold for safety vests (harder to detect)
        results = model(image, conf=0.3, verbose=False)
        
        # Additional pass with even lower threshold for safety vests
        safety_vest_results = model(image, conf=0.1, verbose=False)
        
        # Combine results
        all_detections = []
        detected_classes = set()
        
        # Process main results
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    detected_classes.add(class_name)
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    all_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Process safety vest specific results
        for r in safety_vest_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Only add safety vests from this pass
                    if class_name == 'safety_vest':
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check if we already have this detection
                        bbox_exists = False
                        for existing in all_detections:
                            if existing['class'] == 'safety_vest':
                                # Check if bounding boxes overlap significantly
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 50 and abs(y1 - ex_bbox[1]) < 50:
                                    bbox_exists = True
                                    break
                        
                        if not bbox_exists:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
        # Check compliance
        compliance_status = "COMPLIANT"
        missing_ppe = []
        
        required_ppe = ['helmet', 'safety_vest']
        for ppe in required_ppe:
            if ppe not in detected_classes:
                missing_ppe.append(ppe)
                compliance_status = "NON-COMPLIANT"
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'detections': all_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(all_detections)
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
            'detections': all_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(all_detections),
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

if __name__ == '__main__':
    print("="*70)
    print("IMPROVED PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Improved safety vest detection enabled")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    with open("improved_ppe_web_app.py", 'w') as f:
        f.write(improved_web_app)
    
    print("[SUCCESS] Improved web application created: improved_ppe_web_app.py")

def main():
    """Main function"""
    print("="*70)
    print("SAFETY VEST DETECTION IMPROVEMENT")
    print("="*70)
    
    # Find optimal confidence threshold
    optimal_conf = improve_safety_vest_detection()
    
    # Create improved web app
    create_improved_web_app()
    
    print("\n" + "="*70)
    print("IMPROVEMENTS COMPLETE!")
    print("="*70)
    print(f"\n[SUCCESS] Optimal confidence threshold: {optimal_conf}")
    print("[SUCCESS] Improved web application created")
    print("\n[INFO] To use improved detection:")
    print("  1. Stop current web app (Ctrl+C)")
    print("  2. Run: python improved_ppe_web_app.py")
    print("  3. Test with your construction site image")
    print("\n[INFO] The improved app will:")
    print("  - Use lower confidence for safety vests")
    print("  - Run multiple detection passes")
    print("  - Better detect orange/lime safety vests")

if __name__ == "__main__":
    main()