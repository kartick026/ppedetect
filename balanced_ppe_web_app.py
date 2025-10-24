#!/usr/bin/env python3
"""
Balanced PPE Detection Web Application
Maintains good helmet detection while fixing other PPE items
"""

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model
model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
model = YOLO(model_path)

# Store detection history
detection_history = []

def enhance_for_goggles(image):
    """Enhance image specifically for goggles/glasses detection"""
    # Convert to grayscale for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    # Apply sharpening for better detail
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def enhance_for_gloves(image):
    """Enhance image specifically for gloves detection"""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow/orange colors (common glove colors)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Enhance saturation and brightness for glove colors
    hsv[:,:,1] = np.where(yellow_mask > 0, np.minimum(hsv[:,:,1] * 1.5, 255), hsv[:,:,1])
    hsv[:,:,2] = np.where(yellow_mask > 0, np.minimum(hsv[:,:,2] * 1.2, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def enhance_for_safety_vests(image):
    """Enhance image specifically for safety vests"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Orange color range
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Enhance saturation and brightness for safety vest colors
    hsv[:,:,1] = np.where(orange_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
    hsv[:,:,2] = np.where(orange_mask > 0, np.minimum(hsv[:,:,2] * 1.2, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def merge_detections_balanced(all_detections):
    """Balanced merging to maintain good helmet detection"""
    if not all_detections:
        return []
    
    # Separate by class
    class_detections = {}
    for det in all_detections:
        class_name = det['class']
        if class_name not in class_detections:
            class_detections[class_name] = []
        class_detections[class_name].append(det)
    
    merged_detections = []
    
    for class_name, detections in class_detections.items():
        if not detections:
            continue
            
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Different merging strategies for different classes
        if class_name == 'helmet':
            # For helmets, be more conservative - keep top 2-3 detections
            # This maintains the good helmet detection we had before
            merged_detections.extend(detections[:2])  # Keep top 2 helmet detections
            
        elif class_name == 'safety_vest':
            # For safety vests, be more aggressive in merging (they tend to be detected multiple times)
            # Keep only the best detection
            merged_detections.append(detections[0])
            
        else:  # goggles, gloves
            # For other classes, merge overlapping detections
            merged_class_detections = []
            used_indices = set()
            
            for i, det1 in enumerate(detections):
                if i in used_indices:
                    continue
                    
                similar_detections = [det1]
                used_indices.add(i)
                
                for j, det2 in enumerate(detections[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    # Check overlap
                    if bbox_overlap(det1['bbox'], det2['bbox']) > 0.3:
                        similar_detections.append(det2)
                        used_indices.add(j)
                
                # Keep the best detection from similar ones
                best_detection = max(similar_detections, key=lambda x: x['confidence'])
                merged_class_detections.append(best_detection)
            
            merged_detections.extend(merged_class_detections)
    
    return merged_detections

def bbox_overlap(bbox1, bbox2):
    """Calculate overlap between bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
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

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Balanced PPE detection maintaining good helmet detection"""
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
        
        # BALANCED DETECTION: Different strategies for different PPE
        all_detections = []
        detected_classes = set()
        
        # Strategy 1: Original image with balanced confidence
        # Higher confidence for helmets (they work well), lower for others
        print("[INFO] Strategy 1: Original image with balanced confidence")
        results1 = model(image, conf=0.3, verbose=False)  # Higher confidence for helmets
        for r in results1:
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
        
        # Strategy 2: Enhanced for goggles/glasses (low confidence)
        print("[INFO] Strategy 2: Enhanced for goggles")
        goggles_enhanced = enhance_for_goggles(image)
        results2 = model(goggles_enhanced, conf=0.1, verbose=False)  # Lower confidence for goggles
        for r in results2:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'goggles':  # Only add goggles from this pass
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'goggles':
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 50 and abs(y1 - ex_bbox[1]) < 50:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
        # Strategy 3: Enhanced for gloves (low confidence)
        print("[INFO] Strategy 3: Enhanced for gloves")
        gloves_enhanced = enhance_for_gloves(image)
        results3 = model(gloves_enhanced, conf=0.1, verbose=False)  # Lower confidence for gloves
        for r in results3:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'gloves':  # Only add gloves from this pass
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'gloves':
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 50 and abs(y1 - ex_bbox[1]) < 50:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
        # Strategy 4: Enhanced for safety vests (very low confidence)
        print("[INFO] Strategy 4: Enhanced for safety vests")
        vest_enhanced = enhance_for_safety_vests(image)
        results4 = model(vest_enhanced, conf=0.05, verbose=False)  # Very low confidence for vests
        for r in results4:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'safety_vest':  # Only add safety vests from this pass
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'safety_vest':
                                ex_bbox = existing['bbox']
                                if abs(x1 - ex_bbox[0]) < 100 and abs(y1 - ex_bbox[1]) < 100:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                            detected_classes.add(class_name)
        
        # Balanced merging
        final_detections = merge_detections_balanced(all_detections)
        
        # Update detected classes
        detected_classes = set(det['class'] for det in final_detections)
        
        # Check compliance
        compliance_status = "COMPLIANT"
        missing_ppe = []
        
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        for ppe in required_ppe:
            if ppe not in detected_classes:
                missing_ppe.append(ppe)
                compliance_status = "NON-COMPLIANT"
        
        # Save detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'image_name': file.filename,
            'detections': final_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(final_detections)
        }
        detection_history.append(detection_record)
        
        # Save annotated image
        if results1:
            annotated_image = results1[0].plot()
            output_path = f"static/detections/detection_{len(detection_history)}.jpg"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            detection_record['annotated_image'] = output_path
        
        print(f"[INFO] Final detections: {len(final_detections)}")
        for det in final_detections:
            print(f"  {det['class']}: {det['confidence']:.2f}")
        
        return jsonify({
            'success': True,
            'detections': final_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(final_detections),
            'annotated_image': detection_record.get('annotated_image', '')
        })
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
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
    print("BALANCED PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Balanced detection maintaining good helmet detection")
    print(f"[INFO] Different confidence levels for different PPE")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
