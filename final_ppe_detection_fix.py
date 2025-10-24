#!/usr/bin/env python3
"""
Final PPE Detection Fix
Comprehensive solution for multiple people and all detection issues
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime

class FinalPPEDetectionFix:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize final PPE detection fix"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        print("="*70)
        print("FINAL PPE DETECTION FIX")
        print("="*70)
        print(f"[INFO] Model loaded: {model_path}")
        print(f"[INFO] Classes: {self.model.names}")
        print(f"[INFO] Fixing: Multiple people, All PPE items, Accuracy")
    
    def create_final_web_app(self):
        """Create final comprehensive web app"""
        print("\n[INFO] Creating final comprehensive web application...")
        
        final_app = '''#!/usr/bin/env python3
"""
Final PPE Detection Web Application
Comprehensive solution for all detection issues
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

def enhance_for_all_ppe(image):
    """Comprehensive enhancement for all PPE types"""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for different PPE colors
    # Orange/yellow for safety vests
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Grey for gloves
    lower_grey = np.array([0, 0, 50])
    upper_grey = np.array([180, 30, 200])
    grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)
    
    # Combine all PPE color masks
    ppe_mask = cv2.bitwise_or(orange_mask, cv2.bitwise_or(yellow_mask, grey_mask))
    
    # Enhance saturation and brightness for PPE colors
    hsv[:,:,1] = np.where(ppe_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
    hsv[:,:,2] = np.where(ppe_mask > 0, np.minimum(hsv[:,:,2] * 1.3, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def enhance_for_goggles(image):
    """Enhanced goggles/glasses detection"""
    # Convert to grayscale for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    # Apply sharpening for better detail
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return enhanced

def enhance_for_helmets(image):
    """Enhanced helmet detection"""
    # Convert to LAB color space for better brightness control
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def merge_detections_final(all_detections):
    """Final comprehensive merging strategy"""
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
        
        # For all classes, use conservative merging
        # Keep top detections with good confidence
        if class_name == 'helmet':
            # Keep top 2 helmet detections (for multiple people)
            for det in detections[:2]:
                if det['confidence'] > 0.3:  # Minimum confidence
                    merged_detections.append(det)
        elif class_name == 'safety_vest':
            # Keep top 2 safety vest detections (for multiple people)
            for det in detections[:2]:
                if det['confidence'] > 0.2:  # Lower threshold for vests
                    merged_detections.append(det)
        elif class_name == 'goggles':
            # Keep top 2 goggles detections (for multiple people)
            for det in detections[:2]:
                if det['confidence'] > 0.1:  # Very low threshold for goggles
                    merged_detections.append(det)
        elif class_name == 'gloves':
            # Keep top 2 gloves detections (for multiple people)
            for det in detections[:2]:
                if det['confidence'] > 0.1:  # Very low threshold for gloves
                    merged_detections.append(det)
        else:
            # For other classes, keep best detection
            if detections[0]['confidence'] > 0.3:
                merged_detections.append(detections[0])
    
    return merged_detections

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Final comprehensive PPE detection"""
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
        
        # FINAL COMPREHENSIVE DETECTION
        all_detections = []
        detected_classes = set()
        
        # Strategy 1: Original image with very low confidence
        print("[INFO] Strategy 1: Original image (conf=0.01)")
        results1 = model(image, conf=0.01, verbose=False)  # Very low confidence
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
        
        # Strategy 2: Enhanced for all PPE
        print("[INFO] Strategy 2: Enhanced for all PPE")
        ppe_enhanced = enhance_for_all_ppe(image)
        results2 = model(ppe_enhanced, conf=0.01, verbose=False)  # Very low confidence
        for r in results2:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Check for duplicates
                    is_duplicate = False
                    for existing in all_detections:
                        if existing['class'] == class_name:
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
        
        # Strategy 3: Enhanced for goggles
        print("[INFO] Strategy 3: Enhanced for goggles")
        goggles_enhanced = enhance_for_goggles(image)
        results3 = model(goggles_enhanced, conf=0.01, verbose=False)  # Very low confidence
        for r in results3:
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
        
        # Strategy 4: Enhanced for helmets
        print("[INFO] Strategy 4: Enhanced for helmets")
        helmet_enhanced = enhance_for_helmets(image)
        results4 = model(helmet_enhanced, conf=0.01, verbose=False)  # Very low confidence
        for r in results4:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'helmet':  # Only add helmets from this pass
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == 'helmet':
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
        
        # Final comprehensive merging
        final_detections = merge_detections_final(all_detections)
        
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
    print("FINAL PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Final comprehensive detection")
    print(f"[INFO] Multiple people support")
    print(f"[INFO] All PPE types detection")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        with open("final_ppe_web_app.py", 'w') as f:
            f.write(final_app)
        
        print("[SUCCESS] Final web application created: final_ppe_web_app.py")

def main():
    """Main function"""
    print("="*70)
    print("FINAL PPE DETECTION FIX")
    print("="*70)
    
    try:
        # Initialize fixer
        fixer = FinalPPEDetectionFix()
        
        # Create final web app
        fixer.create_final_web_app()
        
        print("\n" + "="*70)
        print("FINAL DETECTION FIX COMPLETE!")
        print("="*70)
        print(f"\n[SUCCESS] Final comprehensive detection created")
        print(f"[SUCCESS] Multiple people support")
        print(f"[SUCCESS] All PPE types detection")
        print(f"[SUCCESS] Very low confidence thresholds")
        
        print(f"\n[INFO] To use final detection:")
        print("  1. Stop current web app (Ctrl+C)")
        print("  2. Run: python final_ppe_web_app.py")
        print("  3. Test with your construction site image")
        
        print(f"\n[INFO] Final fix features:")
        print("  - Multiple people detection (2+ workers)")
        print("  - All PPE types: helmet, safety_vest, goggles, gloves")
        print("  - Very low confidence (0.01) for maximum detection")
        print("  - Comprehensive enhancement for all PPE")
        print("  - Smart merging for multiple detections")
        
    except Exception as e:
        print(f"[ERROR] Fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
