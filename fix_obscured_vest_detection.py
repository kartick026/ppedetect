#!/usr/bin/env python3
"""
Fix Obscured Safety Vest Detection
Specialized solution for partially hidden safety vests
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime

class ObscuredVestDetector:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize obscured vest detector"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        print("="*70)
        print("FIXING OBSCURED SAFETY VEST DETECTION")
        print("="*70)
        print(f"[INFO] Model loaded: {model_path}")
        print(f"[INFO] Classes: {self.model.names}")
    
    def detect_obscured_vests(self, image_path):
        """Detect safety vests even when partially obscured"""
        print(f"\n[INFO] Processing: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None
        
        all_detections = []
        
        # Strategy 1: Very low confidence for safety vests
        print("  [INFO] Strategy 1: Very low confidence (0.01)")
        results1 = self.model(image, conf=0.01, verbose=False)
        detections1 = self.extract_detections(results1)
        all_detections.extend(detections1)
        
        # Strategy 2: Enhanced contrast for reflective strips
        print("  [INFO] Strategy 2: Enhanced contrast for reflective strips")
        enhanced = self.enhance_reflective_strips(image)
        results2 = self.model(enhanced, conf=0.01, verbose=False)
        detections2 = self.extract_detections(results2)
        all_detections.extend(detections2)
        
        # Strategy 3: Color segmentation for orange/yellow
        print("  [INFO] Strategy 3: Color segmentation for orange/yellow")
        color_enhanced = self.enhance_safety_colors(image)
        results3 = self.model(color_enhanced, conf=0.01, verbose=False)
        detections3 = self.extract_detections(results3)
        all_detections.extend(detections3)
        
        # Strategy 4: Brightness boost for hidden vests
        print("  [INFO] Strategy 4: Brightness boost for hidden vests")
        bright = self.boost_brightness(image, 1.5)
        results4 = self.model(bright, conf=0.01, verbose=False)
        detections4 = self.extract_detections(results4)
        all_detections.extend(detections4)
        
        # Strategy 5: Edge detection for vest outlines
        print("  [INFO] Strategy 5: Edge detection for vest outlines")
        edges = self.enhance_edges(image)
        results5 = self.model(edges, conf=0.01, verbose=False)
        detections5 = self.extract_detections(results5)
        all_detections.extend(detections5)
        
        # Filter and merge detections
        final_detections = self.merge_vest_detections(all_detections)
        
        # Count safety vests
        safety_vest_count = sum(1 for det in final_detections if det['class'] == 'safety_vest')
        print(f"  [RESULT] Found {safety_vest_count} safety vests")
        
        # Save result
        if results1:
            annotated = results1[0].plot()
            output_path = f"obscured_vest_detection_{os.path.basename(image_path).split('.')[0]}.jpg"
            cv2.imwrite(output_path, annotated)
            print(f"  [SUCCESS] Result saved: {output_path}")
        
        return final_detections, safety_vest_count
    
    def extract_detections(self, results):
        """Extract detections from YOLO results"""
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        return detections
    
    def merge_vest_detections(self, detections):
        """Merge safety vest detections, keeping the best ones"""
        if not detections:
            return []
        
        # Separate safety vests from other detections
        vest_detections = [d for d in detections if d['class'] == 'safety_vest']
        other_detections = [d for d in detections if d['class'] != 'safety_vest']
        
        # For safety vests, use more aggressive merging
        merged_vests = []
        used_indices = set()
        
        for i, vest1 in enumerate(vest_detections):
            if i in used_indices:
                continue
                
            similar_vests = [vest1]
            used_indices.add(i)
            
            for j, vest2 in enumerate(vest_detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check overlap (lower threshold for vests)
                if self.bbox_overlap(vest1['bbox'], vest2['bbox']) > 0.2:
                    similar_vests.append(vest2)
                    used_indices.add(j)
            
            # Keep the best vest detection
            if len(similar_vests) > 1:
                best_vest = max(similar_vests, key=lambda x: x['confidence'])
                merged_vests.append(best_vest)
            else:
                merged_vests.append(vest1)
        
        # Combine with other detections
        return merged_vests + other_detections
    
    def bbox_overlap(self, bbox1, bbox2):
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
    
    def enhance_reflective_strips(self, image):
        """Enhance image to highlight reflective strips on safety vests"""
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for bright/reflective areas
        lower_bright = np.array([0, 0, 200])  # Very bright areas
        upper_bright = np.array([180, 30, 255])
        bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Enhance brightness in bright areas
        hsv[:,:,2] = np.where(bright_mask > 0, np.minimum(hsv[:,:,2] * 1.3, 255), hsv[:,:,2])
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def enhance_safety_colors(self, image):
        """Enhance orange and yellow colors typical of safety vests"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Orange color range
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Yellow color range
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        safety_mask = cv2.bitwise_or(orange_mask, yellow_mask)
        
        # Enhance saturation and brightness for safety colors
        hsv[:,:,1] = np.where(safety_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
        hsv[:,:,2] = np.where(safety_mask > 0, np.minimum(hsv[:,:,2] * 1.2, 255), hsv[:,:,2])
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def boost_brightness(self, image, factor):
        """Boost image brightness"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=30)
    
    def enhance_edges(self, image):
        """Enhance edges to detect vest outlines"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert back to BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Blend with original image
        enhanced = cv2.addWeighted(image, 0.7, edges_bgr, 0.3, 0)
        
        return enhanced
    
    def create_fixed_web_app(self):
        """Create web app with fixed obscured vest detection"""
        print("\n[INFO] Creating fixed web application...")
        
        fixed_app = '''#!/usr/bin/env python3
"""
Fixed PPE Detection Web Application
Specialized for obscured safety vests
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

def enhance_reflective_strips(image):
    """Enhance image to highlight reflective strips on safety vests"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for bright/reflective areas
    lower_bright = np.array([0, 0, 200])
    upper_bright = np.array([180, 30, 255])
    bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Enhance brightness in bright areas
    hsv[:,:,2] = np.where(bright_mask > 0, np.minimum(hsv[:,:,2] * 1.3, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def enhance_safety_colors(image):
    """Enhance orange and yellow colors typical of safety vests"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Orange color range
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Yellow color range
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    safety_mask = cv2.bitwise_or(orange_mask, yellow_mask)
    
    # Enhance saturation and brightness for safety colors
    hsv[:,:,1] = np.where(safety_mask > 0, np.minimum(hsv[:,:,1] * 2.0, 255), hsv[:,:,1])
    hsv[:,:,2] = np.where(safety_mask > 0, np.minimum(hsv[:,:,2] * 1.2, 255), hsv[:,:,2])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def boost_brightness(image, factor):
    """Boost image brightness"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=30)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Fixed PPE detection with specialized obscured vest detection"""
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
        
        # FIXED DETECTION: Multiple specialized strategies
        all_detections = []
        detected_classes = set()
        
        # Strategy 1: Very low confidence for all PPE
        results1 = model(image, conf=0.01, verbose=False)
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
        
        # Strategy 2: Enhanced reflective strips for safety vests
        enhanced_strips = enhance_reflective_strips(image)
        results2 = model(enhanced_strips, conf=0.01, verbose=False)
        for r in results2:
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
        
        # Strategy 3: Enhanced safety colors for vests
        color_enhanced = enhance_safety_colors(image)
        results3 = model(color_enhanced, conf=0.01, verbose=False)
        for r in results3:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'safety_vest':
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
        
        # Strategy 4: Brightness boost for hidden vests
        bright = boost_brightness(image, 1.5)
        results4 = model(bright, conf=0.01, verbose=False)
        for r in results4:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'safety_vest':
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
            'detections': all_detections,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(all_detections)
        }
        detection_history.append(detection_record)
        
        # Save annotated image
        if results1:
            annotated_image = results1[0].plot()
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
    print("FIXED PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Specialized for obscured safety vests")
    print(f"[INFO] Very low confidence thresholds")
    print(f"[INFO] Multiple enhancement strategies")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        with open("fixed_ppe_web_app.py", 'w') as f:
            f.write(fixed_app)
        
        print("[SUCCESS] Fixed web application created: fixed_ppe_web_app.py")

def main():
    """Main function"""
    print("="*70)
    print("FIXING OBSCURED SAFETY VEST DETECTION")
    print("="*70)
    
    try:
        # Initialize detector
        detector = ObscuredVestDetector()
        
        # Create fixed web app
        detector.create_fixed_web_app()
        
        print("\n" + "="*70)
        print("OBSCURED VEST DETECTION FIXED!")
        print("="*70)
        print(f"\n[SUCCESS] Specialized detection created")
        print(f"[SUCCESS] Very low confidence thresholds (0.01)")
        print(f"[SUCCESS] Enhanced reflective strip detection")
        print(f"[SUCCESS] Color enhancement for orange/yellow vests")
        print(f"[SUCCESS] Brightness boost for hidden vests")
        print(f"[SUCCESS] Fixed web application ready")
        
        print(f"\n[INFO] To use the fix:")
        print("  1. Stop current web app (Ctrl+C)")
        print("  2. Run: python fixed_ppe_web_app.py")
        print("  3. Test with your construction site image")
        
        print(f"\n[INFO] Fix features:")
        print("  - Confidence threshold: 0.01 (very sensitive)")
        print("  - Reflective strip enhancement")
        print("  - Orange/yellow color boost")
        print("  - Brightness enhancement")
        print("  - Duplicate removal with larger tolerance")
        
    except Exception as e:
        print(f"[ERROR] Fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
