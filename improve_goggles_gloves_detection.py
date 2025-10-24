#!/usr/bin/env python3
"""
Improve Safety Goggles and Gloves Detection
Enhanced detection for all PPE items including goggles and gloves
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime

class EnhancedPPEDetector:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize enhanced PPE detector"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        print("="*70)
        print("ENHANCED PPE DETECTION - ALL ITEMS")
        print("="*70)
        print(f"[INFO] Model loaded: {model_path}")
        print(f"[INFO] Classes: {self.model.names}")
        print(f"[INFO] Detecting: Helmet, Safety Vest, Goggles, Gloves")
    
    def detect_ppe_enhanced(self, image_path, confidence=0.1):
        """Enhanced PPE detection with multiple strategies"""
        print(f"\n[INFO] Detecting PPE in: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None
        
        # Strategy 1: Original image with low confidence
        results1 = self.model(image, conf=confidence, verbose=False)
        detections = self.extract_detections(results1)
        
        # Strategy 2: Enhanced contrast for better detection
        enhanced_contrast = self.enhance_contrast(image)
        results2 = self.model(enhanced_contrast, conf=confidence*0.5, verbose=False)
        detections.extend(self.extract_detections(results2))
        
        # Strategy 3: Brightness adjusted for goggles/gloves
        bright_image = self.adjust_brightness(image, 1.3)
        results3 = self.model(bright_image, conf=confidence*0.5, verbose=False)
        detections.extend(self.extract_detections(results3))
        
        # Strategy 4: Color enhanced for safety vests
        color_enhanced = self.enhance_colors(image)
        results4 = self.model(color_enhanced, conf=confidence*0.5, verbose=False)
        detections.extend(self.extract_detections(results4))
        
        # Strategy 5: Sharpened for goggles/gloves details
        sharpened = self.sharpen_image(image)
        results5 = self.model(sharpened, conf=confidence*0.5, verbose=False)
        detections.extend(self.extract_detections(results5))
        
        # Remove duplicates and merge similar detections
        final_detections = self.merge_detections(detections)
        
        # Analyze results
        detected_classes = set(det['class'] for det in final_detections)
        class_counts = {}
        for det in final_detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  Detected classes: {list(detected_classes)}")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
        
        # Check compliance
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
        
        compliance_status = "COMPLIANT" if not missing_ppe else "NON-COMPLIANT"
        
        print(f"  Compliance: {compliance_status}")
        if missing_ppe:
            print(f"  Missing: {', '.join(missing_ppe)}")
        
        # Save annotated image
        if results1:
            annotated_image = results1[0].plot()
            output_path = f"enhanced_detection_{os.path.basename(image_path).split('.')[0]}.jpg"
            cv2.imwrite(output_path, annotated_image)
            print(f"  Annotated image saved: {output_path}")
        
        return {
            'image_path': image_path,
            'detections': final_detections,
            'detected_classes': list(detected_classes),
            'class_counts': class_counts,
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(final_detections)
        }
    
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
    
    def merge_detections(self, detections):
        """Merge similar detections to avoid duplicates"""
        if not detections:
            return []
        
        merged = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
                
            similar_detections = [det1]
            used_indices.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check if detections are similar (same class and overlapping bbox)
                if (det1['class'] == det2['class'] and 
                    self.bbox_overlap(det1['bbox'], det2['bbox']) > 0.3):
                    similar_detections.append(det2)
                    used_indices.add(j)
            
            # Merge similar detections (keep highest confidence)
            if len(similar_detections) > 1:
                best_det = max(similar_detections, key=lambda x: x['confidence'])
                merged.append(best_det)
            else:
                merged.append(det1)
        
        return merged
    
    def bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
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
    
    def enhance_contrast(self, image):
        """Enhance image contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def adjust_brightness(self, image, factor):
        """Adjust image brightness"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def enhance_colors(self, image):
        """Enhance colors for safety vests"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for orange/yellow colors (safety vest colors)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask_orange, mask_yellow)
        
        # Enhance saturation for safety vest colors
        hsv[:,:,1] = np.where(mask > 0, np.minimum(hsv[:,:,1] * 1.5, 255), hsv[:,:,1])
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def sharpen_image(self, image):
        """Apply sharpening filter for better detail detection"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def create_enhanced_web_app(self):
        """Create enhanced web app with all PPE detection"""
        print("\n[INFO] Creating enhanced web application...")
        
        enhanced_app = '''#!/usr/bin/env python3
"""
Enhanced PPE Detection Web Application
Detects: Helmet, Safety Vest, Goggles, Gloves
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

def enhance_contrast(image):
    """Enhance image contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def enhance_colors(image):
    """Enhance colors for safety vests"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for orange/yellow colors (safety vest colors)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_orange, mask_yellow)
    
    # Enhance saturation for safety vest colors
    hsv[:,:,1] = np.where(mask > 0, np.minimum(hsv[:,:,1] * 1.5, 255), hsv[:,:,1])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_brightness(image, factor):
    """Adjust image brightness"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def sharpen_image(image):
    """Apply sharpening filter"""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Enhanced PPE detection for all items"""
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
        
        # ENHANCED DETECTION: Multiple strategies for all PPE
        all_detections = []
        detected_classes = set()
        
        # Strategy 1: Original image with low confidence
        results1 = model(image, conf=0.1, verbose=False)
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
        
        # Strategy 2: Enhanced contrast for better detection
        enhanced = enhance_contrast(image)
        results2 = model(enhanced, conf=0.05, verbose=False)
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
        
        # Strategy 3: Brightness adjusted for goggles/gloves
        bright = adjust_brightness(image, 1.3)
        results3 = model(bright, conf=0.05, verbose=False)
        for r in results3:
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
        
        # Strategy 4: Color enhanced for safety vests
        color_enhanced = enhance_colors(image)
        results4 = model(color_enhanced, conf=0.05, verbose=False)
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
        
        # Strategy 5: Sharpened for goggles/gloves details
        sharpened = sharpen_image(image)
        results5 = model(sharpened, conf=0.05, verbose=False)
        for r in results5:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name in ['goggles', 'gloves']:  # Only add goggles/gloves from this pass
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Check for duplicates
                        is_duplicate = False
                        for existing in all_detections:
                            if existing['class'] == class_name:
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
        
        # Check compliance - ALL PPE required
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
    print("ENHANCED PPE DETECTION WEB APPLICATION")
    print("="*70)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    print(f"[INFO] Detecting: Helmet, Safety Vest, Goggles, Gloves")
    print(f"[INFO] Enhanced detection with multiple strategies")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        with open("enhanced_all_ppe_web_app.py", 'w') as f:
            f.write(enhanced_app)
        
        print("[SUCCESS] Enhanced web application created: enhanced_all_ppe_web_app.py")
    
    def test_enhanced_detection(self, image_path):
        """Test enhanced detection on a single image"""
        result = self.detect_ppe_enhanced(image_path)
        return result

def main():
    """Main function"""
    print("="*70)
    print("ENHANCED PPE DETECTION - ALL ITEMS")
    print("="*70)
    
    try:
        # Initialize enhanced detector
        detector = EnhancedPPEDetector()
        
        # Create enhanced web app
        detector.create_enhanced_web_app()
        
        print("\n" + "="*70)
        print("ENHANCED DETECTION SETUP COMPLETE!")
        print("="*70)
        print(f"\n[SUCCESS] Enhanced detection created")
        print(f"[SUCCESS] Detects: Helmet, Safety Vest, Goggles, Gloves")
        print(f"[SUCCESS] Multiple detection strategies")
        print(f"[SUCCESS] Enhanced web application ready")
        
        print(f"\n[INFO] To use enhanced detection:")
        print("  1. Stop current web app (Ctrl+C)")
        print("  2. Run: python enhanced_all_ppe_web_app.py")
        print("  3. Test with your construction site image")
        
        print(f"\n[INFO] Enhanced features:")
        print("  - Detects ALL PPE items (helmet, vest, goggles, gloves)")
        print("  - Multiple detection strategies")
        print("  - Image preprocessing for better detection")
        print("  - Duplicate removal")
        print("  - Enhanced compliance checking")
        
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
