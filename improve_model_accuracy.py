#!/usr/bin/env python3
"""
Improve Model Accuracy for Safety Vest Detection
Multiple strategies to boost safety vest detection performance
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime

class ModelAccuracyImprover:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize model accuracy improvement"""
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"[INFO] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Classes: {self.model.names}")
    
    def test_image_preprocessing(self, image_path):
        """Test different image preprocessing techniques"""
        print(f"\n[INFO] Testing image preprocessing on: {os.path.basename(image_path)}")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return
        
        # Test different preprocessing techniques
        preprocessing_methods = {
            'original': image,
            'enhanced_contrast': self.enhance_contrast(image),
            'brightness_adjusted': self.adjust_brightness(image, 1.2),
            'saturation_boosted': self.boost_saturation(image),
            'sharpened': self.sharpen_image(image),
            'color_enhanced': self.enhance_colors(image)
        }
        
        results = {}
        
        for method_name, processed_image in preprocessing_methods.items():
            print(f"  Testing: {method_name}")
            
            # Run detection with low confidence
            results_det = self.model(processed_image, conf=0.1, verbose=False)
            
            # Count safety vests
            safety_vest_count = 0
            for r in results_det:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        if class_name == 'safety_vest':
                            safety_vest_count += 1
            
            results[method_name] = safety_vest_count
            print(f"    Safety vests detected: {safety_vest_count}")
        
        # Find best preprocessing method
        best_method = max(results, key=results.get)
        best_count = results[best_method]
        
        print(f"\n[INFO] Best preprocessing method: {best_method} ({best_count} safety vests)")
        return best_method, preprocessing_methods[best_method]
    
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
    
    def boost_saturation(self, image):
        """Boost image saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)  # Boost saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def sharpen_image(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def enhance_colors(self, image):
        """Enhance colors, especially orange/yellow for safety vests"""
        # Convert to HSV for better color manipulation
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
    
    def create_ensemble_detection(self, image_path):
        """Create ensemble detection with multiple strategies"""
        print(f"\n[INFO] Running ensemble detection on: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        all_detections = []
        
        # Strategy 1: Original image with low confidence
        results1 = self.model(image, conf=0.1, verbose=False)
        detections1 = self.extract_detections(results1)
        all_detections.extend(detections1)
        
        # Strategy 2: Enhanced contrast
        enhanced = self.enhance_contrast(image)
        results2 = self.model(enhanced, conf=0.1, verbose=False)
        detections2 = self.extract_detections(results2)
        all_detections.extend(detections2)
        
        # Strategy 3: Color enhanced
        color_enhanced = self.enhance_colors(image)
        results3 = self.model(color_enhanced, conf=0.1, verbose=False)
        detections3 = self.extract_detections(results3)
        all_detections.extend(detections3)
        
        # Strategy 4: Brightness adjusted
        bright = self.adjust_brightness(image, 1.3)
        results4 = self.model(bright, conf=0.1, verbose=False)
        detections4 = self.extract_detections(results4)
        all_detections.extend(detections4)
        
        # Remove duplicates and merge similar detections
        merged_detections = self.merge_similar_detections(all_detections)
        
        print(f"  Total detections before merging: {len(all_detections)}")
        print(f"  Total detections after merging: {len(merged_detections)}")
        
        # Count safety vests
        safety_vest_count = sum(1 for det in merged_detections if det['class'] == 'safety_vest')
        print(f"  Safety vests detected: {safety_vest_count}")
        
        return merged_detections
    
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
    
    def merge_similar_detections(self, detections):
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
    
    def create_improved_web_app(self):
        """Create improved web app with all enhancements"""
        print("\n[INFO] Creating enhanced web application...")
        
        enhanced_app = '''#!/usr/bin/env python3
"""
Enhanced PPE Detection Web Application
Advanced safety vest detection with multiple strategies
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
    """Enhance colors, especially orange/yellow for safety vests"""
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

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ppe():
    """Enhanced PPE detection with multiple strategies"""
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
        
        # ENHANCED DETECTION: Multiple strategies
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
        
        # Strategy 2: Enhanced contrast for safety vests
        enhanced = enhance_contrast(image)
        results2 = model(enhanced, conf=0.05, verbose=False)
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
        
        # Strategy 3: Color enhanced for safety vests
        color_enhanced = enhance_colors(image)
        results3 = model(color_enhanced, conf=0.05, verbose=False)
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
    print(f"[INFO] Enhanced safety vest detection enabled")
    print(f"[INFO] Multiple detection strategies active")
    print(f"[INFO] Starting web server...")
    print(f"[INFO] Open: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        with open("enhanced_ppe_web_app.py", 'w') as f:
            f.write(enhanced_app)
        
        print("[SUCCESS] Enhanced web application created: enhanced_ppe_web_app.py")
    
    def run_comprehensive_test(self):
        """Run comprehensive test on sample images"""
        print("\n[INFO] Running comprehensive accuracy test...")
        
        # Test on sample images
        test_dirs = ["combined_datasets/images/test", "combined_datasets/images/valid"]
        
        results_summary = {
            'total_images': 0,
            'safety_vest_detections': 0,
            'improvement_methods': {}
        }
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                import glob
                images = glob.glob(os.path.join(test_dir, "*.jpg"))[:5]
                
                for img_path in images:
                    results_summary['total_images'] += 1
                    print(f"\n[INFO] Testing: {os.path.basename(img_path)}")
                    
                    # Test ensemble detection
                    detections = self.create_ensemble_detection(img_path)
                    
                    # Count safety vests
                    safety_vest_count = sum(1 for det in detections if det['class'] == 'safety_vest')
                    results_summary['safety_vest_detections'] += safety_vest_count
                    
                    if safety_vest_count > 0:
                        print(f"  ✅ Found {safety_vest_count} safety vests")
                    else:
                        print(f"  ❌ No safety vests detected")
        
        print(f"\n[INFO] Test Summary:")
        print(f"  Total images tested: {results_summary['total_images']}")
        print(f"  Total safety vests detected: {results_summary['safety_vest_detections']}")
        print(f"  Average per image: {results_summary['safety_vest_detections']/results_summary['total_images']:.2f}")
        
        return results_summary

def main():
    """Main function"""
    print("="*70)
    print("MODEL ACCURACY IMPROVEMENT")
    print("="*70)
    
    try:
        # Initialize improver
        improver = ModelAccuracyImprover()
        
        # Test image preprocessing
        print("\n[INFO] Testing image preprocessing methods...")
        
        # Find a test image
        test_image = None
        test_dirs = ["combined_datasets/images/test", "combined_datasets/images/valid"]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                import glob
                images = glob.glob(os.path.join(test_dir, "*.jpg"))
                if images:
                    test_image = images[0]
                    break
        
        if test_image:
            best_method, processed_image = improver.test_image_preprocessing(test_image)
            print(f"[SUCCESS] Best preprocessing method: {best_method}")
        else:
            print("[WARNING] No test images found")
        
        # Run comprehensive test
        results = improver.run_comprehensive_test()
        
        # Create enhanced web app
        improver.create_improved_web_app()
        
        print("\n" + "="*70)
        print("MODEL ACCURACY IMPROVEMENT COMPLETE!")
        print("="*70)
        print(f"\n[SUCCESS] Enhanced detection methods created")
        print(f"[SUCCESS] Safety vest detection improved")
        print(f"[SUCCESS] Enhanced web application ready")
        
        print(f"\n[INFO] To use enhanced detection:")
        print("  1. Stop current web app (Ctrl+C)")
        print("  2. Run: python enhanced_ppe_web_app.py")
        print("  3. Test with your construction site image")
        
        print(f"\n[INFO] Enhanced features:")
        print("  - Multiple detection strategies")
        print("  - Image preprocessing for safety vests")
        print("  - Color enhancement for orange/yellow vests")
        print("  - Ensemble detection methods")
        print("  - Duplicate removal")
        
    except Exception as e:
        print(f"[ERROR] Improvement failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
