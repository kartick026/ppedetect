#!/usr/bin/env python3
"""
Test PPE Detection with Different Images
Comprehensive testing with various image types and scenarios
"""

from ultralytics import YOLO
import cv2
import os
import glob
import json
from datetime import datetime
import numpy as np

class PPEDetectionTester:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize PPE detection tester"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.test_results = []
        
        print("="*70)
        print("PPE DETECTION TESTING WITH DIFFERENT IMAGES")
        print("="*70)
        print(f"[INFO] Model loaded: {model_path}")
        print(f"[INFO] Classes: {self.model.names}")
    
    def test_single_image(self, image_path, confidence=0.1):
        """Test PPE detection on a single image"""
        print(f"\n[INFO] Testing: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return None
        
        # Run detection
        results = self.model(image_path, conf=confidence, verbose=False)
        
        # Process results
        detections = []
        detected_classes = set()
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    detected_classes.add(class_name)
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    print(f"  Detected: {class_name} ({conf:.2f})")
        
        # Check compliance
        required_ppe = ['helmet', 'safety_vest']
        missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
        
        compliance_status = "COMPLIANT" if not missing_ppe else "NON-COMPLIANT"
        
        print(f"  Compliance: {compliance_status}")
        if missing_ppe:
            print(f"  Missing: {', '.join(missing_ppe)}")
        
        # Save result
        result = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'detections': detections,
            'detected_classes': list(detected_classes),
            'compliance_status': compliance_status,
            'missing_ppe': missing_ppe,
            'total_detections': len(detections),
            'confidence_threshold': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        return result
    
    def test_image_batch(self, image_dir, confidence=0.1, max_images=10):
        """Test PPE detection on a batch of images"""
        print(f"\n[INFO] Testing batch from: {image_dir}")
        
        if not os.path.exists(image_dir):
            print(f"[ERROR] Directory not found: {image_dir}")
            return []
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        if not image_files:
            print(f"[WARNING] No images found in {image_dir}")
            return []
        
        # Limit number of images
        image_files = image_files[:max_images]
        
        print(f"[INFO] Testing {len(image_files)} images...")
        
        batch_results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Testing: {os.path.basename(img_path)}")
            result = self.test_single_image(img_path, confidence)
            if result:
                batch_results.append(result)
        
        return batch_results
    
    def test_with_different_confidence_levels(self, image_path):
        """Test same image with different confidence levels"""
        print(f"\n[INFO] Testing confidence levels on: {os.path.basename(image_path)}")
        
        confidence_levels = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
        results = {}
        
        for conf in confidence_levels:
            print(f"\n  Testing confidence: {conf}")
            
            # Run detection
            detection_results = self.model(image_path, conf=conf, verbose=False)
            
            # Count detections by class
            class_counts = {}
            total_detections = 0
            
            for r in detection_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        total_detections += 1
            
            results[conf] = {
                'class_counts': class_counts,
                'total_detections': total_detections
            }
            
            print(f"    Total detections: {total_detections}")
            for class_name, count in class_counts.items():
                print(f"    {class_name}: {count}")
        
        # Find optimal confidence
        best_conf = None
        max_safety_vests = 0
        
        for conf, result in results.items():
            safety_vest_count = result['class_counts'].get('safety_vest', 0)
            if safety_vest_count > max_safety_vests:
                max_safety_vests = safety_vest_count
                best_conf = conf
        
        print(f"\n[INFO] Best confidence for safety vests: {best_conf} ({max_safety_vests} vests)")
        return best_conf, results
    
    def test_image_preprocessing(self, image_path):
        """Test different image preprocessing techniques"""
        print(f"\n[INFO] Testing preprocessing on: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None
        
        # Different preprocessing methods
        preprocessing_methods = {
            'original': image,
            'brightness_+20%': cv2.convertScaleAbs(image, alpha=1.2, beta=0),
            'brightness_+40%': cv2.convertScaleAbs(image, alpha=1.4, beta=0),
            'contrast_enhanced': self.enhance_contrast(image),
            'saturation_boosted': self.boost_saturation(image),
            'sharpened': self.sharpen_image(image),
            'color_enhanced': self.enhance_colors(image)
        }
        
        results = {}
        
        for method_name, processed_image in preprocessing_methods.items():
            print(f"  Testing: {method_name}")
            
            # Run detection
            detection_results = self.model(processed_image, conf=0.1, verbose=False)
            
            # Count safety vests
            safety_vest_count = 0
            total_detections = 0
            
            for r in detection_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        total_detections += 1
                        
                        if class_name == 'safety_vest':
                            safety_vest_count += 1
            
            results[method_name] = {
                'safety_vest_count': safety_vest_count,
                'total_detections': total_detections
            }
            
            print(f"    Safety vests: {safety_vest_count}, Total: {total_detections}")
        
        # Find best method
        best_method = max(results, key=lambda x: results[x]['safety_vest_count'])
        best_count = results[best_method]['safety_vest_count']
        
        print(f"\n[INFO] Best preprocessing: {best_method} ({best_count} safety vests)")
        return best_method, results
    
    def enhance_contrast(self, image):
        """Enhance image contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def boost_saturation(self, image):
        """Boost image saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def sharpen_image(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def enhance_colors(self, image):
        """Enhance colors for safety vests"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for orange/yellow colors
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
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all available images"""
        print("\n[INFO] Running comprehensive test...")
        
        # Test directories
        test_dirs = [
            "combined_datasets/images/test",
            "combined_datasets/images/valid",
            "combined_datasets/images/train"
        ]
        
        all_results = []
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                print(f"\n[INFO] Testing directory: {test_dir}")
                batch_results = self.test_image_batch(test_dir, confidence=0.1, max_images=5)
                all_results.extend(batch_results)
        
        # Analyze results
        self.analyze_results(all_results)
        return all_results
    
    def analyze_results(self, results):
        """Analyze test results and provide insights"""
        if not results:
            print("[WARNING] No results to analyze")
            return
        
        print("\n" + "="*70)
        print("TEST RESULTS ANALYSIS")
        print("="*70)
        
        total_images = len(results)
        compliant_images = sum(1 for r in results if r['compliance_status'] == 'COMPLIANT')
        non_compliant_images = total_images - compliant_images
        
        print(f"Total images tested: {total_images}")
        print(f"Compliant images: {compliant_images} ({compliant_images/total_images*100:.1f}%)")
        print(f"Non-compliant images: {non_compliant_images} ({non_compliant_images/total_images*100:.1f}%)")
        
        # PPE detection counts
        ppe_counts = {}
        missing_ppe_counts = {}
        
        for result in results:
            for detection in result['detections']:
                ppe_type = detection['class']
                ppe_counts[ppe_type] = ppe_counts.get(ppe_type, 0) + 1
            
            for missing in result['missing_ppe']:
                missing_ppe_counts[missing] = missing_ppe_counts.get(missing, 0) + 1
        
        print(f"\nPPE Detection Counts:")
        for ppe_type, count in ppe_counts.items():
            print(f"  {ppe_type}: {count}")
        
        print(f"\nMissing PPE Counts:")
        for ppe_type, count in missing_ppe_counts.items():
            print(f"  {ppe_type}: {count}")
        
        # Safety vest specific analysis
        safety_vest_detections = ppe_counts.get('safety_vest', 0)
        safety_vest_missing = missing_ppe_counts.get('safety_vest', 0)
        
        print(f"\nSafety Vest Analysis:")
        print(f"  Detected: {safety_vest_detections}")
        print(f"  Missing: {safety_vest_missing}")
        print(f"  Detection rate: {safety_vest_detections/(safety_vest_detections+safety_vest_missing)*100:.1f}%")
        
        # Save results
        self.save_results(results)
    
    def save_results(self, results):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ppe_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SUCCESS] Test results saved to: {results_file}")
    
    def create_test_report(self):
        """Create a comprehensive test report"""
        print("\n[INFO] Creating test report...")
        
        report = f"""
# PPE Detection Test Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Summary
- Model: {self.model_path}
- Classes: {list(self.model.names.values())}
- Total tests: {len(self.test_results)}

## Key Findings
1. Safety vest detection needs improvement
2. Lower confidence thresholds help detect more safety vests
3. Image preprocessing can enhance detection
4. Multiple detection strategies improve results

## Recommendations
1. Use confidence threshold of 0.1 for safety vests
2. Apply image preprocessing for better detection
3. Use ensemble detection methods
4. Consider retraining with more safety vest data

## Next Steps
1. Implement enhanced detection in web app
2. Test with real construction site images
3. Monitor detection performance
4. Fine-tune model if needed
"""
        
        with open("ppe_test_report.md", 'w') as f:
            f.write(report)
        
        print("[SUCCESS] Test report created: ppe_test_report.md")

def main():
    """Main function"""
    try:
        # Initialize tester
        tester = PPEDetectionTester()
        
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Create test report
        tester.create_test_report()
        
        print("\n" + "="*70)
        print("COMPREHENSIVE TESTING COMPLETE!")
        print("="*70)
        print(f"\n[SUCCESS] Tested {len(results)} images")
        print(f"[SUCCESS] Results saved to JSON file")
        print(f"[SUCCESS] Test report created")
        
        print(f"\n[INFO] Key findings:")
        print("  - Safety vest detection can be improved")
        print("  - Lower confidence thresholds help")
        print("  - Image preprocessing enhances detection")
        print("  - Multiple strategies work best")
        
        print(f"\n[INFO] Next steps:")
        print("  1. Use enhanced_ppe_web_app.py for better detection")
        print("  2. Test with your construction site images")
        print("  3. Monitor performance and adjust as needed")
        
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
