#!/usr/bin/env python3
"""
Test Safety Vest Detection with Different Settings
"""

from ultralytics import YOLO
import cv2
import os
import glob

def test_safety_vest_detection():
    """Test safety vest detection with different confidence levels"""
    print("="*70)
    print("TESTING SAFETY VEST DETECTION")
    print("="*70)
    
    # Load model
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    model = YOLO(model_path)
    
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    
    # Test different confidence levels
    confidence_levels = [0.5, 0.3, 0.2, 0.1, 0.05]
    
    for conf in confidence_levels:
        print(f"\n[INFO] Testing confidence threshold: {conf}")
        
        safety_vest_count = 0
        total_detections = 0
        
        # Test on sample images
        test_dirs = ["combined_datasets/images/test", "combined_datasets/images/valid"]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = glob.glob(os.path.join(test_dir, "*.jpg"))[:3]  # Test first 3 images
                
                for img_path in images:
                    print(f"  Testing: {os.path.basename(img_path)}")
                    
                    # Run detection
                    results = model(img_path, conf=conf, verbose=False)
                    
                    # Count detections
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                class_name = model.names[cls]
                                total_detections += 1
                                
                                if class_name == 'safety_vest':
                                    safety_vest_count += 1
                                    conf_score = float(box.conf[0])
                                    print(f"    ✅ Safety vest detected: {conf_score:.2f}")
        
        print(f"  Results: {safety_vest_count} safety vests out of {total_detections} total detections")
        
        if safety_vest_count > 0:
            print(f"  ✅ SUCCESS: Found safety vests with confidence {conf}")
            return conf
        else:
            print(f"  ❌ No safety vests found with confidence {conf}")
    
    print(f"\n[WARNING] No safety vests detected even with very low confidence")
    print("[INFO] This might indicate:")
    print("  1. Safety vests not visible in test images")
    print("  2. Model needs retraining with more safety vest data")
    print("  3. Different image preprocessing needed")
    
    return 0.1  # Return lowest confidence as fallback

def create_quick_fix_script():
    """Create a quick fix script for immediate use"""
    print("\n[INFO] Creating quick fix script...")
    
    quick_fix = '''#!/usr/bin/env python3
"""
Quick Fix for Safety Vest Detection
Use this to test with lower confidence threshold
"""

from ultralytics import YOLO
import cv2
import sys

def detect_ppe_with_low_confidence(image_path, confidence=0.1):
    """Detect PPE with lower confidence for safety vests"""
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    model = YOLO(model_path)
    
    print(f"[INFO] Detecting PPE in: {image_path}")
    print(f"[INFO] Using confidence threshold: {confidence}")
    
    # Run detection
    results = model(image_path, conf=confidence)
    
    # Process results
    detections = []
    detected_classes = set()
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                detected_classes.add(class_name)
                
                detections.append({
                    'class': class_name,
                    'confidence': conf
                })
                
                print(f"  Detected: {class_name} ({conf:.2f})")
    
    # Check compliance
    required_ppe = ['helmet', 'safety_vest']
    missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
    
    if missing_ppe:
        print(f"\\n❌ NON-COMPLIANT: Missing {', '.join(missing_ppe)}")
    else:
        print(f"\\n✅ COMPLIANT: All required PPE detected")
    
    # Save result
    if results:
        output_path = f"ppe_detection_result_{confidence}.jpg"
        results[0].save(output_path)
        print(f"\\n[INFO] Result saved: {output_path}")
    
    return detections, missing_ppe

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        
        detect_ppe_with_low_confidence(image_path, confidence)
    else:
        print("Usage: python quick_fix.py <image_path> [confidence]")
        print("Example: python quick_fix.py construction_site.jpg 0.1")
'''
    
    with open("quick_fix_safety_vest.py", 'w') as f:
        f.write(quick_fix)
    
    print("[SUCCESS] Quick fix script created: quick_fix_safety_vest.py")

def main():
    """Main function"""
    # Test current detection
    optimal_conf = test_safety_vest_detection()
    
    # Create quick fix
    create_quick_fix_script()
    
    print("\n" + "="*70)
    print("SAFETY VEST DETECTION ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n[INFO] Recommended confidence threshold: {optimal_conf}")
    print("\n[INFO] Quick fixes available:")
    print("  1. Use improved web app: python improved_ppe_web_app.py")
    print("  2. Test single image: python quick_fix_safety_vest.py <image_path>")
    print("  3. Lower confidence in current app")
    
    print("\n[INFO] To fix your current issue:")
    print("  1. Stop current web app (Ctrl+C)")
    print("  2. Run: python improved_ppe_web_app.py")
    print("  3. Test with your construction site image")

if __name__ == "__main__":
    main()
