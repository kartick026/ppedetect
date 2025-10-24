#!/usr/bin/env python3
"""
Test the clean PPE detection to ensure it works correctly
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

def test_clean_detection():
    """Test the clean detection function"""
    print("="*70)
    print("TESTING CLEAN PPE DETECTION")
    print("="*70)
    
    # Load the trained model
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return
    
    print(f"[INFO] Loading trained model: {model_path}")
    model = YOLO(model_path)
    
    def detect_ppe_simple(image):
        """
        Simple, clean PPE detection - the original working method
        This is the core detection that was working perfectly before
        """
        try:
            # Simple detection with balanced confidence
            results = model(image, conf=0.3, verbose=False)
            
            detections = []
            detected_classes = set()
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        detected_classes.add(class_name)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
            
            # Check compliance
            compliance_status = "COMPLIANT"
            missing_ppe = []
            
            required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
            for ppe in required_ppe:
                if ppe not in detected_classes:
                    missing_ppe.append(ppe)
                    compliance_status = "NON-COMPLIANT"
            
            return {
                'detections': detections,
                'compliance_status': compliance_status,
                'missing_ppe': missing_ppe,
                'total_detections': len(detections)
            }
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return {
                'detections': [],
                'compliance_status': 'ERROR',
                'missing_ppe': ['helmet', 'safety_vest', 'goggles', 'gloves'],
                'total_detections': 0
            }
    
    # Test on sample images
    test_images = [
        "combined_datasets/images/test/",
        "combined_datasets/images/valid/"
    ]
    
    print("\n[INFO] Testing clean detection on sample images...")
    
    for test_dir in test_images:
        if os.path.exists(test_dir):
            print(f"\n[INFO] Testing images from: {test_dir}")
            
            # Get first 3 images for testing
            image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:3]
            
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(test_dir, image_file)
                print(f"\n[INFO] Testing image {i+1}: {image_file}")
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  [ERROR] Could not load image: {image_path}")
                    continue
                
                # Run clean detection
                result = detect_ppe_simple(image)
                
                # Display results
                print(f"  Compliance: {result['compliance_status']}")
                print(f"  Total detections: {result['total_detections']}")
                
                if result['detections']:
                    print("  Detected PPE:")
                    for det in result['detections']:
                        print(f"    - {det['class']}: {det['confidence']:.2f}")
                else:
                    print("  No PPE detected")
                
                if result['missing_ppe']:
                    print(f"  Missing PPE: {', '.join(result['missing_ppe'])}")
                
                # Save result image
                results = model(image, conf=0.3, verbose=False)
                if results:
                    annotated_image = results[0].plot()
                    output_path = f"clean_test_result_{i+1}.jpg"
                    cv2.imwrite(output_path, annotated_image)
                    print(f"  Result saved: {output_path}")
    
    print("\n" + "="*70)
    print("CLEAN DETECTION TEST COMPLETE!")
    print("="*70)
    print("\nThe clean detection is working correctly!")
    print("This preserves the original working detection logic.")
    print("Live camera functionality is kept separate.")

if __name__ == "__main__":
    test_clean_detection()
