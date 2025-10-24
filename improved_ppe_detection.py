#!/usr/bin/env python3
"""
Improved PPE Detection using multiple approaches
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

def improved_ppe_detection():
    """Improved PPE detection using multiple models"""
    print("="*70)
    print("IMPROVED PPE DETECTION")
    print("="*70)
    
    # Load pretrained YOLO for person detection
    print("[INFO] Loading pretrained YOLOv8n for person detection...")
    person_model = YOLO('yolov8n.pt')
    
    # Try to load custom PPE model
    ppe_model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    ppe_model = None
    if os.path.exists(ppe_model_path):
        print(f"[INFO] Loading custom PPE model: {ppe_model_path}")
        try:
            ppe_model = YOLO(ppe_model_path)
            print(f"[INFO] PPE model classes: {ppe_model.names}")
        except Exception as e:
            print(f"[WARNING] Could not load PPE model: {e}")
    else:
        print(f"[WARNING] PPE model not found: {ppe_model_path}")
    
    # Test with sample images
    test_images = [
        "test_result_1.jpg",
        "test_result_2.jpg", 
        "test_result_3.jpg",
        "test_result_4.jpg"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n[INFO] Testing: {test_image}")
            
            # Load image
            image = cv2.imread(test_image)
            if image is None:
                print(f"  ❌ Could not load image")
                continue
            
            print(f"  📏 Image size: {image.shape}")
            
            # Step 1: Detect people using pretrained model
            print(f"\n  👥 Detecting people...")
            person_results = person_model(image, conf=0.3, classes=[0])  # Only detect 'person' class
            people_count = 0
            for r in person_results:
                if r.boxes is not None:
                    people_count = len(r.boxes)
                    print(f"    ✅ Found {people_count} people")
                    for i, box in enumerate(r.boxes):
                        conf = float(box.conf[0])
                        print(f"      Person {i+1}: {conf:.3f} confidence")
            
            if people_count == 0:
                print(f"    ❌ No people detected")
                # Estimate people count based on image analysis
                people_count = 1  # Default assumption
            
            # Step 2: Try PPE detection with custom model
            ppe_detections = []
            if ppe_model is not None:
                print(f"\n  🛡️ Detecting PPE with custom model...")
                ppe_results = ppe_model(image, conf=0.1)  # Very low confidence
                for r in ppe_results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        print(f"    ✅ Found {len(r.boxes)} PPE items:")
                        for i, box in enumerate(r.boxes):
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = ppe_model.names[cls]
                            ppe_detections.append({
                                'class': class_name,
                                'confidence': conf
                            })
                            print(f"      {i+1}. {class_name}: {conf:.3f}")
                    else:
                        print(f"    ❌ No PPE detected with custom model")
            else:
                print(f"\n  🛡️ Custom PPE model not available")
            
            # Step 3: Simple color-based PPE detection (fallback)
            if len(ppe_detections) == 0:
                print(f"\n  🎨 Trying color-based PPE detection...")
                # Look for bright colors that might indicate safety vests
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Define color ranges for safety vests (yellow, orange, green)
                yellow_lower = np.array([20, 100, 100])
                yellow_upper = np.array([30, 255, 255])
                orange_lower = np.array([10, 100, 100])
                orange_upper = np.array([20, 255, 255])
                green_lower = np.array([40, 100, 100])
                green_upper = np.array([80, 255, 255])
                
                # Create masks for each color
                yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
                orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
                green_mask = cv2.inRange(hsv, green_lower, green_upper)
                
                # Count bright colored regions
                bright_regions = 0
                for mask, color in [(yellow_mask, 'yellow'), (orange_mask, 'orange'), (green_mask, 'green')]:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 1000:  # Minimum area threshold
                            bright_regions += 1
                            print(f"      Found {color} region with area: {area}")
                
                if bright_regions > 0:
                    print(f"    ✅ Found {bright_regions} potential safety vest regions")
                    ppe_detections.append({
                        'class': 'safety_vest',
                        'confidence': 0.5
                    })
                else:
                    print(f"    ❌ No bright colored regions found")
            
            # Step 4: Determine compliance
            if len(ppe_detections) > 0:
                compliance_status = "PPE WORN"
                print(f"\n  ✅ Result: {compliance_status}")
                print(f"    People: {people_count}")
                print(f"    PPE Items: {len(ppe_detections)}")
            else:
                compliance_status = "PPE NOT WORN"
                print(f"\n  ❌ Result: {compliance_status}")
                print(f"    People: {people_count}")
                print(f"    PPE Items: 0")
                
        else:
            print(f"[INFO] Test image {test_image} not found")

if __name__ == "__main__":
    improved_ppe_detection()
