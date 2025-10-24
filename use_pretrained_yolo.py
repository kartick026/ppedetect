#!/usr/bin/env python3
"""
Use pretrained YOLO model for PPE detection
"""

from ultralytics import YOLO
import cv2
import os

def test_pretrained_model():
    """Test with pretrained YOLO model"""
    print("="*70)
    print("TESTING PRETRAINED YOLO MODEL")
    print("="*70)
    
    # Load pretrained YOLOv8 model
    print("[INFO] Loading pretrained YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # This will download if not present
    
    print(f"[INFO] Model classes: {model.names}")
    
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
            
            # Test with different confidence thresholds
            for conf_threshold in [0.1, 0.3, 0.5]:
                print(f"\n  🔍 Testing with confidence: {conf_threshold}")
                
                results = model(image, conf=conf_threshold, verbose=False)
                
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        print(f"    ✅ Found {len(r.boxes)} detections:")
                        for i, box in enumerate(r.boxes):
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = model.names[cls]
                            print(f"      {i+1}. {class_name}: {conf:.3f}")
                    else:
                        print(f"    ❌ No detections found")
        else:
            print(f"[INFO] Test image {test_image} not found")

if __name__ == "__main__":
    test_pretrained_model()
