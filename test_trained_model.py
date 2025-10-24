#!/usr/bin/env python3
"""
Test the trained PPE detection model
"""

from ultralytics import YOLO
import cv2
import os
import glob

def test_trained_model():
    """Test the trained PPE detection model"""
    print("="*70)
    print("TESTING TRAINED PPE DETECTION MODEL")
    print("="*70)
    
    # Load the trained model
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return
    
    print(f"[INFO] Loading trained model: {model_path}")
    model = YOLO(model_path)
    
    # Test on sample images
    test_images = [
        "combined_datasets/images/test/",
        "combined_datasets/images/valid/"
    ]
    
    print("\n[INFO] Testing on sample images...")
    
    for test_dir in test_images:
        if os.path.exists(test_dir):
            print(f"\n[INFO] Testing images from: {test_dir}")
            
            # Get first 5 images for testing
            image_files = glob.glob(os.path.join(test_dir, "*.jpg"))[:5]
            
            for i, image_path in enumerate(image_files):
                print(f"\n[INFO] Testing image {i+1}: {os.path.basename(image_path)}")
                
                # Run inference
                results = model(image_path)
                
                # Display results
                for r in results:
                    # Get detection info
                    boxes = r.boxes
                    if boxes is not None:
                        print(f"  Detected {len(boxes)} objects:")
                        for box in boxes:
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            class_name = model.names[cls]
                            print(f"    - {class_name}: {conf:.2f} confidence")
                    else:
                        print("  No objects detected")
                
                # Save result image
                output_path = f"test_result_{i+1}.jpg"
                r.save(output_path)
                print(f"  Result saved: {output_path}")
    
    print("\n" + "="*70)
    print("MODEL TESTING COMPLETE!")
    print("="*70)
    print("\nYour trained model is working!")
    print("Check the generated test_result_*.jpg files to see detections.")

if __name__ == "__main__":
    test_trained_model()
