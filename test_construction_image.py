#!/usr/bin/env python3
"""
Test PPE detection on construction site image
"""

from ultralytics import YOLO
import cv2
import os

def test_construction_image():
    """Test PPE detection on construction site image"""
    print("="*70)
    print("TESTING PPE DETECTION ON CONSTRUCTION SITE IMAGE")
    print("="*70)
    
    # Load the trained model
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return
    
    print(f"[INFO] Loading trained model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Model classes: {model.names}")
    
    # Look for construction site image
    possible_names = [
        "construction_site_image.jpg",
        "construction_site_image.png", 
        "construction.jpg",
        "construction.png",
        "site.jpg",
        "site.png"
    ]
    
    image_path = None
    for name in possible_names:
        if os.path.exists(name):
            image_path = name
            break
    
    if not image_path:
        print("[INFO] No construction site image found with common names.")
        print("[INFO] Please save your image as 'construction_site_image.jpg' and run again.")
        print("\n[INFO] Testing with a sample image instead...")
        
        # Use a sample image for demonstration
        sample_images = [
            "combined_datasets/images/test",
            "combined_datasets/images/valid"
        ]
        
        for sample_dir in sample_images:
            if os.path.exists(sample_dir):
                import glob
                images = glob.glob(os.path.join(sample_dir, "*.jpg"))[:3]
                for img in images:
                    print(f"\n[INFO] Testing: {os.path.basename(img)}")
                    results = model(img, conf=0.3)  # Lower confidence for more detections
                    
                    for r in results:
                        if r.boxes is not None:
                            print(f"  Detected {len(r.boxes)} objects:")
                            for box in r.boxes:
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                class_name = model.names[cls]
                                print(f"    - {class_name}: {conf:.2f} confidence")
                        else:
                            print("  No PPE detected")
                    
                    # Save result
                    output_path = f"construction_test_{os.path.basename(img)}"
                    r.save(output_path)
                    print(f"  Result saved: {output_path}")
                break
    else:
        print(f"[INFO] Found construction image: {image_path}")
        print(f"[INFO] Processing construction site image...")
        
        # Run detection with lower confidence threshold for better detection
        results = model(image_path, conf=0.3, verbose=True)
        
        print(f"\n[INFO] Detection Results:")
        for r in results:
            if r.boxes is not None:
                print(f"  Detected {len(r.boxes)} PPE items:")
                for box in r.boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.names[cls]
                    print(f"    - {class_name}: {conf:.2f} confidence")
            else:
                print("  No PPE detected")
        
        # Save annotated result
        output_path = "construction_site_ppe_detection.jpg"
        results[0].save(output_path)
        print(f"\n[INFO] Annotated image saved: {output_path}")
    
    print("\n" + "="*70)
    print("CONSTRUCTION SITE PPE DETECTION COMPLETE!")
    print("="*70)
    print("\nYour model can detect:")
    print("  - Hard hats (helmets)")
    print("  - Safety vests")
    print("  - Safety goggles")
    print("  - Safety gloves")
    print("\nFor the construction site image with 7 workers in PPE,")
    print("the model should detect multiple hard hats and safety vests!")

if __name__ == "__main__":
    test_construction_image()
