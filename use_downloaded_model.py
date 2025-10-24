#!/usr/bin/env python3
"""
Use Any Downloaded Pretrained PPE Model
Works with models from Roboflow, Kaggle, or GitHub
"""

from ultralytics import YOLO
import glob
import os
from pathlib import Path

print("="*70)
print("PRETRAINED PPE MODEL TESTER")
print("="*70)

# Help user find their downloaded model
print("\n[INFO] Looking for pretrained models in current directory...")

# Search for .pt files
pt_files = glob.glob('**/*.pt', recursive=True)
pt_files = [f for f in pt_files if 'yolov8' in f or 'best' in f or 'model' in f.lower()]

if pt_files:
    print(f"\n[OK] Found {len(pt_files)} potential model files:")
    for i, model_path in enumerate(pt_files[:10], 1):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"  {i}. {model_path} ({size_mb:.1f} MB)")
    
    # Use the first best.pt or largest file
    best_model = None
    for f in pt_files:
        if 'best.pt' in f:
            best_model = f
            break
    
    if not best_model:
        best_model = pt_files[0]
    
    print(f"\n[INFO] Using model: {best_model}")
    
else:
    print("\n[WARNING] No .pt model files found!")
    print("\nOptions:")
    print("  1. Download a model from:")
    print("     - Roboflow: https://universe.roboflow.com/")
    print("     - Kaggle: See EASY_DOWNLOAD_GUIDE.md")
    print("  2. Or specify model path manually below")
    
    # Use base YOLO as fallback
    best_model = 'yolov8n.pt'
    print(f"\n[INFO] Using base YOLOv8n model instead")

# Load the model
try:
    print(f"\n[INFO] Loading model: {best_model}")
    model = YOLO(best_model)
    print("[OK] Model loaded successfully!")
    
    # Show model info
    print(f"\n[INFO] Model Information:")
    print(f"  Model file: {best_model}")
    print(f"  File size: {os.path.getsize(best_model) / (1024*1024):.1f} MB")
    
    # Test on sample images
    print("\n[INFO] Testing model on sample images...")
    test_images = glob.glob('combined_datasets/images/test/*.jpg')[:5]
    
    if test_images:
        print(f"[INFO] Found {len(test_images)} test images")
        
        for i, img_path in enumerate(test_images[:3], 1):
            print(f"\n[INFO] Testing image {i}/3: {Path(img_path).name}")
            
            # Run detection
            results = model(img_path, conf=0.5)
            
            # Show detections
            detections = len(results[0].boxes)
            print(f"  Detections: {detections} objects found")
            
            # Save result
            output_dir = Path('pretrained_model_results')
            output_dir.mkdir(exist_ok=True)
            
            result_path = output_dir / f"result_{i}_{Path(img_path).name}"
            results[0].save(str(result_path))
            print(f"  Saved: {result_path}")
        
        print(f"\n[OK] Results saved to: pretrained_model_results/")
        print("\n[INFO] Check the saved images to see the detections!")
        
    else:
        print("[WARNING] No test images found in combined_datasets/images/test/")
        print("[INFO] You can test manually:")
        print("  results = model('path/to/your/image.jpg')")
        print("  results[0].show()")
    
    # Model summary
    print("\n" + "="*70)
    print("MODEL READY TO USE!")
    print("="*70)
    print(f"\nYour model: {best_model}")
    print("\nUsage:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model}')")
    print(f"  results = model('image.jpg')")
    print(f"  results[0].show()")
    
    print("\n[INFO] Model is ready for:")
    print("  - Real-time detection")
    print("  - Batch processing")
    print("  - Video analysis")
    print("  - Deployment")

except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    print("\n[INFO] Make sure you have a valid .pt model file")
    print("[INFO] Download options:")
    print("  1. Roboflow: https://universe.roboflow.com/")
    print("  2. See: EASY_DOWNLOAD_GUIDE.md")

if __name__ == "__main__":
    pass




