#!/usr/bin/env python3
"""
Transfer Learning: Use Pretrained YOLO + Your Dataset
FASTEST way to get good results (1-2 hours training instead of 20+ hours!)
"""

from ultralytics import YOLO
import torch

print("="*70)
print("TRANSFER LEARNING - PPE DETECTION")
print("Using pretrained YOLO + Your dataset = Fast & Accurate!")
print("="*70)

# Download pretrained model (trained on COCO - 80 classes)
print("\n[INFO] Step 1: Downloading pretrained YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Auto-downloads from Ultralytics
print("[OK] Pretrained model downloaded!")

print("\n[INFO] This model is already trained on:")
print("  - 80 object classes (COCO dataset)")
print("  - 118,000+ images")
print("  - Can detect: person, backpack, handbag, etc.")

print("\n[INFO] Step 2: Fine-tuning on YOUR PPE dataset...")
print("[INFO] This adapts the model to detect:")
print("  - Helmet")
print("  - Safety Vest")
print("  - Goggles")
print("  - Gloves")

# Transfer learning configuration
print("\n[INFO] Starting transfer learning...")
print("[INFO] Estimated time: 1-2 hours (much faster than 20+ hours!)")

try:
    results = model.train(
        data='ppe_detection_dataset.yaml',
        
        # Quick training
        epochs=10,              # Only 10 epochs needed!
        imgsz=416,              # Slightly larger for better quality
        batch=2,                # Try batch 2
        workers=4,
        
        # Transfer learning optimizations
        pretrained=True,        # Keep COCO weights
        freeze=10,              # Freeze first 10 layers (backbone)
        lr0=0.001,             # Lower LR (we're fine-tuning, not training from scratch)
        lrf=0.01,
        
        # Settings
        device=0,
        project='ppe_transfer_learning',
        name='finetune_10epochs',
        exist_ok=True,
        patience=5,
        
        # Memory optimizations (keep these)
        mosaic=0.0,
        cache=False,
        amp=False,
        
        verbose=True,
        plots=True,
    )
    
    print("\n" + "="*70)
    print("TRANSFER LEARNING COMPLETE!")
    print("="*70)
    print(f"\nModel saved: {results.save_dir}/weights/best.pt")
    print("\nThis model:")
    print("  ✅ Trained in 1-2 hours (vs 20+ hours)")
    print("  ✅ Uses pretrained COCO knowledge")
    print("  ✅ Adapted to your PPE dataset")
    print("  ✅ Expected accuracy: 65-75% mAP@0.5")
    
    # Test the model
    print("\n[INFO] Testing model on a sample image...")
    test_model = YOLO(f"{results.save_dir}/weights/best.pt")
    
    # Get a test image
    import glob
    test_images = glob.glob('combined_datasets/images/test/*.jpg')[:5]
    
    if test_images:
        print(f"\n[INFO] Running inference on {len(test_images)} test images...")
        for img_path in test_images[:2]:  # Test on 2 images
            result = test_model(img_path)
            print(f"  Tested: {img_path}")
    
    print("\n[SUCCESS] Transfer learning completed!")
    print(f"Use your model: {results.save_dir}/weights/best.pt")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n[INFO] Batch=2 caused OOM. Retrying with batch=1...")
        torch.cuda.empty_cache()
        
        results = model.train(
            data='ppe_detection_dataset.yaml',
            epochs=10, imgsz=320, batch=1, workers=4,
            pretrained=True, freeze=10, lr0=0.001,
            device=0, project='ppe_transfer_learning',
            name='finetune_batch1', mosaic=0.0, cache=False
        )
        print(f"[OK] Completed with batch=1")
    else:
        raise e

if __name__ == "__main__":
    # Run transfer learning
    pass




