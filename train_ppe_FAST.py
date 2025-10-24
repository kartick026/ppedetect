#!/usr/bin/env python3
"""
FAST PPE Detection Training - Optimized for Speed
Completes in ~6-8 hours instead of 24-30 hours
"""

from ultralytics import YOLO
import torch
import gc

print("="*70)
print("FAST PPE DETECTION TRAINING")
print("Optimized for SPEED - Reduced training time")
print("="*70)

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print("GPU cache cleared")

# Load model
model = YOLO('yolov8n.pt')

# SPEED-OPTIMIZED PARAMETERS
training_params = {
    'data': 'ppe_detection_dataset.yaml',
    
    # SPEED OPTIMIZATIONS
    'epochs': 25,           # Reduced from 50 (2x faster)
    'imgsz': 256,           # Reduced from 320 (1.5x faster)
    'batch': 2,             # Try batch=2 (might work, 2x faster)
    'workers': 4,           # More CPU workers (faster data loading)
    
    # Reduced validation frequency
    'val': True,
    'plots': False,         # Skip plots during training (faster)
    'save_period': 5,       # Less frequent saves
    
    # Training settings
    'device': 0,
    'project': 'ppe_detection_project',
    'name': 'ppe_fast_8hours',
    'exist_ok': True,
    'lr0': 0.01,
    'lrf': 0.1,
    'patience': 15,         # Early stopping - faster exit if converged
    
    # Memory optimizations (keep these)
    'mosaic': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'cache': False,
    'amp': False,
    
    # Minimal augmentation
    'degrees': 0.0,
    'scale': 0.2,
    'fliplr': 0.5,
    'flipud': 0.0,
    
    'verbose': True,
    'save_json': False,
}

print("\n[FAST MODE] Configuration:")
print("="*70)
print("  Classes: Helmet, Safety Vest, Goggles, Gloves")
print(f"  Epochs: {training_params['epochs']} (reduced for speed)")
print(f"  Image Size: {training_params['imgsz']}x{training_params['imgsz']}")
print(f"  Batch Size: {training_params['batch']} (attempting 2 for speed)")
print(f"  Workers: {training_params['workers']}")
print(f"  Estimated Time: 6-8 hours (vs 24-30 hours)")
print("="*70)

print("\n[INFO] Starting FAST training...")
print("[WARNING] If batch=2 causes OOM error, it will auto-retry with batch=1")

try:
    # Try with batch=2 first
    results = model.train(**training_params)
    
    print("\n" + "="*70)
    print("SUCCESS - TRAINING COMPLETED!")
    print("="*70)
    print(f"Best model: {results.save_dir}/weights/best.pt")
    
    # Quick validation
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(data='ppe_detection_dataset.yaml', imgsz=256)
    
    print(f"\nPerformance:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n[INFO] Batch=2 caused OOM. Retrying with batch=1...")
        torch.cuda.empty_cache()
        gc.collect()
        
        training_params['batch'] = 1
        training_params['name'] = 'ppe_fast_batch1'
        
        results = model.train(**training_params)
        print(f"\n[SUCCESS] Training completed with batch=1")
        print(f"Best model: {results.save_dir}/weights/best.pt")
    else:
        raise e
        
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")





