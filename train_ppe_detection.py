#!/usr/bin/env python3
"""
PPE Detection Training - Helmet, Safety Vest, Goggles, Gloves
Optimized for 4GB GPU (GTX 1650)
"""

from ultralytics import YOLO
import torch
import gc

print("="*70)
print("PPE DETECTION TRAINING")
print("Detecting: Helmet, Safety Vest, Goggles, Gloves")
print("="*70)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nGPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU cache cleared")
else:
    print("\nWARNING: No GPU detected, training will be very slow")

# Load YOLOv8n model (nano - optimized for 4GB GPU)
print("\n[INFO] Loading YOLOv8n (Nano) model...")
model = YOLO('yolov8n.pt')

# Training parameters optimized for 4GB GPU
training_params = {
    'data': 'ppe_detection_dataset.yaml',
    'epochs': 25,  # Reduced from 50 - trains 2x faster!
    'imgsz': 320,
    'batch': 1,
    'device': 0,
    'project': 'ppe_detection_project',
    'name': 'ppe_yolov8n_fast',
    'exist_ok': True,
    'save_period': 5,  # Save checkpoints more frequently
    'workers': 4,  # Increased from 2 - utilize CPU better
    
    # Learning rate
    'lr0': 0.01,
    'lrf': 0.1,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 2,
    
    # Memory-saving: Disable heavy augmentations
    'mosaic': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'degrees': 0.0,
    'scale': 0.2,
    'fliplr': 0.5,
    'flipud': 0.0,
    
    # Memory optimization
    'cache': False,
    'amp': False,
    
    # Validation
    'val': True,
    'patience': 20,
    'save_json': False,
    'verbose': True,
    'plots': True,
}

print("\n[INFO] Training Configuration:")
print("="*70)
print(f"  Detection Classes:")
print(f"    - Class 0: Helmet")
print(f"    - Class 1: Safety Vest")
print(f"    - Class 2: Goggles")
print(f"    - Class 3: Gloves")
print(f"\n  Model Settings:")
print(f"    Model: YOLOv8n (Nano)")
print(f"    Image Size: {training_params['imgsz']}x{training_params['imgsz']}")
print(f"    Batch Size: {training_params['batch']}")
print(f"    Epochs: {training_params['epochs']}")
print(f"    Workers: {training_params['workers']}")
print(f"\n  Memory Optimizations:")
print(f"    Mosaic Augmentation: DISABLED")
print(f"    Image Caching: DISABLED")
print(f"    AMP Training: DISABLED")
print("="*70)

print("\n[INFO] Starting PPE Detection Training...")
print("[INFO] Estimated time: 2-3 hours")
print("[INFO] Monitor progress below...\n")

try:
    # Train the model
    results = model.train(**training_params)
    
    # Training completed
    print("\n" + "="*70)
    print("SUCCESS - TRAINING COMPLETED!")
    print("="*70)
    print(f"\nBest model saved: {results.save_dir}/weights/best.pt")
    print(f"Results directory: {results.save_dir}")
    
    # Validate the best model
    print("\n[INFO] Validating best model on test set...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(data='ppe_detection_dataset.yaml', imgsz=320)
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS")
    print("="*70)
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")
    print("="*70)
    
    print("\n[SUCCESS] PPE Detection model training complete!")
    print(f"\nTo use your model for detection:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{results.save_dir}/weights/best.pt')")
    print(f"  results = model('image.jpg')")
    print(f"  results[0].show()")
    
except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user")
    print("[INFO] Partial results saved in ppe_detection_project/")
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    print("\n[TROUBLESHOOTING]")
    print("  1. Ensure no other applications are using GPU memory")
    print("  2. Try restarting your computer")
    print("  3. Check dataset paths are correct")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory Status:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

