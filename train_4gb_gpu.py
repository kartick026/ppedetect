#!/usr/bin/env python3
"""
Memory-Optimized YOLO Training for 4GB GPU (GTX 1650)
Ultra-low memory configuration for successful training
"""

from ultralytics import YOLO
import torch
import gc

print("="*60)
print("YOLO Training - Optimized for 4GB GPU (GTX 1650)")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("WARNING: No GPU detected, training will be very slow")

# Load smaller YOLOv8n model (nano - smallest)
print("\n[INFO] Loading YOLOv8n (Nano) model...")
model = YOLO('yolov8n.pt')

# Memory-optimized training parameters for 4GB GPU
training_params = {
    'data': 'glove_detection_dataset.yaml',
    'epochs': 50,           # Reduced epochs
    'imgsz': 320,           # Very small image size (was 416)
    'batch': 1,             # Smallest possible batch
    'device': 0,
    'project': 'glove_detection_project',
    'name': 'run_4gb_optimized',
    'exist_ok': True,
    'save_period': 10,
    'workers': 2,           # Reduced workers (was 8)
    
    # Learning rate
    'lr0': 0.01,
    'lrf': 0.1,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 2,     # Reduced warmup
    
    # CRITICAL: Disable memory-intensive augmentations
    'mosaic': 0.0,          # Disable mosaic (uses lots of memory)
    'mixup': 0.0,           # Disable mixup
    'copy_paste': 0.0,      # Disable copy-paste
    'degrees': 0.0,         # Minimal augmentation
    'scale': 0.2,           # Reduced scaling
    'fliplr': 0.5,          # Keep horizontal flip only
    'flipud': 0.0,
    
    # Memory optimization
    'cache': False,         # Don't cache images in RAM
    'amp': False,           # Disable AMP (was causing issues)
    
    # Validation
    'val': True,
    'patience': 20,         # Early stopping
    'save_json': False,     # Reduce disk I/O
    'verbose': True,
    'plots': True,
}

print("\n[INFO] Training Configuration:")
print(f"  Model: YOLOv8n (Nano - smallest)")
print(f"  Image Size: {training_params['imgsz']}")
print(f"  Batch Size: {training_params['batch']}")
print(f"  Epochs: {training_params['epochs']}")
print(f"  Workers: {training_params['workers']}")
print(f"  Mosaic Augmentation: DISABLED (saves memory)")
print(f"  Cache: DISABLED (saves RAM)")
print("\n[INFO] This configuration is optimized for 4GB GPU memory")
print("="*60)

try:
    print("\n[INFO] Starting training...")
    print("[INFO] Training will take approximately 2-3 hours")
    print("[INFO] You can monitor progress in the terminal\n")
    
    # Train the model
    results = model.train(**training_params)
    
    # Training completed
    print("\n" + "="*60)
    print("[SUCCESS] Training completed successfully!")
    print("="*60)
    print(f"Model saved to: {results.save_dir}/weights/best.pt")
    print(f"Results saved to: {results.save_dir}")
    
    # Validation on best model
    print("\n[INFO] Running validation on best model...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(data='glove_detection_dataset.yaml', imgsz=320)
    
    print("\n[RESULTS] Model Performance:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    print("\n[SUCCESS] Training pipeline completed!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Latest model: {results.save_dir}/weights/last.pt")
    
except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user")
    print("[INFO] Partial results may be available in glove_detection_project/")
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    print("\n[INFO] Troubleshooting tips:")
    print("  1. Close all other applications to free up memory")
    print("  2. Restart your computer to clear system memory")
    print("  3. If still failing, your GPU may need even smaller settings")
    print("  4. Consider using CPU training (much slower but no memory limit)")
    
    # Print memory stats if available
    if torch.cuda.is_available():
        print(f"\n[INFO] GPU Memory Stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")





