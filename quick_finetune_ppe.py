#!/usr/bin/env python3
"""
Quick Fine-tuning for PPE Detection
Optimized for 1-2 hour training with pretrained YOLO model
"""

from ultralytics import YOLO
import torch
import os
import time
from datetime import datetime

print("="*70)
print("QUICK FINE-TUNING FOR PPE DETECTION")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def quick_finetune_ppe():
    """
    Quick fine-tuning optimized for 1-2 hours
    Uses pretrained YOLOv8n + your PPE data
    """
    print("\n[INFO] Starting quick fine-tuning process...")
    print("[INFO] This will take 1-2 hours for optimal results")
    
    # Load pretrained YOLOv8n model
    print("\n[STEP 1] Loading pretrained YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Auto-downloads if not exists
    print("[OK] Pretrained model loaded!")
    
    # Quick fine-tune parameters (optimized for speed + accuracy)
    print("\n[STEP 2] Starting fine-tuning with optimized parameters...")
    print("[INFO] Parameters optimized for 1-2 hour training:")
    print("  - Model: YOLOv8n (pretrained on COCO)")
    print("  - Epochs: 20 (quick but effective)")
    print("  - Image size: 416 (good balance)")
    print("  - Batch size: 4 (memory efficient)")
    print("  - Learning rate: 0.001 (fine-tuning rate)")
    print("  - Freeze layers: 10 (faster training)")
    
    start_time = time.time()
    
    # Fine-tune with optimized parameters
    results = model.train(
        data='ppe_detection_dataset.yaml',
        
        # Training duration (quick but effective)
        epochs=20,              # 20 epochs for quick fine-tuning
        patience=10,            # Early stopping if no improvement
        
        # Model architecture
        imgsz=416,             # Good balance of speed/accuracy
        batch=4,               # Memory efficient for 4GB GPU
        
        # Learning parameters
        lr0=0.001,            # Lower learning rate for fine-tuning
        momentum=0.937,        # Standard momentum
        weight_decay=0.0005,   # L2 regularization
        
        # Optimization settings
        workers=4,             # Parallel data loading
        device=0,             # GPU 0
        project='ppe_quick_finetune',
        name='yolov8n_ppe_20epochs',
        exist_ok=True,
        
        # Fine-tuning specific settings
        pretrained=True,      # Start with COCO weights
        freeze=10,            # Freeze first 10 layers (faster)
        
        # Data augmentation (reduced for faster convergence)
        mosaic=0.3,           # Reduced mosaic (was 1.0)
        mixup=0.0,            # Disabled for speed
        copy_paste=0.0,        # Disabled for speed
        
        # Validation
        val=True,             # Validate during training
        save_period=5,        # Save checkpoint every 5 epochs
        
        # Monitoring
        verbose=True,         # Show progress
        plots=True,           # Generate training plots
        save=True,            # Save model checkpoints
        
        # Memory optimizations
        amp=True,             # Automatic Mixed Precision (faster)
        cache=False,          # Disable image caching (saves memory)
    )
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # minutes
    
    print(f"\n[SUCCESS] Fine-tuning completed!")
    print(f"Training time: {training_time:.1f} minutes")
    print(f"Model saved: {results.save_dir}/weights/best.pt")
    
    return results, model

def test_finetuned_model(model_path):
    """
    Test the fine-tuned model on sample images
    """
    print("\n[STEP 3] Testing fine-tuned model...")
    
    # Load the fine-tuned model
    model = YOLO(model_path)
    
    # Test on a few sample images
    test_images = [
        'combined_datasets/images/test',
    ]
    
    print("[INFO] Running inference on test images...")
    results = model(test_images, conf=0.5, save=True)
    
    print("[OK] Test results saved!")
    return results

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("QUICK FINE-TUNING SETUP")
    print("="*70)
    
    # Check if dataset exists
    if not os.path.exists('ppe_detection_dataset.yaml'):
        print("[ERROR] Dataset config not found!")
        print("Make sure ppe_detection_dataset.yaml exists")
        return
    
    if not os.path.exists('combined_datasets'):
        print("[ERROR] Dataset directory not found!")
        print("Make sure combined_datasets/ exists with your data")
        return
    
    print("[OK] Dataset found and ready!")
    
    # Start fine-tuning
    try:
        results, model = quick_finetune_ppe()
        
        # Test the model
        model_path = f"{results.save_dir}/weights/best.pt"
        test_finetuned_model(model_path)
        
        print("\n" + "="*70)
        print("FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nYour fine-tuned model is ready:")
        print(f"Model path: {model_path}")
        print(f"Results directory: {results.save_dir}")
        
        print("\nNext steps:")
        print("1. Check training results in the results directory")
        print("2. Test the model on your images")
        print("3. Adjust confidence threshold if needed")
        print("4. Deploy for your use case")
        
        print(f"\nTraining plots: {results.save_dir}/results.png")
        print(f"Confusion matrix: {results.save_dir}/confusion_matrix.png")
        
    except Exception as e:
        print(f"\n[ERROR] Fine-tuning failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if you have enough GPU memory")
        print("2. Verify dataset paths are correct")
        print("3. Try reducing batch size to 2 or 1")
        print("4. Check if all dependencies are installed")

if __name__ == "__main__":
    main()

