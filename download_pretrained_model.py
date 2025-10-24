#!/usr/bin/env python3
"""
Download and Use Pretrained PPE Detection Models
Skip training completely by using existing models
"""

from ultralytics import YOLO
import torch
import os

print("="*70)
print("PRETRAINED PPE DETECTION MODEL DOWNLOADER")
print("="*70)

# Option 1: Use YOLO pretrained on COCO then fine-tune
def download_base_yolo_models():
    """Download base YOLO models pretrained on COCO dataset"""
    print("\n[INFO] Downloading base YOLO models pretrained on COCO...")
    
    models = {
        'yolov8n.pt': 'Nano - Fastest, smallest',
        'yolov8s.pt': 'Small - Good balance',
        'yolov8m.pt': 'Medium - Better accuracy',
        'yolov8l.pt': 'Large - High accuracy',
    }
    
    downloaded = []
    for model_name, description in models.items():
        try:
            print(f"\n[INFO] Downloading {model_name} ({description})...")
            model = YOLO(model_name)  # Auto-downloads if not exists
            print(f"[OK] {model_name} downloaded successfully!")
            downloaded.append(model_name)
        except Exception as e:
            print(f"[ERROR] Failed to download {model_name}: {e}")
    
    return downloaded

# Option 2: Fine-tune for just 5-10 epochs (quick adaptation)
def quick_finetune(model_name='yolov8n.pt'):
    """Quickly fine-tune pretrained model on your PPE dataset (1-2 hours)"""
    print(f"\n[INFO] Quick fine-tuning {model_name} on your PPE data...")
    print("[INFO] This will take 1-2 hours instead of 20+ hours")
    
    model = YOLO(model_name)
    
    # Quick fine-tune parameters
    results = model.train(
        data='ppe_detection_dataset.yaml',
        epochs=10,          # Just 10 epochs for quick adaptation
        imgsz=320,
        batch=2,
        workers=4,
        device=0,
        project='ppe_pretrained_models',
        name='quick_finetune',
        exist_ok=True,
        patience=5,
        
        # Start with pretrained weights
        pretrained=True,   # Keep COCO knowledge
        
        # Freeze early layers (faster training)
        freeze=10,         # Freeze first 10 layers
        
        # Reduced augmentation for faster convergence
        mosaic=0.0,
        mixup=0.0,
        lr0=0.001,        # Lower learning rate for fine-tuning
        
        verbose=True,
    )
    
    print(f"\n[SUCCESS] Fine-tuned model ready!")
    print(f"Model saved: {results.save_dir}/weights/best.pt")
    return results

# Option 3: Download community PPE models
def download_community_ppe_models():
    """Guide to download community-trained PPE models"""
    print("\n[INFO] Community PPE Detection Models")
    print("="*70)
    print("\nPopular pretrained PPE models from Ultralytics Hub:")
    print("\n1. Official YOLO Models (General Object Detection):")
    print("   - Already includes 'person' class")
    print("   - Can detect basic objects")
    print("   - Good starting point")
    
    print("\n2. Custom PPE Models from Community:")
    print("   - Search: https://hub.ultralytics.com/")
    print("   - Look for: 'PPE', 'Safety Equipment', 'Construction Safety'")
    print("   - Download .pt files directly")
    
    print("\n3. Roboflow Universe:")
    print("   - URL: https://universe.roboflow.com/")
    print("   - Search: 'PPE detection', 'hard hat', 'safety vest'")
    print("   - Can download pretrained models directly")
    
    print("\n[INFO] To use a downloaded model:")
    print("   model = YOLO('path/to/downloaded_model.pt')")
    print("   results = model('image.jpg')")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("CHOOSE AN OPTION:")
    print("="*70)
    print("\n1. Download base YOLO models (pretrained on COCO)")
    print("2. Quick fine-tune on your data (1-2 hours)")
    print("3. Get community PPE models info")
    print("4. All of the above")
    
    choice = "2"  # Default: Quick fine-tune (most practical)
    
    print(f"\n[INFO] Running option: Quick fine-tune (1-2 hours)")
    print("[INFO] This combines pretrained knowledge + your PPE data")
    
    # Download base model
    print("\n" + "="*70)
    print("STEP 1: Downloading Base Model")
    print("="*70)
    model = YOLO('yolov8n.pt')
    print("[OK] Base model downloaded!")
    
    # Show info about community models
    print("\n" + "="*70)
    print("STEP 2: Community Resources")
    print("="*70)
    download_community_ppe_models()
    
    print("\n" + "="*70)
    print("STEP 3: Quick Fine-tune Option")
    print("="*70)
    print("\nYou can now:")
    print("  A) Use base yolov8n.pt as-is (detects generic objects)")
    print("  B) Quick fine-tune for 1-2 hours (recommended)")
    print("  C) Download community PPE model from links above")
    
    print("\n[INFO] To quick fine-tune (1-2 hours):")
    print("  quick_finetune('yolov8n.pt')")
    
    return model

if __name__ == "__main__":
    model = main()
    
    print("\n" + "="*70)
    print("PRETRAINED MODEL READY!")
    print("="*70)
    print("\nYou now have yolov8n.pt downloaded.")
    print("\nNext steps:")
    print("  1. Use it directly for general object detection")
    print("  2. Fine-tune on your PPE data (run quick_finetune)")
    print("  3. Or download community PPE models from links above")




