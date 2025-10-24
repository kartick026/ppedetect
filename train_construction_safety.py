#!/usr/bin/env python3
"""
Train PPE Detection Model with Construction Safety Dataset
"""

from ultralytics import YOLO
import torch
import os

def train_construction_safety_model():
    """Train the model with Construction Safety dataset"""
    print("🚀 Training PPE Detection Model")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists("datasets/ppe-balanced/train/images"):
        print("❌ Dataset not found. Please download the Construction Safety dataset first.")
        print("Run: python integrate_roboflow.py")
        return False
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8m model
    print("Loading YOLOv8m model...")
    model = YOLO("yolov8m.pt")
    
    # Training configuration optimized for Construction Safety dataset
    training_config = {
        'data': 'data.yaml',
        'epochs': 100,
        'imgsz': 960,
        'batch': 16,
        'device': device,
        'project': 'runs/detect',
        'name': 'construction_safety_ppe',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 10,
        'cache': False,
        'amp': True,
        'verbose': True,
        # Augmentation parameters for construction safety
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("Starting training with Construction Safety dataset...")
    print(f"Model: YOLOv8m")
    print(f"Image size: {training_config['imgsz']}")
    print(f"Epochs: {training_config['epochs']}")
    print(f"Batch size: {training_config['batch']}")
    
    try:
        # Start training
        results = model.train(**training_config)
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: runs/detect/construction_safety_ppe/weights/best.pt")
        
        # Validate the model
        print("\nValidating model...")
        validation_results = model.val(
            data='data.yaml',
            imgsz=960,
            batch=16,
            conf=0.25,
            iou=0.45,
            device=device
        )
        
        print(f"mAP50: {validation_results.box.map50:.3f}")
        print(f"mAP50-95: {validation_results.box.map:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    train_construction_safety_model()
