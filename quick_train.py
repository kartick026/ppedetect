#!/usr/bin/env python3
"""
Quick training script for enhanced PPE detection
"""

from ultralytics import YOLO
import torch

def main():
    print("🚀 Starting Enhanced PPE Training")
    print("="*50)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8m model
    model = YOLO("yolov8m.pt")
    
    # Training configuration
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=960,
        batch=16,
        device=device,
        project="runs/detect",
        name="ppe_detector_v2",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        cache=False,
        amp=True,
        verbose=True
    )
    
    print("✅ Training completed!")
    print("📁 Model saved to: runs/detect/ppe_detector_v2/weights/best.pt")

if __name__ == "__main__":
    main()
