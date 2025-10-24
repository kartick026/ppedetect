#!/usr/bin/env python3
"""
Enhanced PPE Detection Training Script
Using YOLOv8m with improved configuration for better accuracy
"""

import os
import yaml
from ultralytics import YOLO
import torch

def setup_roboflow_dataset():
    """Setup Roboflow dataset for Construction PPE Detection"""
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow
        rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # Replace with actual API key
        
        # Download dataset
        project = rf.workspace("construction-ppe").project("construction-ppe-detection")
        dataset = project.version(3).download("yolov8")
        
        print(f"[INFO] Dataset downloaded to: {dataset.location}")
        return dataset.location
        
    except ImportError:
        print("[WARNING] Roboflow not installed. Install with: pip install roboflow")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        return None

def create_data_yaml(dataset_path):
    """Create data.yaml configuration file"""
    data_config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['helmet', 'vest', 'gloves', 'glasses']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    print("[INFO] Created data.yaml configuration")
    return 'data.yaml'

def apply_augmentations():
    """Apply dataset augmentations for better training"""
    augmentation_config = {
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 15.0,  # Rotation degrees
        'translate': 0.1, # Translation
        'scale': 0.5,     # Scale range (0.5-1.5)
        'shear': 0.0,     # Shear
        'perspective': 0.0, # Perspective
        'flipud': 0.0,    # Vertical flip
        'fliplr': 0.5,    # Horizontal flip (50%)
        'mosaic': 1.0,    # Mosaic augmentation
        'mixup': 0.0,     # Mixup augmentation
        'copy_paste': 0.0 # Copy-paste augmentation
    }
    return augmentation_config

def train_ppe_model():
    """Train the enhanced PPE detection model"""
    print("="*70)
    print("ENHANCED PPE DETECTION TRAINING")
    print("="*70)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # Load YOLOv8m model
    print("[INFO] Loading YOLOv8m model...")
    model = YOLO("yolov8m.pt")
    
    # Training configuration
    training_config = {
        'data': 'data.yaml',
        'epochs': 100,
        'imgsz': 960,           # Higher resolution for small objects
        'batch': 16,            # Batch size
        'device': device,
        'workers': 8,
        'project': 'runs/detect',
        'name': 'ppe_detector_v2',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',    # Better optimizer
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.01,            # Final learning rate
        'momentum': 0.937,      # SGD momentum
        'weight_decay': 0.0005, # Weight decay
        'warmup_epochs': 3,     # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,   # Warmup bias learning rate
        'box': 7.5,             # Box loss gain
        'cls': 0.5,             # Class loss gain
        'dfl': 1.5,             # DFL loss gain
        'pose': 12.0,           # Pose loss gain
        'kobj': 2.0,            # Keypoint object loss gain
        'label_smoothing': 0.0, # Label smoothing
        'nbs': 64,              # Nominal batch size
        'overlap_mask': True,   # Overlap mask
        'mask_ratio': 4,        # Mask ratio
        'dropout': 0.0,         # Dropout
        'val': True,            # Validate during training
        'plots': True,          # Save plots
        'save': True,           # Save checkpoints
        'save_period': 10,      # Save checkpoint every N epochs
        'cache': False,         # Cache images
        'rect': False,          # Rectangular training
        'cos_lr': False,        # Cosine LR scheduler
        'close_mosaic': 10,     # Close mosaic augmentation
        'resume': False,        # Resume training
        'amp': True,            # Automatic Mixed Precision
        'fraction': 1.0,        # Dataset fraction
        'profile': False,       # Profile ONNX and TensorRT speeds
        'freeze': None,         # Freeze layers
        'multi_scale': False,   # Multi-scale training
        'single_cls': False,   # Single class training
        'augment': True,        # Augmentation
        'verbose': True,        # Verbose output
        'seed': 0,              # Random seed
        'deterministic': True,  # Deterministic training
        'workers': 8,           # Number of workers
        'patience': 50,         # Early stopping patience
        'save_txt': False,      # Save results to txt
        'save_conf': False,     # Save confidences
        'save_crop': False,     # Save cropped images
        'show_labels': True,    # Show labels
        'show_conf': True,      # Show confidences
        'vid_stride': 1,        # Video frame stride
        'stream_buffer': False, # Stream buffer
        'line_width': 3,        # Line width
        'visualize': False,     # Visualize features
        'augment': True,        # Augmentation
        'agnostic_nms': False,  # Agnostic NMS
        'retina_masks': False,  # Retina masks
        'format': 'torchscript', # Export format
        'keras': False,         # Keras export
        'optimize': False,      # TorchScript optimization
        'int8': False,          # INT8 quantization
        'dynamic': False,       # Dynamic axes
        'simplify': False,      # Simplify model
        'opset': None,          # ONNX opset version
        'workspace': 4,         # TensorRT workspace size
        'nms': False,           # NMS in ONNX export
        'batch': 1,             # Batch size for export
    }
    
    # Add augmentation parameters
    augmentation_params = apply_augmentations()
    training_config.update(augmentation_params)
    
    print("[INFO] Starting training with enhanced configuration...")
    print(f"[INFO] Model: YOLOv8m")
    print(f"[INFO] Image size: {training_config['imgsz']}")
    print(f"[INFO] Epochs: {training_config['epochs']}")
    print(f"[INFO] Batch size: {training_config['batch']}")
    print(f"[INFO] Device: {device}")
    
    # Start training
    try:
        results = model.train(**training_config)
        print("[SUCCESS] Training completed successfully!")
        print(f"[INFO] Best model saved to: runs/detect/ppe_detector_v2/weights/best.pt")
        return results
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return None

def validate_model():
    """Validate the trained model"""
    print("\n" + "="*50)
    print("MODEL VALIDATION")
    print("="*50)
    
    try:
        # Load the best model
        model_path = "runs/detect/ppe_detector_v2/weights/best.pt"
        model = YOLO(model_path)
        
        # Validate on test set
        results = model.val(
            data='data.yaml',
            imgsz=960,
            batch=16,
            conf=0.25,
            iou=0.45,
            max_det=300,
            half=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,
            verbose=True
        )
        
        print("[INFO] Validation completed!")
        print(f"[INFO] mAP50: {results.box.map50:.3f}")
        print(f"[INFO] mAP50-95: {results.box.map:.3f}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return None

def main():
    """Main training pipeline"""
    print("🚀 Enhanced PPE Detection Training Pipeline")
    print("="*70)
    
    # Check if data.yaml exists, if not create it
    if not os.path.exists('data.yaml'):
        print("[INFO] data.yaml not found. Please ensure your dataset is properly configured.")
        print("[INFO] Expected structure:")
        print("  data.yaml")
        print("  ├── train/images/")
        print("  ├── train/labels/")
        print("  ├── valid/images/")
        print("  ├── valid/labels/")
        print("  ├── test/images/")
        print("  └── test/labels/")
        return
    
    # Train the model
    results = train_ppe_model()
    
    if results:
        # Validate the model
        validation_results = validate_model()
        
        if validation_results:
            print("\n🎉 Training pipeline completed successfully!")
            print("📁 Model saved to: runs/detect/ppe_detector_v2/weights/best.pt")
            print("📊 Check the results in: runs/detect/ppe_detector_v2/")
        else:
            print("❌ Validation failed")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()
