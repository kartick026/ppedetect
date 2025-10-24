#!/usr/bin/env python3
"""
Integrate Construction Safety Dataset
Dataset: kartick025/construction-safety-n0gkb-vd2dp-instant-1
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path

def download_construction_safety_dataset():
    """Download the Construction Safety dataset"""
    print("="*70)
    print("CONSTRUCTION SAFETY DATASET INTEGRATION")
    print("="*70)
    print("Dataset: kartick025/construction-safety-n0gkb-vd2dp-instant-1")
    print("="*70)
    
    # Dataset information
    dataset_info = {
        "name": "Construction Safety Dataset",
        "workspace": "kartick025",
        "project": "construction-safety", 
        "version": 1,
        "classes": ["helmet", "vest", "gloves", "glasses"],
        "format": "yolov8"
    }
    
    print(f"[INFO] Dataset: {dataset_info['name']}")
    print(f"[INFO] Classes: {dataset_info['classes']}")
    print(f"[INFO] Format: {dataset_info['format']}")
    
    # Create dataset directory structure
    dataset_path = "datasets/ppe-balanced"
    create_dataset_structure(dataset_path)
    
    # Instructions for manual download
    print("\n" + "="*50)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*50)
    print("1. Go to: https://app.roboflow.com/kartick025/construction-safety/1")
    print("2. Click 'Download' -> 'YOLOv8'")
    print("3. Extract the downloaded zip file")
    print("4. Copy contents to datasets/ppe-balanced/")
    print("="*50)
    
    # Create sample data structure
    create_sample_data_structure(dataset_path)
    
    # Update data.yaml
    update_data_yaml(dataset_path)
    
    print(f"\n[SUCCESS] Dataset structure created at: {dataset_path}")
    print("[INFO] Please download the actual dataset and place it in the directory")

def create_dataset_structure(dataset_path):
    """Create the dataset directory structure"""
    directories = [
        f"{dataset_path}/train/images",
        f"{dataset_path}/train/labels",
        f"{dataset_path}/valid/images", 
        f"{dataset_path}/valid/labels",
        f"{dataset_path}/test/images",
        f"{dataset_path}/test/labels"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Created directory: {directory}")

def create_sample_data_structure(dataset_path):
    """Create sample data structure for testing"""
    # Create sample data.yaml
    data_yaml_content = f"""# Construction Safety Dataset Configuration
path: {os.path.abspath(dataset_path)}
train: train/images
val: valid/images
test: test/images

# Number of classes
nc: 4

# Class names
names:
  0: helmet
  1: vest
  2: gloves
  3: glasses
"""
    
    with open('data.yaml', 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    
    print("[INFO] Created data.yaml configuration")

def update_data_yaml(dataset_path):
    """Update data.yaml with correct paths"""
    data_config = {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['helmet', 'vest', 'gloves', 'glasses']
    }
    
    with open('data.yaml', 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(data_config, f, default_flow_style=False)
    
    print("[INFO] Updated data.yaml configuration")

def create_roboflow_integration_script():
    """Create a script for Roboflow integration"""
    script_content = '''#!/usr/bin/env python3
"""
Roboflow Construction Safety Dataset Integration
"""

import os
import shutil
from pathlib import Path

def integrate_roboflow_dataset():
    """Integrate the downloaded Roboflow dataset"""
    
    # Check if roboflow is installed
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system("pip install roboflow")
        from roboflow import Roboflow
    
    # Initialize Roboflow
    print("Enter your Roboflow API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("No API key provided. Please get one from: https://app.roboflow.com/settings/api")
        return False
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Download the Construction Safety dataset
        print("Downloading Construction Safety dataset...")
        project = rf.workspace("kartick025").project("construction-safety")
        dataset = project.version(1).download("yolov8")
        
        print(f"Dataset downloaded to: {dataset.location}")
        
        # Move to our datasets directory
        target_dir = "datasets/ppe-balanced"
        
        # Remove existing dataset if it exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # Move the downloaded dataset
        shutil.move(dataset.location, target_dir)
        print(f"Dataset moved to: {target_dir}")
        
        # Verify the dataset
        verify_dataset_structure(target_dir)
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def verify_dataset_structure(dataset_path):
    """Verify the dataset structure"""
    required_dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels"
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"Missing directory: {dir_path}")
            return False
    
    # Count files
    train_images = len([f for f in os.listdir(os.path.join(dataset_path, "train/images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
    train_labels = len([f for f in os.listdir(os.path.join(dataset_path, "train/labels")) if f.endswith('.txt')])
    
    print(f"Dataset structure verified:")
    print(f"  Train images: {train_images}")
    print(f"  Train labels: {train_labels}")
    
    return True

if __name__ == "__main__":
    integrate_roboflow_dataset()
'''
    
    with open('integrate_roboflow.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("[INFO] Created integrate_roboflow.py for automated download")

def create_training_script_with_dataset():
    """Create a training script specifically for the Construction Safety dataset"""
    training_script = '''#!/usr/bin/env python3
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
        print("\\nValidating model...")
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
'''
    
    with open('train_construction_safety.py', 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    print("[INFO] Created train_construction_safety.py")

def main():
    """Main function"""
    print("🚀 Construction Safety Dataset Integration")
    print("="*60)
    
    # Create dataset structure
    download_construction_safety_dataset()
    
    # Create integration scripts
    create_roboflow_integration_script()
    create_training_script_with_dataset()
    
    print("\n" + "="*60)
    print("✅ INTEGRATION COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Get your Roboflow API key from: https://app.roboflow.com/settings/api")
    print("2. Run: python integrate_roboflow.py")
    print("3. Train the model: python train_construction_safety.py")
    print("4. Start the API: python start_api.py")
    print("="*60)

if __name__ == "__main__":
    main()
