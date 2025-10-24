#!/usr/bin/env python3
"""
Download Construction Safety Dataset with API Key
Dataset: kartick025/construction-safety-n0gkb-vd2dp-instant-1
"""

import os
import shutil
from pathlib import Path

def download_with_api_key():
    """Download the Construction Safety dataset using the provided API key"""
    api_key = "HgnvZhJ7BJaxCF94uuTI"
    
    print("="*70)
    print("CONSTRUCTION SAFETY DATASET DOWNLOAD")
    print("="*70)
    print("Dataset: kartick025/construction-safety-n0gkb-vd2dp-instant-1")
    print("API Key: Provided")
    print("="*70)
    
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow with the API key
        print("[INFO] Initializing Roboflow with API key...")
        rf = Roboflow(api_key=api_key)
        
        # Download the Construction Safety dataset
        print("[INFO] Downloading Construction Safety dataset...")
        project = rf.workspace("kartick025").project("construction-safety")
        dataset = project.version(1).download("yolov8")
        
        print(f"[SUCCESS] Dataset downloaded to: {dataset.location}")
        
        # Move to our datasets directory
        target_dir = "datasets/ppe-balanced"
        
        # Remove existing dataset if it exists
        if os.path.exists(target_dir):
            print(f"[INFO] Removing existing dataset at {target_dir}")
            shutil.rmtree(target_dir)
        
        # Move the downloaded dataset
        print(f"[INFO] Moving dataset to {target_dir}...")
        shutil.move(dataset.location, target_dir)
        print(f"[SUCCESS] Dataset moved to {target_dir}")
        
        # Verify the dataset structure
        verify_dataset_structure(target_dir)
        
        # Update data.yaml
        update_data_yaml(target_dir)
        
        return True
        
    except ImportError:
        print("[ERROR] Roboflow not installed. Installing...")
        os.system("pip install roboflow")
        print("[INFO] Please run the script again after installation")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        return False

def verify_dataset_structure(dataset_path):
    """Verify the dataset structure"""
    required_dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels"
    ]
    
    print("\n[INFO] Verifying dataset structure...")
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"[WARNING] Missing directory: {dir_path}")
        else:
            print(f"[OK] Found directory: {dir_path}")
    
    # Count files
    try:
        train_images = len([f for f in os.listdir(os.path.join(dataset_path, "train/images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
        train_labels = len([f for f in os.listdir(os.path.join(dataset_path, "train/labels")) if f.endswith('.txt')])
        
        valid_images = len([f for f in os.listdir(os.path.join(dataset_path, "valid/images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
        valid_labels = len([f for f in os.listdir(os.path.join(dataset_path, "valid/labels")) if f.endswith('.txt')])
        
        test_images = len([f for f in os.listdir(os.path.join(dataset_path, "test/images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
        test_labels = len([f for f in os.listdir(os.path.join(dataset_path, "test/labels")) if f.endswith('.txt')])
        
        print(f"\n[INFO] Dataset statistics:")
        print(f"  Train images: {train_images}")
        print(f"  Train labels: {train_labels}")
        print(f"  Valid images: {valid_images}")
        print(f"  Valid labels: {valid_labels}")
        print(f"  Test images: {test_images}")
        print(f"  Test labels: {test_labels}")
        
        total_images = train_images + valid_images + test_images
        total_labels = train_labels + valid_labels + test_labels
        
        print(f"  Total images: {total_images}")
        print(f"  Total labels: {total_labels}")
        
        if total_images > 0 and total_labels > 0:
            print("[SUCCESS] Dataset structure verified!")
            return True
        else:
            print("[WARNING] Dataset appears to be empty")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error verifying dataset: {e}")
        return False

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
    
    print(f"[INFO] Updated data.yaml configuration")
    print(f"[INFO] Dataset path: {os.path.abspath(dataset_path)}")

def main():
    """Main function"""
    print("🚀 Construction Safety Dataset Download")
    print("="*60)
    
    # Download the dataset
    success = download_with_api_key()
    
    if success:
        print("\n" + "="*60)
        print("✅ DATASET DOWNLOAD COMPLETED!")
        print("="*60)
        print("Next steps:")
        print("1. Train the model: python train_construction_safety.py")
        print("2. Start the API: python start_api.py")
        print("3. Open browser: http://localhost:8000")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ DATASET DOWNLOAD FAILED!")
        print("="*60)
        print("Please check the error messages above and try again.")
        print("="*60)

if __name__ == "__main__":
    main()
