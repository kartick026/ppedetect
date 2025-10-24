#!/usr/bin/env python3
"""
Download Construction Safety Dataset from Roboflow
Dataset: kartick025/construction-safety-n0gkb-vd2dp-instant-1
"""

import os
import shutil
from pathlib import Path

def download_construction_safety_dataset():
    """Download the Construction Safety dataset from Roboflow"""
    print("="*70)
    print("CONSTRUCTION SAFETY DATASET DOWNLOAD")
    print("="*70)
    print("Dataset: kartick025/construction-safety-n0gkb-vd2dp-instant-1")
    print("="*70)
    
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow
        print("[INFO] Initializing Roboflow...")
        rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # Replace with your API key
        
        # Download the specific dataset
        print("[INFO] Downloading Construction Safety dataset...")
        project = rf.workspace("kartick025").project("construction-safety")
        dataset = project.version(1).download("yolov8")
        
        print(f"[SUCCESS] Dataset downloaded to: {dataset.location}")
        
        # Move to our datasets directory
        target_dir = "datasets/ppe-balanced"
        if os.path.exists(dataset.location):
            print(f"[INFO] Moving dataset to {target_dir}...")
            
            # Remove existing dataset if it exists
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            # Move the downloaded dataset
            shutil.move(dataset.location, target_dir)
            print(f"[SUCCESS] Dataset moved to {target_dir}")
            
            # Update data.yaml
            update_data_yaml(target_dir)
            
            return True
        else:
            print(f"[ERROR] Dataset not found at {dataset.location}")
            return False
            
    except ImportError:
        print("[ERROR] Roboflow not installed. Install with: pip install roboflow")
        print("[INFO] Manual download instructions:")
        print("1. Get your API key from: https://app.roboflow.com/settings/api")
        print("2. Install roboflow: pip install roboflow")
        print("3. Run this script again")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[INFO] Manual download instructions:")
        print("1. Go to: https://app.roboflow.com/kartick025/construction-safety/1")
        print("2. Click 'Download' -> 'YOLOv8'")
        print("3. Extract to datasets/ppe-balanced/")
        return False

def update_data_yaml(dataset_path):
    """Update data.yaml with the correct paths"""
    data_config = {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['helmet', 'vest', 'gloves', 'glasses']
    }
    
    with open('data.yaml', 'w') as f:
        import yaml
        yaml.dump(data_config, f)
    
    print("[INFO] Updated data.yaml configuration")
    print(f"[INFO] Dataset path: {os.path.abspath(dataset_path)}")

def verify_dataset_structure():
    """Verify the dataset structure"""
    dataset_path = "datasets/ppe-balanced"
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        return False
    
    required_dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels", 
        "test/images", "test/labels"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"[WARNING] Missing directories: {missing_dirs}")
        return False
    
    # Count files
    train_images = len([f for f in os.listdir(os.path.join(dataset_path, "train/images")) if f.endswith(('.jpg', '.jpeg', '.png'))])
    train_labels = len([f for f in os.listdir(os.path.join(dataset_path, "train/labels")) if f.endswith('.txt')])
    
    print(f"[INFO] Dataset structure verified:")
    print(f"  Train images: {train_images}")
    print(f"  Train labels: {train_labels}")
    
    return True

def create_roboflow_script():
    """Create a script for manual Roboflow download"""
    script_content = '''#!/usr/bin/env python3
"""
Manual Roboflow Dataset Download Script
Replace YOUR_API_KEY with your actual Roboflow API key
"""

from roboflow import Roboflow
import os
import shutil

# Replace with your actual API key
API_KEY = "YOUR_API_KEY"

def download_dataset():
    # Initialize Roboflow
    rf = Roboflow(api_key=API_KEY)
    
    # Download the Construction Safety dataset
    project = rf.workspace("kartick025").project("construction-safety")
    dataset = project.version(1).download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Move to datasets directory
    target_dir = "datasets/ppe-balanced"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    shutil.move(dataset.location, target_dir)
    print(f"Dataset moved to: {target_dir}")

if __name__ == "__main__":
    download_dataset()
'''
    
    with open('download_roboflow_manual.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("[INFO] Created download_roboflow_manual.py")
    print("[INFO] Edit the file and add your Roboflow API key")

def main():
    """Main function"""
    print("🚀 Construction Safety Dataset Integration")
    print("="*50)
    
    # Check if dataset already exists
    if os.path.exists("datasets/ppe-balanced"):
        print("[INFO] Dataset directory already exists")
        if verify_dataset_structure():
            print("[SUCCESS] Dataset structure is valid")
            return
        else:
            print("[WARNING] Dataset structure is incomplete")
    
    # Try to download the dataset
    success = download_construction_safety_dataset()
    
    if not success:
        print("\n" + "="*50)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*50)
        print("1. Get your Roboflow API key from: https://app.roboflow.com/settings/api")
        print("2. Install roboflow: pip install roboflow")
        print("3. Edit download_roboflow_manual.py with your API key")
        print("4. Run: python download_roboflow_manual.py")
        print("="*50)
        
        # Create manual download script
        create_roboflow_script()
    
    # Verify the dataset
    if verify_dataset_structure():
        print("\n✅ Dataset ready for training!")
        print("Next steps:")
        print("1. Train the model: python quick_train.py")
        print("2. Start the API: python start_api.py")
    else:
        print("\n❌ Dataset verification failed")

if __name__ == "__main__":
    main()
