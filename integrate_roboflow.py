#!/usr/bin/env python3
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
