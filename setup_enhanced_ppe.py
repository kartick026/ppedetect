#!/usr/bin/env python3
"""
Enhanced PPE Detection System Setup Script
Automated setup for the complete system
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "datasets/ppe-balanced",
        "datasets/ppe-balanced/train/images",
        "datasets/ppe-balanced/train/labels", 
        "datasets/ppe-balanced/valid/images",
        "datasets/ppe-balanced/valid/labels",
        "datasets/ppe-balanced/test/images",
        "datasets/ppe-balanced/test/labels",
        "static/uploads",
        "static/detections",
        "static/results",
        "templates",
        "runs/detect"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Created directory: {directory}")

def create_data_yaml():
    """Create data.yaml configuration file"""
    data_config = {
        'path': 'datasets/ppe-balanced',
        'train': 'train/images',
        'val': 'valid/images', 
        'test': 'test/images',
        'nc': 4,
        'names': ['helmet', 'vest', 'gloves', 'glasses']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    print("[INFO] Created data.yaml configuration")

def install_requirements():
    """Install Python requirements"""
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("[WARNING] Some packages may have failed to install")
        return False
    return True

def download_roboflow_dataset():
    """Download dataset from Roboflow (requires API key)"""
    print("\n" + "="*60)
    print("ROBOFLOW DATASET SETUP")
    print("="*60)
    print("To download the Construction PPE Detection dataset:")
    print("1. Get your API key from: https://app.roboflow.com/settings/api")
    print("2. Run: python -c \"from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); project = rf.workspace('construction-ppe').project('construction-ppe-detection'); dataset = project.version(3).download('yolov8')\"")
    print("3. Move the downloaded dataset to datasets/ppe-balanced/")
    print("="*60)

def create_training_script():
    """Create the training script"""
    training_script = '''#!/usr/bin/env python3
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
'''
    
    with open('quick_train.py', 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    print("[INFO] Created quick_train.py")

def create_startup_script():
    """Create startup script for the API"""
    startup_script = '''#!/usr/bin/env python3
"""
Start the Enhanced PPE Detection API
"""

import uvicorn
from app import app

if __name__ == "__main__":
    print("🚀 Starting Enhanced PPE Detection API")
    print("="*50)
    print("API Documentation: http://localhost:8000/docs")
    print("Web Interface: http://localhost:8000")
    print("="*50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'''
    
    with open('start_api.py', 'w', encoding='utf-8') as f:
        f.write(startup_script)
    
    print("[INFO] Created start_api.py")

def create_docker_setup():
    """Create Docker configuration"""
    dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads static/detections static/results templates

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "start_api.py"]
'''

    with open('Dockerfile', 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    
    docker_compose_content = '''version: '3.8'

services:
  ppe-detection:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./static:/app/static
      - ./runs:/app/runs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''

    with open('docker-compose.yml', 'w', encoding='utf-8') as f:
        f.write(docker_compose_content)
    
    print("[INFO] Created Docker configuration")

def main():
    """Main setup function"""
    print("🚀 Enhanced PPE Detection System Setup")
    print("="*60)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create data.yaml
    print("\n📝 Creating data.yaml...")
    create_data_yaml()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("[WARNING] Some packages failed to install. Please check manually.")
    
    # Create scripts
    print("\n🔧 Creating scripts...")
    create_training_script()
    create_startup_script()
    create_docker_setup()
    
    # Dataset instructions
    download_roboflow_dataset()
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Download your dataset to datasets/ppe-balanced/")
    print("2. Train the model: python quick_train.py")
    print("3. Start the API: python start_api.py")
    print("4. Open browser: http://localhost:8000")
    print("="*60)

if __name__ == "__main__":
    main()
