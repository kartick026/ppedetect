#!/usr/bin/env python3
"""
PPE Detection System Configuration
Centralized configuration for the application
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
DATASETS_DIR = PROJECT_ROOT / "src" / "datasets"

# Model configuration
MODEL_CONFIG = {
    "ppe_model_path": "src/models/ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt",
    "person_model_path": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100
}

# Dataset configuration
DATASET_CONFIG = {
    "train_path": "src/datasets/combined_datasets/images/train",
    "valid_path": "src/datasets/combined_datasets/images/valid", 
    "test_path": "src/datasets/combined_datasets/images/test",
    "classes": ["helmet", "safety_vest", "goggles", "gloves"],
    "num_classes": 4
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "threaded": True
}

# Camera configuration
CAMERA_CONFIG = {
    "default_camera": 0,
    "resolution": (640, 480),
    "fps": 30,
    "buffer_size": 1
}

# Detection configuration
DETECTION_CONFIG = {
    "ppe_classes": ["helmet", "safety_vest", "goggles", "gloves"],
    "person_class": "person",
    "compliance_required": [],  # Empty - any PPE detection is compliant
    "optional_ppe": ["helmet", "safety_vest", "goggles", "gloves"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/ppe_detection.log"
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "static" / "css",
        PROJECT_ROOT / "static" / "js", 
        PROJECT_ROOT / "static" / "images",
        TEMPLATES_DIR / "main",
        TEMPLATES_DIR / "live",
        TEMPLATES_DIR / "components",
        MODELS_DIR,
        DATASETS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directories()
