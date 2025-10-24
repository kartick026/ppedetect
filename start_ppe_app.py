#!/usr/bin/env python3
"""
Start the PPE Detection Web Application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask-cors', 'ultralytics', 'opencv-python', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model():
    """Check if the trained model exists"""
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a model first or download a pretrained model.")
        return False
    
    print(f"✅ Model found: {model_path}")
    return True

def start_app():
    """Start the PPE detection web application"""
    print("="*70)
    print("STARTING PPE DETECTION WEB APPLICATION")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Check model
    if not check_model():
        return False
    
    print("\n[INFO] Starting Flask application...")
    print("[INFO] Web interface will be available at: http://localhost:5000")
    print("[INFO] Press Ctrl+C to stop the application")
    print("="*70)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "ppe_web_app.py"])
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_app()
