#!/usr/bin/env python3
"""
PPE Detection System - Application Starter
Convenient script to start the application with proper configuration
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Start the PPE Detection System"""
    print("=" * 60)
    print("🛡️  PPE Detection System")
    print("=" * 60)
    print("Starting application...")
    
    try:
        # Import and run the main application
        from src.web_app.main import app, load_model
        
        # Load model first
        if load_model():
            print("✅ Model loaded successfully")
            print("🌐 Starting web server...")
            print("📱 Open your browser and go to: http://localhost:5000")
            print("🛑 Press Ctrl+C to stop the server")
            print("=" * 60)
            
            # Start the Flask app
            app.run(
                host='0.0.0.0',
                port=5000,
                debug=True,
                threaded=True
            )
        else:
            print("❌ Failed to load model. Please check your model files.")
            print("📁 Expected model path: src/models/ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r config/requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
