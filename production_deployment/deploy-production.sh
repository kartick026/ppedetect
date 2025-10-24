#!/bin/bash
# Production Deployment Script

echo "Deploying PPE Detection System to production..."

# Set environment variables
export FLASK_ENV=production
export MODEL_PATH=ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt

# Install dependencies
pip install -r requirements.txt

# Start with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 ppe_web_app:app

echo "Production deployment complete!"
