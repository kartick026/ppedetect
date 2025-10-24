#!/usr/bin/env python3
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
