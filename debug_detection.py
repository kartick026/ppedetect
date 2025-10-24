#!/usr/bin/env python3
"""
Debug PPE Detection Issues
"""

import requests
import json
import os
from pathlib import Path

def debug_detection():
    """Debug detection issues with detailed output"""
    print("="*70)
    print("DEBUGGING PPE DETECTION ISSUES")
    print("="*70)
    
    # Check if Flask app is running
    try:
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✅ Flask app is running on port 5000")
        else:
            print("❌ Flask app not responding on port 5000")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app. Make sure it's running:")
        print("   python ppe_web_app.py")
        return False
    
    # Test with a sample image
    test_images = [
        "test_result_1.jpg",
        "test_result_2.jpg", 
        "test_result_3.jpg",
        "test_result_4.jpg"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n[DEBUG] Testing with: {test_image}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'image': f}
                    response = requests.post('http://localhost:5000/detect', files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✅ Detection successful!")
                    print(f"  📊 Results:")
                    print(f"    - Total detections: {result.get('total_detections', 0)}")
                    print(f"    - People count: {result.get('num_people', 0)}")
                    print(f"    - Compliance: {result.get('compliance_status', 'UNKNOWN')}")
                    print(f"    - Missing PPE: {result.get('missing_ppe', [])}")
                    
                    if result.get('detections'):
                        print(f"    - Detection details:")
                        for i, det in enumerate(result['detections'][:5]):  # Show first 5
                            print(f"      {i+1}. {det['class']}: {det['confidence']:.2f} confidence")
                    else:
                        print(f"    - No detections found")
                        
                else:
                    print(f"  ❌ Detection failed: {response.status_code}")
                    print(f"  Error: {response.text}")
                    
            except Exception as e:
                print(f"  ❌ Error testing {test_image}: {e}")
        else:
            print(f"[DEBUG] Test image {test_image} not found")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE!")
    print("="*70)
    print("\nIf you're still seeing incorrect counts:")
    print("1. Check if the model is properly trained")
    print("2. Try adjusting confidence threshold in ppe_web_app.py")
    print("3. Verify the model file exists: ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt")

if __name__ == "__main__":
    debug_detection()
