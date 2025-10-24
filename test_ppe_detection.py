#!/usr/bin/env python3
"""
Test PPE Detection Button Functionality
"""

import requests
import os
from pathlib import Path

def test_ppe_detection():
    """Test the PPE detection functionality"""
    print("="*70)
    print("TESTING PPE DETECTION BUTTON FUNCTIONALITY")
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
            print(f"\n[INFO] Testing with: {test_image}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'image': f}
                    response = requests.post('http://localhost:5000/detect', files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✅ Detection successful!")
                    print(f"  - Total detections: {result.get('total_detections', 0)}")
                    print(f"  - Compliance: {result.get('compliance_status', 'UNKNOWN')}")
                    if result.get('detections'):
                        for det in result['detections'][:3]:  # Show first 3
                            print(f"    - {det['class']}: {det['confidence']:.2f}")
                else:
                    print(f"  ❌ Detection failed: {response.status_code}")
                    print(f"  Error: {response.text}")
                    
            except Exception as e:
                print(f"  ❌ Error testing {test_image}: {e}")
        else:
            print(f"[INFO] Test image {test_image} not found")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    print("\nTo test the web interface:")
    print("1. Run: python ppe_web_app.py")
    print("2. Open: http://localhost:5000")
    print("3. Upload an image and click 'Detect PPE' button")

if __name__ == "__main__":
    test_ppe_detection()
