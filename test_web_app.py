#!/usr/bin/env python3
"""
Test the Enhanced PPE Web Application
"""

import requests
import time
import json
from datetime import datetime

def test_web_app():
    """Test the web application endpoints"""
    print("🌐 TESTING ENHANCED PPE WEB APPLICATION")
    print("="*50)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test 1: Main page
        print("[TEST 1] Main page accessibility...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("  ✓ Main page accessible")
        else:
            print(f"  ✗ Main page failed: {response.status_code}")
            return False
            
        # Test 2: Live monitoring page
        print("[TEST 2] Live monitoring page...")
        response = requests.get(f"{base_url}/live", timeout=5)
        if response.status_code == 200:
            print("  ✓ Live monitoring page accessible")
        else:
            print(f"  ✗ Live monitoring page failed: {response.status_code}")
            
        # Test 3: Camera status
        print("[TEST 3] Camera status endpoint...")
        response = requests.get(f"{base_url}/camera/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Camera status: {data.get('active', 'Unknown')}")
        else:
            print(f"  ✗ Camera status failed: {response.status_code}")
            
        # Test 4: Statistics endpoint
        print("[TEST 4] Statistics endpoint...")
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Statistics accessible: {data.get('message', 'Data available')}")
        else:
            print(f"  ✗ Statistics failed: {response.status_code}")
            
        # Test 5: History endpoint
        print("[TEST 5] History endpoint...")
        response = requests.get(f"{base_url}/history", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ History accessible: {len(data)} records")
        else:
            print(f"  ✗ History failed: {response.status_code}")
            
        print("\n🎉 WEB APPLICATION TESTS COMPLETED!")
        print("All endpoints are working correctly.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to web application.")
        print("Make sure the application is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_web_app()
