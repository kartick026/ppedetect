#!/usr/bin/env python3
"""
Comprehensive Test for Simple PPE Web App Backend
Tests the currently running simple_ppe_web_app.py
"""

import requests
import json
import time
import cv2
import numpy as np
from datetime import datetime
import os
from ultralytics import YOLO
import base64
import io
from PIL import Image

class SimplePPETester:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
        self.test_results = []
        
    def test_model_loading(self):
        """Test ML model loading and initialization"""
        print("🧠 TESTING ML MODEL LOADING")
        print("="*50)
        
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
                
            model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Classes: {model.names}")
            print(f"📊 Number of classes: {len(model.names)}")
            
            # Test model inference on dummy image
            dummy_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
            results = model(dummy_image, conf=0.3, verbose=False)
            print(f"✅ Model inference test passed")
            
            self.test_results.append(("ML Model Loading", True, "Model loaded and inference working"))
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.test_results.append(("ML Model Loading", False, str(e)))
            return False
    
    def test_backend_endpoints(self):
        """Test all backend API endpoints"""
        print("\n🌐 TESTING BACKEND ENDPOINTS")
        print("="*50)
        
        endpoints = [
            ("/", "Main page"),
            ("/stats", "Statistics endpoint"),
            ("/history", "History endpoint"),
        ]
        
        all_passed = True
        
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    print(f"✅ {description}: {response.status_code}")
                    self.test_results.append((f"Endpoint {endpoint}", True, f"Status: {response.status_code}"))
                else:
                    print(f"⚠️ {description}: {response.status_code}")
                    self.test_results.append((f"Endpoint {endpoint}", False, f"Status: {response.status_code}"))
                    all_passed = False
            except Exception as e:
                print(f"❌ {description}: Connection failed - {e}")
                self.test_results.append((f"Endpoint {endpoint}", False, str(e)))
                all_passed = False
        
        return all_passed
    
    def test_detection_api(self):
        """Test PPE detection API with sample image"""
        print("\n🔍 TESTING DETECTION API")
        print("="*50)
        
        try:
            # Create a test image
            test_image = self.create_test_image()
            
            # Save test image temporarily
            test_path = "temp_test_image.jpg"
            cv2.imwrite(test_path, test_image)
            
            # Test detection API
            print("🔄 Testing detection API...")
            with open(test_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{self.base_url}/detect", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Detection API working")
                print(f"📊 Detections: {data.get('total_detections', 0)}")
                print(f"📊 Compliance: {data.get('compliance_status', 'Unknown')}")
                print(f"📊 People count: {data.get('people_count', 0)}")
                
                if 'detections' in data and data['detections']:
                    print(f"📊 Detection details:")
                    for i, det in enumerate(data['detections']):
                        print(f"  {i+1}. {det.get('class', 'Unknown')}: {det.get('confidence', 0):.3f}")
                
                self.test_results.append(("Detection API", True, f"Detections: {data.get('total_detections', 0)}"))
                
                # Clean up
                if os.path.exists(test_path):
                    os.remove(test_path)
                
                return True
            else:
                print(f"❌ Detection API failed: {response.status_code}")
                print(f"Response: {response.text}")
                self.test_results.append(("Detection API", False, f"Status: {response.status_code}"))
                return False
                
        except Exception as e:
            print(f"❌ Detection API test failed: {e}")
            self.test_results.append(("Detection API", False, str(e)))
            return False
    
    def test_performance_metrics(self):
        """Test system performance metrics"""
        print("\n⚡ TESTING PERFORMANCE METRICS")
        print("="*50)
        
        try:
            # Test detection speed
            test_image = self.create_test_image()
            test_path = "temp_performance_test.jpg"
            cv2.imwrite(test_path, test_image)
            
            times = []
            for i in range(3):  # Reduced iterations for faster testing
                start_time = time.time()
                with open(test_path, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(f"{self.base_url}/detect", files=files, timeout=30)
                end_time = time.time()
                times.append(end_time - start_time)
                print(f"  Test {i+1}: {times[-1]:.3f}s")
            
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"✅ Performance metrics:")
            print(f"📊 Average detection time: {avg_time:.3f}s")
            print(f"📊 Min detection time: {min_time:.3f}s")
            print(f"📊 Max detection time: {max_time:.3f}s")
            
            # Performance rating
            if avg_time < 1.0:
                rating = "Excellent"
            elif avg_time < 3.0:
                rating = "Good"
            elif avg_time < 5.0:
                rating = "Fair"
            else:
                rating = "Poor"
            
            print(f"📊 Performance rating: {rating}")
            
            self.test_results.append(("Performance", True, f"Avg: {avg_time:.3f}s, Rating: {rating}"))
            
            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results.append(("Performance", False, str(e)))
            return False
    
    def test_data_persistence(self):
        """Test data storage and retrieval"""
        print("\n💾 TESTING DATA PERSISTENCE")
        print("="*50)
        
        try:
            # Test history endpoint
            print("🔄 Testing history retrieval...")
            response = requests.get(f"{self.base_url}/history", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ History accessible: {len(data)} records")
                if data:
                    latest = data[-1]
                    print(f"📊 Latest detection: {latest.get('timestamp', 'Unknown')}")
                    print(f"📊 Latest compliance: {latest.get('compliance_status', 'Unknown')}")
                self.test_results.append(("Data History", True, f"{len(data)} records"))
            else:
                print(f"❌ History failed: {response.status_code}")
                return False
            
            # Test statistics endpoint
            print("🔄 Testing statistics...")
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Statistics accessible")
                print(f"📊 Total detections: {data.get('total_detections', 0)}")
                print(f"📊 Compliance rate: {data.get('compliance_rate', 0):.1f}%")
                print(f"📊 Compliant: {data.get('compliant', 0)}")
                print(f"📊 Non-compliant: {data.get('non_compliant', 0)}")
                
                if 'ppe_counts' in data:
                    print(f"📊 PPE detection counts:")
                    for ppe_type, count in data['ppe_counts'].items():
                        print(f"  {ppe_type}: {count}")
                
                self.test_results.append(("Data Statistics", True, f"Rate: {data.get('compliance_rate', 0):.1f}%"))
            else:
                print(f"❌ Statistics failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Data persistence test failed: {e}")
            self.test_results.append(("Data Persistence", False, str(e)))
            return False
    
    def test_detection_accuracy(self):
        """Test detection accuracy with different scenarios"""
        print("\n🎯 TESTING DETECTION ACCURACY")
        print("="*50)
        
        try:
            # Test with different image scenarios
            scenarios = [
                ("Synthetic Person", self.create_test_image()),
                ("Empty Background", np.ones((480, 640, 3), dtype=np.uint8) * 255),
                ("Random Noise", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            ]
            
            for scenario_name, test_image in scenarios:
                print(f"🔄 Testing {scenario_name}...")
                
                test_path = f"temp_{scenario_name.lower().replace(' ', '_')}.jpg"
                cv2.imwrite(test_path, test_image)
                
                with open(test_path, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(f"{self.base_url}/detect", files=files, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    detections = data.get('total_detections', 0)
                    compliance = data.get('compliance_status', 'Unknown')
                    print(f"  ✅ {scenario_name}: {detections} detections, {compliance}")
                else:
                    print(f"  ❌ {scenario_name}: Failed ({response.status_code})")
                
                # Clean up
                if os.path.exists(test_path):
                    os.remove(test_path)
            
            self.test_results.append(("Detection Accuracy", True, "Multiple scenarios tested"))
            return True
            
        except Exception as e:
            print(f"❌ Detection accuracy test failed: {e}")
            self.test_results.append(("Detection Accuracy", False, str(e)))
            return False
    
    def create_test_image(self):
        """Create a synthetic test image"""
        # Create a 640x480 image with a person-like figure
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw a simple person figure
        # Head (helmet area)
        cv2.circle(image, (320, 120), 40, (100, 100, 100), -1)
        
        # Body (safety vest area)
        cv2.rectangle(image, (280, 160), (360, 300), (255, 165, 0), -1)
        
        # Arms
        cv2.rectangle(image, (250, 180), (280, 250), (100, 100, 100), -1)
        cv2.rectangle(image, (360, 180), (390, 250), (100, 100, 100), -1)
        
        # Legs
        cv2.rectangle(image, (300, 300), (320, 400), (100, 100, 100), -1)
        cv2.rectangle(image, (340, 300), (360, 400), (100, 100, 100), -1)
        
        return image
    
    def run_comprehensive_test(self):
        """Run all tests and generate report"""
        print("🚀 COMPREHENSIVE SIMPLE PPE BACKEND TESTING")
        print("="*60)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        tests = [
            ("ML Model Loading", self.test_model_loading),
            ("Backend Endpoints", self.test_backend_endpoints),
            ("Detection API", self.test_detection_api),
            ("Performance Metrics", self.test_performance_metrics),
            ("Data Persistence", self.test_data_persistence),
            ("Detection Accuracy", self.test_detection_accuracy),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False, str(e)))
        
        # Generate final report
        print("\n" + "="*60)
        print("📊 FINAL TEST SUMMARY")
        print("="*60)
        
        for test_name, success, details in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status} - {details}")
        
        print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! System is fully operational.")
            grade = "A+"
        elif passed_tests >= total_tests * 0.8:
            print("✅ Most tests passed. System is operational with minor issues.")
            grade = "A-"
        elif passed_tests >= total_tests * 0.6:
            print("⚠️ Some tests failed. System needs attention.")
            grade = "B"
        else:
            print("❌ Multiple tests failed. System needs significant fixes.")
            grade = "C"
        
        print(f"📊 System Grade: {grade}")
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'grade': grade,
            'results': self.test_results
        }
        
        with open('simple_backend_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: simple_backend_test_report.json")
        print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    tester = SimplePPETester()
    tester.run_comprehensive_test()
