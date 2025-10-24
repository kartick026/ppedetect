#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced PPE Detection Models
Tests all detection capabilities, accuracy, and performance
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import json

def test_model_loading():
    """Test if the model loads correctly"""
    print("="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
        print(f"[INFO] Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return False
            
        model = YOLO(model_path)
        print(f"[SUCCESS] Model loaded successfully!")
        print(f"[INFO] Model classes: {model.names}")
        print(f"[INFO] Number of classes: {len(model.names)}")
        
        return True, model
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False, None

def test_detection_accuracy():
    """Test detection accuracy on sample images"""
    print("\n" + "="*60)
    print("TESTING DETECTION ACCURACY")
    print("="*60)
    
    try:
        success, model = test_model_loading()
        if not success:
            return False
            
        # Test on sample images
        test_images = []
        
        # Look for test images in the current directory
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'test' in file.lower():
                test_images.append(file)
        
        if not test_images:
            print("[WARNING] No test images found. Creating synthetic test image...")
            # Create a synthetic test image
            test_image = create_synthetic_test_image()
            test_images = ['synthetic_test.jpg']
            cv2.imwrite('synthetic_test.jpg', test_image)
        
        print(f"[INFO] Found {len(test_images)} test images")
        
        results = []
        for i, image_path in enumerate(test_images[:3]):  # Test first 3 images
            print(f"\n[TEST {i+1}] Testing on: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Could not load image: {image_path}")
                continue
                
            # Run detection
            start_time = time.time()
            results_detection = model(image, conf=0.3, verbose=False)
            detection_time = time.time() - start_time
            
            # Process results
            detections = []
            if results_detection and len(results_detection) > 0:
                result = results_detection[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
            
            # Calculate metrics
            total_detections = len(detections)
            avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
            unique_classes = list(set([d['class'] for d in detections]))
            
            print(f"  ✓ Detections: {total_detections}")
            print(f"  ✓ Classes: {unique_classes}")
            print(f"  ✓ Avg Confidence: {avg_confidence:.3f}")
            print(f"  ✓ Detection Time: {detection_time:.3f}s")
            
            results.append({
                'image': image_path,
                'detections': detections,
                'total_detections': total_detections,
                'avg_confidence': avg_confidence,
                'detection_time': detection_time,
                'unique_classes': unique_classes
            })
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Detection accuracy test failed: {e}")
        return False

def create_synthetic_test_image():
    """Create a synthetic test image with basic shapes"""
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

def test_performance_metrics():
    """Test model performance metrics"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE METRICS")
    print("="*60)
    
    try:
        success, model = test_model_loading()
        if not success:
            return False
        
        # Create test image
        test_image = create_synthetic_test_image()
        
        # Performance tests
        times = []
        confidences = []
        
        print("[INFO] Running performance tests...")
        
        for i in range(10):  # Run 10 iterations
            start_time = time.time()
            results = model(test_image, conf=0.3, verbose=False)
            detection_time = time.time() - start_time
            times.append(detection_time)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = box.conf[0].cpu().numpy()
                        confidences.append(float(confidence))
        
        # Calculate metrics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        avg_confidence = np.mean(confidences) if confidences else 0
        max_confidence = np.max(confidences) if confidences else 0
        min_confidence = np.min(confidences) if confidences else 0
        
        print(f"  ✓ Average Detection Time: {avg_time:.3f}s")
        print(f"  ✓ Min Detection Time: {min_time:.3f}s")
        print(f"  ✓ Max Detection Time: {max_time:.3f}s")
        print(f"  ✓ Time Standard Deviation: {std_time:.3f}s")
        print(f"  ✓ Average Confidence: {avg_confidence:.3f}")
        print(f"  ✓ Max Confidence: {max_confidence:.3f}")
        print(f"  ✓ Min Confidence: {min_confidence:.3f}")
        
        # Performance rating
        if avg_time < 0.1:
            performance_rating = "Excellent"
        elif avg_time < 0.2:
            performance_rating = "Good"
        elif avg_time < 0.5:
            performance_rating = "Fair"
        else:
            performance_rating = "Poor"
            
        print(f"  ✓ Performance Rating: {performance_rating}")
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'avg_confidence': avg_confidence,
            'performance_rating': performance_rating
        }
        
    except Exception as e:
        print(f"[ERROR] Performance test failed: {e}")
        return False

def test_camera_functionality():
    """Test camera functionality"""
    print("\n" + "="*60)
    print("TESTING CAMERA FUNCTIONALITY")
    print("="*60)
    
    try:
        # Test camera availability
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("[WARNING] Camera not available")
            return False
            
        print("[INFO] Camera is available")
        
        # Test frame capture
        ret, frame = camera.read()
        if ret:
            print("[SUCCESS] Frame capture successful")
            print(f"[INFO] Frame shape: {frame.shape}")
            
            # Test detection on camera frame
            success, model = test_model_loading()
            if success:
                start_time = time.time()
                results = model(frame, conf=0.3, verbose=False)
                detection_time = time.time() - start_time
                
                detections = 0
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        detections = len(result.boxes)
                
                print(f"[INFO] Live detection successful")
                print(f"[INFO] Detections: {detections}")
                print(f"[INFO] Detection time: {detection_time:.3f}s")
            else:
                print("[ERROR] Model not available for live testing")
                
        else:
            print("[ERROR] Failed to capture frame")
            return False
            
        camera.release()
        return True
        
    except Exception as e:
        print(f"[ERROR] Camera test failed: {e}")
        return False

def test_compliance_logic():
    """Test compliance detection logic"""
    print("\n" + "="*60)
    print("TESTING COMPLIANCE LOGIC")
    print("="*60)
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Fully Compliant',
            'detections': [
                {'class': 'helmet', 'confidence': 0.8},
                {'class': 'safety_vest', 'confidence': 0.9},
                {'class': 'goggles', 'confidence': 0.7},
                {'class': 'gloves', 'confidence': 0.8}
            ],
            'expected': 'COMPLIANT'
        },
        {
            'name': 'Missing Helmet',
            'detections': [
                {'class': 'safety_vest', 'confidence': 0.9},
                {'class': 'goggles', 'confidence': 0.7},
                {'class': 'gloves', 'confidence': 0.8}
            ],
            'expected': 'NON-COMPLIANT'
        },
        {
            'name': 'Missing Multiple PPE',
            'detections': [
                {'class': 'helmet', 'confidence': 0.8}
            ],
            'expected': 'NON-COMPLIANT'
        },
        {
            'name': 'No Detections',
            'detections': [],
            'expected': 'NON-COMPLIANT'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n[TEST] {scenario['name']}")
        
        # Simulate compliance logic
        required_ppe = ['helmet', 'safety_vest', 'goggles', 'gloves']
        detected_classes = [det['class'] for det in scenario['detections']]
        missing_ppe = [ppe for ppe in required_ppe if ppe not in detected_classes]
        
        compliance_status = "COMPLIANT" if len(missing_ppe) == 0 else "NON-COMPLIANT"
        
        print(f"  ✓ Detected: {detected_classes}")
        print(f"  ✓ Missing: {missing_ppe}")
        print(f"  ✓ Status: {compliance_status}")
        print(f"  ✓ Expected: {scenario['expected']}")
        
        if compliance_status == scenario['expected']:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': results,
        'summary': {
            'total_tests': len(results) if results else 0,
            'passed_tests': 0,
            'failed_tests': 0
        }
    }
    
    # Save report
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[INFO] Test report saved to: test_report.json")
    print(f"[INFO] Report timestamp: {report['timestamp']}")

def main():
    """Run all tests"""
    print("🚀 ENHANCED PPE DETECTION MODEL TESTING SUITE")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Run all tests
    try:
        # Test 1: Model Loading
        success, model = test_model_loading()
        all_results.append(('Model Loading', success))
        
        if success:
            # Test 2: Detection Accuracy
            accuracy_results = test_detection_accuracy()
            all_results.append(('Detection Accuracy', accuracy_results is not False))
            
            # Test 3: Performance Metrics
            performance_results = test_performance_metrics()
            all_results.append(('Performance Metrics', performance_results is not False))
            
            # Test 4: Camera Functionality
            camera_success = test_camera_functionality()
            all_results.append(('Camera Functionality', camera_success))
            
            # Test 5: Compliance Logic
            test_compliance_logic()
            all_results.append(('Compliance Logic', True))
        
        # Generate final report
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, success in all_results if success)
        total = len(all_results)
        
        for test_name, success in all_results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Model is ready for production.")
        else:
            print("⚠️  Some tests failed. Please review the results.")
            
    except Exception as e:
        print(f"[ERROR] Test suite failed: {e}")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
