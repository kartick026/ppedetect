#!/usr/bin/env python3
"""
Real-time PPE Monitoring System
Live camera feed with PPE detection
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
import json
from datetime import datetime
import os
import threading
import queue

class RealTimePPEMonitor:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize real-time PPE monitoring system"""
        self.model_path = model_path
        self.model = None
        self.cap = None
        self.running = False
        self.detection_queue = queue.Queue()
        self.alerts = []
        
        # Load model
        self.load_model()
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.required_ppe = ['helmet', 'safety_vest']  # Required PPE for compliance
        
    def load_model(self):
        """Load the trained PPE detection model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        print(f"[INFO] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Classes: {self.model.names}")
    
    def start_monitoring(self, camera_id=0, save_video=False, output_path="ppe_monitoring_output.mp4"):
        """Start real-time monitoring"""
        print("="*70)
        print("REAL-TIME PPE MONITORING SYSTEM")
        print("="*70)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[INFO] Camera initialized: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if needed
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[INFO] Recording to: {output_path}")
        
        # Start monitoring
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        print(f"[INFO] Starting real-time monitoring...")
        print(f"[INFO] Press 'q' to quit, 's' to save screenshot, 'a' to show alerts")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Run PPE detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process detections
                detections = self.process_detections(results, frame)
                
                # Draw results on frame
                annotated_frame = self.draw_detections(frame, results, detections)
                
                # Add monitoring info
                annotated_frame = self.add_monitoring_info(annotated_frame, frame_count, start_time)
                
                # Display frame
                cv2.imshow('PPE Monitoring System', annotated_frame)
                
                # Save video if recording
                if out is not None:
                    out.write(annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_path = f"ppe_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"[INFO] Screenshot saved: {screenshot_path}")
                elif key == ord('a'):
                    self.show_alerts()
                elif key == ord('c'):
                    self.clear_alerts()
        
        except KeyboardInterrupt:
            print("\n[INFO] Monitoring stopped by user")
        
        finally:
            # Cleanup
            self.running = False
            if self.cap:
                self.cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            elapsed_time = time.time() - start_time
            print(f"\n[INFO] Monitoring session complete!")
            print(f"[INFO] Frames processed: {frame_count}")
            print(f"[INFO] Duration: {elapsed_time:.1f} seconds")
            print(f"[INFO] Average FPS: {frame_count/elapsed_time:.1f}")
    
    def process_detections(self, results, frame):
        """Process detection results and check compliance"""
        detections = []
        detected_classes = set()
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    detected_classes.add(class_name)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Check compliance
        missing_ppe = []
        for required in self.required_ppe:
            if required not in detected_classes:
                missing_ppe.append(required)
        
        # Generate alert if non-compliant
        if missing_ppe:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'NON_COMPLIANT',
                'missing_ppe': missing_ppe,
                'detected_ppe': list(detected_classes),
                'frame_count': len(detections)
            }
            self.alerts.append(alert)
            print(f"[ALERT] Missing PPE: {', '.join(missing_ppe)}")
        
        return detections
    
    def draw_detections(self, frame, results, detections):
        """Draw detection results on frame"""
        if results and results[0].boxes is not None:
            # Use YOLO's built-in plotting
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()
        
        # Add compliance status
        if detections:
            detected_classes = {d['class'] for d in detections}
            missing_ppe = [p for p in self.required_ppe if p not in detected_classes]
            
            if missing_ppe:
                # Non-compliant - red background
                cv2.rectangle(annotated_frame, (10, 10), (400, 80), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "NON-COMPLIANT", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Missing: {', '.join(missing_ppe)}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Compliant - green background
                cv2.rectangle(annotated_frame, (10, 10), (300, 60), (0, 255, 0), -1)
                cv2.putText(annotated_frame, "COMPLIANT", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return annotated_frame
    
    def add_monitoring_info(self, frame, frame_count, start_time):
        """Add monitoring information to frame"""
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add controls info
        cv2.putText(frame, "Controls: Q=Quit, S=Screenshot, A=Alerts, C=Clear", 
                   (frame.shape[1] - 400, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def show_alerts(self):
        """Display recent alerts"""
        print("\n" + "="*50)
        print("RECENT ALERTS")
        print("="*50)
        
        if not self.alerts:
            print("No alerts in current session")
            return
        
        # Show last 10 alerts
        recent_alerts = self.alerts[-10:]
        for i, alert in enumerate(recent_alerts, 1):
            print(f"{i}. {alert['timestamp']} - {alert['type']}")
            print(f"   Missing: {', '.join(alert['missing_ppe'])}")
            print(f"   Detected: {', '.join(alert['detected_ppe'])}")
            print()
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        print("[INFO] All alerts cleared")
    
    def save_alerts_report(self, filename="ppe_alerts_report.json"):
        """Save alerts to JSON file"""
        if not self.alerts:
            print("[INFO] No alerts to save")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.alerts, f, indent=2)
        
        print(f"[INFO] Alerts saved to: {filename}")

def main():
    """Main function"""
    print("="*70)
    print("REAL-TIME PPE MONITORING SYSTEM")
    print("="*70)
    
    try:
        # Initialize monitor
        monitor = RealTimePPEMonitor()
        
        print("\n[INFO] Starting real-time monitoring...")
        print("[INFO] Make sure your camera is connected and working")
        print("[INFO] The system will detect helmets and safety vests")
        
        # Start monitoring
        monitor.start_monitoring(
            camera_id=0,  # Use default camera
            save_video=True,  # Record monitoring session
            output_path="ppe_monitoring_session.mp4"
        )
        
        # Save alerts report
        monitor.save_alerts_report()
        
    except Exception as e:
        print(f"[ERROR] Monitoring failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
