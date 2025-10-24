#!/usr/bin/env python3
"""
PPE Detection Inference Script
Use your trained model for real-world PPE detection
"""

from ultralytics import YOLO
import cv2
import os
import argparse
import time
from pathlib import Path

class PPEDetector:
    def __init__(self, model_path="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"):
        """Initialize the PPE detector with trained model"""
        self.model_path = model_path
        self.model = None
        self.class_names = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        print(f"[INFO] Loading trained model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Classes: {self.class_names}")
    
    def detect_image(self, image_path, conf_threshold=0.5, save_result=True):
        """Detect PPE in a single image"""
        print(f"\n[INFO] Processing: {os.path.basename(image_path)}")
        
        # Run inference
        results = self.model(image_path, conf=conf_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = self.class_names[cls]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    print(f"  ✓ {class_name}: {conf:.2f} confidence")
            else:
                print("  No PPE detected")
        
        # Save result if requested
        if save_result and results:
            output_path = f"ppe_detection_{int(time.time())}.jpg"
            results[0].save(output_path)
            print(f"  Result saved: {output_path}")
        
        return detections
    
    def detect_batch(self, input_dir, output_dir="ppe_results", conf_threshold=0.5):
        """Detect PPE in multiple images"""
        print(f"\n[INFO] Batch processing images from: {input_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        print(f"[INFO] Found {len(image_files)} images to process")
        
        results_summary = {}
        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")
            
            # Run detection
            detections = self.detect_image(str(image_path), conf_threshold, save_result=False)
            
            # Save result with original name
            results = self.model(str(image_path))
            if results:
                output_path = os.path.join(output_dir, f"detected_{image_path.name}")
                results[0].save(output_path)
                print(f"  Result saved: {output_path}")
            
            # Count detections by class
            for detection in detections:
                class_name = detection['class']
                if class_name not in results_summary:
                    results_summary[class_name] = 0
                results_summary[class_name] += 1
        
        # Print summary
        print(f"\n[INFO] Batch processing complete!")
        print(f"[INFO] Results saved in: {output_dir}")
        print(f"[INFO] Detection summary:")
        for class_name, count in results_summary.items():
            print(f"  - {class_name}: {count} detections")
    
    def detect_video(self, video_path, output_path="ppe_detection_output.mp4", conf_threshold=0.5):
        """Detect PPE in video"""
        print(f"\n[INFO] Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress every 30 frames
                print(f"[INFO] Processing frame {frame_count}/{total_frames}")
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Draw results on frame
            if results and results[0].boxes is not None:
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
            else:
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"[INFO] Video processing complete!")
        print(f"[INFO] Output saved: {output_path}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="PPE Detection Inference")
    parser.add_argument("--mode", choices=["image", "batch", "video"], default="image",
                       help="Detection mode")
    parser.add_argument("--input", required=True, help="Input image/video/directory path")
    parser.add_argument("--output", help="Output path (for batch/video mode)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--model", default="ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt",
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = PPEDetector(args.model)
        
        if args.mode == "image":
            # Single image detection
            detector.detect_image(args.input, args.conf)
            
        elif args.mode == "batch":
            # Batch processing
            output_dir = args.output or "ppe_results"
            detector.detect_batch(args.input, output_dir, args.conf)
            
        elif args.mode == "video":
            # Video processing
            output_path = args.output or "ppe_detection_output.mp4"
            detector.detect_video(args.input, output_path, args.conf)
            
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    # Example usage without command line arguments
    print("="*70)
    print("PPE DETECTION INFERENCE")
    print("="*70)
    
    try:
        detector = PPEDetector()
        
        print("\n[INFO] Available modes:")
        print("1. Single image detection")
        print("2. Batch processing")
        print("3. Video processing")
        
        # Test with a sample image
        test_image = "combined_datasets/images/test"
        if os.path.exists(test_image):
            sample_images = [f for f in os.listdir(test_image) if f.endswith('.jpg')][:3]
            for img in sample_images:
                img_path = os.path.join(test_image, img)
                detector.detect_image(img_path)
        
        print("\n" + "="*70)
        print("USAGE EXAMPLES:")
        print("="*70)
        print("Single image: python ppe_detection_inference.py --mode image --input image.jpg")
        print("Batch:        python ppe_detection_inference.py --mode batch --input /path/to/images/")
        print("Video:        python ppe_detection_inference.py --mode video --input video.mp4")
        
    except Exception as e:
        print(f"[ERROR] {e}")
