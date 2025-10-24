#!/usr/bin/env python3
"""
YOLO Model Evaluation and Inference Script for Glove Detection
Comprehensive evaluation tools and inference pipeline
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob
import time

class YOLOEvaluator:
    def __init__(self, model_path, dataset_config="glove_detection_dataset.yaml"):
        """Initialize YOLO evaluator"""
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.class_names = {
            0: "glove_type_0",
            1: "glove_type_1", 
            2: "glove_type_2",
            3: "glove_type_3"
        }
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
        
        try:
            self.model = YOLO(model_path)
            print(f"[OK] Loaded model: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
    
    def evaluate_model(self, split='test', save_results=True):
        """Comprehensive model evaluation"""
        if self.model is None:
            print("[ERROR] No model loaded")
            return None
            
        print(f"[INFO] Evaluating model on {split} set...")
        print("="*50)
        
        try:
            # Run validation
            results = self.model.val(
                data=self.dataset_config,
                split=split,
                imgsz=640,
                device='cuda' if self.model.device.type == 'cuda' else 'cpu',
                save_json=True,
                save_hybrid=True,
                plots=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16)
            }
            
            # Per-class metrics
            if hasattr(results.box, 'ap_class_index') and len(results.box.ap_class_index) > 0:
                class_metrics = {}
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if i < len(results.box.ap50):
                        class_metrics[self.class_names[class_idx]] = {
                            'AP50': results.box.ap50[i],
                            'AP50-95': results.box.ap[i] if i < len(results.box.ap) else 0
                        }
            else:
                class_metrics = {}
            
            # Print results
            print(f"[RESULTS] Overall Performance:")
            print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            
            if class_metrics:
                print(f"\n[RESULTS] Per-Class Performance:")
                for class_name, class_metric in class_metrics.items():
                    print(f"  {class_name}:")
                    print(f"    AP@0.5: {class_metric['AP50']:.4f}")
                    print(f"    AP@0.5:0.95: {class_metric['AP50-95']:.4f}")
            
            # Save results if requested
            if save_results:
                self.save_evaluation_results(metrics, class_metrics, split)
            
            return metrics, class_metrics, results
            
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return None, None, None
    
    def save_evaluation_results(self, metrics, class_metrics, split):
        """Save evaluation results to files"""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save overall metrics
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(results_dir / f"{split}_overall_metrics.csv", index=False)
        
        # Save per-class metrics
        if class_metrics:
            class_df = pd.DataFrame(class_metrics).T
            class_df.to_csv(results_dir / f"{split}_class_metrics.csv")
        
        # Save JSON summary
        summary = {
            'model_path': self.model_path,
            'split': split,
            'overall_metrics': metrics,
            'class_metrics': class_metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_dir / f"{split}_evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"[OK] Results saved to {results_dir}/")
    
    def inference_on_image(self, image_path, conf_threshold=0.5, save_result=True):
        """Run inference on a single image"""
        if self.model is None:
            print("[ERROR] No model loaded")
            return None
            
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[ERROR] Could not load image: {image_path}")
                return None
                
            original_image = image.copy()
            
            # Run inference
            results = self.model(image, conf=conf_threshold, verbose=False)
            
            # Process results
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, f"class_{class_id}"),
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Draw bounding box
                    color = self.colors[class_id % len(self.colors)]
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add label
                    label = f"{self.class_names.get(class_id, f'class_{class_id}')} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(image, label, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save result if requested
            if save_result:
                output_path = Path("inference_results") 
                output_path.mkdir(exist_ok=True)
                
                # Save annotated image
                output_file = output_path / f"result_{Path(image_path).name}"
                cv2.imwrite(str(output_file), image)
                
                # Save detection info
                detection_file = output_path / f"detections_{Path(image_path).stem}.json"
                with open(detection_file, 'w') as f:
                    json.dump(detections, f, indent=2)
                
                print(f"[OK] Results saved: {output_file}")
            
            return detections, image, original_image
            
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None, None, None
    
    def batch_inference(self, images_dir, conf_threshold=0.5, max_images=None):
        """Run inference on a batch of images"""
        if self.model is None:
            print("[ERROR] No model loaded")
            return []
            
        print(f"[INFO] Running batch inference on {images_dir}")
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(Path(images_dir) / ext)))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"[INFO] Processing {len(image_files)} images...")
        
        all_detections = []
        results_dir = Path("batch_inference_results")
        results_dir.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(image_files):
            print(f"[PROGRESS] Processing {i+1}/{len(image_files)}: {Path(image_path).name}")
            
            detections, annotated_image, _ = self.inference_on_image(
                image_path, conf_threshold, save_result=False
            )
            
            if detections is not None:
                # Save annotated image
                output_path = results_dir / f"batch_{Path(image_path).name}"
                cv2.imwrite(str(output_path), annotated_image)
                
                # Store results
                result_entry = {
                    'image_path': str(image_path),
                    'detections': detections,
                    'num_detections': len(detections)
                }
                all_detections.append(result_entry)
        
        # Save batch results summary
        summary_file = results_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Create statistics
        total_detections = sum(entry['num_detections'] for entry in all_detections)
        avg_detections = total_detections / len(image_files) if image_files else 0
        
        print(f"\n[RESULTS] Batch Inference Summary:")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {avg_detections:.2f}")
        print(f"  Results saved to: {results_dir}/")
        
        return all_detections
    
    def create_inference_report(self, detections_summary):
        """Create a comprehensive inference report"""
        if not detections_summary:
            return
            
        print(f"\n[INFO] Creating inference report...")
        
        # Analyze detection statistics
        class_counts = {name: 0 for name in self.class_names.values()}
        confidence_scores = []
        images_with_detections = 0
        
        for entry in detections_summary:
            if entry['num_detections'] > 0:
                images_with_detections += 1
                
            for detection in entry['detections']:
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                if class_name in class_counts:
                    class_counts[class_name] += 1
                confidence_scores.append(confidence)
        
        # Create report
        report = {
            'total_images': len(detections_summary),
            'images_with_detections': images_with_detections,
            'detection_rate': images_with_detections / len(detections_summary),
            'total_detections': sum(class_counts.values()),
            'class_distribution': class_counts,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': min(confidence_scores) if confidence_scores else 0,
            'max_confidence': max(confidence_scores) if confidence_scores else 0
        }
        
        # Print report
        print(f"\nINFERENCE REPORT")
        print("="*40)
        print(f"Total images: {report['total_images']}")
        print(f"Images with detections: {report['images_with_detections']}")
        print(f"Detection rate: {report['detection_rate']:.2%}")
        print(f"Total detections: {report['total_detections']}")
        print(f"\nClass distribution:")
        for class_name, count in report['class_distribution'].items():
            percentage = (count / report['total_detections'] * 100) if report['total_detections'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"\nConfidence statistics:")
        print(f"  Average: {report['avg_confidence']:.3f}")
        print(f"  Range: {report['min_confidence']:.3f} - {report['max_confidence']:.3f}")
        
        # Save report
        report_file = Path("inference_results") / "inference_report.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Report saved: {report_file}")
        return report

    def benchmark_speed(self, test_images_dir, num_iterations=100):
        """Benchmark inference speed"""
        if self.model is None:
            print("[ERROR] No model loaded")
            return None
            
        print(f"[INFO] Benchmarking inference speed...")
        
        # Get test images
        image_files = list(Path(test_images_dir).glob("*.jpg"))[:10]  # Use first 10 images
        if not image_files:
            print(f"[ERROR] No images found in {test_images_dir}")
            return None
        
        # Warmup
        print("[INFO] Warming up...")
        for _ in range(5):
            test_image = str(image_files[0])
            self.model(test_image, verbose=False)
        
        # Benchmark
        print(f"[INFO] Running {num_iterations} inference iterations...")
        times = []
        
        for i in range(num_iterations):
            image_path = str(image_files[i % len(image_files)])
            
            start_time = time.time()
            results = self.model(image_path, verbose=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        fps = 1 / avg_time
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nSPEED BENCHMARK RESULTS")
        print("="*30)
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.1f}")
        print(f"Min time: {min_time*1000:.2f} ms")
        print(f"Max time: {max_time*1000:.2f} ms")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000
        }

def main():
    """Main evaluation and inference function"""
    
    # Configuration - UPDATE THESE PATHS
    model_path = "glove_detection_project/run_balanced/weights/best.pt"  # Update this path
    test_images_dir = "combined_datasets/images/test"  # Test images directory
    
    print("YOLO GLOVE DETECTION - EVALUATION & INFERENCE")
    print("="*80)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("[INFO] Please train a model first using train_yolo_model.py")
        return
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(model_path)
    
    # 1. Model Evaluation
    print("\n1. MODEL EVALUATION")
    print("-" * 40)
    
    metrics, class_metrics, results = evaluator.evaluate_model('test', save_results=True)
    
    # 2. Single Image Inference Demo
    print("\n\n2. SINGLE IMAGE INFERENCE DEMO")
    print("-" * 40)
    
    # Get a test image for demo
    test_images = list(Path(test_images_dir).glob("*.jpg"))
    if test_images:
        demo_image = test_images[0]
        print(f"[INFO] Running inference on: {demo_image.name}")
        
        detections, annotated_img, original_img = evaluator.inference_on_image(
            demo_image, conf_threshold=0.5
        )
        
        if detections:
            print(f"[INFO] Found {len(detections)} detections:")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.3f}")
    
    # 3. Batch Inference
    print("\n\n3. BATCH INFERENCE")
    print("-" * 40)
    
    batch_results = evaluator.batch_inference(
        test_images_dir, 
        conf_threshold=0.5, 
        max_images=20  # Limit for demo
    )
    
    # 4. Create Inference Report
    if batch_results:
        evaluator.create_inference_report(batch_results)
    
    # 5. Speed Benchmark
    print("\n\n4. SPEED BENCHMARK")
    print("-" * 40)
    
    benchmark_results = evaluator.benchmark_speed(test_images_dir, num_iterations=50)
    
    print("\n" + "="*80)
    print("[SUCCESS] Evaluation and inference pipeline completed!")
    print("\nGenerated outputs:")
    print("  - evaluation_results/: Model evaluation metrics")
    print("  - inference_results/: Single image inference results")
    print("  - batch_inference_results/: Batch processing results")

if __name__ == "__main__":
    main()




