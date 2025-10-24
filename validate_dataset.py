#!/usr/bin/env python3
"""
Dataset Validation and Visualization Script for YOLO Glove Detection
This script validates dataset integrity and creates visualizations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from pathlib import Path

class YOLODatasetValidator:
    def __init__(self, dataset_path="combined_datasets"):
        self.dataset_path = Path(dataset_path)
        self.class_names = {
            0: "glove_type_0",
            1: "glove_type_1", 
            2: "glove_type_2",
            3: "glove_type_3"
        }
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR format
        
    def validate_dataset_structure(self):
        """Validate the dataset folder structure"""
        print("[INFO] Validating Dataset Structure...")
        print("="*50)
        
        required_dirs = [
            "images/train", "images/valid", "images/test",
            "labels/train", "labels/valid", "labels/test"
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if full_path.exists():
                file_count = len(list(full_path.glob("*")))
                print(f"[OK] {dir_path}: {file_count} files")
            else:
                print(f"[ERROR] {dir_path}: Directory not found!")
                
    def validate_annotations(self, split="train", sample_size=100):
        """Validate annotation files and check for inconsistencies"""
        print(f"\n[INFO] Validating {split} annotations...")
        print("="*50)
        
        images_dir = self.dataset_path / f"images/{split}"
        labels_dir = self.dataset_path / f"labels/{split}"
        
        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"[STATS] Images: {len(image_files)}, Labels: {len(label_files)}")
        
        # Check for missing pairs
        missing_labels = []
        missing_images = []
        invalid_annotations = []
        class_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        
        sample_files = random.sample(image_files, min(sample_size, len(image_files)))
        
        for img_file in sample_files:
            label_file = labels_dir / (img_file.stem + ".txt")
            
            if not label_file.exists():
                missing_labels.append(img_file.name)
                continue
                
            # Validate annotation format
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_annotations.append(f"{label_file.name}:{line_num+1}")
                        continue
                        
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Check class ID
                    if class_id not in self.class_names:
                        invalid_annotations.append(f"{label_file.name}:{line_num+1} - Invalid class {class_id}")
                        continue
                        
                    # Check coordinate ranges
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        invalid_annotations.append(f"{label_file.name}:{line_num+1} - Invalid coordinates")
                        continue
                        
                    class_distribution[class_id] += 1
                    
            except Exception as e:
                invalid_annotations.append(f"{label_file.name} - {str(e)}")
        
        # Print results
        print(f"[WARN] Missing labels: {len(missing_labels)}")
        print(f"[WARN] Invalid annotations: {len(invalid_annotations)}")
        print(f"\n[STATS] Class Distribution (sample):")
        for class_id, count in class_distribution.items():
            print(f"   {self.class_names[class_id]}: {count}")
            
        if invalid_annotations[:5]:  # Show first 5 errors
            print(f"\n[ERROR] Sample errors:")
            for error in invalid_annotations[:5]:
                print(f"   - {error}")
                
        return len(invalid_annotations) == 0
    
    def visualize_samples(self, split="train", num_samples=9):
        """Visualize random samples with bounding boxes"""
        print(f"\n[INFO] Creating visualization for {num_samples} {split} samples...")
        
        images_dir = self.dataset_path / f"images/{split}"
        labels_dir = self.dataset_path / f"labels/{split}"
        
        image_files = list(images_dir.glob("*.jpg"))
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        # Create subplot grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'YOLO Dataset Samples - {split.upper()} Set', fontsize=16)
        
        if grid_size == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for idx, img_file in enumerate(sample_files):
            # Load image
            img = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            # Load corresponding label
            label_file = labels_dir / (img_file.stem + ".txt")
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                # Draw bounding boxes
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, box_width, box_height = map(float, parts[1:5])
                        
                        # Convert to pixel coordinates
                        x_center *= width
                        y_center *= height
                        box_width *= width
                        box_height *= height
                        
                        # Calculate corner coordinates
                        x1 = int(x_center - box_width/2)
                        y1 = int(y_center - box_height/2)
                        x2 = int(x_center + box_width/2)
                        y2 = int(y_center + box_height/2)
                        
                        # Draw rectangle
                        color = self.colors[class_id % len(self.colors)]
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                        
                        # Add class label
                        label_text = f"{self.class_names[class_id]}"
                        cv2.putText(img_rgb, label_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display image
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"{img_file.name}", fontsize=8)
            axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(len(sample_files), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(f'dataset_samples_{split}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[OK] Visualization saved as 'dataset_samples_{split}.png'")
    
    def generate_statistics(self):
        """Generate comprehensive dataset statistics"""
        print(f"\n[INFO] Generating Dataset Statistics...")
        print("="*50)
        
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            print(f"\n{split.upper()} Set:")
            
            images_dir = self.dataset_path / f"images/{split}"
            labels_dir = self.dataset_path / f"labels/{split}"
            
            image_files = list(images_dir.glob("*.jpg"))
            label_files = list(labels_dir.glob("*.txt"))
            
            # Count annotations and classes
            total_annotations = 0
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            empty_files = 0
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                if not lines or all(not line.strip() for line in lines):
                    empty_files += 1
                    continue
                    
                for line in lines:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                            total_annotations += 1
            
            stats[split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'annotations': total_annotations,
                'empty_files': empty_files,
                'class_distribution': class_counts
            }
            
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            print(f"  Annotations: {total_annotations}")
            print(f"  Empty files: {empty_files}")
            print(f"  Annotations per image: {total_annotations/len(image_files):.2f}")
            
            for class_id, count in class_counts.items():
                percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
                print(f"  {self.class_names[class_id]}: {count} ({percentage:.1f}%)")
        
        return stats
    
    def run_full_validation(self):
        """Run complete dataset validation"""
        print("[INFO] Starting Full Dataset Validation")
        print("="*80)
        
        # 1. Structure validation
        self.validate_dataset_structure()
        
        # 2. Annotation validation for all splits
        valid = True
        for split in ['train', 'valid', 'test']:
            if not self.validate_annotations(split):
                valid = False
        
        # 3. Generate statistics
        stats = self.generate_statistics()
        
        # 4. Create visualizations
        try:
            self.visualize_samples('train', 9)
            self.visualize_samples('valid', 6)
        except Exception as e:
            print(f"[WARN] Visualization error: {e}")
        
        print(f"\n{'='*80}")
        if valid:
            print("[OK] Dataset validation completed successfully!")
            print("[OK] Your dataset is ready for training!")
        else:
            print("[ERROR] Dataset validation found issues!")
            print("[WARN] Please fix the reported errors before training.")
        
        return valid, stats

def main():
    """Main function to run dataset validation"""
    validator = YOLODatasetValidator()
    
    try:
        is_valid, statistics = validator.run_full_validation()
        return is_valid
    except Exception as e:
        print(f"[ERROR] Error during validation: {e}")
        return False

if __name__ == "__main__":
    main()
