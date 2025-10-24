#!/usr/bin/env python3
"""
Comprehensive YOLO Training Script for Glove Detection
Optimized training pipeline with best practices for object detection
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt

class YOLOTrainer:
    def __init__(self, dataset_config="glove_detection_dataset.yaml"):
        """Initialize YOLO trainer with configuration"""
        self.dataset_config = dataset_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        
        # Training configurations for different scenarios
        self.training_configs = {
            'quick_test': {
                'model': 'yolov8n.pt',  # Fastest model for testing
                'epochs': 10,
                'imgsz': 416,
                'batch': 16,
                'lr0': 0.01,
                'description': 'Quick training test (10 epochs)'
            },
            'balanced': {
                'model': 'yolov8s.pt',  # Balanced speed/accuracy
                'epochs': 100,
                'imgsz': 640,
                'batch': 16,
                'lr0': 0.01,
                'description': 'Balanced training (100 epochs)'
            },
            'high_accuracy': {
                'model': 'yolov8m.pt',  # Better accuracy
                'epochs': 200,
                'imgsz': 640,
                'batch': 8,  # Smaller batch for larger model
                'lr0': 0.01,
                'description': 'High accuracy training (200 epochs)'
            },
            'production': {
                'model': 'yolov8l.pt',  # Production ready
                'epochs': 300,
                'imgsz': 832,  # Higher resolution
                'batch': 4,   # Small batch for large model
                'lr0': 0.008,
                'description': 'Production training (300 epochs)'
            }
        }
    
    def optimize_batch_size(self, model_name, target_imgsz=640):
        """Automatically determine optimal batch size"""
        print(f"[INFO] Optimizing batch size for {model_name}...")
        
        # Get GPU memory if available
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[INFO] GPU Memory: {gpu_memory_gb:.1f} GB")
            
            # Estimate batch size based on model and GPU memory
            memory_estimates = {
                'yolov8n.pt': {'640': 32, '832': 24},
                'yolov8s.pt': {'640': 16, '832': 12}, 
                'yolov8m.pt': {'640': 8, '832': 6},
                'yolov8l.pt': {'640': 4, '832': 3},
                'yolov8x.pt': {'640': 2, '832': 2}
            }
            
            base_batch = memory_estimates.get(model_name, {}).get(str(target_imgsz), 8)
            
            # Scale based on available memory
            if gpu_memory_gb >= 24:  # RTX 4090/A6000 class
                batch_multiplier = 2
            elif gpu_memory_gb >= 16:  # RTX 4080 class 
                batch_multiplier = 1.5
            elif gpu_memory_gb >= 12:  # RTX 4070Ti class
                batch_multiplier = 1.2
            elif gpu_memory_gb >= 8:   # RTX 4060Ti class
                batch_multiplier = 1.0
            else:  # Lower end GPUs
                batch_multiplier = 0.5
                
            optimal_batch = max(1, int(base_batch * batch_multiplier))
            print(f"[INFO] Recommended batch size: {optimal_batch}")
            return optimal_batch
        else:
            print(f"[INFO] CPU training - using batch size 2")
            return 2
    
    def get_class_weights(self):
        """Calculate class weights to handle imbalanced dataset"""
        # Based on our validation results:
        # Class 0: 62.4%, Class 1: 6.6%, Class 2: 24.6%, Class 3: 6.3%
        class_frequencies = [0.624, 0.066, 0.246, 0.063]
        
        # Calculate inverse frequency weights
        total_samples = sum(class_frequencies)
        num_classes = len(class_frequencies)
        
        class_weights = []
        for freq in class_frequencies:
            weight = total_samples / (num_classes * freq)
            class_weights.append(weight)
        
        print(f"[INFO] Class weights: {class_weights}")
        return class_weights
    
    def create_training_config(self, training_mode='balanced', custom_params=None):
        """Create optimized training configuration"""
        if training_mode not in self.training_configs:
            print(f"[WARN] Unknown training mode '{training_mode}', using 'balanced'")
            training_mode = 'balanced'
            
        config = self.training_configs[training_mode].copy()
        print(f"[INFO] Training mode: {config['description']}")
        
        # Optimize batch size
        optimal_batch = self.optimize_batch_size(config['model'], config['imgsz'])
        config['batch'] = optimal_batch
        
        # Add custom parameters if provided
        if custom_params:
            config.update(custom_params)
            print(f"[INFO] Applied custom parameters: {custom_params}")
            
        return config
    
    def setup_callbacks(self):
        """Setup training callbacks for monitoring"""
        def on_train_epoch_end(trainer):
            """Callback for end of training epoch"""
            if hasattr(trainer, 'epoch'):
                print(f"[PROGRESS] Epoch {trainer.epoch + 1} completed")
        
        return {'on_train_epoch_end': on_train_epoch_end}
    
    def train_model(self, training_mode='balanced', project_name='glove_detection', 
                   run_name=None, resume=False, custom_params=None):
        """
        Train YOLO model with specified configuration
        
        Args:
            training_mode: 'quick_test', 'balanced', 'high_accuracy', 'production'
            project_name: Project name for organizing runs
            run_name: Specific run name (auto-generated if None)
            resume: Resume from last checkpoint if available
            custom_params: Dictionary of custom training parameters
        """
        
        print(f"[INFO] Starting YOLO training...")
        print(f"[INFO] Mode: {training_mode}")
        print("="*60)
        
        # Create training configuration
        config = self.create_training_config(training_mode, custom_params)
        
        # Initialize model
        model = YOLO(config['model'])
        print(f"[INFO] Loaded model: {config['model']}")
        
        # Setup run name
        if run_name is None:
            run_name = f"{training_mode}_{config['model'].replace('.pt', '')}"
        
        # Training parameters optimized for object detection
        train_params = {
            'data': self.dataset_config,
            'epochs': config['epochs'],
            'imgsz': config['imgsz'],
            'batch': config['batch'],
            'device': self.device,
            'project': project_name,
            'name': run_name,
            'exist_ok': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            
            # Learning rate and optimization
            'lr0': config['lr0'],
            'lrf': 0.1,  # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Data augmentation optimized for detection
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation  
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 0.0,  # Rotation degree range
            'translate': 0.1, # Translation fraction
            'scale': 0.5,    # Scaling gain
            'shear': 0.0,    # Shear degree range
            'perspective': 0.0, # Perspective transformation
            'flipud': 0.0,   # Vertical flip probability
            'fliplr': 0.5,   # Horizontal flip probability
            'mosaic': 1.0,   # Mosaic augmentation probability
            'mixup': 0.0,    # Mixup augmentation probability
            'copy_paste': 0.0, # Copy-paste augmentation probability
            
            # Regularization
            'dropout': 0.0,  # Dropout rate
            'label_smoothing': 0.0,
            
            # Validation and early stopping
            'val': True,
            'patience': 50,  # Early stopping patience
            'save_json': True,  # Save results in JSON format
            'verbose': True,
            
            # Resume training if requested
            'resume': resume
        }
        
        # Add class weights for imbalanced dataset (if supported)
        try:
            class_weights = self.get_class_weights()
            # Note: Ultralytics handles class weighting automatically based on data distribution
        except Exception as e:
            print(f"[WARN] Could not set class weights: {e}")
        
        # Start training
        print(f"[INFO] Starting training with parameters:")
        for key, value in train_params.items():
            if key in ['lr0', 'epochs', 'batch', 'imgsz', 'device']:
                print(f"  {key}: {value}")
        
        try:
            results = model.train(**train_params)
            
            # Training completed successfully
            print("\n" + "="*60)
            print("[SUCCESS] Training completed successfully!")
            print(f"[INFO] Best model saved to: {results.save_dir}")
            print(f"[INFO] Results saved to: {results.save_dir}")
            
            return results, model
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            print("[INFO] Common solutions:")
            print("  - Reduce batch size if out of memory")
            print("  - Check dataset paths and format")
            print("  - Ensure sufficient disk space")
            return None, None
    
    def create_training_summary(self, results):
        """Create a summary of training results"""
        if results is None:
            return
            
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        try:
            # Get results directory
            save_dir = results.save_dir
            print(f"Results directory: {save_dir}")
            
            # Check for key files
            key_files = [
                'weights/best.pt',
                'weights/last.pt', 
                'results.png',
                'confusion_matrix.png',
                'val_batch0_pred.jpg'
            ]
            
            print("\nGenerated files:")
            for file in key_files:
                file_path = Path(save_dir) / file
                if file_path.exists():
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ {file}")
            
            # Read results if available
            results_csv = Path(save_dir) / 'results.csv'
            if results_csv.exists():
                print(f"\n[INFO] Detailed results available in: {results_csv}")
            
        except Exception as e:
            print(f"[WARN] Could not generate summary: {e}")
    
    def validate_model(self, model_path, test_data=None):
        """Validate trained model on test set"""
        print(f"[INFO] Validating model: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            # Use test set if available, otherwise validation set
            data_split = 'test' if test_data else 'val'
            
            results = model.val(
                data=self.dataset_config,
                split=data_split,
                imgsz=640,
                device=self.device,
                save_json=True,
                save_hybrid=True
            )
            
            print(f"[SUCCESS] Validation completed!")
            print(f"[INFO] mAP50: {results.box.map50:.4f}")
            print(f"[INFO] mAP50-95: {results.box.map:.4f}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Validation failed: {e}")
            return None

def main():
    """Main training function with different training scenarios"""
    
    trainer = YOLOTrainer()
    
    print("YOLO GLOVE DETECTION TRAINING")
    print("="*80)
    print("Available training modes:")
    for mode, config in trainer.training_configs.items():
        print(f"  {mode}: {config['description']}")
    
    # Get user choice or use default
    print("\n[INFO] Starting with 'balanced' mode (recommended)")
    
    # You can change this to different modes:
    # 'quick_test' - Fast testing (10 epochs)
    # 'balanced' - Recommended starting point (100 epochs)  
    # 'high_accuracy' - Better results (200 epochs)
    # 'production' - Best results (300 epochs)
    
    training_mode = 'balanced'
    
    # Custom parameters (optional)
    custom_params = {
        # 'epochs': 150,        # Override epochs
        # 'batch': 8,          # Override batch size
        # 'imgsz': 832,        # Override image size
        # 'lr0': 0.005,        # Override learning rate
    }
    
    # Start training
    results, model = trainer.train_model(
        training_mode=training_mode,
        project_name='glove_detection_project',
        run_name=f'run_{training_mode}',
        resume=False,  # Set to True to resume from checkpoint
        custom_params=None  # Use custom_params if needed
    )
    
    if results is not None:
        # Create training summary
        trainer.create_training_summary(results)
        
        # Validate the best model
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            print(f"\n[INFO] Validating best model...")
            validation_results = trainer.validate_model(str(best_model_path))
        
        print(f"\n[SUCCESS] Training pipeline completed!")
        print(f"[INFO] Best model available at: {best_model_path}")
        print(f"[INFO] Use this model for inference and deployment")
        
        return str(best_model_path)
    else:
        print(f"\n[ERROR] Training failed!")
        return None

if __name__ == "__main__":
    best_model_path = main()




