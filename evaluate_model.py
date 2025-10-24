#!/usr/bin/env python3
"""
Evaluate the trained PPE detection model performance
"""

from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    """Evaluate the trained model on test dataset"""
    print("="*70)
    print("PPE DETECTION MODEL EVALUATION")
    print("="*70)
    
    # Load the trained model
    model_path = "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return
    
    print(f"[INFO] Loading trained model: {model_path}")
    model = YOLO(model_path)
    
    # Check if dataset config exists
    dataset_config = "ppe_detection_dataset.yaml"
    if not os.path.exists(dataset_config):
        print(f"[ERROR] Dataset config not found: {dataset_config}")
        return
    
    print(f"[INFO] Evaluating on dataset: {dataset_config}")
    
    # Run validation
    print("\n[INFO] Running validation on test dataset...")
    results = model.val(data=dataset_config, split='test')
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    if hasattr(results, 'box'):
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        print(f"Precision: {results.box.mp:.3f}")
        print(f"Recall: {results.box.mr:.3f}")
    
    # Class-wise results
    if hasattr(results, 'box') and hasattr(results.box, 'maps'):
        print(f"\nClass-wise mAP50:")
        class_names = model.names
        for i, map50 in enumerate(results.box.maps):
            if i < len(class_names):
                print(f"  {class_names[i]}: {map50:.3f}")
    
    # Load training results for comparison
    results_csv = "ppe_quick_finetune/yolov8n_ppe_20epochs/results.csv"
    if os.path.exists(results_csv):
        print(f"\n[INFO] Loading training history from: {results_csv}")
        df = pd.read_csv(results_csv)
        
        print(f"\nTraining Summary:")
        print(f"  Total epochs: {len(df)}")
        print(f"  Final mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.3f}")
        print(f"  Final mAP50-95: {df['metrics/mAP50-95(B)'].iloc[-1]:.3f}")
        print(f"  Best mAP50: {df['metrics/mAP50(B)'].max():.3f}")
        print(f"  Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.3f}")
        
        # Create performance plots
        create_performance_plots(df)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    
    return results

def create_performance_plots(df):
    """Create performance visualization plots"""
    print("\n[INFO] Creating performance plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPE Detection Model Performance', fontsize=16)
    
    # Plot 1: mAP50 over epochs
    axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=2, label='mAP50')
    axes[0, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', linewidth=2, label='mAP50-95')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP')
    axes[0, 0].set_title('Mean Average Precision')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    axes[0, 1].plot(df['epoch'], df['train/box_loss'], 'b-', linewidth=2, label='Box Loss')
    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 'r-', linewidth=2, label='Class Loss')
    axes[0, 1].plot(df['epoch'], df['train/dfl_loss'], 'g-', linewidth=2, label='DFL Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision and Recall
    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], 'b-', linewidth=2, label='Precision')
    axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], 'r-', linewidth=2, label='Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], 'g-', linewidth=2, label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppe_model_performance.png', dpi=300, bbox_inches='tight')
    print("  Performance plots saved: ppe_model_performance.png")
    
    # Create a summary table
    create_summary_table(df)

def create_summary_table(df):
    """Create a summary table of key metrics"""
    print("\n[INFO] Creating performance summary...")
    
    # Key metrics at different epochs
    key_epochs = [1, 5, 10, len(df)]
    summary_data = []
    
    for epoch in key_epochs:
        if epoch <= len(df):
            row = df.iloc[epoch-1]
            summary_data.append({
                'Epoch': epoch,
                'mAP50': f"{row['metrics/mAP50(B)']:.3f}",
                'mAP50-95': f"{row['metrics/mAP50-95(B)']:.3f}",
                'Precision': f"{row['metrics/precision(B)']:.3f}",
                'Recall': f"{row['metrics/recall(B)']:.3f}",
                'Box Loss': f"{row['train/box_loss']:.3f}",
                'Class Loss': f"{row['train/cls_loss']:.3f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('ppe_model_summary.csv', index=False)
    print("\n  Summary saved: ppe_model_summary.csv")

def check_model_health():
    """Check model health and provide recommendations"""
    print("\n" + "="*70)
    print("MODEL HEALTH CHECK")
    print("="*70)
    
    # Check if model files exist
    model_files = [
        "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt",
        "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/last.pt",
        "ppe_quick_finetune/yolov8n_ppe_20epochs/results.csv"
    ]
    
    print("Model Files Status:")
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✓ {file_path} ({size:.1f} MB)")
        else:
            print(f"  ✗ {file_path} (Missing)")
    
    # Check dataset
    dataset_dirs = [
        "combined_datasets/images/train",
        "combined_datasets/images/valid", 
        "combined_datasets/images/test"
    ]
    
    print("\nDataset Status:")
    for dir_path in dataset_dirs:
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
            print(f"  ✓ {dir_path} ({count} images)")
        else:
            print(f"  ✗ {dir_path} (Missing)")
    
    print("\nRecommendations:")
    print("  1. Model is ready for deployment")
    print("  2. Consider fine-tuning with more data if accuracy needs improvement")
    print("  3. Test on real-world images to validate performance")
    print("  4. Consider model optimization for faster inference if needed")

if __name__ == "__main__":
    try:
        # Run evaluation
        results = evaluate_model()
        
        # Check model health
        check_model_health()
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
