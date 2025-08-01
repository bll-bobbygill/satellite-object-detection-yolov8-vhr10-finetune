#!/usr/bin/env python3
"""Script to train YOLOv8 on VHR-10 dataset with reduced epochs for quick testing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yolov8_finetune.train import YOLOv8Trainer
import torch

def main():
    """Main training function with reduced epochs for faster completion."""
    # Paths
    data_yaml = "data/yolo_dataset/dataset.yaml"
    project_dir = "results"
    
    # Check if data YAML exists
    if not Path(data_yaml).exists():
        print(f"Dataset YAML not found at {data_yaml}")
        return
    
    print("Starting YOLOv8 fine-tuning on VHR-10 dataset...")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    # Initialize trainer
    trainer = YOLOv8Trainer(model_size="n", pretrained=True)
    trainer.load_model()
    
    # Training parameters - reduced epochs for faster completion
    training_params = {
        "data_yaml": data_yaml,
        "epochs": 50,  # Reduced from 100 for faster completion
        "imgsz": 640,
        "batch_size": 16,
        "lr0": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "project": project_dir,
        "name": "vhr10_yolov8n_quick",
        "save_period": 10,
        "patience": 25,  # Reduced patience
        "device": "0" if torch.cuda.is_available() else "cpu",
        "workers": 8,
        "cos_lr": True,
        "close_mosaic": 10,
        "cache": False,  # Disable caching to save memory
    }
    
    # Start training
    print(f"Training parameters: {training_params}")
    results = trainer.train(**training_params)
    
    print(f"\nTraining completed! Results saved to: {project_dir}")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Run validation on test set
    print("\nRunning validation on test set...")
    test_results = trainer.validate(split="test")
    
    print(f"\nTest Results:")
    print(f"mAP50: {test_results.box.map50:.4f}")
    print(f"mAP50-95: {test_results.box.map:.4f}")
    
    return results, test_results

if __name__ == "__main__":
    main()