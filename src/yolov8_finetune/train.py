"""Training script for YOLOv8 fine-tuning on VHR-10 dataset."""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml


class YOLOv8Trainer:
    """YOLOv8 trainer for VHR-10 dataset."""
    
    def __init__(self, model_size: str = "n", pretrained: bool = True):
        """
        Initialize trainer.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            pretrained: Whether to use pretrained weights
        """
        self.model_size = model_size
        self.pretrained = pretrained
        self.model = None
        
    def load_model(self, model_path: str = None):
        """Load YOLOv8 model."""
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.model = YOLO(model_path)
        else:
            model_name = f"yolov8{self.model_size}.pt" if self.pretrained else f"yolov8{self.model_size}.yaml"
            print(f"Loading {'pretrained' if self.pretrained else 'fresh'} YOLOv8{self.model_size} model")
            self.model = YOLO(model_name)
        
        return self.model
    
    def train(self, 
              data_yaml: str,
              epochs: int = 100,
              imgsz: int = 640,
              batch_size: int = 16,
              lr0: float = 0.01,
              weight_decay: float = 0.0005,
              momentum: float = 0.937,
              project: str = "runs/detect",
              name: str = "vhr10_finetune",
              save_period: int = 10,
              patience: int = 50,
              **kwargs):
        """
        Train the model.
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            lr0: Initial learning rate
            weight_decay: Weight decay
            momentum: SGD momentum
            project: Project directory
            name: Experiment name
            save_period: Save checkpoint every N epochs
            patience: Early stopping patience
            **kwargs: Additional training arguments
        """
        if self.model is None:
            self.load_model()
        
        print(f"Starting training with the following parameters:")
        print(f"  Data: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr0}")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Start training
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            weight_decay=weight_decay,
            momentum=momentum,
            project=project,
            name=name,
            save_period=save_period,
            patience=patience,
            verbose=True,
            **kwargs
        )
        
        print("Training completed!")
        return results
    
    def validate(self, data_yaml: str = None, **kwargs):
        """Validate the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Running validation...")
        results = self.model.val(data=data_yaml, **kwargs)
        return results
    
    def export_model(self, format: str = "onnx", **kwargs):
        """Export the model to different formats."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Exporting model to {format} format...")
        results = self.model.export(format=format, **kwargs)
        return results


def main():
    """Main training function."""
    # Paths
    data_yaml = "data/yolo_dataset/dataset.yaml"
    project_dir = "results"
    
    # Check if data YAML exists
    if not Path(data_yaml).exists():
        print(f"Dataset YAML not found at {data_yaml}")
        print("Please run data preparation first: python -m yolov8_finetune.data_prep")
        return
    
    # Initialize trainer
    trainer = YOLOv8Trainer(model_size="n", pretrained=True)
    trainer.load_model()
    
    # Training parameters
    training_params = {
        "data_yaml": data_yaml,
        "epochs": 100,
        "imgsz": 640,
        "batch_size": 16,
        "lr0": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "project": project_dir,
        "name": "vhr10_yolov8n",
        "save_period": 10,
        "patience": 50,
        "amp": True,  # Automatic Mixed Precision
        "cache": False,  # Cache images for faster training
        "device": "0" if torch.cuda.is_available() else "cpu",
        "workers": 8,
        "cos_lr": True,  # Cosine learning rate scheduler
        "close_mosaic": 10,  # Close mosaic augmentation in last N epochs
    }
    
    # Start training
    results = trainer.train(**training_params)
    
    # Validate on test set
    print("\nRunning validation on test set...")
    val_results = trainer.validate(split="test")
    
    print(f"\nTraining completed! Results saved to: {project_dir}")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()