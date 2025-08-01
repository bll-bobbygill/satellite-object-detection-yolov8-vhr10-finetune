"""Evaluation script for YOLOv8 model on VHR-10 dataset."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from ultralytics import YOLO
import torch


class YOLOv8Evaluator:
    """Comprehensive evaluation of YOLOv8 model performance."""
    
    def __init__(self, model_path: str, data_yaml: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to dataset YAML file
        """
        self.model_path = Path(model_path)
        self.data_yaml = data_yaml
        self.model = None
        self.class_names = [
            "airplane", "ship", "storage_tank", "baseball_diamond", 
            "tennis_court", "basketball_court", "ground_track_field", 
            "harbor", "bridge", "vehicle"
        ]
        self.results = {}
        
    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        self.model = YOLO(str(self.model_path))
        return self.model
    
    def run_validation(self, split: str = "test", conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        """Run validation on specified dataset split."""
        if self.model is None:
            self.load_model()
        
        print(f"Running validation on {split} split...")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        
        # Run validation
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True,
            save_json=True,
            save_hybrid=True
        )
        
        # Store results
        self.results[split] = {
            'metrics': results,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        
        return results
    
    def extract_metrics(self, results) -> Dict:
        """Extract key metrics from validation results."""
        metrics = {}
        
        # Overall metrics
        if hasattr(results, 'box'):
            box_metrics = results.box
            metrics['mAP50'] = float(box_metrics.map50)
            metrics['mAP50-95'] = float(box_metrics.map)
            metrics['precision'] = float(box_metrics.mp)
            metrics['recall'] = float(box_metrics.mr)
            
            # Per-class metrics
            if hasattr(box_metrics, 'ap'):
                metrics['per_class_AP50'] = box_metrics.ap50.tolist() if box_metrics.ap50 is not None else []
                metrics['per_class_AP50-95'] = box_metrics.ap.tolist() if box_metrics.ap is not None else []
        
        return metrics
    
    def generate_metrics_report(self, split: str = "test") -> Dict:
        """Generate comprehensive metrics report."""
        if split not in self.results:
            print(f"No results found for {split} split. Running validation...")
            self.run_validation(split=split)
        
        results = self.results[split]['metrics']
        metrics = self.extract_metrics(results)
        
        # Create detailed report
        report = {
            'dataset_split': split,
            'confidence_threshold': self.results[split]['conf_threshold'],
            'iou_threshold': self.results[split]['iou_threshold'],
            'overall_metrics': {
                'mAP@0.5': metrics.get('mAP50', 0.0),
                'mAP@0.5:0.95': metrics.get('mAP50-95', 0.0),
                'Precision': metrics.get('precision', 0.0),
                'Recall': metrics.get('recall', 0.0)
            },
            'per_class_metrics': {}
        }
        
        # Per-class metrics
        if 'per_class_AP50' in metrics:
            for i, (class_name, ap50, ap50_95) in enumerate(zip(
                self.class_names, 
                metrics['per_class_AP50'], 
                metrics['per_class_AP50-95']
            )):
                report['per_class_metrics'][class_name] = {
                    'AP@0.5': float(ap50),
                    'AP@0.5:0.95': float(ap50_95)
                }
        
        return report
    
    def create_visualizations(self, output_dir: str = "results/evaluation"):
        """Create visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Generate report first
        report = self.generate_metrics_report()
        
        # 1. Overall metrics bar plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        metrics_names = list(report['overall_metrics'].keys())
        metrics_values = list(report['overall_metrics'].values())
        
        bars = ax.bar(metrics_names, metrics_values, alpha=0.7)
        ax.set_title('Overall Model Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class AP@0.5 plot
        if report['per_class_metrics']:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            classes = list(report['per_class_metrics'].keys())
            ap50_values = [report['per_class_metrics'][cls]['AP@0.5'] for cls in classes]
            
            bars = ax.barh(classes, ap50_values, alpha=0.7)
            ax.set_title('Per-Class Average Precision @ IoU=0.5', fontsize=16, fontweight='bold')
            ax.set_xlabel('AP@0.5', fontsize=12)
            ax.set_xlim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, ap50_values):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{value:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'per_class_ap50.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. AP@0.5 vs AP@0.5:0.95 comparison
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ap50_values = [report['per_class_metrics'][cls]['AP@0.5'] for cls in classes]
            ap50_95_values = [report['per_class_metrics'][cls]['AP@0.5:0.95'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, ap50_values, width, label='AP@0.5', alpha=0.7)
            bars2 = ax.bar(x + width/2, ap50_95_values, width, label='AP@0.5:0.95', alpha=0.7)
            
            ax.set_title('AP@0.5 vs AP@0.5:0.95 by Class', fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Precision', fontsize=12)
            ax.set_xlabel('Object Classes', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ap_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
    
    def save_detailed_report(self, output_file: str = "results/evaluation/detailed_report.json"):
        """Save detailed evaluation report to JSON."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report
        report = self.generate_metrics_report()
        
        # Add model information
        report['model_info'] = {
            'model_path': str(self.model_path),
            'model_size': self.model_path.stem,
            'parameters': self.get_model_parameters() if self.model else None
        }
        
        # Add training information if available
        if self.model and hasattr(self.model, 'trainer'):
            trainer = self.model.trainer
            if trainer and hasattr(trainer, 'args'):
                report['training_info'] = {
                    'epochs': getattr(trainer.args, 'epochs', None),
                    'batch_size': getattr(trainer.args, 'batch', None),
                    'learning_rate': getattr(trainer.args, 'lr0', None),
                    'image_size': getattr(trainer.args, 'imgsz', None)
                }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved to: {output_file}")
        return report
    
    def get_model_parameters(self) -> Dict:
        """Get model parameter information."""
        if self.model is None:
            return None
        
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def print_summary(self):
        """Print evaluation summary."""
        report = self.generate_metrics_report()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Dataset Split: {report['dataset_split']}")
        print(f"Confidence Threshold: {report['confidence_threshold']}")
        print(f"IoU Threshold: {report['iou_threshold']}")
        print("\nOVERALL METRICS:")
        print("-"*30)
        for metric, value in report['overall_metrics'].items():
            print(f"{metric:20s}: {value:.4f}")
        
        if report['per_class_metrics']:
            print("\nPER-CLASS METRICS (AP@0.5):")
            print("-"*40)
            for class_name, metrics in report['per_class_metrics'].items():
                print(f"{class_name:20s}: {metrics['AP@0.5']:.4f}")
        
        # Model info
        if self.model:
            model_info = self.get_model_parameters()
            if model_info:
                print("\nMODEL INFORMATION:")
                print("-"*30)
                print(f"Total Parameters: {model_info['total_parameters']:,}")
                print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
        
        print("="*60)


def main():
    """Main evaluation function."""
    # Configuration
    model_path = "results/vhr10_yolov8n/weights/best.pt"
    data_yaml = "data/yolo_dataset/dataset.yaml"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first: python -m yolov8_finetune.train")
        return
    
    # Initialize evaluator
    evaluator = YOLOv8Evaluator(model_path, data_yaml)
    
    # Run evaluation
    print("Starting comprehensive evaluation...")
    
    # Evaluate on test set
    evaluator.run_validation(split="test", conf_threshold=0.25, iou_threshold=0.5)
    
    # Generate visualizations
    evaluator.create_visualizations()
    
    # Save detailed report
    evaluator.save_detailed_report()
    
    # Print summary
    evaluator.print_summary()
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()