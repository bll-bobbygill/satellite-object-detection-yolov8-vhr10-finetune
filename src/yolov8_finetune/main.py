"""Main execution script for YOLOv8 fine-tuning pipeline."""

import argparse
import sys
from pathlib import Path

from .data_prep import VHR10DataPrep
from .train import YOLOv8Trainer
from .evaluate import YOLOv8Evaluator


def run_data_preparation(data_dir: str = "data", output_dir: str = "data/yolo_dataset"):
    """Run data preparation step."""
    print("Starting data preparation...")
    
    prep = VHR10DataPrep(data_dir, output_dir)
    splits = prep.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    prep.create_dataset_yaml(Path(output_dir))
    
    print("Data preparation completed!")
    return True


def run_training(data_yaml: str = "data/yolo_dataset/dataset.yaml", 
                model_size: str = "n",
                epochs: int = 100,
                batch_size: int = 16,
                project: str = "results"):
    """Run training step."""
    print("Starting model training...")
    
    if not Path(data_yaml).exists():
        print(f"Dataset YAML not found at {data_yaml}")
        print("Running data preparation first...")
        run_data_preparation()
    
    trainer = YOLOv8Trainer(model_size=model_size, pretrained=True)
    trainer.load_model()
    
    training_params = {
        "data_yaml": data_yaml,
        "epochs": epochs,
        "imgsz": 640,
        "batch_size": batch_size,
        "lr0": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "project": project,
        "name": f"vhr10_yolov8{model_size}",
        "save_period": 10,
        "patience": 50,
        "amp": True,
        "cache": False,
        "workers": 8,
        "cos_lr": True,
        "close_mosaic": 10,
    }
    
    results = trainer.train(**training_params)
    
    # Validate on test set
    print("\nRunning validation on test set...")
    trainer.validate(split="test")
    
    print("Training completed!")
    return results


def run_evaluation(model_path: str, 
                  data_yaml: str = "data/yolo_dataset/dataset.yaml",
                  output_dir: str = "results/evaluation"):
    """Run evaluation step."""
    print("Starting model evaluation...")
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return False
    
    evaluator = YOLOv8Evaluator(model_path, data_yaml)
    
    # Run comprehensive evaluation
    evaluator.run_validation(split="test", conf_threshold=0.25, iou_threshold=0.5)
    evaluator.create_visualizations(output_dir)
    evaluator.save_detailed_report(f"{output_dir}/detailed_report.json")
    evaluator.print_summary()
    
    print("Evaluation completed!")
    return True


def run_full_pipeline(model_size: str = "n", epochs: int = 100, batch_size: int = 16):
    """Run the complete pipeline: data prep -> training -> evaluation."""
    print("Starting full YOLOv8 fine-tuning pipeline...")
    
    # Step 1: Data preparation
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    run_data_preparation()
    
    # Step 2: Training
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING")
    print("="*50)
    results = run_training(
        model_size=model_size, 
        epochs=epochs, 
        batch_size=batch_size
    )
    
    # Step 3: Evaluation
    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)
    model_path = f"results/vhr10_yolov8{model_size}/weights/best.pt"
    run_evaluation(model_path)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Trained model: {model_path}")
    print("Evaluation results: results/evaluation/")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-tuning on VHR-10 Dataset")
    parser.add_argument("command", choices=["data", "train", "eval", "all"], 
                       help="Command to run")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                       help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-path", help="Path to trained model (for evaluation)")
    parser.add_argument("--data-yaml", default="data/yolo_dataset/dataset.yaml",
                       help="Path to dataset YAML file")
    
    args = parser.parse_args()
    
    try:
        if args.command == "data":
            run_data_preparation()
        
        elif args.command == "train":
            run_training(
                data_yaml=args.data_yaml,
                model_size=args.model_size,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        
        elif args.command == "eval":
            if not args.model_path:
                model_path = f"results/vhr10_yolov8{args.model_size}/weights/best.pt"
            else:
                model_path = args.model_path
            
            run_evaluation(model_path, args.data_yaml)
        
        elif args.command == "all":
            run_full_pipeline(
                model_size=args.model_size,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()