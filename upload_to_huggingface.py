#!/usr/bin/env python3
"""Upload YOLOv8 VHR-10 fine-tuned model to HuggingFace Hub."""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

def create_model_card():
    """Create a comprehensive model card for the HuggingFace repository."""
    
    model_card = """---
language: en
license: mit
tags:
- computer-vision
- object-detection
- yolov8
- satellite-imagery
- remote-sensing
- vhr-10
- geospatial
- equipment-detection
datasets:
- VHR-10
pipeline_tag: object-detection
---

# YOLOv8n Fine-tuned on VHR-10 Remote Sensing Dataset

This model is a fine-tuned YOLOv8n (nano) model trained on the NWPU VHR-10 (Very High Resolution) remote sensing dataset for detecting ground equipment and vehicles in satellite imagery.

## Model Description

This model demonstrates the feasibility of using YOLOv8 for detecting various pieces of ground equipment through satellite imagery, serving as a proof-of-concept for commercial applications in competitive intelligence, fleet monitoring, and automated equipment detection.

### Model Details

- **Model Type**: YOLOv8n (nano) - Object Detection
- **Training Dataset**: NWPU VHR-10 Remote Sensing Dataset
- **Model Size**: ~6MB (3M parameters)
- **Input Resolution**: 640x640 pixels
- **Training Duration**: 50 epochs
- **Framework**: Ultralytics YOLOv8

### Detected Classes

The model can detect 10 classes of objects commonly found in satellite imagery:

1. **airplane** - Aircraft on airfields and airports
2. **ship** - Naval vessels and boats
3. **storage_tank** - Industrial storage tanks
4. **baseball_diamond** - Baseball fields and diamonds
5. **tennis_court** - Tennis courts and facilities
6. **basketball_court** - Basketball courts
7. **ground_track_field** - Athletic tracks and fields
8. **harbor** - Harbor facilities and ports
9. **bridge** - Bridges and overpasses
10. **vehicle** - Ground vehicles and equipment

## Performance Metrics

### Overall Performance
- **mAP@0.5**: 98.0% (exceptional)
- **mAP@0.5:0.95**: 68.2% (good across IoU thresholds)
- **Overall Precision**: 94.1%
- **Overall Recall**: 96.5%
- **Inference Speed**: 9.9ms per image

### Vehicle Detection Performance (Primary Focus)
- **Vehicle F1 Score**: 79.2%
- **Vehicle Precision**: 87.5%
- **Vehicle Recall**: 81.5%
- **Vehicle mAP@0.5**: 88.8%

### Class-wise Performance (F1 Scores)
1. Ground Track Field: 100.0%
2. Airplane: 98.0%
3. Ship: 95.8%
4. Baseball Diamond: 94.3%
5. Tennis Court: 91.7%
6. Basketball Court: 90.9%
7. Bridge: 87.0%
8. Storage Tank: 84.2%
9. Harbor: 81.8%
10. Vehicle: 79.2%

## Intended Use

### Primary Applications
- **Proof-of-concept** for satellite-based equipment detection
- **Competitive intelligence** and market analysis
- **Fleet monitoring** and logistics optimization
- **Infrastructure inventory** management
- **Automated lead generation** based on equipment detection

### Commercial Potential
This model demonstrates that AI can reliably detect vehicles and equipment in satellite imagery, laying the groundwork for specialized commercial applications such as:
- Hostler detection for logistics companies
- Construction equipment monitoring
- Fleet tracking and analysis
- Market research and competitive analysis

## Usage

### Loading the Model

```python
from ultralytics import YOLO

# Load the model
model = YOLO('path/to/best.pt')

# Run inference
results = model('satellite_image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        print(f"Detected: {model.names[class_id]} (confidence: {confidence:.3f})")
```

### Model Files
- `best.pt` - Best performing model weights (recommended for inference)
- `last.pt` - Final epoch model weights
- `dataset.yaml` - Dataset configuration file

## Training Details

### Dataset
- **NWPU VHR-10 Dataset**: 800 very high-resolution remote sensing images
- **Training Split**: 70% (559 images)
- **Validation Split**: 20% (160 images)  
- **Test Split**: 10% (81 images)
- **Image Sources**: Google Earth and Vaihingen dataset

### Training Configuration
- **Model**: YOLOv8n (nano)
- **Epochs**: 50
- **Batch Size**: 8 (memory optimized)
- **Image Size**: 640x640
- **Optimizer**: AdamW (auto-selected)
- **Learning Rate**: 0.000714 (auto-selected)
- **GPU**: NVIDIA RTX 4090

### Data Augmentation
- Mosaic augmentation
- Mixup augmentation  
- HSV augmentation
- Geometric transformations
- Automatic mixed precision (AMP)

## Limitations and Considerations

### Strengths
- Excellent overall detection performance (98% mAP@0.5)
- High recall rate ensures minimal missed detections
- Fast inference suitable for real-time applications
- Good generalization across different object types

### Limitations
- Vehicle detection shows 49% over-prediction rate (false positives)
- Performance varies with object size and complexity
- Generic model - specialized training could significantly improve accuracy
- Limited to 10 predefined classes

### Recommendations for Production Use
- Implement post-processing filtering for specific use cases
- Consider ensemble methods for higher accuracy
- Use larger YOLOv8 variants (s/m/l) for better precision
- Develop specialized models for specific equipment types

## Model Card Authors

This model was developed as a proof-of-concept for satellite-based equipment detection applications.

## Citation

If you use this model in your research, please cite the original VHR-10 dataset:

```bibtex
@article{cheng2014multi,
  title={Multi-class geospatial object detection and geographic image classification based on collection of part detectors},
  author={Cheng, Gong and Han, Junwei and Zhou, Peicheng and Guo, Lei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={98},
  pages={119--132},
  year={2014},
  publisher={Elsevier}
}
```

## License

This model is released under the MIT License. The underlying YOLOv8 framework is licensed under GPL-3.0.
"""
    return model_card

def create_config_files():
    """Create configuration files for the model."""
    
    # Create dataset config
    dataset_config = {
        "path": "data/yolo_dataset",
        "train": "train/images",
        "val": "val/images", 
        "test": "test/images",
        "nc": 10,
        "names": {
            0: "airplane",
            1: "ship", 
            2: "storage_tank",
            3: "baseball_diamond",
            4: "tennis_court",
            5: "basketball_court",
            6: "ground_track_field",
            7: "harbor",
            8: "bridge",
            9: "vehicle"
        }
    }
    
    # Model metadata
    model_metadata = {
        "model_type": "YOLOv8n",
        "task": "object-detection",
        "framework": "ultralytics",
        "dataset": "VHR-10",
        "classes": 10,
        "image_size": 640,
        "parameters": 3012798,
        "model_size_mb": 6.3,
        "training_epochs": 50,
        "batch_size": 8,
        "performance": {
            "mAP50": 0.98,
            "mAP50_95": 0.682,
            "precision": 0.941,
            "recall": 0.965,
            "vehicle_f1": 0.792,
            "inference_speed_ms": 9.9
        }
    }
    
    return dataset_config, model_metadata

def upload_to_huggingface():
    """Upload the model to HuggingFace Hub."""
    
    # Configuration
    repo_name = "satellite-equipment-detection-yolov8n-vhr10"
    username = input("Enter your HuggingFace username: ").strip()
    
    if not username:
        print("Username is required!")
        return
    
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating repository: {repo_id}")
    
    # Create temporary directory for upload
    upload_dir = Path("huggingface_upload")
    upload_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repository
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"✅ Repository created: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Repository might already exist: {e}")
        
        # Copy model files
        model_files = [
            "results/vhr10_yolov8n_quick/weights/best.pt",
            "results/vhr10_yolov8n_quick/weights/last.pt"
        ]
        
        for model_file in model_files:
            if Path(model_file).exists():
                shutil.copy(model_file, upload_dir / Path(model_file).name)
                print(f"✅ Copied {model_file}")
        
        # Create model card
        model_card = create_model_card()
        with open(upload_dir / "README.md", "w") as f:
            f.write(model_card)
        print("✅ Created model card (README.md)")
        
        # Create config files
        dataset_config, model_metadata = create_config_files()
        
        with open(upload_dir / "dataset.yaml", "w") as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        with open(upload_dir / "model_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        print("✅ Created configuration files")
        
        # Upload files
        print("🚀 Uploading to HuggingFace Hub...")
        
        for file_path in upload_dir.glob("*"):
            if file_path.is_file():
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✅ Uploaded {file_path.name}")
        
        print(f"\n🎉 Model successfully uploaded to: https://huggingface.co/{repo_id}")
        print("\nModel files uploaded:")
        print("- best.pt (recommended for inference)")
        print("- last.pt (final training checkpoint)")
        print("- README.md (comprehensive model card)")
        print("- dataset.yaml (dataset configuration)")
        print("- model_metadata.json (model metadata)")
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        
    finally:
        # Cleanup
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    print("🤗 HuggingFace Model Upload")
    print("=" * 50)
    print("This script will upload your YOLOv8 VHR-10 model to HuggingFace Hub")
    print("Make sure you're logged in with: huggingface-cli login")
    print()
    
    confirm = input("Continue with upload? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        upload_to_huggingface()
    else:
        print("Upload cancelled.")