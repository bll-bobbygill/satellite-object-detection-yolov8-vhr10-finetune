"""Data preparation utilities for VHR-10 dataset."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
from PIL import Image


class VHR10DataPrep:
    """Prepare VHR-10 dataset for YOLO training."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.class_names = [
            "airplane", "ship", "storage_tank", "baseball_diamond", 
            "tennis_court", "basketball_court", "ground_track_field", 
            "harbor", "bridge", "vehicle"
        ]
        
    def convert_bbox_to_yolo(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bounding box from (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height)."""
        x1, y1, x2, y2 = bbox
        
        # Calculate center coordinates and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize by image dimensions
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return x_center, y_center, width, height
    
    def parse_ground_truth_file(self, gt_file: Path) -> List[Tuple[int, int, int, int, int]]:
        """Parse ground truth annotation file."""
        annotations = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: (x1,y1),(x2,y2),class_id
                parts = line.split(',')
                if len(parts) != 5:
                    continue
                    
                try:
                    x1 = int(parts[0].replace('(', ''))
                    y1 = int(parts[1].replace(')', ''))
                    x2 = int(parts[2].replace('(', ''))
                    y2 = int(parts[3].replace(')', ''))
                    class_id = int(parts[4]) - 1  # Convert to 0-based indexing
                    
                    annotations.append((x1, y1, x2, y2, class_id))
                except ValueError:
                    continue
                    
        return annotations
    
    def create_yolo_annotation(self, img_path: Path, gt_path: Path, output_path: Path):
        """Create YOLO format annotation file."""
        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Parse ground truth
            annotations = self.parse_ground_truth_file(gt_path)
            
            # Write YOLO format annotations
            with open(output_path, 'w') as f:
                for x1, y1, x2, y2, class_id in annotations:
                    x_center, y_center, width, height = self.convert_bbox_to_yolo(
                        (x1, y1, x2, y2), img_width, img_height
                    )
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Create empty annotation file if image can't be processed
            output_path.touch()
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
        """Split dataset into train/validation/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(seed)
        
        # Get all positive images (those with annotations)
        positive_dir = self.data_dir / "VHR-10" / "positive image set"
        ground_truth_dir = self.data_dir / "VHR-10" / "ground truth"
        
        # Find images that have corresponding ground truth files and are valid images
        img_files = []
        for img_file in positive_dir.glob("*.jpg"):
            gt_file = ground_truth_dir / f"{img_file.stem}.txt"
            if gt_file.exists():
                # Check if it's actually a valid image file
                try:
                    with Image.open(img_file):
                        pass
                    img_files.append(img_file.stem)
                except Exception:
                    print(f"Skipping invalid image file: {img_file}")
                    continue
        
        # Add negative images (also check validity)
        negative_dir = self.data_dir / "VHR-10" / "negative image set"
        negative_files = []
        for img_file in negative_dir.glob("*.jpg"):
            try:
                with Image.open(img_file):
                    pass
                negative_files.append(img_file.stem)
            except Exception:
                print(f"Skipping invalid negative image file: {img_file}")
                continue
        
        print(f"Found {len(img_files)} positive images and {len(negative_files)} negative images")
        
        # Shuffle and split
        random.shuffle(img_files)
        random.shuffle(negative_files)
        
        # Split positive images
        n_pos = len(img_files)
        train_pos_end = int(n_pos * train_ratio)
        val_pos_end = int(n_pos * (train_ratio + val_ratio))
        
        train_pos = img_files[:train_pos_end]
        val_pos = img_files[train_pos_end:val_pos_end]
        test_pos = img_files[val_pos_end:]
        
        # Split negative images
        n_neg = len(negative_files)
        train_neg_end = int(n_neg * train_ratio)
        val_neg_end = int(n_neg * (train_ratio + val_ratio))
        
        train_neg = negative_files[:train_neg_end]
        val_neg = negative_files[train_neg_end:val_neg_end]
        test_neg = negative_files[val_neg_end:]
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process each split
        splits = {
            'train': (train_pos, train_neg),
            'val': (val_pos, val_neg),
            'test': (test_pos, test_neg)
        }
        
        for split_name, (pos_files, neg_files) in splits.items():
            print(f"Processing {split_name} split: {len(pos_files)} positive, {len(neg_files)} negative")
            
            # Process positive images
            for img_stem in pos_files:
                # Copy image
                src_img = positive_dir / f"{img_stem}.jpg"
                dst_img = self.output_dir / split_name / 'images' / f"{img_stem}.jpg"
                shutil.copy2(src_img, dst_img)
                
                # Create YOLO annotation
                gt_file = ground_truth_dir / f"{img_stem}.txt"
                label_file = self.output_dir / split_name / 'labels' / f"{img_stem}.txt"
                self.create_yolo_annotation(src_img, gt_file, label_file)
            
            # Process negative images
            for img_stem in neg_files:
                # Copy image
                src_img = negative_dir / f"{img_stem}.jpg"
                dst_img = self.output_dir / split_name / 'images' / f"neg_{img_stem}.jpg"
                shutil.copy2(src_img, dst_img)
                
                # Create empty annotation file for negative images
                label_file = self.output_dir / split_name / 'labels' / f"neg_{img_stem}.txt"
                label_file.touch()
        
        print("Dataset split completed!")
        return splits
    
    def create_dataset_yaml(self, dataset_path: Path):
        """Create dataset.yaml file for YOLO training."""
        yaml_content = f"""# VHR-10 dataset configuration
path: {dataset_path.absolute()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 10

# Class names
names:
"""
        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_file = dataset_path / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"Dataset YAML created at: {yaml_file}")
        return yaml_file


def main():
    """Main function to prepare the dataset."""
    data_dir = Path("data")
    output_dir = Path("data/yolo_dataset")
    
    prep = VHR10DataPrep(data_dir, output_dir)
    
    # Split dataset
    splits = prep.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    # Create dataset YAML
    prep.create_dataset_yaml(output_dir)
    
    print("\nDataset preparation completed!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()