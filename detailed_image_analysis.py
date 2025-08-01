#!/usr/bin/env python3
"""Detailed per-image analysis of YOLOv8 evaluation results."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
import os

def parse_yolo_label(label_file):
    """Parse YOLO format label file to extract class counts."""
    class_counts = {i: 0 for i in range(10)}  # 10 classes
    
    if not label_file.exists():
        return class_counts
    
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:  # class_id x_center y_center width height
                    class_id = int(parts[0])
                    if 0 <= class_id < 10:
                        class_counts[class_id] += 1
    except Exception as e:
        print(f"Error parsing {label_file}: {e}")
    
    return class_counts

def get_predictions_for_image(model, image_path, conf_threshold=0.25):
    """Get model predictions for a single image."""
    try:
        results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            return {i: 0 for i in range(10)}
        
        result = results[0]
        pred_counts = {i: 0 for i in range(10)}
        
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for class_id in classes:
                if 0 <= class_id < 10:
                    pred_counts[class_id] += 1
        
        return pred_counts
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return {i: 0 for i in range(10)}

def create_detailed_analysis():
    """Create detailed per-image analysis."""
    
    # Class names
    class_names = [
        "airplane", "ship", "storage_tank", "baseball_diamond", 
        "tennis_court", "basketball_court", "ground_track_field", 
        "harbor", "bridge", "vehicle"
    ]
    
    # Paths
    model_path = "results/vhr10_yolov8n_quick/weights/best.pt"
    test_images_dir = Path("data/yolo_dataset/test/images")
    test_labels_dir = Path("data/yolo_dataset/test/labels")
    
    # Load model
    print("Loading trained model...")
    model = YOLO(model_path)
    
    # Get all test images
    image_files = sorted(list(test_images_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} test images")
    
    # Create data structure for analysis
    analysis_data = []
    
    print("Analyzing each test image...")
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        # Get corresponding label file
        label_file = test_labels_dir / (image_file.stem + ".txt")
        
        # Parse ground truth
        gt_counts = parse_yolo_label(label_file)
        
        # Get predictions
        pred_counts = get_predictions_for_image(model, image_file)
        
        # Create row data
        row_data = {"image": image_file.name}
        
        # Add ground truth counts
        for class_id, class_name in enumerate(class_names):
            row_data[f"GT_{class_name}"] = gt_counts[class_id]
        
        # Add predicted counts
        for class_id, class_name in enumerate(class_names):
            row_data[f"PRED_{class_name}"] = pred_counts[class_id]
        
        # Calculate totals and accuracy for this image
        gt_total = sum(gt_counts.values())
        pred_total = sum(pred_counts.values())
        
        # Calculate per-image metrics
        if gt_total > 0:
            # Calculate correct predictions (intersection of GT and PRED)
            correct_predictions = 0
            for class_id in range(10):
                correct_predictions += min(gt_counts[class_id], pred_counts[class_id])
            
            precision = correct_predictions / pred_total if pred_total > 0 else 0
            recall = correct_predictions / gt_total if gt_total > 0 else 0
            
            row_data["GT_total"] = gt_total
            row_data["PRED_total"] = pred_total
            row_data["precision"] = precision
            row_data["recall"] = recall
        else:
            row_data["GT_total"] = 0
            row_data["PRED_total"] = pred_total
            row_data["precision"] = 1.0 if pred_total == 0 else 0.0
            row_data["recall"] = 1.0 if pred_total == 0 else 0.0
        
        analysis_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Calculate summary statistics
    print("\n" + "="*120)
    print("DETAILED PER-IMAGE ANALYSIS RESULTS")
    print("="*120)
    
    # Display the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\nPER-IMAGE GROUND TRUTH vs PREDICTED COUNTS:")
    print("-" * 120)
    
    # Create a more readable format
    display_columns = ["image"]
    for class_name in class_names:
        display_columns.extend([f"GT_{class_name}", f"PRED_{class_name}"])
    display_columns.extend(["GT_total", "PRED_total", "precision", "recall"])
    
    # Show only images with objects for readability
    df_with_objects = df[df['GT_total'] > 0].copy()
    
    print(f"\nShowing {len(df_with_objects)} images with ground truth objects:")
    print(df_with_objects[display_columns].to_string(index=False, float_format='{:.3f}'.format))
    
    # Summary statistics
    print(f"\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    total_gt = df['GT_total'].sum()
    total_pred = df['PRED_total'].sum()
    avg_precision = df[df['GT_total'] > 0]['precision'].mean()
    avg_recall = df[df['GT_total'] > 0]['recall'].mean()
    
    print(f"Total Ground Truth Objects: {total_gt}")
    print(f"Total Predicted Objects: {total_pred}")
    print(f"Average Per-Image Precision: {avg_precision:.3f}")
    print(f"Average Per-Image Recall: {avg_recall:.3f}")
    
    # Class-wise summary
    print(f"\nCLASS-WISE TOTALS:")
    print("-" * 60)
    print(f"{'Class':<20} {'GT Count':<10} {'Pred Count':<12} {'Difference':<10}")
    print("-" * 60)
    
    for class_name in class_names:
        gt_col = f"GT_{class_name}"
        pred_col = f"PRED_{class_name}"
        gt_sum = df[gt_col].sum()
        pred_sum = df[pred_col].sum()
        diff = pred_sum - gt_sum
        print(f"{class_name:<20} {gt_sum:<10} {pred_sum:<12} {diff:+<10}")
    
    # Vehicle-specific analysis
    print(f"\n" + "="*120)
    print("VEHICLE-SPECIFIC ANALYSIS")
    print("="*120)
    
    vehicle_gt = df['GT_vehicle'].sum()
    vehicle_pred = df['PRED_vehicle'].sum()
    
    # Images with vehicles
    vehicle_images = df[df['GT_vehicle'] > 0].copy()
    
    print(f"Images with vehicles: {len(vehicle_images)}")
    print(f"Total vehicles (GT): {vehicle_gt}")
    print(f"Total vehicles (Predicted): {vehicle_pred}")
    print(f"Vehicle detection rate: {vehicle_pred/vehicle_gt*100:.1f}%")
    
    if len(vehicle_images) > 0:
        print(f"\nVEHICLE DETECTION BY IMAGE:")
        print("-" * 80)
        vehicle_cols = ['image', 'GT_vehicle', 'PRED_vehicle', 'precision', 'recall']
        print(vehicle_images[vehicle_cols].to_string(index=False, float_format='{:.3f}'.format))
    
    # Error analysis
    print(f"\n" + "="*120)
    print("ERROR ANALYSIS")
    print("="*120)
    
    # False positives and false negatives
    false_positive_images = df[df['PRED_total'] > df['GT_total']]
    false_negative_images = df[df['PRED_total'] < df['GT_total']]
    
    print(f"Images with over-prediction (FP): {len(false_positive_images)}")
    print(f"Images with under-prediction (FN): {len(false_negative_images)}")
    
    if len(false_positive_images) > 0:
        print(f"\nOver-predicted images:")
        fp_cols = ['image', 'GT_total', 'PRED_total']
        print(false_positive_images[fp_cols].head(10).to_string(index=False))
    
    if len(false_negative_images) > 0:
        print(f"\nUnder-predicted images:")
        fn_cols = ['image', 'GT_total', 'PRED_total']
        print(false_negative_images[fn_cols].head(10).to_string(index=False))
    
    # Save detailed results
    output_file = "detailed_per_image_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    df = create_detailed_analysis()