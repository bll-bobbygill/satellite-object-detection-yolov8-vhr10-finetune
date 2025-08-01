#!/usr/bin/env python3
"""Class-wise accuracy analysis for YOLOv8 model."""

import pandas as pd
import numpy as np

def calculate_class_wise_accuracy():
    """Calculate detailed accuracy metrics for each class."""
    
    # Read the detailed CSV file
    try:
        df = pd.read_csv("detailed_per_image_analysis.csv")
    except FileNotFoundError:
        print("Error: detailed_per_image_analysis.csv not found. Please run detailed_image_analysis.py first.")
        return
    
    class_names = [
        "airplane", "ship", "storage_tank", "baseball_diamond", 
        "tennis_court", "basketball_court", "ground_track_field", 
        "harbor", "bridge", "vehicle"
    ]
    
    # Test results from validation (from previous analysis)
    validation_metrics = {
        'airplane': {'precision': 0.995, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.743},
        'ship': {'precision': 0.946, 'recall': 0.957, 'mAP50': 0.963, 'mAP50_95': 0.665},
        'storage_tank': {'precision': 0.922, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.63},
        'baseball_diamond': {'precision': 0.957, 'recall': 0.976, 'mAP50': 0.992, 'mAP50_95': 0.792},
        'tennis_court': {'precision': 0.97, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.709},
        'basketball_court': {'precision': 0.902, 'recall': 0.923, 'mAP50': 0.986, 'mAP50_95': 0.714},
        'ground_track_field': {'precision': 1.0, 'recall': 0.98, 'mAP50': 0.995, 'mAP50_95': 0.872},
        'harbor': {'precision': 0.9, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.696},
        'bridge': {'precision': 0.939, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.483},
        'vehicle': {'precision': 0.875, 'recall': 0.815, 'mAP50': 0.888, 'mAP50_95': 0.511}
    }
    
    # Create class-wise analysis
    class_analysis = []
    
    for class_name in class_names:
        gt_col = f"GT_{class_name}"
        pred_col = f"PRED_{class_name}"
        
        # Basic counts
        gt_total = df[gt_col].sum()
        pred_total = df[pred_col].sum()
        
        # Images with this class
        images_with_class = len(df[df[gt_col] > 0])
        
        # Calculate detection metrics
        if gt_total > 0:
            # True Positives: minimum of GT and Pred for each image, summed
            true_positives = sum(min(row[gt_col], row[pred_col]) for _, row in df.iterrows())
            
            # False Positives: predictions that exceed ground truth
            false_positives = max(0, pred_total - true_positives)
            
            # False Negatives: ground truth objects that weren't detected
            false_negatives = gt_total - true_positives
            
            # Calculate metrics
            precision_calc = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall_calc = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision_calc * recall_calc) / (precision_calc + recall_calc) if (precision_calc + recall_calc) > 0 else 0
            
            # Detection rate
            detection_rate = pred_total / gt_total if gt_total > 0 else 0
            
            # Counting accuracy (how close predicted count is to actual count)
            counting_accuracy = true_positives / gt_total if gt_total > 0 else 0
            
        else:
            # No ground truth objects for this class
            true_positives = 0
            false_positives = pred_total
            false_negatives = 0
            precision_calc = 1.0 if pred_total == 0 else 0.0
            recall_calc = 1.0  # No objects to miss
            f1_score = 1.0 if pred_total == 0 else 0.0
            detection_rate = float('inf') if pred_total > 0 else 1.0
            counting_accuracy = 1.0 if pred_total == 0 else 0.0
        
        # Get validation metrics
        val_metrics = validation_metrics.get(class_name, {})
        
        class_analysis.append({
            'Class': class_name,
            'GT_Objects': gt_total,
            'Pred_Objects': pred_total,
            'Images_with_Class': images_with_class,
            'True_Positives': true_positives,
            'False_Positives': false_positives,
            'False_Negatives': false_negatives,
            'Precision_Calculated': precision_calc,
            'Recall_Calculated': recall_calc,
            'F1_Score': f1_score,
            'Detection_Rate': detection_rate,
            'Counting_Accuracy': counting_accuracy,
            'Validation_Precision': val_metrics.get('precision', 0),
            'Validation_Recall': val_metrics.get('recall', 0),
            'Validation_mAP50': val_metrics.get('mAP50', 0),
            'Validation_mAP50_95': val_metrics.get('mAP50_95', 0)
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(class_analysis)
    
    # Display comprehensive table
    print("="*150)
    print("COMPREHENSIVE CLASS-WISE ACCURACY ANALYSIS")
    print("="*150)
    
    print("\nTABLE 1: BASIC OBJECT COUNTS AND DETECTION PERFORMANCE")
    print("-"*150)
    basic_cols = ['Class', 'GT_Objects', 'Pred_Objects', 'Images_with_Class', 'Detection_Rate', 'Counting_Accuracy']
    basic_df = results_df[basic_cols].copy()
    basic_df['Detection_Rate'] = basic_df['Detection_Rate'].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
    basic_df['Counting_Accuracy'] = basic_df['Counting_Accuracy'].apply(lambda x: f"{x:.3f}")
    print(basic_df.to_string(index=False))
    
    print("\nTABLE 2: DETAILED ACCURACY METRICS")
    print("-"*150)
    accuracy_cols = ['Class', 'True_Positives', 'False_Positives', 'False_Negatives', 'Precision_Calculated', 'Recall_Calculated', 'F1_Score']
    accuracy_df = results_df[accuracy_cols].copy()
    for col in ['Precision_Calculated', 'Recall_Calculated', 'F1_Score']:
        accuracy_df[col] = accuracy_df[col].apply(lambda x: f"{x:.3f}")
    print(accuracy_df.to_string(index=False))
    
    print("\nTABLE 3: VALIDATION METRICS COMPARISON")
    print("-"*150)
    validation_cols = ['Class', 'Validation_Precision', 'Validation_Recall', 'Validation_mAP50', 'Validation_mAP50_95']
    validation_df = results_df[validation_cols].copy()
    for col in ['Validation_Precision', 'Validation_Recall', 'Validation_mAP50', 'Validation_mAP50_95']:
        validation_df[col] = validation_df[col].apply(lambda x: f"{x:.3f}")
    print(validation_df.to_string(index=False))
    
    # Performance ranking
    print("\nCLASS PERFORMANCE RANKING")
    print("-"*80)
    
    # Sort by F1 score for overall performance
    ranked_df = results_df.sort_values('F1_Score', ascending=False)
    
    print("Ranked by F1 Score (Overall Performance):")
    print(f"{'Rank':<5} {'Class':<20} {'F1 Score':<10} {'Precision':<12} {'Recall':<10} {'Counting Acc':<15}")
    print("-"*80)
    
    for i, (_, row) in enumerate(ranked_df.iterrows(), 1):
        class_name = row['Class']
        f1 = row['F1_Score']
        prec = row['Precision_Calculated']
        rec = row['Recall_Calculated']
        count_acc = row['Counting_Accuracy']
        
        print(f"{i:<5} {class_name:<20} {f1:<10.3f} {prec:<12.3f} {rec:<10.3f} {count_acc:<15.3f}")
    
    # Best and worst performers
    print("\nPERFORMACE ANALYSIS")
    print("-"*80)
    
    best_performer = ranked_df.iloc[0]
    worst_performer = ranked_df.iloc[-1]
    
    print(f"🏆 BEST PERFORMER: {best_performer['Class']}")
    print(f"   F1 Score: {best_performer['F1_Score']:.3f}")
    print(f"   Precision: {best_performer['Precision_Calculated']:.3f}")
    print(f"   Recall: {best_performer['Recall_Calculated']:.3f}")
    print(f"   Counting Accuracy: {best_performer['Counting_Accuracy']:.3f}")
    
    print(f"\n⚠️  NEEDS IMPROVEMENT: {worst_performer['Class']}")
    print(f"   F1 Score: {worst_performer['F1_Score']:.3f}")
    print(f"   Precision: {worst_performer['Precision_Calculated']:.3f}")
    print(f"   Recall: {worst_performer['Recall_Calculated']:.3f}")
    print(f"   Counting Accuracy: {worst_performer['Counting_Accuracy']:.3f}")
    
    # Class-specific insights
    print("\nCLASS-SPECIFIC INSIGHTS")
    print("-"*80)
    
    # Over-detection issues
    over_detectors = results_df[results_df['Detection_Rate'] > 1.2]  # More than 20% over-detection
    if len(over_detectors) > 0:
        print("Classes with significant over-detection:")
        for _, row in over_detectors.iterrows():
            rate = row['Detection_Rate']
            if rate != float('inf'):
                print(f"• {row['Class']}: {rate:.1f}x detection rate ({(rate-1)*100:.0f}% over-detection)")
    
    # Under-detection issues
    under_detectors = results_df[results_df['Detection_Rate'] < 0.8]  # Less than 80% detection
    if len(under_detectors) > 0:
        print("\nClasses with under-detection:")
        for _, row in under_detectors.iterrows():
            rate = row['Detection_Rate']
            print(f"• {row['Class']}: {rate:.1f}x detection rate ({(1-rate)*100:.0f}% under-detection)")
    
    # Perfect performers
    perfect_performers = results_df[(results_df['Precision_Calculated'] >= 0.95) & 
                                  (results_df['Recall_Calculated'] >= 0.95) & 
                                  (results_df['GT_Objects'] > 0)]
    if len(perfect_performers) > 0:
        print("\nNear-perfect performers (≥95% precision and recall):")
        for _, row in perfect_performers.iterrows():
            print(f"• {row['Class']}: {row['Precision_Calculated']:.1%} precision, {row['Recall_Calculated']:.1%} recall")
    
    # Save detailed results
    output_file = "class_wise_accuracy_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed class-wise analysis saved to: {output_file}")
    
    print("\n" + "="*150)
    print("END OF CLASS-WISE ACCURACY ANALYSIS")
    print("="*150)
    
    return results_df

if __name__ == "__main__":
    df = calculate_class_wise_accuracy()