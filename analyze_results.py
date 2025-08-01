#!/usr/bin/env python3
"""Quick analysis of YOLOv8 evaluation results."""

import json
import pandas as pd
from pathlib import Path

def analyze_test_results():
    """Analyze test results and provide comprehensive evaluation."""
    
    # Test results from the evaluation output
    test_results = {
        'overall': {
            'total_images': 81,
            'total_instances': 300,
            'precision': 0.941,
            'recall': 0.965,
            'mAP50': 0.98,
            'mAP50_95': 0.682
        },
        'per_class': {
            'airplane': {'images': 10, 'instances': 74, 'precision': 0.995, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.743},
            'ship': {'images': 5, 'instances': 23, 'precision': 0.946, 'recall': 0.957, 'mAP50': 0.963, 'mAP50_95': 0.665},
            'storage_tank': {'images': 1, 'instances': 8, 'precision': 0.922, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.63},
            'baseball_diamond': {'images': 19, 'instances': 41, 'precision': 0.957, 'recall': 0.976, 'mAP50': 0.992, 'mAP50_95': 0.792},
            'tennis_court': {'images': 7, 'instances': 22, 'precision': 0.97, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.709},
            'basketball_court': {'images': 11, 'instances': 20, 'precision': 0.902, 'recall': 0.923, 'mAP50': 0.986, 'mAP50_95': 0.714},
            'ground_track_field': {'images': 16, 'instances': 16, 'precision': 1.0, 'recall': 0.98, 'mAP50': 0.995, 'mAP50_95': 0.872},
            'harbor': {'images': 2, 'instances': 9, 'precision': 0.9, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.696},
            'bridge': {'images': 6, 'instances': 10, 'precision': 0.939, 'recall': 1.0, 'mAP50': 0.995, 'mAP50_95': 0.483},
            'vehicle': {'images': 11, 'instances': 77, 'precision': 0.875, 'recall': 0.815, 'mAP50': 0.888, 'mAP50_95': 0.511}
        }
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE YOLOv8 VHR-10 EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"- Test Images: {test_results['overall']['total_images']}")
    print(f"- Total Object Instances: {test_results['overall']['total_instances']}")
    
    print(f"\nOVERALL PERFORMANCE METRICS:")
    print(f"- Precision: {test_results['overall']['precision']:.3f} (94.1%)")
    print(f"- Recall: {test_results['overall']['recall']:.3f} (96.5%)")
    print(f"- mAP@0.5: {test_results['overall']['mAP50']:.3f} (98.0%)")
    print(f"- mAP@0.5:0.95: {test_results['overall']['mAP50_95']:.3f} (68.2%)")
    
    print(f"\nPER-CLASS DETAILED ANALYSIS:")
    print("-" * 80)
    print(f"{'Class':<18} {'Images':<7} {'Objects':<8} {'Precision':<10} {'Recall':<8} {'mAP@0.5':<8} {'mAP@0.5:0.95':<10}")
    print("-" * 80)
    
    for class_name, metrics in test_results['per_class'].items():
        print(f"{class_name:<18} {metrics['images']:<7} {metrics['instances']:<8} "
              f"{metrics['precision']:<10.3f} {metrics['recall']:<8.3f} "
              f"{metrics['mAP50']:<8.3f} {metrics['mAP50_95']:<10.3f}")
    
    # Vehicle-specific analysis
    vehicle_metrics = test_results['per_class']['vehicle']
    
    print(f"\n" + "="*80)
    print("VEHICLE DETECTION ANALYSIS FOR SATELLITE IMAGERY")
    print("="*80)
    
    print(f"\nVEHICLE CLASS PERFORMANCE:")
    print(f"- Test Images with Vehicles: {vehicle_metrics['images']}")
    print(f"- Total Vehicle Instances: {vehicle_metrics['instances']}")
    print(f"- Precision: {vehicle_metrics['precision']:.3f} (87.5%)")
    print(f"- Recall: {vehicle_metrics['recall']:.3f} (81.5%)")
    print(f"- mAP@0.5: {vehicle_metrics['mAP50']:.3f} (88.8%)")
    print(f"- mAP@0.5:0.95: {vehicle_metrics['mAP50_95']:.3f} (51.1%)")
    
    print(f"\nVEHICLE COUNTING CAPABILITY ASSESSMENT:")
    
    # Calculate expected vs detected vehicles
    expected_vehicles = vehicle_metrics['instances']
    detected_rate = vehicle_metrics['recall']
    precision_rate = vehicle_metrics['precision']
    
    expected_detections = expected_vehicles * detected_rate
    true_positives = expected_detections * precision_rate
    false_positives = expected_detections * (1 - precision_rate)
    missed_vehicles = expected_vehicles * (1 - detected_rate)
    
    print(f"- Expected Vehicles in Test Set: {expected_vehicles}")
    print(f"- Estimated True Detections: {true_positives:.1f}")
    print(f"- Estimated False Positives: {false_positives:.1f}")
    print(f"- Estimated Missed Vehicles: {missed_vehicles:.1f}")
    print(f"- Vehicle Counting Accuracy: {true_positives/expected_vehicles:.1%}")
    
    print(f"\nSTRENGTHS:")
    print(f"✅ Excellent overall mAP@0.5 (98.0%) - model detects objects well at standard IoU")
    print(f"✅ High precision (94.1%) - low false positive rate")
    print(f"✅ Excellent performance on structured targets (sports courts, track fields)")
    print(f"✅ Perfect recall for several classes (airplane, storage_tank, tennis_court, harbor, bridge)")
    print(f"✅ Model generalizes well across different object types and scales")
    
    print(f"\nWEAKNESSES:")
    print(f"⚠️  Lower mAP@0.5:0.95 (68.2%) - struggles with higher IoU thresholds")
    print(f"⚠️  Vehicle detection has lowest performance metrics")
    print(f"⚠️  Vehicle recall (81.5%) means ~18.5% of vehicles are missed")
    print(f"⚠️  Bridge detection has lowest mAP@0.5:0.95 (48.3%)")
    print(f"⚠️  Small object detection may be challenging")
    
    print(f"\nVEHICLE DETECTION SUITABILITY FOR SATELLITE IMAGERY:")
    
    # Suitability assessment
    vehicle_suitability_score = (vehicle_metrics['mAP50'] + vehicle_metrics['recall'] + vehicle_metrics['precision']) / 3
    
    print(f"\nOverall Vehicle Detection Score: {vehicle_suitability_score:.3f} ({vehicle_suitability_score*100:.1f}%)")
    
    if vehicle_suitability_score >= 0.9:
        suitability = "EXCELLENT"
        recommendation = "Highly suitable for operational vehicle counting"
    elif vehicle_suitability_score >= 0.8:
        suitability = "GOOD"
        recommendation = "Suitable for vehicle counting with minor limitations"
    elif vehicle_suitability_score >= 0.7:
        suitability = "FAIR"
        recommendation = "Usable but may require additional validation"
    else:
        suitability = "POOR"
        recommendation = "Requires significant improvement"
    
    print(f"\nSUITABILITY RATING: {suitability}")
    print(f"RECOMMENDATION: {recommendation}")
    
    print(f"\nVEHICLE COUNTING CONSIDERATIONS:")
    print(f"📊 For every 100 vehicles in satellite imagery:")
    print(f"   - Model will detect ~{vehicle_metrics['recall']*100:.0f} vehicles")
    print(f"   - Of detected vehicles, ~{vehicle_metrics['precision']*100:.0f}% will be correct")
    print(f"   - Expected counting accuracy: ~{(true_positives/expected_vehicles)*100:.0f}%")
    print(f"   - Expected false alarms: ~{(false_positives/expected_vehicles)*100:.0f} per 100 actual vehicles")
    
    print(f"\nIMPROVEMENT RECOMMENDATIONS:")
    print(f"🔧 Increase training epochs or adjust learning rate for vehicle class")
    print(f"🔧 Augment dataset with more diverse vehicle examples")
    print(f"🔧 Consider class-weighted loss to improve vehicle detection")
    print(f"🔧 Use larger model (YOLOv8s/m) for better small object detection")
    print(f"🔧 Implement post-processing filtering for vehicle-specific false positives")
    
    print("="*80)
    
    return test_results

if __name__ == "__main__":
    results = analyze_test_results()