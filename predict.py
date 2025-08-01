#!/usr/bin/env python3
"""
Replicate prediction interface for YOLOv8 VHR-10 satellite equipment detection model.
"""

import os
import tempfile
from typing import List
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from cog import BasePredictor, Input, Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class Predictor(BasePredictor):
    """Replicate predictor for YOLOv8 satellite equipment detection."""
    
    def setup(self) -> None:
        """Load the model on startup."""
        print("🚀 Loading YOLOv8 VHR-10 satellite equipment detection model...")
        
        try:
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id="bluelabel/satellite-equipment-detection-yolov8n-vhr10",
                filename="best.pt",
                cache_dir="/tmp/model_cache"
            )
            
            # Load the model
            self.model = YOLO(model_path)
            print("✅ Model loaded successfully!")
            
            # Define class names and colors
            self.class_names = {
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
            
            # Define colors for each class (RGB)
            self.class_colors = {
                0: (255, 0, 0),      # airplane - red
                1: (0, 0, 255),      # ship - blue
                2: (255, 165, 0),    # storage_tank - orange
                3: (0, 255, 0),      # baseball_diamond - green
                4: (255, 20, 147),   # tennis_court - deep pink
                5: (75, 0, 130),     # basketball_court - indigo
                6: (255, 255, 0),    # ground_track_field - yellow
                7: (0, 255, 255),    # harbor - cyan
                8: (128, 0, 128),    # bridge - purple
                9: (255, 69, 0)      # vehicle - orange red
            }
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(
        self,
        image: Path = Input(
            description="Input satellite image for equipment detection",
            default=None
        ),
        confidence_threshold: float = Input(
            description="Confidence threshold for detections",
            default=0.25,
            ge=0.01,
            le=1.0
        ),
        iou_threshold: float = Input(
            description="IoU threshold for non-maximum suppression",
            default=0.45,
            ge=0.01,
            le=1.0
        ),
        max_detections: int = Input(
            description="Maximum number of detections to return",
            default=100,
            ge=1,
            le=300
        ),
        draw_labels: bool = Input(
            description="Draw class labels on the output image",
            default=True
        ),
        draw_confidence: bool = Input(
            description="Draw confidence scores on the output image", 
            default=True
        )
    ) -> Path:
        """
        Run YOLOv8 inference on satellite imagery to detect ground equipment.
        
        Returns:
            Path: Annotated image with detected equipment highlighted
        """
        
        if image is None:
            raise ValueError("No image provided")
        
        print(f"🔍 Processing image: {image}")
        print(f"📊 Settings: conf={confidence_threshold}, iou={iou_threshold}, max_det={max_detections}")
        
        try:
            # Load and process the image
            pil_image = Image.open(image).convert('RGB')
            
            # Run inference
            results = self.model.predict(
                source=pil_image,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False
            )
            
            # Process results
            result = results[0]
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu()
                
                for i, box in enumerate(boxes):
                    # Extract box information
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
            
            print(f"✅ Found {len(detections)} detections")
            
            # Create annotated image
            annotated_image = self._draw_detections(
                pil_image, 
                detections, 
                draw_labels=draw_labels,
                draw_confidence=draw_confidence
            )
            
            # Save output
            output_path = Path(tempfile.mkdtemp()) / "output.jpg"
            annotated_image.save(output_path, "JPEG", quality=95)
            
            # Print detection summary
            self._print_detection_summary(detections)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise
    
    def _draw_detections(self, image: Image.Image, detections: List[dict], 
                        draw_labels: bool = True, draw_confidence: bool = True) -> Image.Image:
        """Draw detection boxes and labels on the image."""
        
        # Create a copy of the image to draw on
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Prepare label text
            label_parts = []
            if draw_labels:
                label_parts.append(class_name)
            if draw_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label_text = " ".join(label_parts)
                
                # Calculate text size and background
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw label background
                label_y = max(y1 - text_height - 10, 0)
                draw.rectangle(
                    [x1, label_y, x1 + text_width + 10, label_y + text_height + 8],
                    fill=color
                )
                
                # Draw label text
                draw.text(
                    (x1 + 5, label_y + 4),
                    label_text,
                    fill=(255, 255, 255),
                    font=font
                )
        
        return draw_image
    
    def _print_detection_summary(self, detections: List[dict]) -> None:
        """Print a summary of detections to the console."""
        
        if not detections:
            print("📋 No objects detected")
            return
        
        # Count detections by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"📋 Detection Summary ({len(detections)} total objects):")
        for class_name, count in sorted(class_counts.items()):
            print(f"   • {class_name}: {count}")
        
        # Print individual detections with high confidence
        high_conf_detections = [d for d in detections if d['confidence'] > 0.7]
        if high_conf_detections:
            print(f"🎯 High Confidence Detections (>{0.7:.1f}):")
            for detection in sorted(high_conf_detections, key=lambda x: x['confidence'], reverse=True)[:10]:
                print(f"   • {detection['class_name']}: {detection['confidence']:.3f}")