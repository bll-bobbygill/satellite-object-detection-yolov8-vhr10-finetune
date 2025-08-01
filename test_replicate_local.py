#!/usr/bin/env python3
"""
Local testing script for the Replicate deployment.
Test your model locally before pushing to Replicate.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import tempfile

# Add current directory to path to import predict.py
sys.path.insert(0, str(Path(__file__).parent))

try:
    from predict import Predictor
except ImportError as e:
    print(f"❌ Error importing predict.py: {e}")
    print("Make sure you're in the correct directory and have all dependencies installed.")
    sys.exit(1)


def test_local_prediction():
    """Test the Replicate predictor locally."""
    
    print("🧪 Testing Replicate predictor locally...")
    
    # Initialize predictor
    try:
        predictor = Predictor()
        predictor.setup()
        print("✅ Predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return False
    
    # Test with a sample image (you can replace this with your own test image)
    test_image_path = None
    
    # Look for test images in common locations
    possible_test_images = [
        "test_satellite_image.jpg",
        "sample_image.jpg",
        "data/yolo_dataset/test/images/004.jpg",  # Use a test image from dataset
        "data/yolo_dataset/val/images/004.jpg",   # Or validation image
    ]
    
    for img_path in possible_test_images:
        if Path(img_path).exists():
            test_image_path = img_path
            break
    
    if not test_image_path:
        print("❌ No test image found. Please provide a test satellite image.")
        print("You can:")
        print("1. Download a satellite image from Google Earth")
        print("2. Use one of your dataset images")
        print("3. Place a test image named 'test_satellite_image.jpg' in this directory")
        return False
    
    print(f"🖼️ Using test image: {test_image_path}")
    
    try:
        # Run prediction
        result_path = predictor.predict(
            image=Path(test_image_path),
            confidence_threshold=0.25,
            iou_threshold=0.45,
            max_detections=100,
            draw_labels=True,
            draw_confidence=True
        )
        
        print(f"✅ Prediction completed!")
        print(f"📁 Result saved to: {result_path}")
        
        # Optionally open the result (uncomment if you want to view the image)
        # import subprocess
        # subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(result_path)])
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False


def create_sample_test_image():
    """Create a simple test image if none exists."""
    
    test_path = Path("test_satellite_image.jpg")
    if test_path.exists():
        return str(test_path)
    
    print("🎨 Creating sample test image...")
    
    # Create a simple colored image as a placeholder
    img = Image.new('RGB', (640, 640), color='lightblue')
    img.save(test_path, 'JPEG')
    
    print(f"✅ Created sample image: {test_path}")
    print("Note: This is just a placeholder. Use real satellite imagery for meaningful tests.")
    
    return str(test_path)


if __name__ == "__main__":
    print("🚀 Local Replicate Testing")
    print("=" * 50)
    
    # Check if we have a test image, create one if not
    test_images = [p for p in Path(".").glob("*.jpg") if p.is_file()]
    if not test_images:
        create_sample_test_image()
    
    # Run the test
    success = test_local_prediction()
    
    if success:
        print("\n🎉 Local test successful!")
        print("You're ready to deploy to Replicate with:")
        print("   cog push r8.im/yourusername/satellite-equipment-detection")
    else:
        print("\n😞 Local test failed. Please check the errors above.")
        print("Make sure all dependencies are installed and the model can be downloaded.")
    
    print("\n📖 Next steps:")
    print("1. Fix any issues shown above")
    print("2. Test with real satellite images")
    print("3. Deploy to Replicate when ready")
    print("4. Share the Replicate URL for easy testing")