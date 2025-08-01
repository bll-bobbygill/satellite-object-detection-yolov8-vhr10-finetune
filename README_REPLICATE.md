# Deploying YOLOv8 VHR-10 Model to Replicate

This guide explains how to deploy your fine-tuned YOLOv8 satellite equipment detection model to Replicate for easy testing and demonstration.

## 🚀 Quick Start

### Prerequisites

1. **Replicate Account**: Sign up at [https://replicate.com](https://replicate.com)
2. **Cog Installed**: Install Cog (Replicate's deployment tool)

```bash
# Install Cog (requires Python 3.8+)
pip install cog
```

### Deploy to Replicate

1. **Push your model to Replicate:**

```bash
# Login to Replicate
cog login

# Push the model (replace 'yourusername' with your Replicate username)
cog push r8.im/yourusername/satellite-equipment-detection
```

2. **Test locally first (optional):**

```bash
# Test the model locally before pushing
cog predict -i image=@path/to/your/satellite/image.jpg
```

## 🎯 Model Features

### Input Parameters

- **image**: Upload satellite images (JPG, PNG)
- **confidence_threshold**: Adjust detection sensitivity (0.01-1.0, default: 0.25)
- **iou_threshold**: Control overlapping detections (0.01-1.0, default: 0.45)
- **max_detections**: Maximum objects to detect (1-300, default: 100)
- **draw_labels**: Show class names on output image
- **draw_confidence**: Show confidence scores on output image

### Detected Classes

The model can identify 10 types of objects in satellite imagery:

1. 🛩️ **Airplane** - Aircraft on airfields
2. 🚢 **Ship** - Naval vessels and boats  
3. ⛽ **Storage Tank** - Industrial storage facilities
4. ⚾ **Baseball Diamond** - Baseball fields
5. 🎾 **Tennis Court** - Tennis facilities
6. 🏀 **Basketball Court** - Basketball facilities
7. 🏃 **Ground Track Field** - Athletic tracks
8. 🏠 **Harbor** - Port facilities
9. 🌉 **Bridge** - Bridges and overpasses
10. 🚗 **Vehicle** - Ground vehicles and equipment

## 🎨 Output Features

- **Color-coded bounding boxes** for each class
- **Confidence scores** displayed on detections
- **Class labels** for easy identification
- **Detection summary** in console output
- **High-quality annotated images**

## 📊 Model Performance

- **Overall mAP@0.5**: 98.0%
- **Vehicle Detection F1**: 79.2%
- **Inference Speed**: ~10ms per image
- **Model Size**: 6.3MB (YOLOv8n)

## 🔧 Advanced Usage

### API Integration

Once deployed, you can use the model via Replicate's API:

```python
import replicate

# Run the model
output = replicate.run(
    "yourusername/satellite-equipment-detection:latest",
    input={
        "image": open("satellite_image.jpg", "rb"),
        "confidence_threshold": 0.3,
        "draw_labels": True
    }
)

# Download the result
with open("result.jpg", "wb") as f:
    f.write(output.read())
```

### Webhook Integration

```python
# For asynchronous processing
prediction = replicate.predictions.create(
    version="yourusername/satellite-equipment-detection:latest",
    input={"image": "https://example.com/satellite.jpg"},
    webhook="https://yourapp.com/webhook"
)
```

## 💡 Use Cases

### Commercial Applications
- **Competitive Intelligence**: Detect competitor equipment in logistics yards
- **Fleet Monitoring**: Track vehicle deployments across facilities
- **Market Research**: Analyze equipment distribution patterns
- **Lead Generation**: Identify potential customers with aging fleets

### Demo Scenarios
- **OrangeEV Pitch**: Show hostler detection capabilities in logistics facilities
- **Infrastructure Analysis**: Count vehicles in parking lots and depots
- **Asset Monitoring**: Track equipment across multiple locations
- **Proof of Concept**: Demonstrate AI viability for satellite-based detection

## 🎭 Testing Tips

### Good Test Images
- **High resolution satellite images** (Google Earth screenshots work well)
- **Clear weather conditions** for best detection
- **Multiple object types** to showcase capabilities
- **Logistics facilities** for vehicle detection demos

### Parameter Tuning
- **Lower confidence** (0.1-0.2) to catch more objects
- **Higher confidence** (0.5-0.8) for only clear detections
- **Adjust IoU threshold** to control overlapping detections

## 🔍 Troubleshooting

### Common Issues

1. **"No objects detected"**
   - Lower the confidence threshold to 0.1-0.2
   - Ensure image contains objects from the 10 trained classes
   - Check image quality and resolution

2. **Too many false positives**
   - Increase confidence threshold to 0.4-0.6
   - Increase IoU threshold to 0.6-0.7

3. **Model loading errors**
   - Verify HuggingFace model repository is accessible
   - Check internet connection for model download

## 📈 Model Limitations

- **Best performance** on structured objects (sports facilities, track fields)
- **Vehicle detection** may have some false positives (49% over-detection rate)
- **Small objects** may be missed in low-resolution images
- **Training limited** to 10 specific classes

## 🎯 Next Steps

1. **Deploy to Replicate** using the provided configuration
2. **Test with your satellite images** using the web UI
3. **Share the Replicate URL** for easy demonstrations
4. **Integrate via API** for production applications
5. **Collect feedback** for future model improvements

## 📞 Support

For deployment issues or questions:
- Check Replicate documentation: [https://replicate.com/docs](https://replicate.com/docs)
- Verify Cog configuration in `cog.yaml`
- Test prediction script locally first

Ready to deploy? Run `cog push r8.im/yourusername/satellite-equipment-detection` to get started!