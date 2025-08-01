#!/bin/bash

# Deployment script for YOLOv8 VHR-10 model to Replicate
# This script helps you deploy your satellite equipment detection model

echo "🚀 YOLOv8 VHR-10 Replicate Deployment Script"
echo "=============================================="

# Check if cog is installed
if ! command -v cog &> /dev/null; then
    echo "❌ Cog is not installed. Please install it first:"
    echo "   pip install cog"
    echo "   or visit: https://github.com/replicate/cog"
    exit 1
fi

echo "✅ Cog is installed"

# Check if logged in to Replicate
if ! cog --help | grep -q "login"; then
    echo "⚠️  Make sure you're logged in to Replicate:"
    echo "   cog login"
    echo ""
fi

# Get username
echo "📝 Enter your Replicate username:"
read -p "Username: " REPLICATE_USERNAME

if [ -z "$REPLICATE_USERNAME" ]; then
    echo "❌ Username is required"
    exit 1
fi

MODEL_NAME="satellite-equipment-detection"
FULL_MODEL_NAME="r8.im/$REPLICATE_USERNAME/$MODEL_NAME"

echo ""
echo "🎯 Model will be deployed as: $FULL_MODEL_NAME"
echo ""

# Ask for confirmation
echo "📋 Pre-deployment checklist:"
echo "   ✅ cog.yaml configured"
echo "   ✅ predict.py implemented" 
echo "   ✅ Model available on HuggingFace"
echo "   ✅ Dependencies specified"
echo ""

read -p "Ready to deploy? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Test locally first (optional)
echo ""
read -p "🧪 Test locally first? (Y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "🧪 Running local test..."
    
    if python test_replicate_local.py; then
        echo "✅ Local test passed!"
    else
        echo "❌ Local test failed!"
        read -p "Continue with deployment anyway? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Deployment cancelled"
            exit 1
        fi
    fi
fi

# Deploy to Replicate
echo ""
echo "🚀 Deploying to Replicate..."
echo "This may take several minutes..."

if cog push "$FULL_MODEL_NAME"; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📋 Your model is now available at:"
    echo "   https://replicate.com/$REPLICATE_USERNAME/$MODEL_NAME"
    echo ""
    echo "🧪 Test your model:"
    echo "   1. Visit the URL above"
    echo "   2. Upload a satellite image"
    echo "   3. Adjust confidence threshold as needed"
    echo "   4. Click 'Run' to see detections"
    echo ""
    echo "🔗 API Usage:"
    echo "   replicate run $FULL_MODEL_NAME image=@your_image.jpg"
    echo ""
    echo "💡 Perfect for OrangeEV demos and client presentations!"
    
else
    echo ""
    echo "❌ Deployment failed!"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "   1. Check your internet connection"
    echo "   2. Verify you're logged in: cog login"
    echo "   3. Make sure the model name doesn't already exist"
    echo "   4. Check cog.yaml and predict.py for errors"
    echo ""
    echo "📖 Need help? Check the Replicate documentation:"
    echo "   https://replicate.com/docs"
    
    exit 1
fi

# Optional: Open the model page
echo ""
read -p "🌐 Open model page in browser? (Y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    MODEL_URL="https://replicate.com/$REPLICATE_USERNAME/$MODEL_NAME"
    
    # Try to open in browser (works on macOS and most Linux distros)
    if command -v open &> /dev/null; then
        open "$MODEL_URL"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$MODEL_URL"
    else
        echo "🔗 Visit: $MODEL_URL"
    fi
fi

echo ""
echo "✨ Deployment complete! Happy detecting! 🛰️"