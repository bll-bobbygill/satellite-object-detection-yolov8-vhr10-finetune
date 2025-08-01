# 🚀 Manual Deployment to Replicate

Since the automated script had issues with the cog command-line tool, here are the manual deployment instructions:

## Step 1: Install Cog Command-Line Tool

Try one of these methods:

### Method A: Using curl (recommended)
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

### Method B: Using conda
```bash
conda install -c conda-forge cog
```

### Method C: Direct download
```bash
# Download from: https://github.com/replicate/cog/releases
# Choose the appropriate binary for your system
```

## Step 2: Create Replicate Account

1. Go to [https://replicate.com](https://replicate.com)
2. Sign up for a free account
3. Note your username

## Step 3: Login to Replicate

```bash
cog login
```

## Step 4: Deploy Your Model

```bash
# Replace 'yourusername' with your actual Replicate username
cog push r8.im/yourusername/satellite-equipment-detection
```

## Step 5: Test Your Deployed Model

Once deployed, you can:

1. **Visit your model page**: `https://replicate.com/yourusername/satellite-equipment-detection`
2. **Upload a satellite image** (try a Google Earth screenshot)
3. **Adjust settings**:
   - Confidence threshold: 0.2-0.4 (lower = more detections)
   - IoU threshold: 0.45 (default is fine)
   - Draw labels: Yes
   - Draw confidence: Yes
4. **Click "Run"** and watch the magic happen!

## Alternative: Use Replicate Web Interface

If command-line deployment doesn't work, you can also:

1. Create a new model on replicate.com
2. Upload the files manually through their web interface
3. Use the provided `cog.yaml` and `predict.py` files

## Expected Results

Your deployed model will:
- ✅ Automatically download from HuggingFace Hub
- ✅ Detect 10 types of objects in satellite imagery
- ✅ Draw color-coded bounding boxes
- ✅ Show confidence scores and labels
- ✅ Provide detection summaries
- ✅ Return high-quality annotated images

## Perfect for OrangeEV Demo

The web interface is perfect for:
- Client demonstrations
- Testing different satellite images
- Adjusting detection sensitivity
- Downloading results for presentations
- API integration development

## Need Help?

If you encounter issues:
1. Check Replicate's documentation: https://replicate.com/docs
2. Verify all files are present (cog.yaml, predict.py)
3. Test locally first with `test_replicate_local.py`
4. Make sure your HuggingFace model is publicly accessible

Your model is ready to deploy - all the hard work is done! 🎉