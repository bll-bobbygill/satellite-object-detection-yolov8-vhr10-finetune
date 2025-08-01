#!/usr/bin/env python3
"""Standalone script to evaluate the trained YOLOv8 model."""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yolov8_finetune.evaluate import main

if __name__ == "__main__":
    main()