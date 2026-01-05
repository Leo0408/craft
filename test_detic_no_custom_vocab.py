"""
Test DETIC detector without custom vocabulary to diagnose the issue
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np

# Load the detector
from perception.detic_clip_detector import DeticClipDetector

# Load your test image
# Assuming you have the image from demo2.ipynb
print("🔍 Testing DETIC without custom vocabulary...")
print("=" * 60)

# Initialize detector
detector = DeticClipDetector(
    detic_threshold=0.3,
    clip_threshold=0.25,
    enable_tracking=False
)

# Load test image (you'll need to provide the image path)
# For now, this is a template - update with your actual image loading code
print("\n⚠️  Please update this script with your image loading code")
print("   Then run it to test DETIC with default vocabulary")

# Example usage (uncomment and modify):
# rgb_pil = Image.open("path/to/your/image.jpg")
# object_list = ["coffee machine", "purple cup", "blue cup with handle", "table on the left of sink"]
# 
# # Test WITHOUT custom vocabulary
# print("\n1️⃣  Testing WITHOUT custom vocabulary (debug mode):")
# detections = detector.detect_objects(
#     rgb_pil, 
#     object_list, 
#     use_custom_vocab=False,  # Disable custom vocabulary
#     debug_mode=True          # Enable debug output
# )
# print(f"\n✅ Found {len(detections)} detections without custom vocab")
# 
# # Test WITH custom vocabulary
# print("\n2️⃣  Testing WITH custom vocabulary (debug mode):")
# detections2 = detector.detect_objects(
#     rgb_pil, 
#     object_list, 
#     use_custom_vocab=True,   # Enable custom vocabulary
#     debug_mode=True          # Enable debug output
# )
# print(f"\n✅ Found {len(detections2)} detections with custom vocab")

