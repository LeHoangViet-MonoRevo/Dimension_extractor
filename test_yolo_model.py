import glob
import random

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the model
model = YOLO("./section_detection_best.pt")
# model = YOLO("../best_shap_detection.pt")

# Image path
img_path = "84b1b33011a7b22d219aaf627f41868a_2.jpg"

# Run inference
results = model(img_path)

# Visualize result
for result in results:
    # Get a NumPy image with bounding boxes drawn
    img_with_boxes = result.plot()

    # Convert BGR (used by OpenCV) to RGB (used by matplotlib)
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Show the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("YOLOv8 Detection Result")
    plt.show()
