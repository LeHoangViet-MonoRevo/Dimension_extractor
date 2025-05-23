from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import random

# Load YOLO models
section_detector_path = "section_detection_best.pt"
dimension_line_detector_path = "dimension_line_detector.pt"
section_detector = YOLO(section_detector_path)
dimension_line_detector = YOLO(dimension_line_detector_path)

# Load image
img_dirs = glob.glob("../image_company/*/*")
# image_path = "../image_company/113/2d7573f20f7cb2b911135543119fc7d7_0.jpg"
image_path = random.choice(img_dirs)
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for matplotlib

# Run section detection
section_results = section_detector(image)[0]

# Prepare to store dimension line detections
dimension_line_detections = []

# Process each detected section
for box in section_results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    section_crop = image[y1:y2, x1:x2]

    # Run dimension line detection on cropped section
    dim_results = dimension_line_detector(section_crop)[0]
    
    # Optional: Adjust the coordinates back to original image space
    if dim_results.boxes is not None:
        for dim_box in dim_results.boxes.xyxy:
            dx1, dy1, dx2, dy2 = map(int, dim_box)
            # Adjust to original image coordinates
            abs_box = [dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1]
            dimension_line_detections.append(abs_box)

# Visualize results using matplotlib
fig, ax = plt.subplots(1, figsize=(12, 10))
ax.imshow(image_rgb)

# Draw each dimension line bounding box
for bbox in dimension_line_detections:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                             edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

ax.set_title("Detected Dimension Lines")
plt.axis("off")
plt.tight_layout()
plt.show()
