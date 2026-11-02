import cv2
import os
import numpy as np

FRAME_DIR = "data/processed/frames"

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

for img_name in os.listdir(FRAME_DIR):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(FRAME_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    restored = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(img_path, restored)

print("✅ Frame Restoration Done (in-place)")
