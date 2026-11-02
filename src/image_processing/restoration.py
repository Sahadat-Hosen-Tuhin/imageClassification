import cv2
import os
import numpy as np

# =========================
# DIRECTORIES
# =========================
INPUT_DIR = "data/processed/denoised"
OUTPUT_DIR = "data/processed/restored"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# RESTORATION (SHARPENING)
# =========================
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

for img_name in os.listdir(INPUT_DIR):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # --- Apply sharpening filter ---
    restored = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), restored)

print("✅ Image Restoration Completed (Color-preserving sharpening)")
