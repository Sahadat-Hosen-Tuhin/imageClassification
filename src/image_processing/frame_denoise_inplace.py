import cv2
import os

FRAME_DIR = "data/processed/frames"

for img_name in os.listdir(FRAME_DIR):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(FRAME_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    denoised = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(img_path, denoised)

print("✅ Frame Denoising Done (in-place)")
