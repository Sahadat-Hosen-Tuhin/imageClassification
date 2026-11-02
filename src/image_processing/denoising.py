import cv2
import os

INPUT_DIR = "data/processed/enhanced"
OUTPUT_DIR = "data/processed/denoised"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    denoised = cv2.GaussianBlur(img, (5, 5), 0)

    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, denoised)

print("✅ Image Denoising Completed")

