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

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 🔥 overwrite same image
    cv2.imwrite(img_path, enhanced)

print("✅ Frame Enhancement Done (in-place)")
 