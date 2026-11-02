import cv2
import os

# =========================
# DIRECTORIES
# =========================
INPUT_DIR = "data/raw/images"
OUTPUT_DIR = "data/processed/enhanced"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# IMAGE ENHANCEMENT (CLAHE)
# =========================
for img_name in os.listdir(INPUT_DIR):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # --- Convert to LAB color space ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # --- Apply CLAHE on L channel ---
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    cl = clahe.apply(l)

    # --- Merge channels back ---
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # --- Save enhanced image ---
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), enhanced)

print("✅ Image Enhancement Completed (CLAHE-based, color preserved)")
