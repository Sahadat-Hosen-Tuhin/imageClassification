import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Allow absolute imports
sys.path.append(os.path.abspath("."))

from src.classification.model import get_model


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/resnet18_classification.pth"
TEST_DIR = "data/test_images"
CONFIDENCE_THRESHOLD = 0.70   # 70%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["effective", "ineffective"]


# =========================
# PREPROCESSING FUNCTION
# (Same logic as training)
# =========================
def preprocess_image(img_bgr):
    """
    Apply enhancement, denoising and restoration
    """

    # --- Enhancement (CLAHE) ---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # --- Denoising ---
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # --- Restoration (Sharpen) ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    restored = cv2.filter2D(denoised, -1, kernel)

    return restored


# =========================
# PYTORCH TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================
# LOAD MODEL
# =========================
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# =========================
# PREDICTION
# =========================
print("\n🔮 Prediction Results (with confidence)\n")

with torch.no_grad():
    for img_name in os.listdir(TEST_DIR):

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(TEST_DIR, img_name)

        # Load image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # 🔥 Apply SAME preprocessing as training
        processed = preprocess_image(img_bgr)

        # Convert BGR → RGB → PIL
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Torch tensor
        image_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Model forward pass
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)
        confidence_value = confidence.item()

        # Threshold logic
        if confidence_value < CONFIDENCE_THRESHOLD:
            label = "Uncertain"
        else:
            label = class_names[pred.item()]

        print(
            f"🖼️ {img_name:25s} ➜ "
            f"🧠 {label:10s} ({confidence_value * 100:.2f}%)"
        )

print("\n✅ Prediction Completed")
