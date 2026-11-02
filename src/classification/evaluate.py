import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import sys, os
sys.path.append(os.path.abspath("."))

from src.classification.dataset import get_dataloaders
from src.classification.model import get_model

DATA_DIR = "data/classification"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_loader = get_dataloaders(DATA_DIR)

model = get_model()
model.load_state_dict(torch.load("models/resnet18_classification.pth"))
model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))