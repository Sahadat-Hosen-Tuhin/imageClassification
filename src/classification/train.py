import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model

DATA_DIR = "data/classification"
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders(DATA_DIR)

model = get_model()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "models/resnet18_classification.pth")
print("✅ Model Training Completed & Saved")
