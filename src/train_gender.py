import matplotlib.pyplot as plt

losses = []
accuracies = []
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import MultiTaskModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

gender_map = {
    "male": 0,
    "female": 1
}

dataset = AudioDataset(
    data_dir="data/gender_processed",
    label_file="data/gender_processed/labels.csv",
    label_map=gender_map,
    label_column="gender"
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset size:", len(dataset))

# MODEL
model = MultiTaskModel(num_genres=8, num_moods=4)
model.to(device)

# Load previous best if exists
if os.path.exists("gender_model_best.pth"):
    model.load_state_dict(torch.load("gender_model_best.pth", map_location=device))
    print("✅ Loaded previous best model — continuing training!")

# LOSS + OPTIMIZER + SCHEDULER
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

epochs = 50
best_acc = 0

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        gender_output = outputs["gender"]

        loss = loss_fn(gender_output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (torch.argmax(gender_output, dim=1) == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    losses.append(total_loss)
    accuracies.append(acc)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f} | Accuracy: {acc:.2f}%")

    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "gender_model_best.pth")
        print(f"  ✅ Best model saved! Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "gender_model.pth")
print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")