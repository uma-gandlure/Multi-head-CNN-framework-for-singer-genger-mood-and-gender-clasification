import matplotlib.pyplot as plt

losses = []
accuracies = []
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import AudioDataset
from model import MultiTaskModel
import numpy as np
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# MOOD LABEL MAP
mood_map = {
    "sad": 0,
    "happy": 1,
    "calm": 2,
    "energetic": 3
}

num_moods = len(mood_map)

dataset = AudioDataset(
    data_dir="data/mood_processed",
    label_file="data/mood_processed/labels.csv",
    label_map=mood_map,
    label_column="mood"
)

# Fix mood imbalance with weighted sampler
df = pd.read_csv("data/mood_processed/labels.csv")
labels = [mood_map[l] for l in df['mood'].str.lower()]
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[l] for l in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
print("Dataset size:", len(dataset))

# MODEL
model = MultiTaskModel(num_genres=8, num_moods=4)
model.to(device)

# Load previous best if exists
if os.path.exists("mood_model_best.pth"):
    model.load_state_dict(torch.load("mood_model_best.pth", map_location=device))
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
        mood_output = outputs["mood"]

        loss = loss_fn(mood_output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (torch.argmax(mood_output, dim=1) == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    losses.append(total_loss)
    accuracies.append(acc)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f} | Accuracy: {acc:.2f}%")

    # Scheduler step
    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mood_model_best.pth")
        print(f"  ✅ Best model saved! Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "mood_model.pth")
print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")