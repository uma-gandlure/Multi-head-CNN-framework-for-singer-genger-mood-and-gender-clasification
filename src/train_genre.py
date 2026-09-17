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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# FIXED label map — lowercase to match dataset.py
genre_map = {
    "electronic": 0,
    "experimental": 1,
    "folk": 2,
    "hip-hop": 3,
    "instrumental": 4,
    "international": 5,
    "pop": 6,
    "rock": 7
}

num_genres = len(genre_map)

dataset = AudioDataset(
    data_dir="data/genre_processed",
    label_file="data/genre_processed/labels.csv",
    label_map=genre_map,
    label_column="genre"
)

# -------------------------------
# WEIGHTED SAMPLER — fixes imbalance
# -------------------------------
df = pd.read_csv("data/genre_processed/labels.csv")
df['genre_lower'] = df['genre'].str.lower()
labels = [genre_map[l] for l in df['genre_lower']]

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

# -------------------------------
# MODEL
# -------------------------------
model = MultiTaskModel(num_genres=8, num_moods=4)
model.to(device)

import os
if os.path.exists("genre_model_best.pth"):
    model.load_state_dict(torch.load("genre_model_best.pth", map_location=device))
    print("✅ Loaded previous best model — continuing training!")

# -------------------------------
# LOSS + OPTIMIZER + SCHEDULER
# -------------------------------
loss_fn = nn.CrossEntropyLoss()
epochs = 50

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5,
)


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
        genre_output = outputs["genre"]

        loss = loss_fn(genre_output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (torch.argmax(genre_output, dim=1) == y).sum().item()
        total += y.size(0)

    scheduler.step(best_acc)

    acc = 100 * correct / total
    losses.append(total_loss)
    accuracies.append(acc)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f} | Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "genre_model_best.pth")
        print(f"  ✅ Best model saved! Accuracy: {acc:.2f}%")

# Save final
torch.save(model.state_dict(), "genre_model.pth")
print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")