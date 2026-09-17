import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, num_genres, num_moods):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.genre_head = nn.Linear(128, num_genres)
        self.mood_head = nn.Linear(128, num_moods)
        self.gender_head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return {
            "genre": self.genre_head(x),
            "mood": self.mood_head(x),
            "gender": self.gender_head(x)
        }