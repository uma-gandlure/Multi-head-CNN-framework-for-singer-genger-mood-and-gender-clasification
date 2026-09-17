import torch
import librosa
import numpy as np
from model import MultiTaskModel

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LABEL MAPS — must match train maps exactly
genre_labels = [
    "electronic", "experimental", "folk", "hip-hop",
    "instrumental", "international", "pop", "rock"
]

mood_labels = ["sad", "happy", "calm", "energetic"]

gender_labels = ["male", "female"]

# LOAD MODELS
genre_model = MultiTaskModel(num_genres=8, num_moods=4)
mood_model = MultiTaskModel(num_genres=8, num_moods=4)
gender_model = MultiTaskModel(num_genres=8, num_moods=4)

genre_model.load_state_dict(torch.load("models/genre_model_best.pth", map_location=device))
mood_model.load_state_dict(torch.load("models/mood_model_best.pth", map_location=device))
gender_model.load_state_dict(torch.load("models/gender_model_best.pth", map_location=device))

genre_model.to(device).eval()
mood_model.to(device).eval()
gender_model.to(device).eval()

# PREPROCESS — 5 sec clip + normalize like training
def extract_mel(file_path, clip_duration=5, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)

    clip_samples = clip_duration * sr

    # Take clip from middle of song
    mid = len(y) // 2
    start = max(0, mid - clip_samples // 2)
    end = start + clip_samples

    y_clip = y[start:end]

    # Pad if too short
    if len(y_clip) < clip_samples:
        y_clip = np.pad(y_clip, (0, clip_samples - len(y_clip)))

    mel = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    # Normalize — must match dataset.py
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db

# PREDICT
def predict(file_path):
    mel = extract_mel(file_path)

    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        genre_out = genre_model(x)["genre"]
        mood_out = mood_model(x)["mood"]
        gender_out = gender_model(x)["gender"]

    genre = genre_labels[torch.argmax(genre_out).item()]
    mood = mood_labels[torch.argmax(mood_out).item()]
    gender = gender_labels[torch.argmax(gender_out).item()]

    print("\n===== RESULT =====")
    print("Genre  :", genre)
    print("Mood   :", mood)
    print("Gender :", gender)

# RUN
if __name__ == "__main__":
    file_path = input("Enter audio file path: ")
    predict(file_path)