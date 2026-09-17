import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================================
# paths
# ================================

DATA_DIR = "data/genre/train"
OUTPUT_DIR = "data/genre_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# audio parameters
# ================================

SR = 22050
CLIP_SECONDS = 5
CLIP_LENGTH = SR * CLIP_SECONDS


# ================================
# mel spectrogram
# ================================

def extract_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db


# ================================
# processing
# ================================

records = []

for genre in os.listdir(DATA_DIR):

    genre_folder = os.path.join(DATA_DIR, genre)

    if not os.path.isdir(genre_folder):
        continue

    for file in tqdm(os.listdir(genre_folder), desc=f"Processing {genre}"):

        path = os.path.join(genre_folder, file)

        try:

            y, sr = librosa.load(path, sr=SR)

            for i in range(0, len(y), CLIP_LENGTH):

                clip = y[i:i+CLIP_LENGTH]

                if len(clip) < CLIP_LENGTH:
                    continue

                mel = extract_mel(clip)

                name = f"{file}_{i}.npy"

                save_path = os.path.join(OUTPUT_DIR, name)

                np.save(save_path, mel)

                records.append({
                    "file": name,
                    "genre": genre
                })

        except:
            print("Error:", path)


# ================================
# save labels
# ================================

labels = pd.DataFrame(records)

labels.to_csv(
    os.path.join(OUTPUT_DIR, "labels.csv"),
    index=False
)

print("Genre preprocessing finished.")