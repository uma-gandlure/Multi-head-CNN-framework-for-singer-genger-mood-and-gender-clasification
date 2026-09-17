import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/gender"
OUTPUT_DIR = "data/gender_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 22050       
CLIP_SECONDS = 5
CLIP_LENGTH = SR * CLIP_SECONDS


def extract_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db


records = []

for gender in ["male", "female"]:

    folder = os.path.join(DATA_DIR, gender)

    for file in tqdm(os.listdir(folder)):

        path = os.path.join(folder, file)

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
                    "gender": gender
                })

        except:
            print("Error:", path)


labels = pd.DataFrame(records)

labels.to_csv(
    os.path.join(OUTPUT_DIR, "labels.csv"),
    index=False
)

print("Gender preprocessing finished.")  