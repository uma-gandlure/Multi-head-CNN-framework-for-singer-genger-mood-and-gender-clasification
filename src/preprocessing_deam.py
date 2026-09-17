import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ================================
# paths
# ================================

AUDIO_DIR = "data/mood/DEAM/DEAM_audio/MEMD_audio"
ANNOTATION_DIR = "data/mood/DEAM/DEAM_Annotations"
OUTPUT_DIR = "dataset/deam_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# load annotations
# ================================

valence = pd.read_csv(os.path.join(ANNOTATION_DIR, "valence.csv"))
arousal = pd.read_csv(os.path.join(ANNOTATION_DIR, "arousal.csv"))

# merge annotation tables
df = pd.merge(valence, arousal, on="song_id")




# ================================
# convert valence + arousal → mood
# ================================

def mood_label(v, a):
    if v >= 0 and a >= 0:  # DEAM uses negative values too
        return "happy"
    elif v < 0 and a >= 0:
        return "energetic"
    elif v >= 0 and a < 0:
        return "calm"
    else:
        return "sad"

# Compute mean valence/arousal across ALL time samples per song
valence_cols = [col for col in df.columns if '_x' in col]  # All valence samples
arousal_cols = [col for col in df.columns if '_y' in col]  # All arousal samples

df['valence_mean'] = df[valence_cols].mean(axis=1, skipna=True)
df['arousal_mean'] = df[arousal_cols].mean(axis=1, skipna=True)

# Label moods using averages
df["mood"] = df.apply(
    lambda x: mood_label(x["valence_mean"], x["arousal_mean"]), 
    axis=1
)

print("Mood distribution:", df['mood'].value_counts())



# ================================
# audio parameters
# ================================

SR = 22050
CLIP_SECONDS = 5
CLIP_LENGTH = SR * CLIP_SECONDS


# ================================
# mel spectrogram extraction
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
# process songs
# ================================

data_records = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    song_id = str(int(row["song_id"]))   # FIX HERE
    mood = row["mood"]

    audio_path = os.path.join(AUDIO_DIR, song_id + ".mp3")

    if not os.path.exists(audio_path):
        continue

    try:
        y, sr = librosa.load(audio_path, sr=SR)

        # split into clips
        for i in range(0, len(y), CLIP_LENGTH):

            clip = y[i:i+CLIP_LENGTH]

            if len(clip) < CLIP_LENGTH:
                continue

            mel = extract_mel(clip)

            file_name = f"{song_id}_{i}.npy"
            save_path = os.path.join(OUTPUT_DIR, file_name)

            np.save(save_path, mel)

            data_records.append({
                "file": file_name,
                "mood": mood
            })

    except Exception as e:
        print("Error:", audio_path)


# ================================
# save label file
# ================================

label_df = pd.DataFrame(data_records)

label_df.to_csv(
    os.path.join(OUTPUT_DIR, "labels.csv"),
    index=False
)

print("Preprocessing finished.")
df = pd.read_csv("dataset/deam_processed/labels.csv")
print(len(df))