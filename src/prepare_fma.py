import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# PATHS (EDIT IF NEEDED)
FMA_AUDIO_PATH = "data/genre/train/fma_small"
FMA_METADATA_PATH = "data/genre/train/fma_metadata/tracks.csv"
OUTPUT_PATH = "data/genre"

# Load metadata
tracks = pd.read_csv(FMA_METADATA_PATH, header=[0,1], index_col=0)

# Keep only small subset
tracks = tracks[tracks[('set', 'subset')] == 'small']

# Get top-level genre
tracks = tracks.dropna(subset=[('track', 'genre_top')])

# Prepare list
samples = []

for track_id, row in tracks.iterrows():
    genre = row[('track', 'genre_top')]
    folder = f"{track_id:06d}"[:3]
    filename = f"{track_id:06d}.mp3"

    file_path = os.path.join(FMA_AUDIO_PATH, folder, filename)

    if os.path.exists(file_path):
        samples.append((file_path, genre))

# Split data
train, temp = train_test_split(samples, test_size=0.3, stratify=[s[1] for s in samples], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=[s[1] for s in temp], random_state=42)

def copy_files(data, split):
    for path, genre in data:
        dest_dir = os.path.join(OUTPUT_PATH, split, genre)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, dest_dir)

copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

print("Dataset prepared successfully.")