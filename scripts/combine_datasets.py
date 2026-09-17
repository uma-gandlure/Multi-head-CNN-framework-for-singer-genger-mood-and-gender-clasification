import os

print("Current Directory:", os.getcwd())
import pandas as pd

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "multitask")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

# ---------------- Genre ----------------
genre_csv = os.path.join(DATA_DIR, "genre_processed", "labels.csv")
genre_df = pd.read_csv(genre_csv)

for _, row in genre_df.iterrows():
    rows.append({
        "file_path": os.path.join(DATA_DIR, "genre_processed", row["file"]),
        "genre": row["genre"],
        "mood": -1,
        "gender": -1
    })

# ---------------- Mood ----------------
mood_csv = os.path.join(DATA_DIR, "mood_processed", "labels.csv")
mood_df = pd.read_csv(mood_csv)

for _, row in mood_df.iterrows():
    rows.append({
        "file_path": os.path.join(DATA_DIR, "mood_processed", row["file"]),
        "genre": -1,
        "mood": row["mood"],
        "gender": -1
    })

# ---------------- Gender ----------------
gender_csv = os.path.join(DATA_DIR, "gender_processed", "labels.csv")
gender_df = pd.read_csv(gender_csv)

for _, row in gender_df.iterrows():
    rows.append({
        "file_path": os.path.join(DATA_DIR, "gender_processed", row["file"]),
        "genre": -1,
        "mood": -1,
        "gender": row["gender"]
    })

combined_df = pd.DataFrame(rows)

combined_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done! Combined dataset created.")
print(f"Total samples: {len(combined_df)}")
print(f"Saved to: {OUTPUT_CSV}")