import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from dataset import AudioDataset
from model import MultiTaskModel

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device :", device)

# =====================================================
# LABEL MAPS
# =====================================================

genre_map = {
    "electronic":0,
    "experimental":1,
    "folk":2,
    "hip-hop":3,
    "instrumental":4,
    "international":5,
    "pop":6,
    "rock":7
}

mood_map = {
    "sad":0,
    "happy":1,
    "calm":2,
    "energetic":3
}

gender_map = {
    "male":0,
    "female":1
}

# =====================================================
# DATASETS
# =====================================================

genre_dataset = AudioDataset(
    data_dir=os.path.join(DATA_DIR,"genre_processed"),
    label_file=os.path.join(DATA_DIR,"genre_processed","labels.csv"),
    label_map=genre_map,
    label_column="genre"
)

mood_dataset = AudioDataset(
    data_dir=os.path.join(DATA_DIR,"mood_processed"),
    label_file=os.path.join(DATA_DIR,"mood_processed","labels.csv"),
    label_map=mood_map,
    label_column="mood"
)

gender_dataset = AudioDataset(
    data_dir=os.path.join(DATA_DIR,"gender_processed"),
    label_file=os.path.join(DATA_DIR,"gender_processed","labels.csv"),
    label_map=gender_map,
    label_column="gender"
)

genre_loader = DataLoader(
    genre_dataset,
    batch_size=32,
    shuffle=False
)

mood_loader = DataLoader(
    mood_dataset,
    batch_size=32,
    shuffle=False
)

gender_loader = DataLoader(
    gender_dataset,
    batch_size=32,
    shuffle=False
)

print("Datasets Loaded Successfully")

# =====================================================
# LOAD MODELS
# =====================================================

genre_model = MultiTaskModel(8,4).to(device)
mood_model = MultiTaskModel(8,4).to(device)
gender_model = MultiTaskModel(8,4).to(device)

genre_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR,"genre_model_best.pth"),
        map_location=device
    )
)

mood_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR,"mood_model_best.pth"),
        map_location=device
    )
)

gender_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR,"gender_model_best.pth"),
        map_location=device
    )
)

genre_model.eval()
mood_model.eval()
gender_model.eval()

print("Models Loaded Successfully")

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate(model, loader, head_name, task_name, class_names):

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            pred = torch.argmax(outputs[head_name], dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n====================================")
    print(task_name)
    print("====================================")

    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8,6))

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    plt.title(f"{task_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"{task_name.lower()}_confusion.png"
        )
    )

    plt.close()

    return {
        "Task":task_name,
        "Accuracy":round(acc*100,2),
        "Precision":round(prec,4),
        "Recall":round(rec,4),
        "F1-score":round(f1,4)
    }
# =====================================================
# RUN EVALUATION
# =====================================================

results = []

results.append(
    evaluate(
        genre_model,
        genre_loader,
        "genre",
        "Genre",
        list(genre_map.keys())
    )
)

results.append(
    evaluate(
        mood_model,
        mood_loader,
        "mood",
        "Mood",
        list(mood_map.keys())
    )
)

results.append(
    evaluate(
        gender_model,
        gender_loader,
        "gender",
        "Gender",
        list(gender_map.keys())
    )
)

# =====================================================
# SAVE RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "evaluation_table.csv"
    ),
    index=False
)


print("\nEvaluation table saved successfully.")
# =====================================================
# SAVE METRICS REPORT
# =====================================================

with open(os.path.join(RESULT_DIR, "metrics.txt"), "w") as f:

    f.write("MODEL EVALUATION RESULTS\n")
    f.write("=" * 40 + "\n\n")

    for row in results:

        f.write(f"Task       : {row['Task']}\n")
        f.write(f"Accuracy   : {row['Accuracy']}%\n")
        f.write(f"Precision  : {row['Precision']}\n")
        f.write(f"Recall     : {row['Recall']}\n")
        f.write(f"F1-score   : {row['F1-score']}\n")
        f.write("-" * 40 + "\n")

print("\n==========================================")
print(" Evaluation Completed Successfully ")
print("==========================================")

print(f"Results folder : {RESULT_DIR}")

print("\nGenerated Files:")
print("✔ evaluation_table.csv")
print("✔ metrics.txt")
print("✔ genre_confusion.png")
print("✔ mood_confusion.png")
print("✔ gender_confusion.png")