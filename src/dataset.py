import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir, label_file, label_map, label_column):

        self.data_dir = data_dir
        self.df = pd.read_csv(label_file)
        self.label_map = label_map
        self.label_column = label_column
        print(set(self.df[self.label_column].str.lower()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        x = np.load(os.path.join(self.data_dir, row["file"]))

        x = (x - x.mean()) / (x.std() + 1e-6)

        x = torch.tensor(x).unsqueeze(0).float()

        label_str = str(row[self.label_column]).strip().lower()
        label = self.label_map[label_str]

        return x, label
   