import os
import numpy as np
import time

folder = "../data/gender_processed"

# Get the first .npy file automatically
files = [f for f in os.listdir(folder) if f.endswith(".npy")]

file = os.path.join(folder, files[0])

print("Testing:", file)

start = time.time()

x = np.load(file)

print("Shape:", x.shape)
print("Loaded in:", time.time() - start, "seconds")