import torch
from model import MultiTaskModel

device = "cpu"

for name in [
    "../models/genre_model_best.pth",
    "../models/mood_model_best.pth",
    "../models/gender_model_best.pth"
]:
    print("\n", name)

    state = torch.load(name, map_location=device)

    print(type(state))

    if isinstance(state, dict):
        print("Number of keys:", len(state))
        print("First 10 keys:")
        for k in list(state.keys())[:10]:
            print("   ", k)