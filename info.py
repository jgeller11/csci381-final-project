import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
# DEVICE = "cpu"
print(f"Using {DEVICE} device")

seed = 1234567896
torch.manual_seed(seed)
print(f"Using random seed: {seed}")