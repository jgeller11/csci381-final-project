import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
# DEVICE = "cpu"
print(f"Using {DEVICE} device")