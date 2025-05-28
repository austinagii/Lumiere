import torch

def get_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device