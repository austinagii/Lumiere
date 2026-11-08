import torch


def get_device() -> str:
    device = "cpu"

    if torch.backends.mps.is_available():
        device = "mps"

    if torch.cuda.is_available():
        device = "cuda"

    return device
