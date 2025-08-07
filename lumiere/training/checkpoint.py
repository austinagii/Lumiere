import io
from enum import StrEnum, auto

import torch


class CheckpointType(StrEnum):
    EPOCH = auto()
    BEST = auto()
    FINAL = auto()


class Checkpoint(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __bytes__(self):
        buffer = io.BytesIO()
        torch.save(dict(self), buffer)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, bytes: bytes, device: torch.device = torch.device("cpu")):
        return cls(**torch.load(io.BytesIO(bytes), map_location=device))
