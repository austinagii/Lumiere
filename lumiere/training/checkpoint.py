import io
from enum import StrEnum, auto

import torch


class CheckpointType(StrEnum):
    EPOCH = auto()
    BEST = auto()
    FINAL = auto()


class Checkpoint:
    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __bytes__(self):
        buffer = io.BytesIO()
        torch.save(self._data, buffer)
        return buffer.getvalue()

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
