import io
from enum import StrEnum, auto
from typing import Protocol

import torch


class CheckpointType(StrEnum):
    """Represents the type of checkpoint.

    EPOCH: Represents a checkpoint at the end of an epoch.
    BEST: Represents the best checkpoint based on a metric.
    FINAL: Represents the final checkpoint after training is complete.
    """

    EPOCH = auto()
    BEST = auto()
    FINAL = auto()


class Checkpoint(dict):
    """A container of arbitrary key-value pairs representing a model's state.

    A checkpoint is a dictionary of key-value pairs that can be saved and loaded.
    The key-value pairs can be arbitrary, but they are typically used to store the
    state of a model, optimizer, and other training parameters.
    """

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
        """Construct a checkpoint from a bytes object.

        The `bytes` argument is expected to be byte data output from calling `bytes`
        on an existing checkpoint.

        By default, the checkpoint is loaded onto the CPU but can be loaded onto any
        desired device using the `device` argument.

        Example:
            >>> checkpoint = Checkpoint(epoch=7, global_step=128, eval_loss=0.0103243)
            >>> bytes_data = bytes(checkpoint)
            >>> checkpoint_from_bytes = Checkpoint.from_bytes(bytes_data)
            >>> checkpoint_from_bytes == checkpoint
            True

        Args:
            bytes: The bytes object to construct the checkpoint from.
            device: The device to load the checkpoint onto.

        Returns:
            A new checkpoint object.
        """
        return cls(**torch.load(io.BytesIO(bytes), map_location=device))


class Checkpointable(Protocol):
    def to_checkpoint(self) -> Checkpoint: ...
