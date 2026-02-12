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
    state of a model, optimizer, scheduler, and training metrics like epoch number
    and loss values.

    Supports attribute-style access (e.g., `checkpoint.epoch`) in addition to
    dictionary-style access (e.g., `checkpoint["epoch"]`).

    Args:
        **kwargs: Key-value pairs to store in the checkpoint.

    Example:
        ```python
        checkpoint = Checkpoint(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=5,
            loss=0.123
        )
        print(checkpoint.epoch)
        # Output: 5
        ```
    """

    def __init__(self, **kwargs):
        """Initialize a checkpoint with the specified key-value pairs.

        Args:
            **kwargs: Key-value pairs to store in the checkpoint.
        """
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
            ```python
            checkpoint = Checkpoint(epoch=7, global_step=128, eval_loss=0.0103243)
            bytes_data = bytes(checkpoint)
            checkpoint_from_bytes = Checkpoint.from_bytes(bytes_data)
            print(checkpoint_from_bytes == checkpoint)
            # Output: True
            ```

        Args:
            bytes: The bytes object to construct the checkpoint from.
            device: The device to load the checkpoint onto.

        Returns:
            A new checkpoint object.
        """
        return cls(**torch.load(io.BytesIO(bytes), map_location=device))


class Checkpointable(Protocol):
    """Protocol for objects that can be converted to checkpoints.

    Objects implementing this protocol can serialize their state into a `Checkpoint`
    for saving during training.
    """

    def to_checkpoint(self) -> Checkpoint:
        """Convert this object to a checkpoint.

        Returns:
            A `Checkpoint` containing the object's state.
        """
        ...
