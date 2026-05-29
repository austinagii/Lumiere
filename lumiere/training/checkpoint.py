import io
import json
import logging
import time
from enum import StrEnum, auto

import torch

from lumiere.utils import randomizer


CHECKPOINT_INDEX_PATH_TEMPLATE = "runs/{run_name}/artifacts/checkpoints/index.json"
CHECKPOINT_PATH_TEMPLATE = "runs/{run_name}/artifacts/checkpoints/{checkpoint_id}.ckpt"

logger = logging.getLogger(__name__)


class CheckpointTag(StrEnum):
    """A label for easily identifying and tracking checkpoints.

    EPOCH: Identifies a checkpoint generated at the end of an epoch.
    BEST: Identifies the checkpont with the best loss.
    LATEST: Identifies the most recent checkpoint for a run.
    FINAL: Identifies the final checkpoint for a completed training run.
    """

    EPOCH = auto()
    BEST = auto()
    LATEST = auto()
    FINAL = auto()


class Checkpoint(dict):
    """A container of arbitrary key-value pairs representing a model's state.

    A checkpoint is a dictionary of key-value pairs that can be saved and loaded.
    The key-value pairs can be arbitrary, but they are typically used to store the
    state of a model, optimizer, scheduler, and training metrics like epoch number
    and loss values.

    Supports attribute-style access (e.g., `checkpoint.epoch`) in addition to
    dictionary-style access (e.g., `checkpoint["epoch"]`).

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

    def __init__(
        self,
        epoch: int,
        loss: float,
        id: str | None = None,
        created_at: int | None = None,
        **kwargs,
    ):
        """Initialize a checkpoint with the specified key-value pairs.

        Args:
            **kwargs: Key-value pairs to store in the checkpoint.
        """
        super().__init__()

        self.id = id if id else randomizer.random_id()
        self.epoch = epoch
        self.loss = loss
        self.created_at = created_at if created_at else time.time_ns()
        self.update(kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def meta(self):
        return {
            "id": self.id,
            "epoch": self.epoch,
            "loss": self.loss,
            "created_at": self.created_at,
        }

    def to_bytes(self):
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


class CheckpointRepository:
    def __init__(self, client: StorageClient):
        self.client = client

    def insert(self, run_name: str, checkpoint: Checkpoint):
        """Save a checkpoint.

        Raises:
            StorageError: If an error occurred while attempting to save the checkpoint.

        """
        checkpoint_path = CHECKPOINT_PATH_TEMPLATE.format(
            run_name=run_name, checkpoint_id=checkpoint.id
        )
        checkpoint_bytes = checkpoint.to_bytes()
        logger.info("Saving checkpoint '{checkpoint.id}' for run '{run_name}'.")
        num_bytes_written = self.client.save(checkpoint_path, checkpoint_bytes)
        if num_bytes_written < len(checkpoint_bytes):
            raise RuntimeError(
                f"Failed to save all data for checkpoint: '{checkpoint.id}'."
            )

        self._index_checkpoint(run_name, checkpoint.meta())

    def _index_checkpoint(self, run_name: str, checkpoint: Checkpoint):
        checkpoint_dict = checkpoint.to_dict()
        checkpoint_index_path = CHECKPOINT_INDEX_PATH_TEMPLATE.format(run_name=run_name)
        if checkpoint_index_bytes := self.client.load(checkpoint_index_path):
            checkpoint_index = json.loads(checkpoint_index_bytes)
        else:
            checkpoint_index = {}

        epoch_tag = f"{CheckpointTag.EPOCH}:{checkpoint.epoch}"
        checkpoint_index[epoch_tag] = checkpoint_dict

        if latest_checkpoint := checkpoint_index.get(CheckpointTag.LATEST):
            if checkpoint.created_at > latest_checkpoint.created_at:
                checkpoint_index[CheckpointTag.LATEST] = checkpoint
        else:
            checkpoint_index[CheckpointTag.LATEST] = checkpoint

        if best_checkpoint := checkpoint_index.get(CheckpointTag.BEST):
            if checkpoint.loss < best_checkpoint.loss:
                checkpoint_index[CheckpointTag.BEST] = checkpoint
        else:
            checkpoint_index[CheckpointTag.BEST] = checkpoint

        # TODO: Clean up unereferenced checkpoints.

        checkpoint_index_bytes = json.dumps(checkpoint_index, indent=2)
        self.client.save(checkpoint_index_path, checkpoint_index_bytes)

    def get(
        self, run_name: str, checkpoint_tag: CheckpointTag, include_state: bool = True
    ) -> Checkpoint:
        pass
