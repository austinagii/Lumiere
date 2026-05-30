import copy
import io
import itertools
import json
import logging
import time
from enum import StrEnum, auto

import torch

from lumiere.persistence.clients import StorageClient
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


class Checkpoint:
    """A container of arbitrary key-value pairs representing a model's state.

    A checkpoint is a dictionary of key-value pairs that can be saved and loaded.
    The key-value pairs can be arbitrary, but they are typically used to store the
    state of a model, optimizer, scheduler, and training metrics like epoch number
    and loss values.

    Supports attribute-style access (e.g., `checkpoint.epoch`) in addition to
    dictionary-style access (e.g., `checkpoint["epoch"]`).

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
        self.__dict__["_meta"] = {
            "id": id if id else randomizer.random_id(),
            "epoch": epoch,
            "loss": loss,
            "created_at": created_at if created_at else time.time_ns(),
        }
        self.__dict__["_state"] = kwargs

    def __getattr__(self, name):
        if name in self.__dict__["_meta"]:
            return self.__dict__["_meta"][name]
        if name in self.__dict__["_state"]:
            return self.__dict__["_state"][name]
        else:
            raise AttributeError(
                f"Attribute '{name}' not defined for type '{type(self).__name__}"
            )

    def __setattr__(self, name, value):
        self.__dict__["_state"][name] = value

    def meta(self):
        return copy.copy(self._meta)

    def to_bytes(self):
        buffer = io.BytesIO()
        torch.save(
            dict(itertools.chain(self._meta.items(), self._state.items())),
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, bytes: bytes):
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
        return cls(**torch.load(io.BytesIO(bytes)))


class CheckpointEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Checkpoint):
            return {
                "__type__": "Checkpoint",
                "id": obj.id,
                "epoch": obj.epoch,
                "loss": obj.loss,
                "created_at": obj.created_at,
            }
        else:
            return super().default(obj)


def decode_checkpoint(d):
    if "__type__" in d and d["__type__"] == "Checkpoint":
        return Checkpoint(
            id=d["id"], epoch=d["epoch"], loss=d["loss"], created_at=d["created_at"]
        )
    return d


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

        self._index_checkpoint(run_name, checkpoint)

    def _index_checkpoint(self, run_name: str, checkpoint: Checkpoint):
        checkpoint_index_path = CHECKPOINT_INDEX_PATH_TEMPLATE.format(run_name=run_name)
        if checkpoint_index_bytes := self.client.load(checkpoint_index_path):
            checkpoint_index = json.loads(
                checkpoint_index_bytes, object_hook=decode_checkpoint
            )
        else:
            checkpoint_index = {}

        epoch_tag = f"{CheckpointTag.EPOCH}:{checkpoint.epoch}"
        checkpoint_index[epoch_tag] = checkpoint

        # TODO: Use epoch instead of datetime to determine latest checkpoint.
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

        checkpoint_index_json = json.dumps(
            checkpoint_index, cls=CheckpointEncoder, indent=2
        )
        checkpoint_index_bytes = bytes(checkpoint_index_json, "utf-8")
        self.client.save(checkpoint_index_path, checkpoint_index_bytes, overwrite=True)

    def get(
        self, run_name: str, checkpoint_tag: CheckpointTag, include_state: bool = True
    ) -> Checkpoint:
        raise NotImplementedError()
