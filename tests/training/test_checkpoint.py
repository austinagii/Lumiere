import pytest
import torch

from lumiere.training.checkpoint import Checkpoint


@pytest.fixture
def checkpoint() -> Checkpoint:
    return Checkpoint(
        epoch=10, loss=1.59e-5, random_state=42, model_state=torch.tensor([1, 2, 3])
    )


class TestCheckpoint:
    def test_checkpoint_can_be_instantiated_from_arbitrary_key_value_pairs(self):
        pass

    def test_checkpoint_data_can_be_accessed_using_subscript_notation(self, checkpoint: Checkpoint):
        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 1.59e-5
        assert checkpoint["random_state"] == 42 
        assert torch.equal(checkpoint["model_state"], torch.tensor([1, 2, 3]))

    def test_checkpoint_data_can_be_accessed_using_dot_notation(self, checkpoint: Checkpoint):
        assert checkpoint.epoch == 10
        assert checkpoint.loss == 1.59e-5
        assert checkpoint.random_state == 42 
        assert torch.equal(checkpoint.model_state, torch.tensor([1, 2, 3]))

    def test_checkpoint_can_be_converted_to_bytes(self, checkpoint: Checkpoint):
        checkpoint_bytes = checkpoint.to_bytes()
        loaded_checkpoint = Checkpoint.from_bytes(checkpoint_bytes)

        assert isinstance(loaded_checkpoint, Checkpoint)
        assert loaded_checkpoint.epoch == checkpoint.epoch 
        assert loaded_checkpoint.loss == checkpoint.loss 
        assert loaded_checkpoint.random_state == checkpoint.random_state
        assert torch.equal(loaded_checkpoint.model_state, checkpoint.model_state)
