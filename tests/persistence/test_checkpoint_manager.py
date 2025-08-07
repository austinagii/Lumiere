import random
import string
from pathlib import Path

import pytest

from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.errors import PersistenceError
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.persistence.checkpoint import Checkpoint, CheckpointType


@pytest.fixture
def checkpoint():
    return Checkpoint(epoch=1, prev_loss=0.95, best_loss=0.88)


@pytest.fixture
def checkpoint_bytes(checkpoint):
    return bytes(checkpoint)


@pytest.fixture
def run_name():
    vocab = string.ascii_letters + string.digits
    return "".join([random.choice(vocab) for i in range(10)])


@pytest.fixture
def mock_local_storage_client(mocker, tmp_path, checkpoint_bytes):
    local_storage_client = LocalStorageClient(base_dir=tmp_path)

    mocker.patch.object(
        local_storage_client,
        "exists",
        return_value=True,
        autospec=True,
    )
    mocker.patch.object(
        local_storage_client,
        "store",
        return_value=None,
        autospec=True,
    )
    mocker.patch.object(
        local_storage_client,
        "retrieve",
        return_value=checkpoint_bytes,
        autospec=True,
    )
    return local_storage_client


@pytest.fixture
def mock_remote_storage_client(mocker, checkpoint_bytes):
    remote_storage_client = RemoteStorageClient(
        blob_service_client=mocker.MagicMock(),
        container_name="test-container",
    )
    mocker.patch.object(
        remote_storage_client,
        "exists",
        return_value=True,
        autospec=True,
    )
    mocker.patch.object(
        remote_storage_client,
        "store",
        return_value=None,
        autospec=True,
    )
    mocker.patch.object(
        remote_storage_client,
        "retrieve",
        return_value=checkpoint_bytes,
        autospec=True,
    )
    return remote_storage_client


@pytest.fixture
def checkpoint_manager(mock_local_storage_client, mock_remote_storage_client):
    return CheckpointManager(
        remote_storage_client=mock_remote_storage_client,
        local_storage_client=mock_local_storage_client,
    )


class TestCheckpointManager:
    @pytest.mark.parametrize(
        "checkpoint_type",
        [CheckpointType.EPOCH, CheckpointType.BEST, CheckpointType.FINAL],
    )
    def test_save_checkpoint_correctly_saves_checkpoints(
        self, checkpoint_manager, run_name, checkpoint, checkpoint_type
    ):
        checkpoint_manager.save_checkpoint(run_name, checkpoint_type, checkpoint)

        expected_checkpoint_dir = Path(f"runs/{run_name}/checkpoints")
        match checkpoint_type:
            case CheckpointType.EPOCH:
                expected_checkpoint_path = expected_checkpoint_dir / "epoch_0001.pth"
            case CheckpointType.BEST:
                expected_checkpoint_path = expected_checkpoint_dir / "best.pth"
            case CheckpointType.FINAL:
                expected_checkpoint_path = expected_checkpoint_dir / "final.pth"

        # Check that the manager called the local storage client.
        checkpoint_manager.local_storage_client.store.assert_called_once_with(
            expected_checkpoint_path, bytes(checkpoint)
        )

        # Check that the manager called the remote storage client.
        checkpoint_manager.remote_storage_client.store.assert_called_once_with(
            expected_checkpoint_path, bytes(checkpoint)
        )

    @pytest.mark.parametrize(
        "checkpoint_name",
        ["epoch:0001", "best", "final"],
    )
    def test_load_checkpoint_loads_local_checkpoints_if_they_exist(
        self, checkpoint_manager, run_name, checkpoint_name
    ):
        expected_checkpoint_dir = Path(f"runs/{run_name}/checkpoints")
        match checkpoint_name:
            case "epoch:0001":
                expected_checkpoint_path = expected_checkpoint_dir / "epoch_0001.pth"
            case "best":
                expected_checkpoint_path = expected_checkpoint_dir / "best.pth"
            case "final":
                expected_checkpoint_path = expected_checkpoint_dir / "final.pth"

        checkpoint_manager.load_checkpoint(run_name, checkpoint_name)

        checkpoint_manager.local_storage_client.retrieve.assert_called_once_with(
            expected_checkpoint_path
        )

        checkpoint_manager.remote_storage_client.retrieve.assert_not_called()

    def test_load_checkpoint_loads_remote_checkpoints_if_local_does_not_exist(
        self, checkpoint_manager, run_name
    ):
        checkpoint_name = "epoch:0001"

        expected_checkpoint_path = Path(f"runs/{run_name}/checkpoints/epoch_0001.pth")

        checkpoint_manager.local_storage_client.exists.return_value = False
        checkpoint_manager.load_checkpoint(run_name, checkpoint_name)

        checkpoint_manager.local_storage_client.retrieve.assert_not_called()
        checkpoint_manager.remote_storage_client.retrieve.assert_called_once_with(
            expected_checkpoint_path
        )

    def test_load_checkpoint_raises_error_if_checkpoint_does_not_exist(
        self, checkpoint_manager, run_name
    ):
        checkpoint_name = "epoch:0001"

        checkpoint_manager.local_storage_client.exists.return_value = False
        checkpoint_manager.remote_storage_client.exists.return_value = False

        with pytest.raises(PersistenceError):
            checkpoint_manager.load_checkpoint(run_name, checkpoint_name)
