from pathlib import Path

import pytest

from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.training import Checkpoint, CheckpointType


@pytest.fixture(
    params=[CheckpointType.EPOCH, CheckpointType.BEST, CheckpointType.FINAL]
)
def checkpoint(request):
    return Checkpoint(request.param, epoch=1, prev_loss=0.95, best_loss=0.88)


@pytest.fixture
def checkpoint_bytes(checkpoint):
    return bytes(checkpoint)


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
        connection_string="DefaultEndpointsProtocol=https;AccountName=testing;AccountKey=arandomkey==;EndpointSuffix=core.windows.net",
        container_name="testing",
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
    @pytest.mark.parametrize("model_name", ["cnn", "transformer"])
    def test_save_checkpoint_correctly_saves_epoch_checkpoints(
        self, checkpoint_manager, checkpoint, model_name
    ):
        checkpoint_manager.save_checkpoint(model_name, checkpoint)

        expected_checkpoint_dir = Path(model_name)
        match checkpoint.type_:
            case CheckpointType.EPOCH:
                expected_checkpoint_path = expected_checkpoint_dir / "epoch_0001.pth"
            case CheckpointType.BEST:
                expected_checkpoint_path = expected_checkpoint_dir / "best.pth"
            case CheckpointType.FINAL:
                expected_checkpoint_path = expected_checkpoint_dir / "final.pth"

        # Check that the manager called the local storage client.
        checkpoint_manager.local_storage_client.store.assert_called_once_with(
            bytes(checkpoint), expected_checkpoint_path
        )

        # Check that the manager called the remote storage client.
        checkpoint_manager.remote_storage_client.store.assert_called_once_with(
            bytes(checkpoint), expected_checkpoint_path
        )

    @pytest.mark.parametrize("model_name", ["cnn", "transformer"])
    def test_load_checkpoint_retrieves_local_when_it_exists(
        self, checkpoint_manager, checkpoint, model_name
    ):
        expected_checkpoint_dir = Path(model_name)
        match checkpoint.type_:
            case CheckpointType.EPOCH:
                expected_checkpoint_path = expected_checkpoint_dir / "epoch_0001.pth"
            case CheckpointType.BEST:
                expected_checkpoint_path = expected_checkpoint_dir / "best.pth"
            case CheckpointType.FINAL:
                expected_checkpoint_path = expected_checkpoint_dir / "final.pth"

        checkpoint_manager.load_checkpoint(model_name, checkpoint)

        checkpoint_manager.local_storage_client.retrieve.assert_called_once_with(
            expected_checkpoint_path
        )

        checkpoint_manager.remote_storage_client.retrieve.assert_not_called()

    @pytest.mark.parametrize("model_name", ["cnn", "transformer"])
    def test_load_checkpoint_retrieves_remote_when_local_does_not_exist(
        self, checkpoint_manager, checkpoint, model_name
    ):
        checkpoint_manager.local_storage_client.exists.return_value = False

        expected_checkpoint_dir = Path(model_name)
        match checkpoint.type_:
            case CheckpointType.EPOCH:
                expected_checkpoint_path = expected_checkpoint_dir / "epoch_0001.pth"
            case CheckpointType.BEST:
                expected_checkpoint_path = expected_checkpoint_dir / "best.pth"
            case CheckpointType.FINAL:
                expected_checkpoint_path = expected_checkpoint_dir / "final.pth"

        checkpoint_manager.load_checkpoint(model_name, checkpoint)

        checkpoint_manager.local_storage_client.retrieve.assert_not_called()

        checkpoint_manager.remote_storage_client.retrieve.assert_called_once_with(
            expected_checkpoint_path
        )

    def test_load_checkpoint_raises_error_if_checkpoint_does_not_exist(
        self, checkpoint_manager, checkpoint, model_name
    ):
        checkpoint_manager.local_storage_client.exists.return_value = False
        checkpoint_manager.remote_storage_client.exists.return_value = False

        with pytest.raises(PersistenceError):
            checkpoint_manager.load_checkpoint(model_name, checkpoint)
