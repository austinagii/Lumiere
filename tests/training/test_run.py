from pathlib import Path
from typing import Any

import pytest

from lumiere.persistence.clients import (
    AzureBlobStorageClient,
    FileSystemStorageClient,
)
from lumiere.persistence.errors import ArtifactNotFoundError, StorageError
from lumiere.training import Checkpoint, CheckpointType, Run, RunManager
from lumiere.training.config import Config


@pytest.fixture
def dsconfig_file_path(tmp_path: Path) -> Path:
    config = """
    runs:
        checkpoints:
            sources:
                - filesystem 
                - azure-blob
            destinations:
                - azure-blob
                - filesystem 
    """

    config_file_path = tmp_path / "dsconfig.yaml"
    config_file_path.write_text(config)
    return config_file_path


@pytest.fixture(scope="module")
def run_config() -> dict[str, Any]:
    return {
        "tokenizer": {"vocab_size": 1024},
        "model": {"num_blocks": 8, "d_key": 128, "d_value": 128},
    }


@pytest.fixture(scope="module")
def checkpoint():
    return Checkpoint(epoch=1, prev_loss=0.95, best_loss=0.88)


@pytest.fixture
def run_manager(run_config):
    run_manager = RunManager(
        sources=["azure-blob", "filesystem"],
        destinations=["azure-blob", "filesystem"],
    )

    run_manager.run = Run.from_config(run_config)

    return run_manager


class TestRunManager:
    def test_run_manager_can_be_initialized_from_location_list(self):
        run_manager = RunManager(
            sources=["azure-blob", "filesystem"],
            destinations=["azure-blob", "filesystem"],
        )

        assert isinstance(run_manager.storage_clients[0], AzureBlobStorageClient)
        assert isinstance(run_manager.storage_clients[1], FileSystemStorageClient)
        assert isinstance(run_manager.retrieval_clients[0], AzureBlobStorageClient)
        assert isinstance(run_manager.retrieval_clients[1], FileSystemStorageClient)

    def test_run_manager_can_be_initialized_from_yaml_config(
        self, dsconfig_file_path: Path
    ) -> None:
        config = Config.from_yaml(dsconfig_file_path)
        run_manager = RunManager.from_config(config)

        assert isinstance(run_manager.storage_clients[0], FileSystemStorageClient)
        assert isinstance(run_manager.storage_clients[1], AzureBlobStorageClient)
        assert isinstance(run_manager.retrieval_clients[0], AzureBlobStorageClient)
        assert isinstance(run_manager.retrieval_clients[1], FileSystemStorageClient)

    def test_init_run_cascades_to_storage_clients(
        self, mocker, dsconfig_file_path, run_config
    ) -> None:
        run_manager = RunManager(
            sources=["azure-blob", "filesystem"],
            destinations=["azure-blob", "filesystem"],
        )

        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        run_manager.init_run(run_config)

        for client in run_manager.storage_clients:
            client.save_artifact.assert_called_once_with(
                run_manager.run.id, "config.yaml", run_manager.run.config, False
            )

    def test_init_run_raises_error_if_any_storage_client_fails(
        self, mocker, dsconfig_file_path, run_config
    ) -> None:
        run_manager = RunManager(
            sources=["azure-blob", "filesystem"],
            destinations=["azure-blob", "filesystem"],
        )

        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        # Configure one storage client to fail
        run_manager.storage_clients[1].save_artifact.side_effect = StorageError()

        with pytest.raises(StorageError):
            run_manager.init_run(run_config)

        for client in run_manager.storage_clients:
            client.save_artifact.assert_called_once()

    def test_save_checkpoint_cascades_to_storage_clients(
        self, mocker, run_manager, checkpoint
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_checkpoint")

        run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint, epoch_no=checkpoint['epoch'])

        for client in run_manager.storage_clients:
            client.save_checkpoint.assert_called_once_with(
                run_manager.run.id, "epoch:0001", bytes(checkpoint)
            )

    @pytest.mark.parametrize(
        "checkpoint_type, checkpoint_tag",
        [
            (CheckpointType.EPOCH, "epoch:0001"),
            (CheckpointType.BEST, "best"),
            (CheckpointType.FINAL, "final"),
        ],
    )
    def test_save_checkpoint_uses_correct_checkpoint_tag(
        self, mocker, run_manager, checkpoint, checkpoint_type, checkpoint_tag
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_checkpoint")

        if checkpoint_type == CheckpointType.EPOCH:
            run_manager.save_checkpoint(checkpoint_type, checkpoint, epoch_no=checkpoint['epoch'])
        else:
            run_manager.save_checkpoint(checkpoint_type, checkpoint)

        for client in run_manager.storage_clients:
            client.save_checkpoint.assert_called_once_with(
                run_manager.run.id, checkpoint_tag, bytes(checkpoint)
            )

    def test_save_checkpoint_does_not_raise_error_if_any_storage_client_fails(
        self, mocker, run_manager, checkpoint
    ) -> None:
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_checkpoint")

        # Configure one storage client to fail
        run_manager.storage_clients[1].save_checkpoint.side_effect = StorageError()

        run_manager.save_checkpoint(CheckpointType.EPOCH, checkpoint, epoch_no=checkpoint['epoch'])

        for client in run_manager.storage_clients:
            client.save_checkpoint.assert_called_once()

    # ----------------------------------
    # ------ SAVE ARTIFACT TESTS -------
    # ----------------------------------

    def test_save_artifact_saves_to_all_storage_locations(self, mocker, run_manager):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        key = "test-dict"
        artifact = {"test_data": ["Some", "simple", "test", "data"]}

        run_manager.save_artifact(key, artifact)

        # Since RunManager doesn't actually save the artifacts, but instead delegates
        # the actual work to the storage client configured for each location, we just
        # verify that the necessary calls have been dispatched correctly.
        for client in run_manager.storage_clients:
            actual_args = client.save_artifact.call_args.args
            assert actual_args[0] == run_manager.run.id
            assert actual_args[1] == key
            assert actual_args[2] == artifact

    def test_save_artifact_does_not_overwrite_existing_artifacts_by_default(
        self, mocker, run_manager
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        key = "test-dict"
        artifact = {"test_data": ["Some", "simple", "test", "data"]}

        run_manager.save_artifact(key, artifact)

        # Same as the other tests for this method, just verify that the method call
        # is dispatched specifying that the storage client should not overwrite the
        # previous document.
        for client in run_manager.storage_clients:
            assert not client.save_artifact.call_args.kwargs["overwrite"]

    def test_save_artifact_overwrites_existing_artifacts_if_overwrite_is_true(
        self, mocker, run_manager
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        key = "test"
        artifact = {"testdata": 10}

        # Verify that the method call is dispatched to each storage client specifying
        # that artifacts should be overwritten.
        run_manager.save_artifact(key, artifact, overwrite=True)

        for client in run_manager.storage_clients:
            assert client.save_artifact.call_args.kwargs["overwrite"]

    def test_save_artifact_does_not_overwrite_existing_artifacts_if_overwrite_is_false(
        self, mocker, run_manager
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        key = "test"
        artifact = {"testdata": 10}

        # Verify that the method call is dispatched to each storage client specifying
        # that artifacts should not be overwritten.
        run_manager.save_artifact(key, artifact, overwrite=False)

        for client in run_manager.storage_clients:
            assert not client.save_artifact.call_args.kwargs["overwrite"]

    def test_save_artifact_raises_error_if_raise_exception_is_true(
        self, mocker, run_manager
    ):
        mocker.patch.object(
            run_manager.storage_clients[1], "save_artifact", side_effect=Exception()
        )

        with pytest.raises(StorageError):
            run_manager.save_artifact(
                "test", ["some", "test", "data"], raise_exception=True
            )

    def test_save_artifact_does_not_raise_error_if_raise_exception_is_false(
        self, mocker, run_manager
    ):
        for client in run_manager.storage_clients:
            mocker.patch.object(client, "save_artifact")

        # Configure at least one storage client to raise an error.
        mocker.patch.object(
            run_manager.storage_clients[1], "save_artifact", side_effect=Exception()
        )

        # This should not raise an error even if a client raises an error.
        run_manager.save_artifact(
            "test", ["some", "test", "data"], raise_exception=False
        )

        # Double check to make sure that all clients (including the one that errored)
        # were callled.
        for client in run_manager.storage_clients:
            client.save_artifact.assert_called_once()

    # ------------------------------------
    # ------- LOAD_ARTIFACT TESTS --------
    # -------------------------------------

    def test_load_artifact_loads_from_storage_locations_in_configured_order(
        self, mocker, run_manager
    ):
        # Configure each storage client to return a unique object.
        artifact1 = {"test": "data"}
        mocker.patch.object(
            run_manager.retrieval_clients[0], "load_artifact", return_value=artifact1
        )

        artifact2 = {"test": "data2"}
        mocker.patch.object(
            run_manager.retrieval_clients[1], "load_artifact", return_value=artifact2
        )

        # Since our run_manager fixture defines a run manager configured with sources
        # "azure-blob" and "filesystem" in that order. We should expect artifacts are
        # looked up from these sources in the same order.

        # Since azure blob comes first, we should find artifact1 on lookup.
        artifact = run_manager.load_artifact("test")
        assert artifact == artifact1
        assert artifact != artifact2  # Not necessary but for extra safety.

        # If azure blob does not contain the object, then we expect the object should be
        # loaded from the filesystem.
        run_manager.retrieval_clients[0].load_artifact.return_value = None
        artifact = run_manager.load_artifact("test")
        assert artifact == artifact2

    def test_load_artifact_does_not_raise_error_if_location_raises_error(
        self, mocker, run_manager
    ):
        for client in run_manager.retrieval_clients:
            mocker.patch.object(client, "load_artifact", side_effect=Exception())

        mocker.patch.object(
            run_manager.retrieval_clients[-1],
            "load_artifact",
            return_value={"test": "data"},
        )

        run_manager.load_artifact("test")

        for client in run_manager.retrieval_clients:
            client.load_artifact.assert_called_once()

    def test_load_artifact_raises_an_error_if_the_artifact_could_not_be_found(
        self, mocker, run_manager
    ):
        for client in run_manager.retrieval_clients:
            mocker.patch.object(client, "load_artifact", return_value=None)

        with pytest.raises(ArtifactNotFoundError):
            run_manager.load_artifact("test")
