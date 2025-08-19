from pathlib import Path

from lumiere.data.tokenizer import Tokenizer
from lumiere.persistence.errors import PersistenceError
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient


TOKENIZER_PATH_TEMPLATE = "runs/{run_name}/tokenizer.json"


class TokenizerManager:
    def __init__(
        self,
        remote_storage_client: RemoteStorageClient = None,
        local_storage_client: LocalStorageClient = None,
        should_cache: bool = True,
    ):
        self.remote_storage_client = remote_storage_client
        self.local_storage_client = local_storage_client
        self.should_cache = should_cache

    def save_tokenizer(self, run_name: str, tokenizer: Tokenizer) -> None:
        tokenizer_path = self._get_tokenizer_path(run_name)
        tokenizer_bytes = bytes(tokenizer.tokenizer.to_str(), "utf-8")

        if self.local_storage_client is not None:
            self.local_storage_client.store(tokenizer_path, tokenizer_bytes)

        if self.remote_storage_client is not None:
            self.remote_storage_client.store(tokenizer_path, tokenizer_bytes)

    def load_tokenizer(self, run_name: str) -> Tokenizer:
        """Load the tokenizer from local storage or blob storage.

        If the tokenizer is not found in local storage, it will be downloaded from blob
        storage.
        """
        tokenizer_path = self._get_tokenizer_path(run_name)

        # TODO: Allow tokenizer to be overwritten by remote.
        if self.local_storage_client is not None and self.local_storage_client.exists(
            tokenizer_path
        ):
            tokenizer_bytes = self.local_storage_client.retrieve(tokenizer_path)
        else:
            if (
                self.remote_storage_client is not None
                and self.remote_storage_client.exists(tokenizer_path)
            ):
                tokenizer_bytes = self.remote_storage_client.retrieve(tokenizer_path)
            else:
                raise PersistenceError(
                    f"The specified tokenizer could not be found: {run_name}"
                )

            if self.local_storage_client is not None:
                self.local_storage_client.store(tokenizer_path, tokenizer_bytes)

        return Tokenizer.from_bytes(tokenizer_bytes)

    def _get_tokenizer_path(self, run_name: str) -> Path:
        """Returns the path to the specified tokenizer"""
        return TOKENIZER_PATH_TEMPLATE.format(run_name=run_name)
