import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


class RunFinder:
    def __init__(self, blob_service_client: BlobServiceClient):
        self.blob_service_client = blob_service_client

    def find_run(self, run_id: str) -> str:
        """Find the run name for the given run ID."""
        run_name = None
        for run_path in Path("runs").iterdir():
            if run_path.is_dir() and run_id in run_path.name:
                run_name = run_path.name
                break
        if run_name is None:
            blob_client = self.blob_service_client.get_container_client(
                os.getenv("BLOB_STORAGE_CONTAINER_NAME")
            )
            for blob_name in blob_client.list_blob_names():
                if run_id in blob_name:
                    run_name = blob_name.split("/")[1]
                    break
        return run_name
