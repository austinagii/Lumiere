class StorageError(Exception):
    def __init__(self, message: str = None, e: Exception = None) -> None:
        super().__init__(message, e)


class CheckpointNotFoundError(Exception):
    def __init__(self, message: str = None, e: Exception = None) -> None:
        super().__init__(message, e)


class RunNotFoundError(Exception):
    def __init__(self, message: str = None, e: Exception = None) -> None:
        super().__init__(message, e)


class ArtifactNotFoundError(Exception):
    def __init__(self, message: str = None, e: Exception = None) -> None:
        super().__init__(message, e)
