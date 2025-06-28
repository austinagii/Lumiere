class PersistenceError(Exception):
    def __init__(self, message: str, e: Exception = None) -> None:
        super().__init__(message, e)
