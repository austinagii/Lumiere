from typing import Generator, Protocol


class DataLoader(Protocol):
    def iter_train(self) -> Generator[str, None, None]: ...

    def iter_validation(self) -> Generator[str, None, None]: ...
