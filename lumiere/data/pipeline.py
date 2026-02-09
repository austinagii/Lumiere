from typing import Protocol


class Pipeline(Protocol):
    def batches(self, data): ...
