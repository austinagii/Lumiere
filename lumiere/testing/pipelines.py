from collections.abc import Iterable


class IdentityPipeline:
    def batches(self, data: Iterable):
        """An iterator over samples from the dataloader."""
        yield from data
