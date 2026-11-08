from collections.abc import Iterable


class IdentityPipeline:
    def batches(self, data: Iterable):
        """An iterator over samples from the dataloader."""
        yield from data

    def to(self, device):
        """No-op, the identity pipeline holds no device state."""
        return self
