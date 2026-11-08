"""Image data pipeline for batching and normalizing labelled images.

Provides :class:`ImagePipeline`, a :class:`~lumiere.data.Pipeline` implementation
that groups `(image, label)` samples into batches, scales pixel values to the range
[0, 1] and normalizes each channel, producing batches suitable for image
classification training or inference.

Example:
    ```python
    import torch
    from lumiere.data.pipelines import ImagePipeline

    samples = [(torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8), 1)] * 10
    pipeline = ImagePipeline(batch_size=4, mean=[0.5] * 3, std=[0.5] * 3)
    images, labels = next(pipeline.batches(samples))
    print(images.shape, labels.shape)
    # Output: torch.Size([4, 3, 32, 32]) torch.Size([4])
    ```
"""

import functools
from collections.abc import Iterable, Sequence

import torch

from lumiere.data import Pipeline, Preprocessor
from lumiere.internal.registry import discover
from lumiere.utils.validation import validate_integer


@discover(Pipeline, "image")
class ImagePipeline:
    """Performs common preprocessing steps across labelled image data.

    It should be possible to directly pass the outputs of this pipeline directly to a
    model as input for training or inference.

    """

    def __init__(
        self,
        batch_size: int,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        preprocessors: list[Preprocessor] | None = None,
        device: str = "cpu",
    ):
        """Initialise the image pipeline.

        Args:
            batch_size: Number of images packed into each batch. Must be at least 1.
            mean: Optional per-channel mean subtracted from each image after scaling
                pixel values to the range [0, 1]. Defaults to no mean subtraction.
            std: Optional per-channel standard deviation each image is divided by
                after mean subtraction. Defaults to no scaling.
            preprocessors: Optional ordered sequence of
                :class:`~lumiere.data.Preprocessor` callables applied to every
                `(images, labels)` batch before it is yielded. Defaults to no
                preprocessing.
            device: The device batches are created on. Defaults to `"cpu"`.

        Raises:
            ValueError: If `batch_size` is less than 1, if any value in `std` is not
                positive, or if `mean` and `std` have different lengths.
        """
        validate_integer(batch_size, "batch_size", min_value=1)

        if std is not None and any(s <= 0 for s in std):
            raise ValueError("std values must be positive.")
        if mean is not None and std is not None and len(mean) != len(std):
            raise ValueError("mean and std must have the same number of channels.")

        self.batch_size = batch_size
        self.mean = list(mean) if mean is not None else None
        self.std = list(std) if std is not None else None
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.device = device

    def _create_batches(
        self, data: Iterable, num_batches: int | None = None
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Internal method to create raw image batches before preprocessing.

        Returns:
            Iterator of `(images, labels)` tuples.
        """
        if num_batches is not None:
            validate_integer(num_batches, "num_batches", min_value=1)

        images, labels = [], []
        num_batches_created = 0

        for image, label in data:
            images.append(torch.as_tensor(image))
            labels.append(label)

            if len(images) == self.batch_size:
                yield self._collate(images, labels)
                images, labels = [], []
                num_batches_created += 1

                # Return early if we've reached the requested number of batches.
                if num_batches is not None and num_batches_created >= num_batches:
                    return

        # Yield a partial batch if we're at the end of the data.
        if images:
            yield self._collate(images, labels)

    def _collate(
        self, images: list[torch.Tensor], labels: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack, scale and normalize a list of images and labels into a batch."""
        batch = torch.stack(images).to(self.device)

        # Scale integer pixel values (e.g. uint8) to the range [0, 1].
        if not torch.is_floating_point(batch):
            batch = batch.float() / 255.0

        if self.mean is not None:
            batch = batch - self._channel_tensor(self.mean, batch)
        if self.std is not None:
            batch = batch / self._channel_tensor(self.std, batch)

        targets = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        return batch, targets

    @staticmethod
    def _channel_tensor(values: list[float], batch: torch.Tensor) -> torch.Tensor:
        """Return per-channel values shaped for broadcasting across a batch."""
        if len(values) != batch.shape[1]:
            raise ValueError(
                f"Expected {batch.shape[1]} per-channel values, got {len(values)}."
            )

        return torch.tensor(values, dtype=batch.dtype, device=batch.device).view(
            1, -1, 1, 1
        )

    def batches(
        self, data: Iterable, num_batches: int | None = None
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Convert labelled images into normalized image batches for training.

        Images are grouped into batches of `batch_size`. Integer pixel values are
        scaled to the range [0, 1] and, if configured, each channel is normalized
        using `mean` and `std`. After batching, all configured preprocessors are
        applied in sequence.

        Args:
            data: Iterable of `(image, label)` tuples, where each image is a tensor
                of shape `(num_channels, height, width)` and each label is an
                integer class index.
            num_batches: If provided, stop after yielding this many full batches.
                Otherwise yield all batches until the data is exhausted.

        Returns:
            Iterator of preprocessed batches. Before preprocessing, batches are
            tuples of:
                - `images` is a FloatTensor of shape
                  `(N, num_channels, height, width)`.
                - `labels` is a LongTensor of shape `(N,)`.
              For all full batches, `N == batch_size`. The final batch may be smaller
              if there are not enough images to fill it.
        """
        for batch in self._create_batches(data, num_batches):
            yield functools.reduce(lambda x, f: f(*x), self.preprocessors, batch)

    def to(self, device: str):
        """Move the pipeline and its preprocessors to the specified device."""
        self.device = device
        for i in range(len(self.preprocessors)):
            self.preprocessors[i] = self.preprocessors[i].to(device)

        return self
