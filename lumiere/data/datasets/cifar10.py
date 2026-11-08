"""This module provides the CIFAR-10 dataset."""

from collections.abc import Generator

import datasets
import numpy as np
import torch

from lumiere.data import Dataset
from lumiere.internal.registry import discover

from .wikitext import parse_split_percentages


_DATASET_ID = "uoft-cs/cifar10"
_DATASET_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"

# CIFAR-10 has no validation split, so the last images of the train split are held out
# for validation.
_NUM_TRAIN_IMAGES = 45_000
_NUM_VALIDATION_IMAGES = 5_000

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@discover(Dataset, "cifar10")
class CIFAR10Dataset:
    """The CIFAR-10 dataset.

    This class provides access to the 32x32 RGB images and class labels of the CIFAR-10
    dataset across the train, validation and test splits. Since CIFAR-10 does not
    provide a validation split, the last 5,000 images of the original train split are
    held out for validation, leaving 45,000 images for training.

    By default, all splits will be accessible in full. However, splits can be partially
    loaded or omitted by specifying percentages during initialization using a colon-
    separated format (e.g., "50:100:30" for 50% train, 100% validation, 30% test).

    ```python
    dataset = CIFAR10Dataset("50:100:30")
    ```

    This class implements the `Dataset` protocol, allowing the use of subscript notation
    to access an iterator over samples from a specified training split. Each sample is a
    tuple of an image and its class label.

    ```python
    cifar10 = CIFAR10Dataset("50:100:30")
    for image, label in cifar10["train"]:
        print(image.shape, CLASS_NAMES[label])
        # Output: torch.Size([3, 32, 32]) frog
    ```

    """

    def __init__(self, split: str = "100:100:100"):
        """Initialize a CIFAR-10 dataset.

        Args:
            split: Percentages for each training split specified as a colon-separated
                string in the format "train:validation:test" (e.g., "50:100:30").
                Omitted values default to 100% (e.g., "50::30" loads 50% train, 100%
                validation, 30% test). Defaults to "100:100:100".

        Raises:
            ValueError: If all splits are specified to be empty (0%).

        """
        split_percentages = parse_split_percentages(split)
        if not split_percentages:
            raise ValueError("At least one split must contain data.")

        split_specs = {
            "train": lambda p: f"train[:{_NUM_TRAIN_IMAGES * p // 100}]",
            "validation": lambda p: (
                f"train[{_NUM_TRAIN_IMAGES}:"
                f"{_NUM_TRAIN_IMAGES + _NUM_VALIDATION_IMAGES * p // 100}]"
            ),
            "test": lambda p: f"test[:{p}%]",
        }

        splits = datasets.load_dataset(
            _DATASET_ID,
            split=[
                split_specs[split_name](percentage)
                for split_name, percentage in split_percentages.items()
            ],
            revision=_DATASET_REVISION,
        )
        self._splits = {
            split_name: splits[split_ix].with_format("numpy")
            for split_ix, split_name in enumerate(split_percentages.keys())
        }

    def __getitem__(
        self, split_name: str
    ) -> Generator[tuple[torch.Tensor, int], None, None]:
        """Return an iterator over samples in the specified split.

        Args:
            split_name: Name of the split to access. Must be one of "train",
                "validation" or "test".

        Returns:
            Generator yielding tuples of images and class labels from the specified
            split. Each image is a uint8 tensor of shape `(3, 32, 32)` and each label
            is an integer in the range [0, 10).

        Raises:
            KeyError: If the specified split is not available.
        """
        if self._splits.get(split_name) is None:
            raise KeyError(
                f"Invalid split '{split_name}'. Avaliable splits are: "
                + f"[{', '.join(self._splits.keys())}]"
            )

        # Wrap in function to allow exception to be raised on calling __getitem__
        # with invalid split, instead of on access to first element from iterator.
        def _get_split() -> Generator[tuple[torch.Tensor, int], None, None]:
            for sample in self._splits[split_name]:
                # (height, width, channels) -> (channels, height, width)
                image = torch.from_numpy(np.array(sample["img"], dtype=np.uint8))
                image = image.permute(2, 0, 1)
                yield image, int(sample["label"])

        return _get_split()
