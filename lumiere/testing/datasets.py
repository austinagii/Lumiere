from collections.abc import Iterable, Mapping
from typing import Any

from lumiere.data.dataset import dataset


class IdentityDataset:
    """A custom dataset for defining arbitrary splits with text samples.

    Accepts a mapping of split names to iterables of string samples,
    allowing for flexible dataset construction for testing purposes.
    """

    def __init__(self, data: Mapping[str, Iterable[Any]]):
        self.data = data

    def __getitem__(self, split_name):
        """Return an iterable over the text samples for the given split.

        Args:
            split_name: The name of the split to retrieve.

        Returns:
            An iterable yielding string samples for the specified split.

        Raises:
            KeyError: If the split name is not found in the dataset.
        """

        def _get_split():
            yield from self.data[split_name]

        if split_name not in self.data:
            raise KeyError(f"Split '{split_name}' not found")

        return _get_split()


class StringDataset:
    """A custom dataset for defining arbitrary splits with text samples.

    Accepts a mapping of split names to iterables of string samples,
    allowing for flexible dataset construction for testing purposes.
    """

    def __init__(self, data: Mapping[str, Iterable[str]]):
        self.data = data

    def __getitem__(self, split_name):
        """Return an iterable over the text samples for the given split.

        Args:
            split_name: The name of the split to retrieve.

        Returns:
            An iterable yielding string samples for the specified split.

        Raises:
            KeyError: If the split name is not found in the dataset.
        """

        def _get_split():
            yield from self.data[split_name]

        if split_name not in self.data:
            raise KeyError(f"Split '{split_name}' not found")

        return _get_split()


@dataset("lorem-ipsum")
class LoremIpsumDataset:
    """A toy dataset containing Lorem Ipsum."""

    data = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Aliquam erat volutpat. Vivamus eu magna sem.",
        "Morbi nec finibus justo.",
        "Aliquam id feugiat nulla, malesuada euismod libero.",
        "Curabitur id massa nibh.",
    ]

    def __init__(self, source: str, count: int = 10):
        self.source = source
        self.count = count

    def __getitem__(self, split_name):
        def _get_split():
            match split_name:
                case "train":
                    yield from self.data
                case "validation":
                    yield from self.data[1:]
                case _:
                    return

        if split_name not in ["train", "validation"]:
            raise KeyError(f"Split '{split_name}' not found")

        return _get_split()


@dataset("famous-quotes")
class FamousQuotesDataset:
    """A toy dataset containing famous quotes."""

    data = [
        "Be yourself; everyone else is already taken.",
        "In three words I can sum up everything I've learned about life: it goes on.",
        "The only way to do great work is to love what you do.",
        "It is during our darkest moments that we must focus to see the light.",
        "The only impossible journey is the one you never begin.",
    ]

    def __init__(self, tone: str, topics: list[str]):
        self.tone = tone
        self.topics = topics

    def __getitem__(self, split_name):
        def _get_split():
            match split_name:
                case "train":
                    yield from self.data
                case "test":
                    yield from self.data[2:]
                case _:
                    return

        if split_name not in ["train", "test"]:
            raise KeyError(f"Split '{split_name}' not found")

        return _get_split()
