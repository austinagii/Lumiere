"""This module provides the WikiText2 dataset."""

import re
from collections.abc import Generator, Iterable
from typing import Final

import datasets

from lumiere.data import Dataset
from lumiere.internal.registry import discover


_DATASET_ID = "Salesforce/wikitext"
_DATASET_NAME = "wikitext-2-raw-v1"
_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

# Pattern for parsing formatted split percentages.
_SPLIT_PATTERN: Final[re.Pattern] = re.compile(
    r"^(?P<train>100|\d{1,2})?(?::(?P<validation>100|\d{1,2})?)?(?::(?P<test>100|\d{1,2})?)?$"
)
# Patterns for identifying Wikipedia article titles and headers.
_ARTICLE_TITLE_PATTERN: Final[re.Pattern] = re.compile(
    r"^(?:\s*=\s*){1}([^=\s]+(?:\s+[^=\s]+)*)(?:\s*=\s*){1}$"
)
_ARTICLE_HEADER_PATTERN: Final[re.Pattern] = re.compile(
    r"^(?:\s*=\s*){2,}([^=\s]+(?:\s+[^=\s]+)*)(?:\s*=\s*){2,}$"
)


@discover(Dataset, "wikitext")
class WikiText2Dataset:
    """The WikiText-2 dataset.

    This class provides access to complete Wikipedia articles from the WikiText2 dataset
    across the train, validation and test splits. For each article, the title and
    headers are removed, and the article is wrapped in start and end of sequence tokens.

    By default, all splits will be accessible in full. However, splits can be partially
    loaded or omitted by specifying percentages during initialization using a colon-
    separated format (e.g., "50:100:30" for 50% train, 100% validation, 30% test).

    ```python
    dataset = WikiText2Dataset("50:100:30")
    ```

    This class implements the `Dataset` protocol, allowing the use of subscript notation
    to access an iterator over samples from a specified training split.

    ```python
    wikitext = WikiText2Dataset("50:100:30")
    for article in wikitext["train"]:
        print(article)
    ```

    """

    def __init__(
        self,
        split: str = "100:100:100",
    ):
        """Initialize a WikiText2 dataset.

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

        splits = datasets.load_dataset(
            _DATASET_ID,
            _DATASET_NAME,
            split=[
                f"{split_name}[:{percentage}%]"
                for split_name, percentage in split_percentages.items()
            ],
            revision=_DATASET_REVISION,
        )
        self._splits = {
            split_name: splits[split_ix]
            for split_ix, split_name in enumerate(split_percentages.keys())
        }

    def __getitem__(self, split_name: str) -> Generator[str, None, None]:
        """Return an iterator over samples in the specified split.

        Args:
            split_name: Name of the split to access. Must be one of "train",
                "validation" or "test".

        Returns:
            Generator yielding complete aticles from the specified split.

        Raises:
            KeyError: If the specified split is not available.
        """
        if self._splits.get(split_name) is None:
            raise KeyError(
                f"Invalid split '{split_name}'. Avaliable splits are: "
                + f"[{'.'.join(self._splits.keys())}]"
            )

        # Wrap in function to allow exception to be raised on calling __getitem__
        # with invalid split, instead of on access to first element from iterator.
        def _get_split() -> Generator[str, None, None]:
            yield from _iter_articles(self._splits[split_name]["text"])

        return _get_split()


def parse_split_percentages(split: str | None) -> dict[str, int]:
    """Parse split specification string.

    Args:
        split: Split specification in format `"train:validation:test"`.

    Returns:
        Dictionary mapping split names to their percentages. Splits with 0% are
            excluded.

    Raises:
        ValueError: If the split string format is invalid.
    """
    split_percentages = {"train": 100, "validation": 100, "test": 100}

    if split is not None:
        if (match := _SPLIT_PATTERN.match(split)) is None:
            raise ValueError(f"Split '{split}' is incorrectly formatted.")

        # Override default split percentages with those specified. If zero, the
        # split is removed entirely.
        for split_name, percentage in match.groupdict().items():
            if percentage is not None:
                percentage_value = int(percentage)

                if percentage_value == 0:
                    del split_percentages[split_name]
                else:
                    split_percentages[split_name] = percentage_value

    return split_percentages


def _iter_articles(dataset: Iterable[str]) -> Generator[str, None, None]:
    """Return an iterator over full articles in the specified dataset."""
    text_buffer: list[str] = []

    for text in dataset:
        # Ignore empty text.
        if len(text.strip()) == 0:
            continue

        # Prevent samples containing header formatting details.
        if _ARTICLE_HEADER_PATTERN.match(text):
            continue

        # Prevent grouping unrelated text (where text crosses article boundaries).
        if _ARTICLE_TITLE_PATTERN.match(text):
            if len(text_buffer) > 0:
                yield _concat_article(text_buffer)
                text_buffer = []
            continue  # Also prevent samples containing title formatting details.

        text_buffer.append(text)

    # Flush buffer to get last article since loop only yields at article boundaries.
    if text_buffer:
        yield _concat_article(text_buffer)


def _concat_article(text: list[str]) -> str:
    """Concatenate text segments into a single article with special tokens."""
    return f"<|sot|>{''.join(text)}<|eot|>"
