"""This module provides the WikiText2 dataset."""

import re
from collections.abc import Generator, Iterable
from typing import Final

import datasets

from lumiere.data import Dataset
from lumiere.discover import discover


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
    """Loads the WikiText-2 dataset.

    By default, all splits will be accessible in full. However, splits can be partially
    loaded or omitted by specifying percentages during initialization using a colon-
    separated format (e.g., "50:100:30" for 50% train, 100% validation, 30% test).

    Once initialized, samples from a given split can be iterated over using bracket
    notation (e.g., data['train']). Preprocessing is applied to all splits such that
    each sample is a complete Wikipedia article.

    Raises:
        ValueError: If all splits are specified to be empty.

    Example:
        >>> wikitext = WikiText2Dataset("20:30:50")
        >>> for sample in wikitext["train"]:
        ...     # process the sample

    """

    def __init__(
        self,
        split: str | None = None,
    ):
        """Initialize a WikiText2 dataset.

        Args:
            split: Percentages for each split as a colon-separated string in the format
                "train:validation:test" (e.g., "50:100:30"). Omitted values default to
                100% (e.g., "50::30" loads 50% train, 100% validation, 30% test).
                Defaults to "100:100:100".

        Raises:
            ValueError: If all splits are specified to be empty.

        """
        split_percentages = self._get_split_percentages(split)
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

    @staticmethod
    def _get_split_percentages(split: str | None) -> dict[str, int]:
        """Return the percentage of each split to be used."""
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

    def __getitem__(self, split_name: str) -> Generator[str, None, None]:
        """Return an iterator over samples in the specified split."""
        if self._splits.get(split_name) is None:
            raise KeyError(f"Invalid split '{split_name}'.")

        def _get_split():
            yield from self._iter_articles(self._splits[split_name]["text"])

        return _get_split()

    def _iter_articles(self, dataset: Iterable[str]) -> Generator[str, None, None]:
        """Return an iterator over full articles in the specified dataset.

        The dataset is preprocessed such that article titles and headers are
        removed, text samples belonging to the same Wikipedia article are
        concatenated into a single text sequence and each article is wrapped in start
        and end of sequence tokens.
        """
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
                    yield self._concat_article(text_buffer)
                    text_buffer = []
                continue  # Also prevent samples containing title formatting details.

            text_buffer.append(text)

        # Flush buffer to get last article since loop only yields at article boundaries.
        if text_buffer:
            yield self._concat_article(text_buffer)

    @staticmethod
    def _concat_article(text: list[str]) -> str:
        return f"<|sot|>{''.join(text)}<|eot|>"
