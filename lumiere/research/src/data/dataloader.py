import re
from typing import Generator, Protocol

import datasets


class DataLoader(Protocol):
    def iter_train(self) -> Generator[str, None, None]: ...

    def iter_validation(self) -> Generator[str, None, None]: ...


# TODO: Add preprocessing step to remove wikipedia headers e.g.
# " = = The Two Year War = = ". The aim is to allow the model to learn "professional"
# language without wikipedia specific formatting details.
class WikiText2DataLoader:
    """Loads the WikiText-2 dataset.

    The specific splits (along with the percentage of those splits) to be loaded can
    be specified during construction.

    Once loaded, the `iter_train`, `iter_valdation` and `iter_test` methods can be
    used to iterate over the corresponding training split.

    The following preprocessing steps are applied to the dataset:

    1. Aggregate text sequences for a given article into a single string.
    2. Add "start of text" and "end of text" special tokens to each string.

    When iterating over a given split, each entry will be a string corresponding to
    a single wikipedia article.
    """

    # TODO: Default percentage to 'None' to prevent loading splits that aren't needed.
    def __init__(
        self,
        train_dataset_percentage: int = 100,
        validation_dataset_percentage: int = 100,
        test_dataset_percentage: int = 100,
    ):
        """Initialize the dataloader.

        For each split (train, validation, test), the percentage of that split to be
        loaded can be specified as an integer between 1 and 100.

        Args:
            train_dataset_percentage: The percentage of the training split to be loaded.
                Defaults to 100 (The full training split).
            validation_dataset_percentage: The percentage of the training split to be
                loaded. Defaults to 100 (The full validation split).
            train_dataset_percentage: The percentage of the test split to be loaded.
                Defaults to 100 (The full test split).
        """
        self.train_dataset_percentage = train_dataset_percentage
        self.validation_dataset_percentage = validation_dataset_percentage
        self.test_dataset_percentage = test_dataset_percentage

        self.train_dataset, self.validation_dataset, self.test_dataset = (
            datasets.load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=[
                    f"train[:{self.train_dataset_percentage}%]",
                    f"validation[:{self.validation_dataset_percentage}%]",
                    f"test[:{self.validation_dataset_percentage}%]",
                ],
            )
        )

    # TODO: Consider consolidating these iterator methods into a single method.
    def iter_train(self) -> Generator[str, None, None]:
        return self._iter_articles(self.train_dataset)

    def iter_validation(self) -> Generator[str, None, None]:
        return self._iter_articles(self.validation_dataset)

    def iter_test(self) -> Generator[str, None, None]:
        return self._iter_articles(self.test_dataset)

    def _iter_articles(self, dataset) -> Generator[str, None, None]:
        """Groups individual samples into complete articles.

        Articles in WikiText-2 are structured as:
        - Empty string (separator)
        - Article title (e.g., " = Title = \n")
        - Empty string (separator)
        - Content paragraphs...
        - Next article starts with empty string + title

        Each article is wrapped with start and end of sequence tokens.
        """
        current_article: list[str] = []

        for sample in dataset:
            text = sample["text"]

            # Check if this is an article title (starts with " = ")
            if re.match(r"^=[^=]*=$", text.strip()):
                # If we have accumulated content, yield the previous article
                # TODO: Make this empty check explicit.
                if current_article:
                    article_text = "".join(current_article)
                    yield f"<|sot|>{article_text}<|eot|>"
                    current_article = []

                # Start new article with the title
                current_article = [text]

            # Skip empty strings that are just separators
            elif text.strip() == "":
                continue

            # Add content to current article
            else:
                current_article.append(text)

        # Yield the last article if it exists
        if current_article:
            article_text = "".join(current_article)
            yield f"<|sot|>{article_text}<|eot|>"


def get_data_loader(
    dataset_name: str,
    train_dataset_percentage: int = 100,
    validation_dataset_percentage: int = 100,
    test_dataset_percentage: int = 100,
) -> DataLoader:
    if dataset_name == "wikitext":
        return WikiText2DataLoader(
            train_dataset_percentage,
            validation_dataset_percentage,
            test_dataset_percentage,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
