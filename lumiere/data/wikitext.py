import re
from typing import Generator

import datasets


class WikiText2DataLoader:
    def __init__(
        self,
        train_dataset_portion: int = 100,
        validation_dataset_portion: int = 100,
    ):
        self.train_dataset_portion = train_dataset_portion
        self.validation_dataset_portion = validation_dataset_portion

        self.train_dataset, self.validation_dataset = datasets.load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=[
                f"train[:{self.train_dataset_portion}%]",
                f"validation[:{self.validation_dataset_portion}%]",
            ],
        )

    def iter_train(self) -> Generator[str, None, None]:
        return self._iter_articles(self.train_dataset)

    def iter_validation(self) -> Generator[str, None, None]:
        return self._iter_articles(self.validation_dataset)

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
        current_article = []

        for sample in dataset:
            text = sample["text"]

            # Check if this is an article title (starts with " = " and ends with " = \n")
            if re.match(r"^=[^=]*=$", text.strip()):
                # If we have accumulated content, yield the previous article
                if current_article:
                    article_text = "\n".join(current_article).strip()
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
            article_text = "\n".join(current_article).strip()
            yield f"<|sot|>{article_text}<|eot|>"
