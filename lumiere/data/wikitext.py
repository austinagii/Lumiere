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
        return (sample["text"] for sample in self.train_dataset)

    def iter_validation(self) -> Generator[str, None, None]:
        return (sample["text"] for sample in self.validation_dataset)
