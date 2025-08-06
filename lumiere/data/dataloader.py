from typing import Generator, Protocol

from lumiere.data.wikitext import WikiText2DataLoader


class DataLoader(Protocol):
    def iter_train(self) -> Generator[str, None, None]: ...

    def iter_validation(self) -> Generator[str, None, None]: ...


class DataLoaderFactory:
    @staticmethod
    def get_data_loader(
        dataset_name: str,
        train_dataset_portion: int = 100,
        validation_dataset_portion: int = 100,
    ) -> DataLoader:
        if dataset_name == "wikitext":
            return WikiText2DataLoader(
                train_dataset_portion,
                validation_dataset_portion,
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
