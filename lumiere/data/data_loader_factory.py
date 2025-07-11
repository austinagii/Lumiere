from lumiere.data.data_loader import DataLoader
from lumiere.data.wikitext import WikiText2DataLoader


class DataLoaderFactory:
    @staticmethod
    def get_data_loader(
        dataset_name: str,
        dataset_subset: str = None,
        train_dataset_portion: int = 100,
        validation_dataset_portion: int = 100,
    ) -> DataLoader:
        if dataset_name == "wikitext" and dataset_subset == "wikitext-2-raw-v1":
            return WikiText2DataLoader(
                train_dataset_portion,
                validation_dataset_portion,
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
