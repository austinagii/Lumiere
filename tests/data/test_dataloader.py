import pytest

from lumiere.data.dataloader import DataLoaderFactory
from lumiere.data.wikitext import WikiText2DataLoader


class TestDataLoaderFactory:
    def test_creates_wikitext_2_dataloader(self):
        dataloader = DataLoaderFactory.get_data_loader("wikitext")

        assert isinstance(dataloader, WikiText2DataLoader)

    def test_raises_error_for_unknown_dataset(self):
        with pytest.raises(ValueError):
            DataLoaderFactory.get_data_loader("openwebtext")
