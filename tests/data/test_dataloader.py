import pytest

from lumiere.data.dataloader import get_data_loader
from lumiere.data.wikitext import WikiText2DataLoader


class TestDataLoaderFactory:
    def test_creates_wikitext_2_dataloader(self, mocker):
        mocker.patch(
            "datasets.load_dataset",
            return_value=(mocker.Mock(), mocker.Mock()),
        )

        dataloader = get_data_loader("wikitext")

        assert isinstance(dataloader, WikiText2DataLoader)

    def test_raises_error_for_unknown_dataset(self):
        with pytest.raises(ValueError):
            get_data_loader("openwebtext")

    @pytest.mark.integration
    def test_dataset_portion_is_respected(self):
        tiny_dataloader = get_data_loader("wikitext", train_dataset_percentage=1)
        large_dataloader = get_data_loader("wikitext", train_dataset_percentage=100)

        relative_size = len(list(tiny_dataloader.iter_train())) / len(
            list(large_dataloader.iter_train())
        )

        # Allow 5% error to account for any rounding that would've been done when
        # calculating 1% of the original dataset.
        assert (
            round(relative_size * 100, 4) <= 1 and round(relative_size * 100, 4) >= 0.95
        )
