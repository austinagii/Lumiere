from typing import Any

import pytest

from lumiere.data import DataLoader
from lumiere.testing.datasets import FamousQuotesDataset, LoremIpsumDataset


@pytest.fixture
def dataset_config() -> dict[str, Any]:
    """A config for initializing the LoremIpsum and Famous Quotes dataset."""
    return {
        "datasets": [
            {
                "name": "lorem-ipsum",
                "source": "Invitica Morialis Aeturnus",
                "count": 12,
            },
            {
                "name": "famous-quotes",
                "tone": "Inspirational",
                "topics": ["Success", "Career", "Adventure"],
            },
        ]
    }


class TestDataLoader:
    """A suite of tests for the `DataLoader` class."""

    # ===========================================
    # ========== TEST INITIALIZATION ============
    # ===========================================

    def test_init_initializes_datasets_according_to_provided_args(self, dataset_config):
        dataloader = DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

        assert isinstance(dataloader.datasets[0], LoremIpsumDataset)
        assert dataloader.datasets[0].source == "Invitica Morialis Aeturnus"
        assert dataloader.datasets[0].count == 12

        assert isinstance(dataloader.datasets[1], FamousQuotesDataset)
        assert dataloader.datasets[1].tone == "Inspirational"
        assert dataloader.datasets[1].topics == ["Success", "Career", "Adventure"]

    def test_init_raises_an_error_if_specified_dataset_is_not_registered(
        self, dataset_config
    ):
        dataset_config["datasets"][1]["name"] = "arc"

        with pytest.raises(ValueError):
            DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

    def test_init_raises_error_if_invalid_dataset_arg_is_provided(self, dataset_config):
        dataset_config["datasets"][0]["invalid_args"] = "value"

        with pytest.raises(RuntimeError):
            DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

    def test_init_raises_error_if_dataset_could_not_be_initialized(
        self, mocker, dataset_config
    ):
        mocker.patch(
            "tests.data.test_dataset.FamousQuotesDataset.__init__",
            side_effect=FileNotFoundError("A random error occurred."),
        )

        with pytest.raises(RuntimeError):
            DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

    def test_init_raises_an_error_if_specified_merge_mode_does_not_exist(
        self, dataset_config
    ):
        with pytest.raises(ValueError):
            DataLoader.from_config(dataset_config["datasets"], merge_mode="hyperbolic")

    # ===========================================
    # =========== TEST SPLIT ACCESS =============
    # ===========================================

    def test_iteritem_uses_greedy_merge_mode_by_default(self, dataset_config):
        dataloader = DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

        assert list(dataloader["train"]) == [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Aliquam erat volutpat. Vivamus eu magna sem.",
            "Morbi nec finibus justo.",
            "Aliquam id feugiat nulla, malesuada euismod libero.",
            "Curabitur id massa nibh.",
            "Be yourself; everyone else is already taken.",
            "In three words I can sum up everything I've learned about life: it goes on.",
            "The only way to do great work is to love what you do.",
            "It is during our darkest moments that we must focus to see the light.",
            "The only impossible journey is the one you never begin.",
        ]

    def test_iteritem_uses_specified_merge_mode_if_specified(self, dataset_config):
        greedy_dataloader = DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

        assert list(greedy_dataloader["train"]) == [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Aliquam erat volutpat. Vivamus eu magna sem.",
            "Morbi nec finibus justo.",
            "Aliquam id feugiat nulla, malesuada euismod libero.",
            "Curabitur id massa nibh.",
            "Be yourself; everyone else is already taken.",
            "In three words I can sum up everything I've learned about life: it goes on.",
            "The only way to do great work is to love what you do.",
            "It is during our darkest moments that we must focus to see the light.",
            "The only impossible journey is the one you never begin.",
        ]

        circular_dataloader = DataLoader.from_config(dataset_config["datasets"], merge_mode="circular")

        assert list(circular_dataloader["train"]) == [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Be yourself; everyone else is already taken.",
            "Aliquam erat volutpat. Vivamus eu magna sem.",
            "In three words I can sum up everything I've learned about life: it goes on.",
            "Morbi nec finibus justo.",
            "The only way to do great work is to love what you do.",
            "Aliquam id feugiat nulla, malesuada euismod libero.",
            "It is during our darkest moments that we must focus to see the light.",
            "Curabitur id massa nibh.",
            "The only impossible journey is the one you never begin.",
        ]

    def test_getitem_excludes_a_datasets_not_containing_specified_split(
        self, dataset_config
    ):
        dataloader = DataLoader.from_config(dataset_config["datasets"], merge_mode="greedy")

        assert list(dataloader["validation"]) == [
            "Aliquam erat volutpat. Vivamus eu magna sem.",
            "Morbi nec finibus justo.",
            "Aliquam id feugiat nulla, malesuada euismod libero.",
            "Curabitur id massa nibh.",
        ]
