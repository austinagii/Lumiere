import pytest

from lumiere.data.dataloader import WikiText2DataLoader, get_data_loader


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


data = [
    {"text": ""},
    {"text": " = A Test Article = \n"},
    {"text": ""},
    {"text": " This article is a test article.\n"},
    {"text": " It's goal is to test the wikitext dataloader.\n"},
    {"text": ""},
    {"text": " = = Subsection = = \n"},
    {"text": ""},
    {"text": " This is a subsection of the test article.\n"},
    {"text": ""},
    {"text": " = = = Another Subsection = = = \n"},
    {"text": ""},
    {"text": " This is another subsection of the test article.\n"},
    {"text": " This is the end of the test article.\n"},
    {"text": ""},
    {"text": ""},
    {"text": " = A Second Test Article = \n"},
    {"text": ""},
    {"text": " This is a second test article.\n"},
    {"text": ""},
    {"text": " = = Subsection = = \n"},
    {"text": ""},
    {"text": " This is a subsection of the second test article.\n"},
    {"text": ""},
    {"text": " = = = Another Subsection = = = \n"},
    {"text": ""},
    {"text": " This is another subsection of the second test article.\n"},
    {"text": ""},
]


class TestWikitextDataLoader:
    def test_iterates_correctly_over_articles(self, mocker):
        mocker.patch("datasets.load_dataset", return_value=(data, data))

        dataloader = WikiText2DataLoader()

        articles = list(dataloader.iter_train())

        assert len(articles) == 2

        first_article = (
            "<|sot|> = A Test Article = \n"
            + " This article is a test article.\n"
            + " It's goal is to test the wikitext dataloader.\n"
            + " = = Subsection = = \n"
            + " This is a subsection of the test article.\n"
            + " = = = Another Subsection = = = \n"
            + " This is another subsection of the test article.\n"
            + " This is the end of the test article.\n"
            + "<|eot|>"
        )

        second_article = (
            "<|sot|> = A Second Test Article = \n"
            + " This is a second test article.\n"
            + " = = Subsection = = \n"
            + " This is a subsection of the second test article.\n"
            + " = = = Another Subsection = = = \n"
            + " This is another subsection of the second test article.\n"
            + "<|eot|>"
        )

        assert articles[0] == first_article
        assert articles[1] == second_article
