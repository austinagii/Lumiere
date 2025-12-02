from collections.abc import Iterator

import datasets
import pytest

from lumiere.research.src.data.datasets.wikitext import WikiText2Dataset


@pytest.fixture
def mock_wikitext(mocker) -> datasets.Dataset:
    """Return a small dataset of truncated samples from the wikitext 2 dataset."""
    data = [
        "",
        " = Valkyria Chronicles III = \n",
        "",
        " Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルュ . \n",
        " The game began development in 2010 , carrying over a large portion of . \n",
        "",
        " = = Gameplay = = \n",
        "",
        " As with previous Valkyira Chronicles games , Valkyria Chronicles III is . \n",
        "",
        " = = = Release = = = \n",
        "",
        " In September 2010 , a teaser website was revealed by Sega . \n",
        " Unlike its two predecessors , Valkyria Chronicles III was not released . \n",
        "",
        "",
        " = Tower Building of the Little Rock Arsenal = \n",
        "",
        " The Tower Building of the Little Rock Arsenal , also known as U.S. . \n",
        " The building receives its name from its distinct octagonal tower . \n",
        "",
        " = = Construction = = \n",
        "",
        " The arsenal was constructed at the request of Governor James Sevier . \n",
        "",
        " = = Civil War = = \n",
        "",
        " For several years the arsenal , which was owned by the federal . \n",
        " The United States troops at the outposts of the western frontier . \n",
        "",
    ]

    return datasets.Dataset.from_dict({"text": data})


class TestWikitText2wikitext:
    # ==============================================
    # ============ TEST INITIALIZATION =============
    # ==============================================

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "split,expected_args",
        [
            ("20:50:30", ["train[:20%]", "validation[:50%]", "test[:30%]"]),
            ("0:50:30", ["validation[:50%]", "test[:30%]"]),
            ("20:0:30", ["train[:20%]", "test[:30%]"]),
            ("20:50:0", ["train[:20%]", "validation[:50%]"]),
            ("::30", ["train[:100%]", "validation[:100%]", "test[:30%]"]),
            (":30", ["train[:100%]", "validation[:30%]", "test[:100%]"]),
            ("20", ["train[:20%]", "validation[:100%]", "test[:100%]"]),
        ],
    )
    def test_init_loads_specified_splits(self, mocker, split, expected_args):
        spy = mocker.patch("datasets.load_dataset", wraps=datasets.load_dataset)

        WikiText2Dataset(split)

        spy.assert_called_once()

        call_args, call_kwargs = spy.call_args
        assert "Salesforce/wikitext" in call_args
        assert "wikitext-2-raw-v1" in call_args
        assert call_kwargs.get("split") == expected_args

    @pytest.mark.slow
    def test_init_loads_full_splits_by_default(self, mocker):
        spy = mocker.patch("datasets.load_dataset", wraps=datasets.load_dataset)

        WikiText2Dataset()

        spy.assert_called_once()

        call_args, call_kwargs = spy.call_args
        assert call_kwargs.get("split") == [
            "train[:100%]",
            "validation[:100%]",
            "test[:100%]",
        ]

    def test_init_raises_error_if_all_splits_are_empty(self):
        with pytest.raises(ValueError):
            WikiText2Dataset("0:0:0")

    # ==============================================
    # ============ TEST SPLIT ACCESS ===============
    # ==============================================

    @pytest.mark.slow
    def test_getitem_provides_an_iterator_over_the_specified_split(self, mocker):
        wikitext = WikiText2Dataset()

        # By default, all three splits should be available.
        assert isinstance((train_data := wikitext["train"]), Iterator)
        assert isinstance((validation_data := wikitext["validation"]), Iterator)
        assert isinstance((test_data := wikitext["test"]), Iterator)

        # Check for the key terms in the first articles of each split.
        assert "Valkyria Chronicles" in next(train_data)
        assert "Homarus gammarus" in next(validation_data)
        assert "Robert Boulter" in next(test_data)

    def test_getitem_raises_error_if_split_name_is_invalid(self, mocker):
        mock_load = mocker.patch(
            "lumiere.research.src.data.datasets.wikitext.datasets.load_dataset"
        )
        mock_load.return_value = [mocker.Mock(), mocker.Mock(), mocker.Mock()]

        wikitext = WikiText2Dataset()

        with pytest.raises(KeyError):
            wikitext["invalid_split"]

    def test_getitem_produces_an_iterator_over_wikitext_articles(
        self, mocker, mock_wikitext
    ):
        mocker.patch(
            "datasets.load_dataset",
            return_value=(mock_wikitext, mock_wikitext, mock_wikitext),
        )

        wikitext = WikiText2Dataset()
        articles = list(wikitext["train"])
        assert len(articles) == 2

        assert articles[0] == (
            "<|sot|> Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルュ . \n"  # noqa: E501
            + " The game began development in 2010 , carrying over a large portion of . \n"  # noqa: E501
            + " As with previous Valkyira Chronicles games , Valkyria Chronicles III is . \n"  # noqa: E501
            + " In September 2010 , a teaser website was revealed by Sega . \n"
            + " Unlike its two predecessors , Valkyria Chronicles III was not released . \n<|eot|>"  # noqa: E501
        )

        assert articles[1] == (
            "<|sot|> The Tower Building of the Little Rock Arsenal , also known as U.S. . \n"  # noqa: E501
            + " The building receives its name from its distinct octagonal tower . \n"
            + " The arsenal was constructed at the request of Governor James Sevier . \n"  # noqa: E501
            + " For several years the arsenal , which was owned by the federal . \n"
            + " The United States troops at the outposts of the western frontier . \n<|eot|>"  # noqa: E501
        )
