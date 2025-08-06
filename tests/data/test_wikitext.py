from lumiere.data.wikitext import WikiText2DataLoader


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


def test_iterates_correctly_over_articles(mocker):
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
