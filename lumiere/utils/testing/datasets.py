from lumiere.data.dataset import dataset


@dataset("lorem-ipsum")
class LoremIpsumDataset:
    """A toy dataset containing Lorem Ipsum."""

    data = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Aliquam erat volutpat. Vivamus eu magna sem.",
        "Morbi nec finibus justo.",
        "Aliquam id feugiat nulla, malesuada euismod libero.",
        "Curabitur id massa nibh.",
    ]

    def __init__(self, source: str, count: int = 10):
        self.source = source
        self.count = count

    def __getitem__(self, split_name):
        def _get_split():
            match split_name:
                case "train":
                    yield from self.data
                case "validation":
                    yield from self.data[1:]
                case _:
                    return

        if split_name not in ["train", "validation"]:
            raise KeyError(f"Split '{split_name}' not found")

        return _get_split()


@dataset("famous-quotes")
class FamousQuotesDataset:
    """A toy dataset containing famous quotes."""

    data = [
        "Be yourself; everyone else is already taken.",
        "In three words I can sum up everything I've learned about life: it goes on.",
        "The only way to do great work is to love what you do.",
        "It is during our darkest moments that we must focus to see the light.",
        "The only impossible journey is the one you never begin.",
    ]

    def __init__(self, tone: str, topics: list[str]):
        self.tone = tone
        self.topics = topics

    def __getitem__(self, split_name):
        def _get_split():
            match split_name:
                case "train":
                    yield from self.data
                case "test":
                    yield from self.data[2:]
                case _:
                    return

        if split_name not in ["train", "test"]:
            raise KeyError(f"Split '{split_name}' not found")

        return _get_split()
