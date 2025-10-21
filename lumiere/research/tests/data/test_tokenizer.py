from typing import Generator

import pytest

from lumiere.research.src.data.dataloader import get_data_loader
from lumiere.research.src.data.tokenizer import SPECIAL_TOKENS, Tokenizer


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(vocab_size=32, min_frequency=1)

    # Train on a small corpus that contains the words we'll test
    corpus = [
        "the quick brown fox jumped over the lazy dog",
        "the lazy dog jumped over the quick fox",
        "brown fox jumped over lazy dog",
    ]
    tokenizer.train(corpus)

    return tokenizer


class TestTokenizer:
    def test_tokenize_returns_a_list_of_tokens(self, tokenizer):
        text = "the quick brown fox jumped the lazy dog"

        actual_output = tokenizer.tokenize(text)

        assert isinstance(actual_output, list)
        assert len(actual_output) > 0
        assert all(isinstance(token, str) for token in actual_output)

    def test_tokenize_recognizes_special_tokens(self, tokenizer):
        text = "<|sot|> the quick dog <|eot|> <|pad|>"

        actual_output = tokenizer.tokenize(text)

        # Should contain special tokens as separate tokens
        tokens_str = " ".join(actual_output)
        assert "<|sot|>" in tokens_str
        assert "<|eot|>" in tokens_str
        assert "<|pad|>" in tokens_str

    def test_tokenize_returns_an_empty_list_if_the_text_is_empty(self, tokenizer):
        actual_output = tokenizer.tokenize("")

        assert actual_output == []

    def test_tokenize_all_returns_a_nested_list_when_executing_eagerly(self, tokenizer):
        corpus = [
            "the quick dog jumped over the fox",
            "the lazy dog jumped over the fox",
        ]

        actual_output = tokenizer.tokenize_all(corpus)

        assert isinstance(actual_output, list)
        assert len(actual_output) == 2
        assert all(isinstance(tokens, list) for tokens in actual_output)
        assert all(len(tokens) > 0 for tokens in actual_output)

    def test_tokenize_all_returns_a_generator_of_tokens_when_executing_lazily(
        self, tokenizer
    ):
        corpus = [
            "the quick dog jumped over the fox",
            "the lazy dog jumped over the fox",
        ]

        actual_output = tokenizer.tokenize_all(corpus, lazy=True)

        assert isinstance(actual_output, Generator)
        result_list = list(actual_output)
        assert len(result_list) == 2
        assert all(isinstance(tokens, list) for tokens in result_list)
        assert all(len(tokens) > 0 for tokens in result_list)

    def test_tokenize_all_returns_an_empty_output_when_corpus_is_empty_list(
        self, tokenizer
    ):
        corpus = []

        actual_output = tokenizer.tokenize_all(corpus, lazy=True)

        with pytest.raises(StopIteration):
            next(actual_output)

        actual_output = tokenizer.tokenize_all(corpus, lazy=False)

        assert actual_output == []

    def test_tokenize_all_returns_empty_outputs_when_corpus_is_empty(self, tokenizer):
        corpus = [""]
        actual_output = tokenizer.tokenize_all(corpus, lazy=True)

        assert list(actual_output) == [[]]

    def test_encode_returns_a_list_of_token_ids(self, tokenizer):
        # First tokenize to get actual tokens from trained tokenizer
        text = "the quick dog jumped over the fox"
        tokens = tokenizer.tokenize(text)

        actual_output = tokenizer.encode(tokens)

        assert isinstance(actual_output, list)
        assert len(actual_output) <= len(tokens)
        assert all(isinstance(token_id, int) for token_id in actual_output)
        assert all(token_id >= 0 for token_id in actual_output)

    def test_encode_all_returns_a_nested_list_of_token_ids_when_executing_eagerly(
        self, tokenizer
    ):
        # First tokenize to get actual tokens
        text = "the quick dog jumped over the fox"
        tokens = tokenizer.tokenize(text)
        corpus = [tokens]

        actual_output = tokenizer.encode_all(corpus)

        assert isinstance(actual_output, list)
        assert len(actual_output) == 1
        assert all(isinstance(token_ids, list) for token_ids in actual_output)
        assert all(
            all(isinstance(token_id, int) for token_id in token_ids)
            for token_ids in actual_output
        )

    def test_encode_all_returns_a_generator_of_token_ids_when_executing_lazily(
        self, tokenizer
    ):
        # First tokenize to get actual tokens
        text = "the quick dog jumped over the fox"
        tokens = tokenizer.tokenize(text)
        corpus = [tokens]

        actual_output = tokenizer.encode_all(corpus, lazy=True)

        assert isinstance(actual_output, Generator)
        result_list = list(actual_output)
        assert len(result_list) == 1
        assert all(isinstance(token_ids, list) for token_ids in result_list)
        assert all(
            all(isinstance(token_id, int) for token_id in token_ids)
            for token_ids in result_list
        )

    def test_decode_returns_a_string(self, tokenizer):
        # First tokenize then encode to get valid IDs
        text = "the quick dog jumped over the fox"
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(tokens)

        actual_output = tokenizer.decode(token_ids)

        assert isinstance(actual_output, str)
        assert len(actual_output) > 0
        assert actual_output.strip() == text.strip()

    def test_decode_all_returns_a_nested_list_of_strings_when_executing_eagerly(
        self, tokenizer
    ):
        text = [
            "the quick dog jumped over the fox",
            "the lazy dog jumped over the fox",
        ]
        tokens = tokenizer.tokenize_all(text)
        token_ids = tokenizer.encode_all(tokens)

        actual_output = tokenizer.decode_all(token_ids)

        assert isinstance(actual_output, list)
        assert len(actual_output) == 2
        assert all(isinstance(text, str) for text in actual_output)
        assert all(len(text) > 0 for text in actual_output)

    def test_decode_all_returns_a_generator_of_strings_when_executing_lazily(
        self, tokenizer
    ):
        # First tokenize then encode to get valid IDs
        text = "the quick dog jumped over the fox"
        tokens = tokenizer.tokenize(text)
        token_ids = [tokenizer.encode(tokens)]

        actual_output = tokenizer.decode_all(token_ids, lazy=True)

        assert isinstance(actual_output, Generator)
        result_list = list(actual_output)
        assert len(result_list) == 1
        assert all(isinstance(text, str) for text in result_list)

    def test_can_train_from_a_corpus(self):
        tokenizer = Tokenizer(vocab_size=32, min_frequency=2)

        corpus = [
            "the quick dog jumped over the fox",
            "the lazy dog jumped over the fox",
        ]

        tokenizer.train(corpus)

        assert tokenizer.vocab_size == 32

    def test_min_vocab_size_is_alphabet_size_plus_special_tokens(self):
        corpus = [
            "this is a test corpus",
            "it contains a lot of words",
            "some of which are repeated",
        ]  # 18 total characters

        alphabet = set([char for sentence in corpus for char in sentence])
        alphabet.update(set([token.token for token in SPECIAL_TOKENS.values()]))

        tokenizer = Tokenizer(vocab_size=10, min_frequency=2)
        tokenizer.train(corpus)

        assert tokenizer.vocab_size == len(alphabet)

    def test_tokenizer_respects_vocab_size_during_training(self):
        corpus = [
            "this is a test corpus",
            "it contains a lot of words",
            "some of which are repeated",
        ]  # 18 total characters

        # Create the alphabet of the unique characters in the corpus, plus the special
        # tokens
        alphabet = set([char for sentence in corpus for char in sentence])
        alphabet.update(set([token.token for token in SPECIAL_TOKENS.values()]))

        tokenizer = Tokenizer(vocab_size=len(alphabet), min_frequency=2)
        tokenizer.train(corpus)

        assert tokenizer.vocab_size == len(alphabet)

    def test_vocab_size_exceeds_alphabet_size_after_training(self):
        corpus = [
            "this is a test corpus",
            "it contains a lot of words",
            "some of which are repeated",
        ]  # 18 total characters

        # Create the alphabet of the unique characters in the corpus, plus the special
        # tokens
        alphabet = set([char for sentence in corpus for char in sentence])
        alphabet.update(set([token.token for token in SPECIAL_TOKENS.values()]))

        target_vocab_size = len(alphabet) - 5
        tokenizer = Tokenizer(vocab_size=target_vocab_size, min_frequency=2)
        tokenizer.train(corpus)

        assert tokenizer.vocab_size > target_vocab_size

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("data_size", "vocab_size"),
        [
            (1, 256),
            (10, 512),
            (25, 1024),
            (50, 2048),
            (100, 4096),
        ],
    )
    def test_tokenizer_respects_vocab_size(self, data_size, vocab_size):
        tokenizer = Tokenizer(vocab_size=vocab_size, min_frequency=2)

        dataloader = get_data_loader("wikitext", train_dataset_percentage=data_size)

        alphabet = set(
            [char for sentence in dataloader.iter_train() for char in sentence]
        )
        alphabet_size = len(alphabet) + len(SPECIAL_TOKENS)

        tokenizer.train(dataloader.iter_train())

        # This line is important, if the specified vocab size is less than the alphabet
        # size of the training set, then the vocab size will be exceeded during training
        assert vocab_size >= alphabet_size
        assert tokenizer.vocab_size == vocab_size
