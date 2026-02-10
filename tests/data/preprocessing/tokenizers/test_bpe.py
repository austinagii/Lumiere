from collections.abc import Generator

import pytest

from lumiere.data import DataLoader
from lumiere.tokenizers import BPETokenizer, SPECIAL_TOKENS, Tokenizer


@pytest.fixture
def tokenizer():
    tokenizer = BPETokenizer(vocab_size=32, min_frequency=1)

    # Train on a small corpus that contains the words we'll test
    corpus = [
        "the quick brown fox jumped over the lazy dog",
        "the lazy dog jumped over the quick fox",
        "brown fox jumped over lazy dog",
    ]
    tokenizer.train(corpus)

    return tokenizer


class TestBpeTokenizer:
    def test_tokenize_returns_a_list_of_tokens(self, tokenizer):
        text = "the quick brown fox jumped the lazy dog"

        actual_output = tokenizer.tokenize(text)

        assert isinstance(actual_output, list)
        assert len(actual_output) > 0
        assert all(isinstance(token, int) for token in actual_output)

    def test_tokenize_recognizes_special_tokens(self, tokenizer):
        text = "<|sot|> the quick dog <|eot|> <|pad|>"

        actual_output = tokenizer.tokenize(text)

        # Tokenizer should successfully process text containing special tokens
        assert isinstance(actual_output, list)
        assert len(actual_output) > 0
        assert all(isinstance(token_id, int) for token_id in actual_output)

    def test_tokenize_returns_an_empty_list_if_the_text_is_empty(self, tokenizer):
        actual_output = tokenizer.tokenize("")

        assert actual_output == []

    def test_tokenize_all_returns_a_nested_list_when_executing_eagerly(self, tokenizer):
        corpus = [
            "the quick dog jumped over the fox",
            "the lazy dog jumped over the fox",
        ]

        actual_output = list(tokenizer.tokenize_all(corpus))

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

        actual_output = tokenizer.tokenize_all(corpus)

        assert isinstance(actual_output, Generator)
        result_list = list(actual_output)
        assert len(result_list) == 2
        assert all(isinstance(tokens, list) for tokens in result_list)
        assert all(len(tokens) > 0 for tokens in result_list)

    def test_tokenize_all_returns_an_empty_output_when_corpus_is_empty_list(
        self, tokenizer
    ):
        corpus = []

        actual_output = tokenizer.tokenize_all(corpus)

        with pytest.raises(StopIteration):
            next(actual_output)

    def test_tokenize_all_returns_empty_outputs_when_corpus_is_empty(self, tokenizer):
        corpus = [""]
        actual_output = tokenizer.tokenize_all(corpus)

        assert list(actual_output) == [[]]

    def test_encode_returns_a_list_of_token_ids(self, tokenizer):
        # encode() expects string tokens, but we can test it with the underlying tokenizer
        # Get actual string tokens from the underlying tokenizer
        text = "the quick dog jumped over the fox"
        encoding = tokenizer.tokenizer.encode(text)
        string_tokens = encoding.tokens

        actual_output = tokenizer.encode(string_tokens)

        assert isinstance(actual_output, list)
        assert len(actual_output) == len(string_tokens)
        assert all(isinstance(token_id, int) for token_id in actual_output)
        assert all(token_id >= 0 for token_id in actual_output)

    def test_encode_all_returns_a_nested_list_of_token_ids_when_executing_eagerly(
        self, tokenizer
    ):
        # Get string tokens from the underlying tokenizer
        text = "the quick dog jumped over the fox"
        encoding = tokenizer.tokenizer.encode(text)
        string_tokens = encoding.tokens
        corpus = [string_tokens]

        actual_output = list(tokenizer.encode_all(corpus))

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
        # Get string tokens from the underlying tokenizer
        text = "the quick dog jumped over the fox"
        encoding = tokenizer.tokenizer.encode(text)
        string_tokens = encoding.tokens
        corpus = [string_tokens]

        actual_output = tokenizer.encode_all(corpus)

        assert isinstance(actual_output, Generator)
        result_list = list(actual_output)
        assert len(result_list) == 1
        assert all(isinstance(token_ids, list) for token_ids in result_list)
        assert all(
            all(isinstance(token_id, int) for token_id in token_ids)
            for token_ids in result_list
        )

    def test_decode_returns_a_string(self, tokenizer):
        # tokenize() returns IDs, decode() converts them back to text
        text = "the quick dog jumped over the fox"
        token_ids = tokenizer.tokenize(text)

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
        token_ids = tokenizer.tokenize_all(text)

        actual_output = list(tokenizer.decode_all(token_ids))

        assert isinstance(actual_output, list)
        assert len(actual_output) == 2
        assert all(isinstance(text, str) for text in actual_output)
        assert all(len(text) > 0 for text in actual_output)

    def test_decode_all_returns_a_generator_of_strings_when_executing_lazily(
        self, tokenizer
    ):
        # tokenize() returns IDs
        text = "the quick dog jumped over the fox"
        token_ids = [tokenizer.tokenize(text)]

        actual_output = tokenizer.decode_all(token_ids)

        assert isinstance(actual_output, Generator)
        result_list = list(actual_output)
        assert len(result_list) == 1
        assert all(isinstance(text, str) for text in result_list)

    def test_can_train_from_a_corpus(self):
        tokenizer = BPETokenizer(vocab_size=32, min_frequency=2)

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

        tokenizer = BPETokenizer(vocab_size=10, min_frequency=2)
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

        tokenizer = BPETokenizer(vocab_size=len(alphabet), min_frequency=2)
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
        tokenizer = BPETokenizer(vocab_size=target_vocab_size, min_frequency=2)
        tokenizer.train(corpus)

        assert tokenizer.vocab_size > target_vocab_size

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("split,vocab_size"),
        [
            ("1", 256),
            ("10", 512),
            ("25", 1024),
            ("50", 2048),
            ("100", 4096),
        ],
    )
    def test_tokenizer_respects_vocab_size(self, split, vocab_size):
        tokenizer = BPETokenizer(vocab_size=vocab_size, min_frequency=2)

        dataloader = DataLoader.from_config([{"name": "wikitext", "split": split}], merge_mode="greedy")

        alphabet = {char for sentence in dataloader["train"] for char in sentence}
        alphabet_size = len(alphabet) + len(SPECIAL_TOKENS)

        tokenizer.train(dataloader["train"])

        # This line is important, if the specified vocab size is less than the alphabet
        # size of the training set, then the vocab size will be exceeded during training
        assert vocab_size >= alphabet_size
        assert tokenizer.vocab_size == vocab_size
