import pytest

from lumiere.utils.testing.tokenizers import AsciiTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return AsciiTokenizer()


class TestAsciiTokenizer:
    def test_tokenize_returns_ascii_codes_for_each_character(self, tokenizer):
        text = "Hello test!"
        expected_output = [72, 101, 108, 108, 111, 32, 116, 101, 115, 116, 33]

        assert tokenizer.tokenize(text) == expected_output

    def test_tokenize_returns_sentinel_for_non_ascii_characters(self, tokenizer):
        text = "Hello Ā!"
        expected_output = [72, 101, 108, 108, 111, 32, -1, 33]

        assert tokenizer.tokenize(text) == expected_output

    def test_tokenize_all_returns_ascii_codes_for_tokens_in_all_sequences(
        self, tokenizer
    ):
        corpus = ["Natsuki Subaru", "Emilia Tan"]
        expected_output = [
            [78, 97, 116, 115, 117, 107, 105, 32, 83, 117, 98, 97, 114, 117],
            [69, 109, 105, 108, 105, 97, 32, 84, 97, 110],
        ]

        assert list(tokenizer.tokenize_all(corpus)) == expected_output

    def test_decode_converts_token_ascii_codes_to_characters(self, tokenizer):
        token_ids = [72, 101, 108, 108, 111, 32, 116, 101, 115, 116, 33]
        expected_output = "Hello test!"

        assert tokenizer.decode(token_ids) == expected_output

    def test_decode_converts_invalid_token_ids_to_replacement_character(
        self, tokenizer
    ):
        token_ids = [
            [78, 97, 116, 115, 117, 107, 105, 32, 83, 117, 98, 97, 114, 117],
            [69, 109, 105, 108, 105, 97, 32, 84, 97, 110],
        ]
        expected_output = ["Natsuki Subaru", "Emilia Tan"]

        assert list(tokenizer.decode_all(token_ids)) == expected_output
