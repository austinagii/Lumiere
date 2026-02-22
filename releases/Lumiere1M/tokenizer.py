from collections.abc import Generator, Iterable

import tokenizers
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from lumiere.internal.registry import discover
from lumiere.tokenizers import SPECIAL_TOKENS, Serializable, Tokenizer


@discover(Tokenizer, "bpe")
class BPETokenizer(Tokenizer, Serializable):
    """Byte Pair Encoding (BPE) tokenizer.

    A subword tokenization algorithm that learns a vocabulary by iteratively merging
    the most frequent pairs of bytes in a training corpus. Uses HuggingFace's
    `tokenizers` library with byte-level pre-tokenization and NFKC normalization.

    Args:
        vocab_size: The maximum size of the vocabulary. Defaults to `30000`.
        min_frequency: The minimum frequency a byte pair must have to be included
            in the vocabulary. Defaults to `2`.
    """

    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2):
        self.tokenizer = tokenizers.Tokenizer(models.BPE())
        self.tokenizer.normalizer = normalizers.NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=False,
            special_tokens=[token.token for token in SPECIAL_TOKENS.values()],
        )
        self.tokenizer.add_special_tokens(
            [token.token for token in SPECIAL_TOKENS.values()]
        )

    def tokenize(self, text: str) -> list[int]:
        """Tokenize a single text string into token IDs.

        Args:
            text: The text string to tokenize.

        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text).ids

    def tokenize_all(self, corpus: Iterable[str]) -> Generator[list[int], None, None]:
        """Tokenize multiple text strings into token IDs.

        Args:
            corpus: Iterable of text strings to tokenize.

        Returns:
            Generator yielding lists of token IDs for each text.
        """
        return (self.tokenize(text) for text in corpus)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Convert token strings to their corresponding token IDs.

        Args:
            tokens: Iterable of token strings.

        Returns:
            List of token IDs.
        """
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def encode_all(
        self, corpus: Iterable[Iterable[str]]
    ) -> Generator[list[int], None, None]:
        """Convert multiple sequences of token strings to token IDs.

        Args:
            corpus: Iterable of iterables containing token strings.

        Returns:
            Generator yielding lists of token IDs for each sequence.
        """
        return (
            [self.tokenizer.token_to_id(token) for token in text] for text in corpus
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back to a text string.

        Args:
            token_ids: Iterable of token IDs.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(list(token_ids))

    def decode_all(self, corpus: Iterable[Iterable[int]]) -> Generator[str, None, None]:
        """Convert multiple sequences of token IDs back to text strings.

        Args:
            corpus: Iterable of iterables containing token IDs.

        Returns:
            Generator yielding decoded text strings for each sequence.
        """
        return (self.decode(ids) for ids in corpus)

    def train(self, corpus: Iterable[str]):
        """Train the tokenizer on a text corpus.

        Learns the BPE vocabulary from the provided corpus using the configured
        vocabulary size and minimum frequency parameters.

        Args:
            corpus: Iterable of text strings to train on.

        Returns:
            Self for method chaining.
        """
        self.tokenizer.train_from_iterator(corpus, self.trainer)
        return self

    @property
    def vocab_size(self) -> int:
        """The size of the tokenizer's vocabulary.

        Returns:
            Number of tokens in the vocabulary including special tokens.
        """
        return self.tokenizer.get_vocab_size()

    @classmethod
    def from_bytes(cls, bytes: bytes, *args, **kwargs):
        """Deserialize a tokenizer from bytes.

        Args:
            bytes: Serialized tokenizer bytes.
            *args: Additional positional arguments for tokenizer initialization.
            **kwargs: Additional keyword arguments for tokenizer initialization.

        Returns:
            Deserialized `BPETokenizer` instance.
        """
        tokenizer = cls(*args, **kwargs)
        tokenizer.tokenizer = tokenizers.Tokenizer.from_str(bytes.decode("utf-8"))
        return tokenizer

    def __bytes__(self):
        """Serialize the tokenizer to bytes.

        Returns:
            Serialized tokenizer as bytes.
        """
        return bytes(self.tokenizer.to_str(), "utf-8")
