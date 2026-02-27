from collections import OrderedDict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class SpecialToken:
    """A special token with an ID and string representation.

    Attributes:
        id: The unique integer identifier for this special token.
        token: The string representation of the special token.
    """

    id: int
    token: str


SPECIAL_TOKENS = OrderedDict(
    {
        "start_of_text": SpecialToken(0, "<|sot|>"),
        "end_of_text": SpecialToken(1, "<|eot|>"),
        "padding": SpecialToken(2, "<|pad|>"),
    }
)


class Tokenizer(Protocol):
    """Protocol defining the interface for tokenizer implementations.

    Tokenizers convert text strings into sequences of token IDs that can be
    processed by language models. They support both encoding (text to IDs)
    and decoding (IDs to text).
    """

    def tokenize(self, text: str) -> list[int]:
        """Tokenize a text string into a sequence of token IDs.

        Args:
            text: The text string to tokenize.

        Returns:
            List of token IDs representing the tokenized text.
        """
        ...

    def tokenize_all(self, corpus: Iterable[str]) -> Generator[list[int], None, None]:
        """Tokenize multiple text strings into sequences of token IDs.

        Args:
            corpus: Iterable of text strings to tokenize.

        Returns:
            Generator yielding lists of token IDs for each text.
        """
        ...

    # TODO: Consider removing these 'encoding' methods.
    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Convert a sequence of token strings to their integer IDs.

        Args:
            tokens: Iterable of token strings.

        Returns:
            List of token IDs.
        """
        ...

    def encode_all(
        self, corpus: Iterable[Iterable[str]]
    ) -> Generator[list[int], None, None]:
        """Convert multiple sequences of token strings to sequences of integer IDs.

        Args:
            corpus: Iterable of iterables containing token strings.

        Returns:
            Generator yielding lists of token IDs for each sequence.
        """
        ...

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert a sequence of token IDs to their string representation.

        Args:
            token_ids: Iterable of token IDs.

        Returns:
            Decoded text string.
        """
        ...

    def decode_all(self, corpus: Iterable[Iterable[int]]) -> Generator[str, None, None]:
        """Convert multiple sequences of token IDs to their string representations.

        Args:
            corpus: Iterable of iterables containing token IDs.

        Returns:
            Generator yielding decoded text strings for each sequence.
        """
        ...

    def train(self, corpus: Iterable[str]):
        """Train the tokenizer on a text corpus.

        Args:
            corpus: Iterable of text strings to train on.

        Returns:
            Self for method chaining.
        """
        ...

    def load_state(self, path: str | Path) -> None:
        """Load the tokenizer's state from the specified path.

        Args:
            path: The path to artifact(s) containing the tokenizer state.

        Raises:
            FileNotFoundError: If the specified artifact could not be found.
        """
        ...

    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer.

        Returns:
            Number of tokens in the vocabulary including special tokens.
        """
        ...


class Serializable(Protocol):
    """Protocol for objects that can be serialized to and from bytes.

    Enables saving and loading tokenizers from disk or storage.
    """

    def from_bytes(cls, bytes: bytes, *args, **kwargs):
        """Deserialize an object from its byte representation.

        Args:
            bytes: Serialized object bytes.
            *args: Additional positional arguments for initialization.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            Deserialized object instance.
        """
        ...

    def __bytes__(self):
        """Serialize this object to its byte representation.

        Returns:
            Serialized object as bytes.
        """
        ...


class Trainable(Protocol):
    """Protocol for objects that can be trained on datasets.

    Training typically involves learning parameters or vocabularies from data.
    """

    def train(self, dataset: Iterable[Any]) -> None:
        """Train the object on the specified dataset.

        Args:
            dataset: Iterable of training samples.
        """
        pass
