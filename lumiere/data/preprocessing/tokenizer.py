from collections import OrderedDict
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SpecialToken:
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
    def tokenize(self, text: str, to_ids: bool = False) -> list[str] | list[int]:
        """Tokenizes the specified text.

        By default, each token will be returned as it's string representation from the
        original text. If `to_ids` is `True`, then the integer ID of each token is
        returned instead.
        """
        ...

    def tokenize_all(
        self, corpus: Iterable[str], lazy: bool = False, to_ids: bool = False
    ) -> list[list[str]] | Generator[list[str], None, None]:
        """Tokenizes the specified text.

        When `lazy` is `True`, a generator of lists of tokens is returned.
        Otherwise, a list of lists of tokens is returned. If `to_ids` is `True`,
        the tokens are converted to their corresponding IDs.
        """
        ...

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a sequence of tokens to their integer IDs."""
        ...

    def encode_all(
        self, corpus: Sequence[list[str]], lazy: bool = False
    ) -> list[list[int]] | Generator[list[int], None, None]:
        """Convert a nested sequence of tokens to sequences of their integer IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Conver a sequence of toekn IDs to their string representation."""
        ...

    def decode_all(
        self, corpus: Sequence[list[int]], lazy: bool = False
    ) -> list[str] | Generator[str, None, None]:
        """Convert a nested sequence of token ids to their string representations."""
        ...

    def train(self, corpus: Iterable[str]):
        """Train the tokenizer on a sequence of strings."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get the vocab size of the tokenizer."""
        ...

    @classmethod
    def from_bytes(cls, bytes: bytes, *args, **kwargs):
        """Create a tokenizer from its byte representation."""
        ...

    def __bytes__(self):
        """Convert this tokenizer to its byte representation."""
        ...
