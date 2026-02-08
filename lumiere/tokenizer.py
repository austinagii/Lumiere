import contextlib
import importlib
from collections import OrderedDict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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
    def tokenize(self, text: str) -> list[int]:
        """Tokenizes the specified text.

        By default, each token will be returned as it's string representation from the
        original text. If `to_ids` is `True`, then the integer ID of each token is
        returned instead.
        """
        ...

    def tokenize_all(self, corpus: Iterable[str]) -> Generator[list[int], None, None]:
        """Tokenizes the specified text.

        When `lazy` is `True`, a generator of lists of tokens is returned.
        Otherwise, a list of lists of tokens is returned. If `to_ids` is `True`,
        the tokens are converted to their corresponding IDs.
        """
        ...

    # TODO: Consider removing these 'encoding' methods.
    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Convert a sequence of tokens to their integer IDs."""
        ...

    def encode_all(
        self, corpus: Iterable[Iterable[str]]
    ) -> Generator[list[int], None, None]:
        """Convert a nested sequence of tokens to sequences of their integer IDs."""
        ...

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert a sequence of toekn IDs to their string representation."""
        ...

    def decode_all(self, corpus: Iterable[Iterable[int]]) -> Generator[str, None, None]:
        """Convert a nested sequence of token ids to their string representations."""
        ...

    def train(self, corpus: Iterable[str]):
        """Train the tokenizer on a sequence of strings."""
        ...

    def vocab_size(self) -> int:
        """Get the vocab size of the tokenizer."""
        ...


class Serializable(Protocol):
    def from_bytes(cls, bytes: bytes, *args, **kwargs):
        """Create a tokenizer from its byte representation."""
        ...

    def __bytes__(self):
        """Convert this tokenizer to its byte representation."""
        ...


class Trainable(Protocol):
    def train(self, dataset: Iterable[Any]) -> None:
        """Train the class on the specified dataset."""
        pass


# A registry of tokenizers indexed by custom names.
_tokenizer_registry: dict[str, type[Tokenizer]] = {}


def tokenizer(tokenizer_name: str):
    """Decorator to register a tokenizer class in the global registry.

    Registered tokenizers can be retrieved by name using get_tokenizer().

    Args:
        tokenizer_name: Unique identifier for the tokenizer in the registry.

    """

    def decorator(cls):
        register_tokenizer(tokenizer_name, cls)
        return cls

    return decorator


def register_tokenizer(name: str, cls: type[Tokenizer]) -> None:
    _tokenizer_registry[name] = cls


def get_tokenizer(tokenizer_name: str) -> type[Tokenizer] | None:
    """Retrieve a tokenizer class from the registry by name.

    Args:
        tokenizer_name: Registered identifier of the tokenizer to retrieve.

    Returns:
        Tokenizer class if found in the registry, None otherwise.
    """
    if not _tokenizer_registry:  # Refresh the imports.
        tokenizers_dir = Path(__file__).parent / "tokenizers"
        if not tokenizers_dir.exists():
            return None

        module_files = tokenizers_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @tokenizer decorator registration
        for module_file in module_files:
            module_name = f"lumiere.tokenizers.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _tokenizer_registry.get(tokenizer_name)
