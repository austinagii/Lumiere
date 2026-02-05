from collections import OrderedDict
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
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


module_discovery_path = [
    "lumiere.data.datasets",
    "lumiere.data.tokenizers",
    "lumiere.model.components",
]


class TokenizerLoader:
    _registry: Mapping [str, [type[Tokenizer]]] = {}

    @classmethod
    def load(cls, spec: dict[str, Any]) -> Tokenizer:
        tokenizer_type = spec.get("type")
        if tokenizer_type is None:
            raise ValueError("Tokenizer specification missing required 'type' key.")

        tokenizers_dir = Path(__file__).parent / "tokenizers"
        module_files = tokenizers_dir.glob("*.py")
        module_file = None
        for file in module_files:
            if tokenizer_type in file.stem:
                module_file = file
                break

        if module_file is None:
            raise ValueError(f"Tokenizer '{tokenizer_type}' could not be found.")

        module_name = f"lumiere.data.datasets.{module_file.stem}"
        with contextlib.suppress(ImportError):
            importlib.import_module(module_name)

    # TODO: Still need to know what the class name is for initializer.
    return _dataset_registry.get(dataset_name)

        tokenizer_cls = cloader.get(tokenizer_type)

        return tokenizer_cls(**spec)
