from collections.abc import Generator, Iterable, Sequence

from lumiere.data.preprocessing.tokenizer import Tokenizer


class AsciiTokenizer(Tokenizer):
    def tokenize(self, text: str, to_ids: bool = False) -> list[str] | list[int]:
        pass

    def tokenize_all(
        self, corpus: Iterable[str], lazy: bool = False, to_ids: bool = False
    ) -> list[list[str]] | Generator[list[str], None, None]:
        pass

    def encode(self, tokens: list[str]) -> list[int]:
        pass

    def encode_all(
        self, corpus: Sequence[list[str]], lazy: bool = False
    ) -> list[list[int]] | Generator[list[int], None, None]:
        pass

    def decode(self, token_ids: list[int]) -> str:
        pass

    def decode_all(
        self, corpus: Sequence[list[int]], lazy: bool = False
    ) -> list[str] | Generator[str, None, None]:
        pass

    def train(self, corpus: Iterable[str]) -> None:
        pass

    @property
    def vocab_size(self) -> int:
        pass

    @classmethod
    def from_bytes(cls, bytes: bytes, *args, **kwargs):
        pass

    def __bytes__(self):
        pass
