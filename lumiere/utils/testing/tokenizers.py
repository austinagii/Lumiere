from collections.abc import Generator, Iterable, Sequence

from lumiere.data.preprocessing.tokenizer import Tokenizer


MAX_ASCII_CODE = 127


class AsciiTokenizer(Tokenizer):
    def tokenize(self, text: str, to_ids: bool = False) -> list[str] | list[int]:
        return [ord(char) if ord(char) < MAX_ASCII_CODE else -1 for char in text]

    def tokenize_all(
        self, corpus: Iterable[str], lazy: bool = False, to_ids: bool = False
    ) -> (
        list[list[str]] | list[list[int]] | Generator[list[str] | list[int], None, None]
    ):
        return (self.tokenize(text, to_ids=to_ids) for text in corpus)

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
