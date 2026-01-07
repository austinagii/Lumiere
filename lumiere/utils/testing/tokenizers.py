from collections.abc import Generator, Iterable

from lumiere.data.preprocessing.tokenizer import Tokenizer


MAX_ASCII_CODE = 127
INVALID_CHARACTER = "�"


class AsciiTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[int]:
        return [ord(char) if ord(char) < MAX_ASCII_CODE else -1 for char in text]

    def tokenize_all(self, corpus: Iterable[str]) -> Generator[list[int], None, None]:
        return (self.tokenize(text) for text in corpus)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        raise NotImplementedError

    def encode_all(
        self, corpus: Iterable[Iterable[str]]
    ) -> Generator[list[int], None, None]:
        raise NotImplementedError

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(
            [
                chr(_id) if _id <= MAX_ASCII_CODE else INVALID_CHARACTER
                for _id in token_ids
            ]
        )

    def decode_all(self, corpus: Iterable[Iterable[int]]) -> Generator[str, None, None]:
        return (self.decode(ids) for ids in corpus)

    def train(self, corpus: Iterable[str]) -> None:
        pass

    def vocab_size(self) -> int:
        return MAX_ASCII_CODE
