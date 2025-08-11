from collections import OrderedDict
from dataclasses import dataclass
from typing import Generator, Iterable, Sequence

import tokenizers
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers


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


class Tokenizer:
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

    def tokenize(self, text: str) -> list[str]:
        """Returns a list of tokens for the specified text"""
        return self.tokenizer.encode(text).tokens

    def tokenize_all(
        self, corpus: Iterable[str], lazy: bool = False
    ) -> list[list[str]] | Generator[list[str], None, None]:
        """Tokenizes the specified text

        When `lazy` is `True`, a generator of lists of tokens is returned.
        Otherwise, a list of lists of tokens is returned.
        """
        if lazy:
            return (self.tokenize(text) for text in corpus)
        else:
            return [self.tokenize(text) for text in corpus]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def encode_all(
        self, corpus: Sequence[list[str]], lazy: bool = False
    ) -> list[list[int]] | Generator[list[int], None, None]:
        if lazy:
            return (
                [self.tokenizer.token_to_id(token) for token in text] for text in corpus
            )
        else:
            return [
                [self.tokenizer.token_to_id(token) for token in text] for text in corpus
            ]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def decode_all(
        self, corpus: Sequence[list[int]], lazy: bool = False
    ) -> list[str] | Generator[str, None, None]:
        if lazy:
            return (self.decode(ids) for ids in corpus)
        else:
            return [self.decode(ids) for ids in corpus]

    def train(self, corpus: Iterable[str]):
        self.tokenizer.train_from_iterator(corpus, self.trainer)
        return self

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @staticmethod
    def from_bytes(bytes: bytes):
        tokenizer = Tokenizer()
        tokenizer.tokenizer = tokenizers.Tokenizer.from_str(bytes.decode("utf-8"))
        return tokenizer
