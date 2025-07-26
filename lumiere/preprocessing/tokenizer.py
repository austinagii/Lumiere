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
            vocab_size=vocab_size, min_frequency=min_frequency
        )
        self.tokenizer.add_special_tokens(
            [token.token for token in SPECIAL_TOKENS.values()]
        )

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes the specified text"""
        return self.tokenizer.encode(text).tokens

    def tokenize_all(
        self, corpus: Iterable[str], lazy: bool = False
    ) -> list[list[str]] | Generator[list[str], None, None]:
        """Tokenizes the specified text in chunks of the given size

        When `chunk_size` is not specified the encoding behavior becomes dynamic based
        on the type of `text`. If text is a string type, then the entire str is encoded,
        however, if text is a sequence of string types, then an iterator is returned
        where containing the token ids of the encoded text sequence. Note that the
        chunks do not cross sample boundaries and the chunks are not padded.
        """
        if lazy:
            for text in corpus:
                yield self.tokenize(text)
        else:
            return [self.tokenize(text) for text in corpus]

    def encode(self, tokens: list[str]) -> list[str]:
        return self.tokenizer.encode(tokens, is_pretokenized=True).ids

    def encode_all(
        self, corpus: Sequence[str], lazy: bool = False
    ) -> list[list[str]] | Generator[list[str], None, None]:
        # if lazy:
        #     for text in corpus:
        #         yield self.tokenizer.encode(text, is_pretokenized=True).ids
        # else:
        #     return [
        #         self.tokenizer.encode(text, is_pretokenized=True).ids for text in corpus
        #     ]
        return [
            [self.tokenizer.token_to_id(token) for token in text] for text in corpus
        ]

    def decode(self, token_ids: list[str]) -> str:
        return self.tokenizer.decode(token_ids)

    def decode_all(self, corpus: Sequence[list[str]]) -> Generator[str, None, None]:
        for ids in corpus:
            yield self.decode(ids)

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
