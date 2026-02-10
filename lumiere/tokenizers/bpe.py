from collections.abc import Generator, Iterable

import tokenizers
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from lumiere.internal.registry import discover
from lumiere.tokenizer import SPECIAL_TOKENS, Serializable, Tokenizer


@discover(Tokenizer, "bpe")
class BPETokenizer(Tokenizer, Serializable):
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
        """Tokenizes the specified text.

        If `to_ids` is `True`, the tokens are converted to their corresponding IDs.
        """
        return self.tokenizer.encode(text).ids

    def tokenize_all(self, corpus: Iterable[str]) -> Generator[list[int], None, None]:
        """Tokenizes the specified text.

        When `lazy` is `True`, a generator of lists of tokens is returned.
        Otherwise, a list of lists of tokens is returned. If `to_ids` is `True`,
        the tokens are converted to their corresponding IDs.
        """
        return (self.tokenize(text) for text in corpus)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def encode_all(
        self, corpus: Iterable[Iterable[str]]
    ) -> Generator[list[int], None, None]:
        return (
            [self.tokenizer.token_to_id(token) for token in text] for text in corpus
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    def decode_all(self, corpus: Iterable[Iterable[int]]) -> Generator[str, None, None]:
        return (self.decode(ids) for ids in corpus)

    def train(self, corpus: Iterable[str]):
        self.tokenizer.train_from_iterator(corpus, self.trainer)
        return self

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def from_bytes(cls, bytes: bytes, *args, **kwargs):
        tokenizer = cls(*args, **kwargs)
        tokenizer.tokenizer = tokenizers.Tokenizer.from_str(bytes.decode("utf-8"))
        return tokenizer

    def __bytes__(self):
        return bytes(self.tokenizer.to_str(), "utf-8")
