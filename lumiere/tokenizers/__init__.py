from lumiere.tokenizer import (
    SPECIAL_TOKENS,
    Serializable,
    SpecialToken,
    Tokenizer,
    Trainable,
    get_tokenizer,
    register_tokenizer,
    tokenizer,
)

from .loader import load


# Import BPETokenizer with fallback for missing dependencies
try:
    from .bpe import BPETokenizer

    __all__ = [
        "BPETokenizer",
        "load",
        "SPECIAL_TOKENS",
        "Serializable",
        "SpecialToken",
        "Tokenizer",
        "Trainable",
        "tokenizer",
        "register_tokenizer",
        "get_tokenizer",
    ]
except ImportError:
    # tokenizers library not installed
    __all__ = [
        "load",
        "SPECIAL_TOKENS",
        "Serializable",
        "SpecialToken",
        "Tokenizer",
        "Trainable",
        "tokenizer",
        "register_tokenizer",
        "get_tokenizer",
    ]
