"""Utilities for generating random data."""

import secrets
import string


# fmt: off
_ADJECTIVES = [
    "autumn", "billowing", "bitter", "black", "blue", "bold", "broken",
    "calm", "cold", "crimson", "curly", "damp", "dark", "dawn", "delicate",
    "divine", "dry", "empty", "falling", "fancy", "flat", "floral", "frosty",
    "gentle", "green", "hidden", "holy", "icy", "jolly", "late", "lingering",
    "little", "lively", "long", "lucky", "misty", "morning", "muddy", "mute",
    "nameless", "noisy", "old", "patient", "plain", "polished", "proud",
    "purple", "quiet", "rapid", "raspy", "red", "restless", "rough", "round",
    "royal", "shiny", "shrill", "shy", "silent", "small", "snowy", "soft",
    "solitary", "sparkling", "spring", "still", "summer", "super", "sweet",
    "throbbing", "tight", "tiny", "twilight", "wandering", "weathered",
    "white", "wild", "winter", "wispy", "withered", "yellow", "young",
]

_NOUNS = [
    "art", "band", "bar", "base", "bird", "block", "boat", "bonus", "bread",
    "brook", "bush", "butterfly", "cake", "cell", "cherry", "cloud", "credit",
    "darkness", "dawn", "dew", "disk", "dream", "dust", "feather", "field",
    "fire", "firefly", "flower", "fog", "forest", "frog", "frost", "glade",
    "glitter", "gloom", "grass", "hall", "heart", "hill", "haze", "lake",
    "leaf", "limit", "math", "meadow", "mode", "moon", "morning", "mountain",
    "mud", "night", "paper", "pine", "poetry", "pond", "rain", "river",
    "resonance", "rice", "sea", "shadow", "shape", "silence", "sky", "smoke",
    "snow", "sound", "star", "sun", "sunset", "surf", "thunder", "tide",
    "tree", "truth", "union", "violet", "voice", "water", "wave", "wildflower",
    "wind", "wood",
]
# fmt: on


def random_name() -> str:
    """Generate a name name in the format ``adjective-noun-NNNN``.

    Returns:
        A randomly generated name such as ``"silent-river-4821"``.
    """
    adjective = secrets.choice(_ADJECTIVES)
    noun = secrets.choice(_NOUNS)
    number = secrets.randbelow(9000) + 1000  # 1000–9999
    return f"{adjective}-{noun}-{number}"


def random_id(
    n: int = 8,
    include_alpha_upper: bool = True,
    include_alpha_lower: bool = True,
    include_digits: bool = True,
):
    """Generate a random alphanumeric identifier.

    Args:
        n: The length of the identifier. Defaults to `8`.
        include_alpha_upper: Whether to include uppercase alphabet characters in the
            identifier.
        include_alpha_lower: Whether to include lowercase alphabet characters in the
            identifier.
        include_digits: Whether to include digits in the identifier.

    Returns:
        A random string of length `n` containing letters and digits.
    """
    alphabet: list[str] = []

    if include_alpha_upper:
        alphabet.extend(string.ascii_uppercase)

    if include_alpha_lower:
        alphabet.extend(string.ascii_lowercase)

    if include_digits:
        alphabet.extend(string.digits)

    if len(alphabet) == 0:
        raise ValueError(
            "At least one group of characters must be included to generate an id."
        )

    return "".join([secrets.choice(alphabet) for _ in range(n)])
