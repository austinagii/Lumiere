from collections.abc import Iterable
from typing import Any, Optional


def validate_integer(
    value: int,
    name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> None:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")


def validate_boolean(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def validate_probability(value: float, name: str) -> None:
    float_value = float(value)
    if float_value < 0.0 or float_value > 1.0:
        raise ValueError(f"{name} must be a float between 0.0 and 1.0")


def validate_positive_even_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0 or value % 2 != 0:
        raise ValueError(f"{name} must be a positive, even integer")


def validate_iterable(corpus: Iterable, name: str = "corpus") -> None:
    """
    Validate that the input is a non string iterable .

    Args:
        corpus: The corpus to validate
        name: The name of the parameter for error messages

    Raises:
        TypeError: If corpus is a string or not an iterable
    """
    if isinstance(corpus, str):
        raise TypeError(f"Expected {name} to be an iterable of strings, not a string")

    if not isinstance(corpus, Iterable):
        raise TypeError(f"Expected {name} to be an iterable, but got {type(corpus)}")


def validate_string(value: Any, name: str) -> None:
    """Validate that the specified value is a string.

    Args:
        value: The value to be validated.
        name: The value's name.

    Raises:
        TypeError: If the specified value is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"Expected '{name}' to be a string, but got {type(value)}.")
