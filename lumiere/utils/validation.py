def validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_boolean(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def validate_probability(value: float, name: str) -> None:
    if not isinstance(value, float) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a float between 0.0 and 1.0")


def validate_positive_even_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0 or value % 2 != 0:
        raise ValueError(f"{name} must be a positive, even integer")
