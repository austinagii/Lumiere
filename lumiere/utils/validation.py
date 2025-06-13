def validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")

def validate_boolean(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")

def validate_positive_float_or_zero(value: float, name: str) -> None:
    if not isinstance(value, float) or value < 0.0:
        raise ValueError(f"{name} must be a positive float or zero")

def validate_positive_even_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0 or value % 2 != 0:
        raise ValueError(f"{name} must be a positive, even integer")