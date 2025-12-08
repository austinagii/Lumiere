from lumiere.research.src.utils.validation import validate_probability


def test_validate_probability_accepts_zero():
    """Test that validate_probability does not raise an error when given 0.0."""
    # This should not raise any exception
    validate_probability(0, "test_value")
