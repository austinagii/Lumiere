from lumiere.data import Preprocessor
from lumiere.internal.registry import discover


@discover(Preprocessor, "autoregressive")
class AutoregressiveLanguageModellingPreprocessor:
    """Preprocessor for autoregressive language modeling tasks.

    Transforms token sequences into input-target pairs for next-token prediction
    by shifting the sequence. The input becomes all tokens except the last, and
    the target becomes all tokens except the first.

    This preprocessing is essential for causal language models where each token
    predicts the next token in the sequence.

    Args:
        device: The device to move tensors to (`"cpu"`, `"cuda"`, or `"mps"`).

    Example:
        Given tokens `[1, 2, 3, 4, 5]`:
        - Input: `[1, 2, 3, 4]`
        - Target: `[2, 3, 4, 5]`
    """

    def __init__(self, device):
        """Initialize the autoregressive preprocessor.

        Args:
            device: The device to move tensors to (`"cpu"`, `"cuda"`, or `"mps"`).
        """
        self.device = device

    def __call__(self, tokens, padding_mask):
        """Transform a batch of tokens into input-target pairs.

        Args:
            tokens: Tensor of token IDs with shape `(batch_size, context_size)`.
            padding_mask: Boolean mask indicating padding positions with shape
                `(batch_size, context_size)`.

        Returns:
            A tuple containing:
                - Tuple of `(input_tokens, input_padding_mask)` where each has shape
                  `(batch_size, context_size - 1)`.
                - `target_tokens` with shape `(batch_size, context_size - 1)`.

        Shape:
            - Input tokens: `(batch_size, context_size)`
            - Input padding_mask: `(batch_size, context_size)`
            - Output input_tokens: `(batch_size, context_size - 1)`
            - Output input_padding_mask: `(batch_size, context_size - 1)`
            - Output target_tokens: `(batch_size, context_size - 1)`
        """
        # Shift the input tokens to the left by one position to get the targets.
        next_tokens = tokens[:, 1:].to(self.device)
        # Shift the input tokens and its padding mask accordingly.
        tokens = tokens[:, :-1].to(self.device)
        padding_mask = padding_mask[:, :-1].to(self.device)
        return (tokens, padding_mask), next_tokens
