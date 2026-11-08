from functools import total_ordering

import torch
import torch.nn.functional as F


@total_ordering
class Loss:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss

    def backward(self):
        self.loss.backward()

    @property
    def perplexity(self):
        return torch.exp(self.loss)

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __isub__(self, other):
        pass

    def __add__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __iadd__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __eq__(self, other):
        pass

    def __truediv__(self, other):
        pass


def cross_entropy_loss(
    predictions: torch.Tensor | tuple, targets: torch.Tensor
) -> torch.Tensor:
    """Compute cross entropy loss for language modeling or classification.

    Args:
        predictions: Model output logits of shape (batch_size, seq_len, vocab_size)
            for language modeling or (batch_size, num_classes) for classification,
            or a tuple of (logits, attention_weights) where we only use the logits.
        targets: Target token IDs of shape (batch_size, seq_len), or target class
            indices of shape (batch_size,).

    Returns:
        Scalar loss tensor.
    """
    # Handle tuple output from models that return (output, attention_weights)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Flatten predictions and targets for cross entropy
    # predictions: (batch_size * seq_len, vocab_size) or (batch_size, num_classes)
    # targets: (batch_size * seq_len) or (batch_size)
    predictions = predictions.reshape(-1, predictions.size(-1))
    targets = targets.reshape(-1)

    # Compute cross entropy loss
    loss = F.cross_entropy(predictions, targets, reduction="mean")

    return loss
