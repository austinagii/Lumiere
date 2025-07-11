from typing import Iterable, Sequence

import torch


# Separate preprocessing concerns, allows to see what sequences are being read from the
# data.
# 1. Read preprocessed text sequences from the dataset

# General tokenization should be able to work over text or batches of text.
# 2. Tokenize the text sequences

# Read the tokenized text into context sized batches
# Encode the tokens for the transformer


class ContextBatchManager:
    def __init__(
        self,
        context_size: int,
        batch_size: int,
        sliding_window_size: int = 0,
        padding_token: str = None,
    ) -> None:
        self.context_size = context_size
        self.batch_size = batch_size
        self.sliding_window_size = sliding_window_size
        self.padding_token = padding_token

    def to_batches(self, data: Iterable[Sequence[str]]) -> Iterable[torch.tensor]:
        """
        Args:
            data: An iterator over tokenized text sequences  # Not accurate description
        Returns:
            An iterable of tensor batches of shape [batch_size, context_size]
        """
        batch, batch_mask = [], []

        for text in data:
            total_tokens = len(text)
            start_idx = 0

            while start_idx < total_tokens:
                context, mask = [], []

                # Calculate how many tokens to read for this context.
                remaining_tokens = total_tokens - start_idx
                available_space_in_context = self.context_size - len(context)
                num_tokens_to_read = min(remaining_tokens, available_space_in_context)

                # Read tokens into context
                context.extend(text[start_idx : start_idx + num_tokens_to_read])
                mask.extend([1] * len(context))
                start_idx += num_tokens_to_read

                # If at end of text and context is not full, pad the context and mask
                # to the context size.
                if start_idx == total_tokens and len(context) < self.context_size:
                    context.extend(
                        [self.padding_token] * (self.context_size - len(context))
                    )
                    mask.extend([0] * (self.context_size - len(mask)))

                batch.append(context)
                batch_mask.append(mask)

                # Yield the batch when it's full.
                if len(batch) == self.batch_size:
                    yield batch, batch_mask
                    batch, batch_mask = [], []
