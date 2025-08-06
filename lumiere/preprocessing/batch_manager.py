from collections.abc import Iterable, Sequence


class BatchManager:
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

    def to_batches(
        self, data: Iterable[Sequence[str]]
    ) -> Iterable[tuple[list[list[str]], list[list[bool]]]]:
        """Splits tokens into batches.

        The data is split into batches of size `batch_size` where each element of the
        batch is a sequence of tokens of length `context_size`.

        Tokens across sequence boundaries are never merged into a single sequence. If a
        context can only be partially filled using the tokens in the current sequence,
        the context is padded with the padding token.

        The padding mask is a boolean mask of the same shape as the batch, where True
        indicates that the token in the batch at the corresponding index is a padding
        token and False indicates a non-padding token.

        Args:
            data: An iterator over tokenized text sequences  # Not accurate description
        Returns:
            An iterable of tensor batches of shape [batch_size, context_size]
        """
        if isinstance(data, str):
            raise TypeError("Expected input to be an iterable of strings, not a string")

        if not isinstance(data, Iterable):
            raise TypeError(f"Expected data to be an iterable, but got {type(data)}")

        batch, batch_padding_mask = [], []

        for ix, text in enumerate(data):
            if not isinstance(text, Sequence):
                raise TypeError(
                    f"Expected elem {ix} to be Sequence, but got {type(text)}"
                )

            total_tokens = len(text)
            start_idx = 0
            prev_context = []
            while start_idx < total_tokens:
                context, context_padding_mask = [], []

                # Seed the current context with the last `sliding_window_size` tokens
                # from the previous context.
                if self.sliding_window_size > 0:
                    # Do not apply sliding window across text sequence boundaries.
                    if len(prev_context) > 0 and start_idx > 0:
                        context.extend(prev_context[-self.sliding_window_size :])
                        context_padding_mask.extend([False] * self.sliding_window_size)

                # Calculate how many tokens to read for this context.
                remaining_tokens = total_tokens - start_idx
                available_space_in_context = self.context_size - len(context)
                num_tokens_to_read = min(remaining_tokens, available_space_in_context)

                # Read tokens into context
                context.extend(text[start_idx : start_idx + num_tokens_to_read])
                context_padding_mask.extend([False] * num_tokens_to_read)
                start_idx += num_tokens_to_read

                # If at end of text and context is not full, pad the context and mask
                # to the context size.
                if start_idx == total_tokens and len(context) < self.context_size:
                    context.extend(
                        [self.padding_token] * (self.context_size - len(context))
                    )
                    context_padding_mask.extend(
                        [True] * (self.context_size - len(context_padding_mask))
                    )

                batch.append(context)
                batch_padding_mask.append(context_padding_mask)
                prev_context = context

                # Yield the batch when it's full.
                if len(batch) == self.batch_size:
                    yield batch, batch_padding_mask
                    batch, batch_padding_mask = [], []

        # Yield the last batch if it's not empty.
        if len(batch) > 0:
            yield batch, batch_padding_mask
