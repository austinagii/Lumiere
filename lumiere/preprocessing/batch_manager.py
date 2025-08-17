from collections.abc import Iterable, Sequence

import torch

from lumiere.preprocessing.tokenizer import Tokenizer


def to_training_batches(
    corpus: Iterable[str],
    tokenizer: Tokenizer,
    context_size: int,
    batch_size: int,
    sliding_window_size: int = 0,
    padding_token: str | None = None,
    device: torch.device = torch.device("cpu"),
    num_batches: int | None = None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    tokenized_corpus = tokenizer.tokenize_all(corpus, lazy=True)
    batches = to_batches(
        tokenized_corpus,
        context_size + 1,
        batch_size,
        sliding_window_size,
        padding_token,
    )
    for batch_num, batch in enumerate(batches):
        if num_batches is not None and batch_num >= num_batches:
            break

        (tokenized_batch, padding_mask) = batch
        encoded_batch = tokenizer.encode_all(tokenized_batch)
        batch = torch.tensor(encoded_batch, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

        x, y = (batch[:, :-1].to(device), batch[:, 1:].to(device))
        # Slice padding mask to match input sequence length
        padding_mask_input = padding_mask[:, :-1].to(device)
        yield x, padding_mask_input, y


def to_batches(
    tokens: Iterable[Sequence[str]],
    context_size: int,
    batch_size: int,
    sliding_window_size: int = 0,
    padding_token: str = None,
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
    if isinstance(tokens, str):
        raise TypeError("Expected input to be an iterable of strings, not a string")

    if not isinstance(tokens, Iterable):
        raise TypeError(f"Expected data to be an iterable, but got {type(tokens)}")

    batch, batch_padding_mask = [], []

    for ix, text in enumerate(tokens):
        if not isinstance(text, Sequence):
            raise TypeError(f"Expected elem {ix} to be Sequence, but got {type(text)}")

        total_tokens = len(text)
        start_idx = 0
        prev_context = []
        while start_idx < total_tokens:
            context, context_padding_mask = [], []

            # Seed the current context with the last `sliding_window_size` tokens
            # from the previous context.
            if sliding_window_size > 0:
                # Do not apply sliding window across text sequence boundaries.
                if len(prev_context) > 0 and start_idx > 0:
                    context.extend(prev_context[-sliding_window_size:])
                    context_padding_mask.extend([False] * sliding_window_size)

            # Calculate how many tokens to read for this context.
            remaining_tokens = total_tokens - start_idx
            available_space_in_context = context_size - len(context)
            num_tokens_to_read = min(remaining_tokens, available_space_in_context)

            # Read tokens into context
            context.extend(text[start_idx : start_idx + num_tokens_to_read])
            context_padding_mask.extend([False] * num_tokens_to_read)
            start_idx += num_tokens_to_read

            # If at end of text and context is not full, pad the context and mask
            # to the context size.
            if start_idx == total_tokens and len(context) < context_size:
                context.extend([padding_token] * (context_size - len(context)))
                context_padding_mask.extend(
                    [True] * (context_size - len(context_padding_mask))
                )

            batch.append(context)
            batch_padding_mask.append(context_padding_mask)
            prev_context = context

            # Yield the batch when it's full.
            if len(batch) == batch_size:
                yield batch, batch_padding_mask
                batch, batch_padding_mask = [], []

    # Yield the last batch if it's not empty.
    if len(batch) > 0:
        yield batch, batch_padding_mask
