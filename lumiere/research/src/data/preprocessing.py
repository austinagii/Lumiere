from collections.abc import Iterable

import torch

from lumiere.research.src.data.tokenizer import Tokenizer
from lumiere.research.src.utils.validation import (
    validate_integer,
    validate_iterable,
)


def to_training_batches(
    corpus: Iterable[str],
    tokenizer: Tokenizer,
    context_size: int,
    batch_size: int,
    pad_id: int,
    sliding_window_size: int = 0,
    num_batches: int | None = None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a text corpus into fixed-length token batches for training.

    This function tokenizes each text sequence in the corpus and splits it into
    training examples of length `context_size`. Examples are packed into batches
    of size `batch_size`. Each example is padded with `padding_token_id` if it
    does not fully fill the context window. A corresponding boolean mask is
    returned, where `True` indicates a padding position and `False` indicates
    a real token.

    Overlap can optionally be introduced between consecutive contexts within the
    same sequence by setting `sliding_window_size`. When enabled, the last
    `sliding_window_size` tokens of one full context will be copied into the
    beginning of the next context. Overlap is only applied if the previous
    context was completely filled (i.e. contains no padding) and is never applied
    across sequence boundaries.

    Args:
        corpus: Iterable of text strings to tokenize and batch.
        tokenizer: Tokenizer with a `tokenize_all(corpus, lazy=True)` method that
            yields token sequences for each text.
        context_size: Number of tokens per training example (sequence length).
        batch_size: Number of examples per batch.
        padding_token_id: Token id to use for padding incomplete contexts.
        sliding_window_size: Number of overlapping tokens to carry from the end
            of one full context into the start of the next (default: 0).
        num_batches: If provided, stop after yielding this many full batches.
            Otherwise yield all batches until the corpus is exhausted.

    Returns:
        Iterator of `(tokens, is_pad_mask)` where:
            - `tokens` is a LongTensor of shape `(N, context_size)`.
            - `is_pad_mask` is a BoolTensor of the same shape, with `True` marking
              padding positions and `False` marking valid tokens.
          For all full batches, `N == batch_size`. The final batch may be smaller
          if there are not enough examples to fill it.
    """
    validate_iterable(corpus)
    validate_integer(context_size, "context_size", min_value=1)
    validate_integer(batch_size, "batch_size", min_value=1)
    validate_integer(sliding_window_size, "sliding_window_size", min_value=0)
    validate_integer(pad_id, "padding_token_id")
    if num_batches is not None:
        validate_integer(num_batches, "num_batches", min_value=1)

    if sliding_window_size >= context_size:
        raise ValueError(
            "sliding_window_size must be < context_size to guarantee progress."
        )

    def _init_batch():
        # Helper function to initialize a batch and padding mask.
        batch_ = torch.full((batch_size, context_size), pad_id, dtype=torch.long)
        mask_ = torch.full((batch_size, context_size), True, dtype=torch.bool)
        return batch_, mask_

    batch, batch_padding_mask = _init_batch()

    batch_write_ix = 0  # The index of the next free space in the batch.
    context_write_ix = 0  # The index of the next free space in the context.
    num_batches_created = 0  # The number of batches created so far.

    for tokens in tokenizer.tokenize_all(corpus, to_ids=True, lazy=True):
        total_tokens_in_seq = len(tokens)
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        seq_read_ix = 0  # The index of the next token to read from the sequence.

        sliding_window = (
            torch.full((sliding_window_size,), pad_id, dtype=torch.long)
            if sliding_window_size > 0
            else None
        )

        while seq_read_ix < total_tokens_in_seq:
            # If the sliding window is not empty, copy it into the batch.
            if sliding_window is not None and torch.all(sliding_window != pad_id):
                batch[batch_write_ix, :sliding_window_size] = sliding_window
                batch_padding_mask[batch_write_ix, :sliding_window_size] = False
                context_write_ix = sliding_window_size

            # Calculate how many tokens to read for this context.
            remaining_tokens = total_tokens_in_seq - seq_read_ix
            available_space_in_context = context_size - context_write_ix
            num_tokens_to_read = min(remaining_tokens, available_space_in_context)

            # Read tokens into context
            if num_tokens_to_read > 0:
                end_read_ix = seq_read_ix + num_tokens_to_read
                end_write_ix = context_write_ix + num_tokens_to_read
                batch[batch_write_ix, context_write_ix:end_write_ix] = tokens[
                    seq_read_ix:end_read_ix
                ]
                batch_padding_mask[batch_write_ix, context_write_ix:end_write_ix] = (
                    False
                )
                seq_read_ix = end_read_ix
                context_write_ix = end_write_ix

            # Since we don't fill contexts across sequence boundaries, we need to
            # pad the context if we're at the end of the sequence and the context is
            # not full.
            if seq_read_ix == total_tokens_in_seq and context_write_ix < context_size:
                batch[batch_write_ix, context_write_ix:] = pad_id
                batch_padding_mask[batch_write_ix, context_write_ix:] = True

            # At this point, the current context is filled with tokens from the
            # sequence or padded with padding tokens in the event that we've reached
            # the end of the sequence. We can update the sliding window to the last
            # `sliding_window_size` tokens in the context.
            if sliding_window_size > 0:
                sliding_window = batch[batch_write_ix, -sliding_window_size:]

            # Advance to the next context in the batch.
            batch_write_ix += 1
            context_write_ix = 0

            # Yield the batch if it's full.
            if batch_write_ix == batch_size:
                yield batch, batch_padding_mask
                batch, batch_padding_mask = _init_batch()
                batch_write_ix = 0
                num_batches_created += 1

                # Return early if we've reached the requested number of batches.
                if num_batches is not None and num_batches_created >= num_batches:
                    return

    # Yield a partial batch (trimmed to the number of contexts) if we're at the end
    # of the corpus and the batch contains at least one context.
    if batch_write_ix > 0:
        yield batch[:batch_write_ix], batch_padding_mask[:batch_write_ix]

    return
