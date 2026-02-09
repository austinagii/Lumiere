import functools
from collections.abc import Iterable

import torch

from lumiere.data import Pipeline, Preprocessor
from lumiere.discover import discover
from lumiere.tokenizers import Tokenizer
from lumiere.utils.validation import (
    validate_integer,
)


@discover(Pipeline, "text")
class TextPipeline:
    """Performs common preprocessing steps across text data.

    It should be possible to directly pass the outputs of this pipeline directly to a
    model as input for training or inference.

    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int,
        context_size: int,
        pad_id: int,
        sliding_window_size: int,
        preprocessors: Iterable[Preprocessor] | None = None,
    ):
        validate_integer(context_size, "context_size", min_value=1)
        validate_integer(batch_size, "batch_size", min_value=1)
        validate_integer(sliding_window_size, "sliding_window_size", min_value=0)

        if sliding_window_size >= context_size:
            raise ValueError(
                "sliding_window_size must be < context_size to guarantee progress."
            )

        self.tokenizer = tokenizer
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.context_size = context_size
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.sliding_window_size = sliding_window_size

    def _create_batches(
        self,
        data: Iterable,
        num_batches: int | None = None,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Internal method to create raw token batches before preprocessing.

        Returns:
            Iterator of `(tokens, is_pad_mask)` tuples.
        """
        if num_batches is not None:
            validate_integer(num_batches, "num_batches", min_value=1)

        def _create_batch():
            # Helper function to create a fresh batch and padding mask.
            batch_ = torch.full(
                (self.batch_size, self.context_size), self.pad_id, dtype=torch.long
            )
            mask_ = torch.full(
                (self.batch_size, self.context_size), True, dtype=torch.bool
            )
            return batch_, mask_

        batch, batch_padding_mask = _create_batch()
        batch_write_ix = 0  # The index of the next free space in the batch.
        context_write_ix = 0  # The index of the next free space in the context.
        num_batches_created = 0  # The number of batches created so far.

        for tokens in self.tokenizer.tokenize_all(data):
            total_tokens_in_seq = len(tokens)
            tokens = torch.as_tensor(tokens, dtype=torch.long)
            seq_read_ix = 0  # The index of the next token to read from the sequence.

            sliding_window = (
                torch.full((self.sliding_window_size,), self.pad_id, dtype=torch.long)
                if self.sliding_window_size > 0
                else None
            )

            while seq_read_ix < total_tokens_in_seq:
                # If the sliding window is not empty, copy it into the batch. A sliding
                # window should never be partially empty as this would only occur at the
                # end of the sequence. In which case, there are no tokens to form the
                # next sample.
                if sliding_window is not None and torch.all(
                    sliding_window != self.pad_id
                ):
                    batch[batch_write_ix, : self.sliding_window_size] = sliding_window
                    batch_padding_mask[batch_write_ix, : self.sliding_window_size] = (
                        False
                    )
                    context_write_ix = self.sliding_window_size

                # Calculate how many tokens to read for this context.
                remaining_tokens = total_tokens_in_seq - seq_read_ix
                available_space_in_context = self.context_size - context_write_ix
                num_tokens_to_read = min(remaining_tokens, available_space_in_context)

                # Read tokens into context
                if num_tokens_to_read > 0:
                    end_read_ix = seq_read_ix + num_tokens_to_read
                    end_write_ix = context_write_ix + num_tokens_to_read
                    batch[batch_write_ix, context_write_ix:end_write_ix] = tokens[
                        seq_read_ix:end_read_ix
                    ]
                    batch_padding_mask[
                        batch_write_ix, context_write_ix:end_write_ix
                    ] = False
                    seq_read_ix = end_read_ix
                    context_write_ix = end_write_ix

                # Since we don't fill contexts across sequence boundaries, we need to
                # pad the context if we're at the end of the sequence and the context is
                # not full.
                if (
                    seq_read_ix == total_tokens_in_seq
                    and context_write_ix < self.context_size
                ):
                    batch[batch_write_ix, context_write_ix:] = self.pad_id
                    batch_padding_mask[batch_write_ix, context_write_ix:] = True

                # At this point, the current context is filled with tokens from the
                # sequence or padded with padding tokens in the event that we've reached
                # the end of the sequence. We can update the sliding window to the last
                # `sliding_window_size` tokens in the context.
                if self.sliding_window_size > 0:
                    sliding_window = batch[batch_write_ix, -self.sliding_window_size :]

                # Advance to the next context in the batch.
                batch_write_ix += 1
                context_write_ix = 0

                # Yield the batch if it's full.
                if batch_write_ix == self.batch_size:
                    yield batch, batch_padding_mask
                    batch, batch_padding_mask = _create_batch()
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

    def batches(
        self,
        data: Iterable,
        num_batches: int | None = None,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Convert a text corpus into fixed-length token batches for training.

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

        After batching, all configured preprocessors are applied in sequence to
        transform the raw batches into the final format needed for training.

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
            Iterator of preprocessed batches. The structure of each batch depends on
            the preprocessors applied. Before preprocessing, batches are tuples of:
                - `tokens` is a LongTensor of shape `(N, context_size)`.
                - `is_pad_mask` is a BoolTensor of the same shape, with `True` marking
                  padding positions and `False` marking valid tokens.
              For all full batches, `N == batch_size`. The final batch may be smaller
              if there are not enough examples to fill it.
        """
        for batch in self._create_batches(data, num_batches):
            preprocessed_batch = functools.reduce(
                lambda x, f: f(*x),
                self.preprocessors,
                batch,
            )
            yield preprocessed_batch
