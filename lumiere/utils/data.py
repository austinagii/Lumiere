import torch
from typing import Iterator

def to_batches(
    tokenizer, 
    dataset, 
    batch_size=64, 
    context_size=512
    ) -> Iterator[torch.Tensor]:
    """Produces an iterator over batches of tokenized text.

    Args:
        tokenizer: The tokenizer that will encode the text.
        dataset: The dataset to tokenize (each item should have a "text" key).
        batch_size: The number of batches to produce.
        context_size: The size of the context window for each batch.
    """
    batch = torch.zeros((batch_size, context_size), dtype=torch.long)
    context_ix = 0
    write_ix = 0

    for text in (sample["text"] for sample in dataset):
        token_ids = tokenizer.encode(text).ids
        read_ix = 0

        while (num_unread_tokens := len(token_ids) - read_ix) > 0:
            remaining_space_in_context = context_size - write_ix
            num_tokens_to_read = min(remaining_space_in_context, num_unread_tokens) 

            tokens_ids_read = token_ids[read_ix: read_ix + num_tokens_to_read]
            read_ix += num_tokens_to_read

            batch[context_ix, write_ix:write_ix + num_tokens_to_read] = torch.tensor(tokens_ids_read, dtype=torch.long)
            write_ix += num_tokens_to_read

            if write_ix == context_size:
                write_ix = 0
                context_ix += 1

                if context_ix == batch_size:
                    yield batch.clone()
                    batch.zero_()
                    context_ix = 0


def decode_batch(decoder, batch) -> list[str]:
    return [decoder.decode(ids.tolist()) for ids in batch]