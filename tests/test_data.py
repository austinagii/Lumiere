import string

import torch

from prism.data import to_batches

class SimpleTokenizer:
    def __init__(self):
        id_by_token = {}
        for i, token in enumerate(string.ascii_lowercase[:10]):
            id_by_token[token] = i
        self.id_by_token = id_by_token

    def encode(self, text):
        return SimpleEncoding([self.id_by_token[c] for c in text])

    def get_vocab_size(self):
        return len(self.id_by_token)


class SimpleEncoding:
    def __init__(self, ids):
        self.ids_ = ids
    
    @property
    def ids(self):
        return self.ids_

    @ids.setter
    def ids(self, ids):
        self.ids_ = ids


def test_tokenizer():
    tokenizer = SimpleTokenizer()

    dataset = [{"text": "abcabc"}]
    token_ids = tokenizer.encode(dataset[0]["text"]).ids
    assert token_ids == [0, 1, 2, 0, 1, 2]

    batches = to_batches(tokenizer, dataset, batch_size=1, context_size=3)
    expected_batch = torch.tensor([
        [0, 1, 2],
        [0, 1, 2]
    ], dtype=torch.long)

    for batch in batches:
        assert torch.allclose(batch, expected_batch)




