from functools import total_ordering

import torch


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
