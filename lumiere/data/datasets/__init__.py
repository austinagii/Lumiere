"""This package contains various datasets for training models."""

from .cifar10 import CIFAR10Dataset
from .wikitext import WikiText2Dataset


__all__ = ["CIFAR10Dataset", "WikiText2Dataset"]
