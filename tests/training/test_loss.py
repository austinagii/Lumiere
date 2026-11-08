import torch
import torch.nn.functional as F  # noqa: N812

from lumiere.training.loss import cross_entropy_loss


class TestCrossEntropyLoss:
    def test_language_modelling_logits(self):
        logits = torch.randn(2, 5, 7)
        targets = torch.randint(0, 7, (2, 5))

        loss = cross_entropy_loss(logits, targets)

        expected = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
        assert torch.allclose(loss, expected)

    def test_classification_logits(self):
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = cross_entropy_loss(logits, targets)

        assert torch.allclose(loss, F.cross_entropy(logits, targets))

    def test_only_logits_are_used_from_tuple_outputs(self):
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = cross_entropy_loss((logits, torch.randn(4, 1)), targets)

        assert torch.allclose(loss, F.cross_entropy(logits, targets))

    def test_non_contiguous_targets(self):
        logits = torch.randn(2, 4, 7)
        targets = torch.randint(0, 7, (2, 5))[:, 1:]

        loss = cross_entropy_loss(logits, targets)

        expected = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
        assert torch.allclose(loss, expected)
