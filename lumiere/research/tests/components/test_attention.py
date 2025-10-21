import torch

from lumiere.research.src.components.attention import (
    MultiHeadAttention,
    concat_heads,
    split_heads,
    stable_softmax,
)


class TestStableSoftmax:
    def test_input_tensor_of_all_neg_inf_returns_zeros(self):
        x = torch.full((1, 10), -float("inf"))
        expected = torch.zeros((1, 10))
        actual = stable_softmax(x)
        assert torch.allclose(actual, expected)

    def test_input_tensor_of_non_neg_inf_values_returns_softmax(self):
        x = torch.randn((1, 10))
        expected = torch.softmax(x, dim=-1)
        actual = stable_softmax(x)
        assert torch.allclose(actual, expected)

    def test_input_tensor_of_some_neg_inf_values_returns_softmax(self):
        x = torch.tensor([-float("inf"), 0.0, -float("inf"), 0.0])
        expected = torch.tensor([0.0, 0.5, 0.0, 0.5])
        actual = stable_softmax(x)
        assert torch.allclose(actual, expected)


class TestSplitHeads:
    def test_correctly_splits_heads(self):
        # fmt: off
        tensor = torch.tensor([[[1, 2, 3, 4, 5, 6],
                                [7, 8, 9, 10, 11, 12]]])
        # fmt: on

        actual = split_heads(tensor, 2, 3)

        # fmt: off
        expected = torch.tensor([[[[1, 2, 3],
                                   [7, 8, 9]], 
                                  [[4, 5, 6],
                                   [10, 11, 12]]]])
        # fmt: on

        assert torch.allclose(actual, expected)


class TestConcatHeads:
    def test_correctly_concatenates_heads(self):
        # fmt: off
        tensor = torch.tensor([[[[1, 2, 3],
                                 [7, 8, 9]], 
                                [[4, 5, 6],
                                 [10, 11, 12]]]])
        # fmt: on

        actual = concat_heads(tensor)

        expected = torch.tensor([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]])

        assert torch.allclose(actual, expected)


class TestMultiHeadAttention:
    def setup_class(self):
        k1_proj = torch.linspace(0.01, 1.00, 24, dtype=torch.float32).view(4, 6)
        q1_proj = torch.linspace(0.05, 1.30, 24, dtype=torch.float32).view(4, 6)
        v1_proj = torch.linspace(0.08, 1.25, 32, dtype=torch.float32).view(4, 8)

        k2_proj = torch.linspace(0.03, 1.10, 24, dtype=torch.float32).view(4, 6)
        q2_proj = torch.linspace(0.05, 1.20, 24, dtype=torch.float32).view(4, 6)
        v2_proj = torch.linspace(0.07, 1.35, 32, dtype=torch.float32).view(4, 8)

        k_proj = torch.concat([k1_proj, k2_proj], dim=-1)
        q_proj = torch.concat([q1_proj, q2_proj], dim=-1)
        v_proj = torch.concat([v1_proj, v2_proj], dim=-1)
        o_proj = torch.linspace(0, 1.4, 64, dtype=torch.float32).reshape(16, 4)

        self.attention = MultiHeadAttention(
            num_heads=2, embedding_size=4, d_key=6, d_value=8
        )
        self.attention._q_proj.weight = torch.nn.Parameter(q_proj.T)
        self.attention._k_proj.weight = torch.nn.Parameter(k_proj.T)
        self.attention._v_proj.weight = torch.nn.Parameter(v_proj.T)
        self.attention._o_proj.weight = torch.nn.Parameter(o_proj.T)

    def test_attention_scores_are_calculated_correctly(self):
        x = torch.arange(0, 1.6, step=0.1, dtype=torch.float32).view(1, 4, 4)

        expected = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [6.3802, 6.5831, 6.7860, 6.9888],
                    [18.7467, 19.3392, 19.9318, 20.5243],
                    [31.1960, 32.1809, 33.1658, 34.1508],
                ]
            ]
        )

        actual, _ = self.attention(x)

        assert torch.allclose(actual, expected, atol=1e-4)

    def test_attention_scores_are_calculated_correctly_with_padding_mask(self):
        x = torch.arange(0, 1.6, step=0.1, dtype=torch.float32).view(1, 4, 4)

        padding_mask = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
        x[:, 3, :] = 0.0

        expected = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [6.3802, 6.5831, 6.7860, 6.9888],
                    [18.7467, 19.3392, 19.9318, 20.5243],
                    [0.0000, 0.0000, 0.0000, 0.0000],
                ]
            ]
        )

        actual, _ = self.attention(x, padding_mask)

        assert torch.allclose(actual, expected, atol=1e-4)
