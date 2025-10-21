# import torch
# from research.src.components import TransformerBlock
#
#
# def verify_normalization(x: torch.Tensor):
#     assert torch.allclose(torch.norm(x), 1)
#
#
# @pytest.fixture
# def block():
#     return TransformerBlock()
#
#
# class TestTransformerBlock:
#     def test_pre_normalization_is_applied(self, block):
#         norm_verifier = NormVerifier()
#
#         block.attention.register_pre_hook(norm_verifier.verify_normalization)
#
#         assert norm_verifier.is_normalized
