import pytest
from torch.nn import RMSNorm

from lumiere.research.src.components.feedforward import (
    LinearFeedForward,
)
from lumiere.research.src.models.transformer_builder import (
    TransformerBuilder,
    TransformerSpec,
)


toy_model_spec = """
    context_size: 64
    embedding_size: 128
    num_blocks: 4
    embedding:
        type: sinusoidal
        padding_id: -1
    block:
        attention:
            num_heads: 4
            d_key: 32
            d_value: 32
        feedforward:
            type: linear
            embedding_size: 128
            d_ff: 256
            dropout: 0.1
        norm: pre,post
    dropout: 0.1
    normalization: rms
    vocab_size: -1
    padding_id: -1
"""

toy_model_spec_alt = """
    context_size: 64
    embedding_size: 128
    num_blocks: 4
    embedding:
        type: sinusoidal
        padding_id: -1
    block:
        attention:
            num_heads: 4
            d_key: 32
            d_value: 32
        feedforward:
            type: swiglu
            embedding_size: 128
            hidden_size: 256
            dropout: 0.1
        norm: pre,post
    dropout: 0.1
    normalization: rms
    vocab_size: -1
    padding_id: -1
"""

toy_model_spec_current = """
    vocab_size: -1
    context_size: 64
    embedding_size: 128
    num_blocks: 4
    num_heads: 4
    d_key: 32
    d_value: 32
    feedforward:
        type: swiglu
        embedding_size: 128
        hidden_size: 256
        dropout: 0.1
    dropout: 0.1
    padding_id: -1
    norm: pre,post
    normalization: rms
"""


@pytest.fixture
def transformer_spec():
    return TransformerSpec.from_bytes(toy_model_spec_alt)


# def recursive_verify(transformer_spec, transformer):
#     for module_path, field in _dfs(_to_tree(transformer_spec)):
#         transformer.module_path.field == field


class TestTransformerBuilder:
    def test_build_produces_a_transformer_that_matches_provided_spec(
        self, transformer_spec
    ):
        spec_yaml = b"""
            vocab_size: 1024
            context_size: 64
            embedding_size: 128
            num_layers: 4
            num_heads: 4
            d_key: 32
            d_value: 32
            feedforward_factory:
                type: linear
                embedding_size: 128
                d_ff: 256
                dropout: 0.1
            dropout: 0.1
            padding_id: 0
            pre_norm: true
            post_norm: true
            norm_type: rms
        """
        spec = TransformerSpec.from_bytes(spec_yaml)

        # transformer_spec.update({"vocab_size": 1096, "padding_id": 0})
        transformer = TransformerBuilder.build(spec)

        assert transformer.context_size == 64
        assert transformer.num_layers == 4

        # assert isinstance(transformer.attention, SinusoidalEmbedding)
        assert transformer.embedding.context_size == 64
        assert transformer.embedding.vocab_size == 1024
        assert transformer.embedding.embedding_size == 128

        assert len(transformer.blocks) == 4
        for block in transformer.blocks:
            assert block.attention.num_heads == 4
            assert block.attention.embedding_size == 128
            assert block.attention.d_key == 32
            assert block.attention.d_value == 32

            assert isinstance(block.feedforward, LinearFeedForward)
            assert block.feedforward.embedding_size == 128
            assert block.feedforward.d_ff == 256
            assert block.feedforward.dropout == 0.1

            assert isinstance(block.normalization_1, RMSNorm)
            assert isinstance(block.normalization_2, RMSNorm)
            assert isinstance(block.normalization_3, RMSNorm)
            assert block._dropout == 0.1
            assert block._pre_norm
            assert block._post_norm

        assert isinstance(transformer.final_norm, RMSNorm)

    # @pytest.mark.parametrize(
    #     "transformer_spec,expected_modules",
    #     [
    #         (toy_model_spec, {"ff": LinearFeedForward}),
    #         (toy_model_spec_alt, {"swiglu": SwigluFeedForward}),
    #     ],
    # )
    # def test_build_allows_for_transformers_with_dynamic_layers(
    #     self, transformer_spec, expected_modules
    # ):
    #     # transformer_spec.update({"vocab_size": 1096, "padding_id": 0})
    #     transformer = TransformerBuilder.build(transformer_spec)
    #
    #     assert transformer.context_size == 64
    #     assert transformer.num_layers == 4
    #
    #     # assert isinstance(transformer.attention, SinusoidalEmbedding)
    #     assert transformer.embedding.context_size == 64
    #     assert transformer.embedding.vocab_size == 1096
    #     assert transformer.embedding.embedding_size == 128
    #
    #     assert len(transformer.blocks) == 4
    #     for block in transformer.blocks:
    #         assert block.attention.num_heads == 4
    #         assert block.attention.embedding_size == 128
    #         assert block.attention.d_key == 32
    #         assert block.attention.d_value == 32
    #
    #         assert isinstance(block.feedforward, expected_modules["ff"])
    #         assert block.feedforward.embedding_size == 128
    #         assert block.feedforward.d_ff == 256
    #         assert block.feedforward.dropout == 0.1
    #
    #         assert isinstance(block.normalization_1, RMSNorm)
    #         assert isinstance(block.normalization_2, RMSNorm)
    #         assert isinstance(block.normalization_3, RMSNorm)
    #         assert block._dropout == 0.1
    #         assert block._pre_norm
    #         assert block._post_norm
    #
    #     assert isinstance(transformer.final_norm, RMSNorm)

    def test_build_raises_an_error_if_no_spec_is_provided(self):
        pass

    def test_build_raises_an_error_if_provided_spec_has_errors(self):
        pass

    def test_build_uses_top_level_spec_where_layer_spec_is_missing(self):
        pass

    def test_build_uses_layer_spec_if_provided(self):
        pass
