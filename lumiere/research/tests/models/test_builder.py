from torch.nn import RMSNorm

from lumiere.research.src.components.feedforward import (
    LinearFeedForward,
)
from lumiere.research.src.models.transformer_builder import (
    TransformerBuilder,
    TransformerSpec,
)


class TestTransformerBuilder:
    def test_build_produces_a_transformer_that_matches_provided_spec(self):
        spec = TransformerSpec(
            {
                "vocab_size": 1024,
                "context_size": 64,
                "embedding_size": 128,
                "num_layers": 4,
                "num_heads": 4,
                "d_key": 32,
                "d_value": 32,
                "feedforward_factory": {
                    "type": "linear",
                    "embedding_size": 128,
                    "d_ff": 256,
                    "dropout": 0.1,
                },
                "dropout": 0.1,
                "padding_id": 0,
                "pre_norm": True,
                "post_norm": True,
                "norm_type": "rms",
            }
        )
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

    def test_build_inherits_ancestor_args_when_module_args_are_omitted(self):
        spec = TransformerSpec(
            {
                "vocab_size": 1024,
                "context_size": 64,
                "embedding_size": 128,
                "num_layers": 4,
                "num_heads": 4,
                "d_key": 32,
                "d_value": 32,
                "feedforward_factory": {
                    "type": "linear",
                    "d_ff": 256,
                    "dropout": 0.1,
                },
                "dropout": 0.1,
                "padding_id": 0,
                "pre_norm": True,
                "post_norm": True,
                "norm_type": "rms",
            }
        )
        transformer = TransformerBuilder.build(spec)

        for block in transformer.blocks:
            assert block.feedforward.embedding_size == 128

    def test_build_raises_an_error_if_no_spec_is_provided(self):
        pass

    def test_build_raises_an_error_if_provided_spec_has_errors(self):
        pass

    def test_build_uses_top_level_spec_where_layer_spec_is_missing(self):
        pass

    def test_build_uses_layer_spec_if_provided(self):
        pass
