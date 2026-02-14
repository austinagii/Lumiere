from torch.nn import RMSNorm

from lumiere.nn.builder import ModelBuilder
from lumiere.nn.components.feedforward import (
    LinearFeedForward,
)


class TestModelBuilder:
    def test_build_produces_a_transformer_that_matches_provided_spec(self):
        spec = {
            "type": "transformer",
            "vocab_size": 1024,
            "context_size": 64,
            "num_blocks": 4,
            "embedding": {
                "type": "sinusoidal",
                "vocab_size": 1024,
                "context_size": 64,
                "embedding_size": 128,
                "padding_id": 0,
            },
            "block": {
                "type": "standard",
                "attention": {
                    "type": "multihead",
                    "num_heads": 4,
                    "embedding_size": 128,
                    "d_key": 32,
                    "d_value": 32,
                },
                "feedforward": {
                    "type": "linear",
                    "embedding_size": 128,
                    "d_ff": 256,
                    "dropout": 0.1,
                },
                "normalization": {
                    "type": "rms",
                    "normalized_shape": 128,
                },
                "dropout": 0.1,
                "pre_norm": True,
                "post_norm": True,
            },
            "normalization": {
                "type": "rms",
                "normalized_shape": 128,
            },
        }
        transformer = ModelBuilder.build(spec)

        assert transformer.context_size == 64
        assert transformer.num_blocks == 4

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
        spec = {
            "type": "transformer",
            "vocab_size": 1024,
            "context_size": 64,
            "embedding_size": 128,
            "normalized_shape": 128,
            "num_blocks": 4,
            "num_heads": 4,
            "d_key": 32,
            "d_value": 32,
            "embedding": {
                "type": "sinusoidal",
                "padding_id": 0,
            },
            "block": {
                "type": "standard",
                "attention": {
                    "type": "multihead",
                },
                "feedforward": {
                    "type": "linear",
                    "d_ff": 256,
                    "dropout": 0.1,
                },
                "normalization": {
                    "type": "rms",
                },
                "dropout": 0.1,
                "pre_norm": True,
                "post_norm": True,
            },
            "normalization": {
                "type": "rms",
            },
        }
        transformer = ModelBuilder.build(spec)

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
