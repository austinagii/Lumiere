import tempfile
from pathlib import Path

import pytest
from torch.nn import RMSNorm

from lumiere.nn.components.feedforward import (
    LinearFeedForward,
)
from lumiere.nn.builder import (
    TransformerBuilder,
    ModelSpec,
)


class TestModelSpec:
    def test_model_spec_can_be_initialized_from_argument_dict(self):
        args = {
            "context_size": 512,
            "embedding_size": 1024,
            "num_blocks": 6,
            "block": {
                "type": "standard",
                "hidden_size": 768,
                "dropout": 0.1,
                "feedforward": {
                    "type": "linear",
                    "d_ff": 2048,
                },
            },
        }
        spec = ModelSpec(args)

        assert spec.args == args
        assert spec["context_size"] == 512
        assert spec["embedding_size"] == 1024
        assert spec["num_blocks"] == 6
        assert spec["block"]["type"] == "standard"
        assert spec["block"]["hidden_size"] == 768
        assert spec["block.dropout"] == 0.1
        assert spec["block.feedforward.type"] == "linear"
        assert spec["block.feedforward.d_ff"] == 2048

    def test_init_raises_an_error_if_args_is_none(self):
        with pytest.raises(ValueError):
            ModelSpec(None)

    def test_from_yaml_correctly_builds_spec_from_yaml_file(self):
        yaml_content = """
        context_size: 512
        embedding_size: 1024
        num_layers: 6
        block:
            type: standard
            hidden_size: 768
            dropout: 0.1
            feedforward:
                type: linear
                d_ff: 2048
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            spec = ModelSpec.from_yaml(yaml_path)

            assert spec["context_size"] == 512
            assert spec["embedding_size"] == 1024
            assert spec["num_layers"] == 6
            assert spec["block.type"] == "standard"
            assert spec["block.hidden_size"] == 768
            assert spec["block.dropout"] == 0.1
            assert spec["block.feedforward.type"] == "linear"
            assert spec["block.feedforward.d_ff"] == 2048
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_raises_error_if_file_path_is_invalid(self):
        with pytest.raises(ValueError):
            ModelSpec.from_yaml(123)

    def test_from_yaml_raises_error_if_file_path_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            ModelSpec.from_yaml("/path/to/nonexistent/file.yaml")

    def test_getitem_retrieves_argument_value(self):
        spec = ModelSpec(
            {
                "context_size": 512,
                "embedding_size": 1024,
                "num_blocks": 6,
            }
        )

        assert spec["context_size"] == 512
        assert spec["embedding_size"] == 1024
        assert spec["num_blocks"] == 6

    def test_getitem_retrieves_component_arguments_using_dot_notation(self):
        spec = ModelSpec(
            {
                "context_size": 512,
                "embedding_size": 1024,
                "num_blocks": 6,
                "block": {"type": "standard", "hidden_size": 768, "dropout": 0.1},
            }
        )

        assert spec["block.hidden_size"] == 768

    def test_setitem_sets_argument_to_specified_value(self):
        spec = ModelSpec(
            {
                "context_size": 512,
                "embedding_size": 1024,
                "num_blocks": 6,
                "block": {"type": "standard", "hidden_size": 768, "dropout": 0.1},
            }
        )

        spec["context_size"] = 2048
        spec["block.hidden_size"] = 1024

        assert spec["context_size"] == 2048
        assert spec["block.hidden_size"] == 1024

    def test_setitem_creates_ancestor_if_missing(self):
        spec = ModelSpec(
            {
                "context_size": 512,
                "embedding_size": 1024,
                "num_blocks": 6,
            }
        )

        spec["block.feedforward.type"] = "standard"

        assert spec["block.feedforward"] == {"type": "standard"}


class TestTransformerBuilder:
    def test_build_produces_a_transformer_that_matches_provided_spec(self):
        spec = ModelSpec(
            {
                "vocab_size": 1024,
                "context_size": 64,
                "num_blocks": 4,
                "embedding_factory": {
                    "type": "sinusoidal",
                    "vocab_size": 1024,
                    "context_size": 64,
                    "embedding_size": 128,
                    "padding_id": 0,
                },
                "block_factory": {
                    "type": "standard",
                    "attention_factory": {
                        "type": "multihead",
                        "num_heads": 4,
                        "embedding_size": 128,
                        "d_key": 32,
                        "d_value": 32,
                    },
                    "feedforward_factory": {
                        "type": "linear",
                        "embedding_size": 128,
                        "d_ff": 256,
                        "dropout": 0.1,
                    },
                    "normalization_factory": {
                        "type": "rms",
                        "normalized_shape": 128,
                    },
                    "dropout": 0.1,
                    "pre_norm": True,
                    "post_norm": True,
                },
                "normalization_factory": {
                    "type": "rms",
                    "normalized_shape": 128,
                },
            }
        )
        transformer = TransformerBuilder.build(spec)

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
        spec = ModelSpec(
            {
                "vocab_size": 1024,
                "context_size": 64,
                "embedding_size": 128,
                "normalized_shape": 128,
                "num_blocks": 4,
                "num_heads": 4,
                "d_key": 32,
                "d_value": 32,
                "embedding_factory": {
                    "type": "sinusoidal",
                    "padding_id": 0,
                },
                "block_factory": {
                    "type": "standard",
                    "attention_factory": {
                        "type": "multihead",
                    },
                    "feedforward_factory": {
                        "type": "linear",
                        "d_ff": 256,
                        "dropout": 0.1,
                    },
                    "normalization_factory": {
                        "type": "rms",
                    },
                    "dropout": 0.1,
                    "pre_norm": True,
                    "post_norm": True,
                },
                "normalization_factory": {
                    "type": "rms",
                },
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
