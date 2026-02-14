import pytest

from lumiere.training.config import Config


@pytest.fixture
def file(tmp_path):
    content = """
        model:
            name: gpt2
            vocab_size: 50257
            n_layers: 12
            n_heads: 12
            n_embd: 768
        tokenizer:
            name: gpt2
            vocab_size: 50257
        data:
            name: wikitext2
            split: train
        training:
            batch_size: 16
            learning_rate: 0.001
            num_epochs: 10
        logging:
            log_interval: 100
    """

    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


class TestConfig:
    def test_config_can_be_loaded_from_a_file(self, file):
        config = Config.from_file(file)
        assert config["model.name"] == "gpt2"
        assert config["model.vocab_size"] == 50257
        assert config["model.n_layers"] == 12
        assert config["model.n_heads"] == 12
        assert config["model.n_embd"] == 768

        assert config["tokenizer.name"] == "gpt2"
        assert config["tokenizer.vocab_size"] == 50257

        assert config["data.name"] == "wikitext2"
        assert config["data.split"] == "train"

        assert config["training.batch_size"] == 16
        assert config["training.learning_rate"] == 0.001
        assert config["training.num_epochs"] == 10
        assert config["logging.log_interval"] == 100

    def test_an_error_is_raised_if_the_config_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            Config.from_file("non_existent_file.yaml")


# class TestModelSpec:
#     def test_model_spec_can_be_initialized_from_argument_dict(self):
#         args = {
#             "context_size": 512,
#             "embedding_size": 1024,
#             "num_blocks": 6,
#             "block": {
#                 "type": "standard",
#                 "hidden_size": 768,
#                 "dropout": 0.1,
#                 "feedforward": {
#                     "type": "linear",
#                     "d_ff": 2048,
#                 },
#             },
#         }
#         spec = ModelSpec(args)
#
#         assert spec.args == args
#         assert spec["context_size"] == 512
#         assert spec["embedding_size"] == 1024
#         assert spec["num_blocks"] == 6
#         assert spec["block"]["type"] == "standard"
#         assert spec["block"]["hidden_size"] == 768
#         assert spec["block.dropout"] == 0.1
#         assert spec["block.feedforward.type"] == "linear"
#         assert spec["block.feedforward.d_ff"] == 2048
#
#     def test_init_raises_an_error_if_args_is_none(self):
#         with pytest.raises(ValueError):
#             ModelSpec(None)
#
#     def test_from_yaml_correctly_builds_spec_from_yaml_file(self):
#         yaml_content = """
#         context_size: 512
#         embedding_size: 1024
#         num_layers: 6
#         block:
#             type: standard
#             hidden_size: 768
#             dropout: 0.1
#             feedforward:
#                 type: linear
#                 d_ff: 2048
#         """
#         with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
#             f.write(yaml_content)
#             yaml_path = f.name
#
#         try:
#             spec = ModelSpec.from_yaml(yaml_path)
#
#             assert spec["context_size"] == 512
#             assert spec["embedding_size"] == 1024
#             assert spec["num_layers"] == 6
#             assert spec["block.type"] == "standard"
#             assert spec["block.hidden_size"] == 768
#             assert spec["block.dropout"] == 0.1
#             assert spec["block.feedforward.type"] == "linear"
#             assert spec["block.feedforward.d_ff"] == 2048
#         finally:
#             Path(yaml_path).unlink()
#
#     def test_from_yaml_raises_error_if_file_path_is_invalid(self):
#         with pytest.raises(ValueError):
#             ModelSpec.from_yaml(123)
#
#     def test_from_yaml_raises_error_if_file_path_does_not_exist(self):
#         with pytest.raises(FileNotFoundError):
#             ModelSpec.from_yaml("/path/to/nonexistent/file.yaml")
#
#     def test_getitem_retrieves_argument_value(self):
#         spec = ModelSpec(
#             {
#                 "context_size": 512,
#                 "embedding_size": 1024,
#                 "num_blocks": 6,
#             }
#         )
#
#         assert spec["context_size"] == 512
#         assert spec["embedding_size"] == 1024
#         assert spec["num_blocks"] == 6
#
#     def test_getitem_retrieves_component_arguments_using_dot_notation(self):
#         spec = ModelSpec(
#             {
#                 "context_size": 512,
#                 "embedding_size": 1024,
#                 "num_blocks": 6,
#                 "block": {"type": "standard", "hidden_size": 768, "dropout": 0.1},
#             }
#         )
#
#         assert spec["block.hidden_size"] == 768
#
#     def test_setitem_sets_argument_to_specified_value(self):
#         spec = ModelSpec(
#             {
#                 "context_size": 512,
#                 "embedding_size": 1024,
#                 "num_blocks": 6,
#                 "block": {"type": "standard", "hidden_size": 768, "dropout": 0.1},
#             }
#         )
#
#         spec["context_size"] = 2048
#         spec["block.hidden_size"] = 1024
#
#         assert spec["context_size"] == 2048
#         assert spec["block.hidden_size"] == 1024
#
#     def test_setitem_creates_ancestor_if_missing(self):
#         spec = ModelSpec(
#             {
#                 "context_size": 512,
#                 "embedding_size": 1024,
#                 "num_blocks": 6,
#             }
#         )
#
#         spec["block.feedforward.type"] = "standard"
#
#         assert spec["block.feedforward"] == {"type": "standard"}
