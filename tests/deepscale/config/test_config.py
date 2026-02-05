import pytest

from lumiere.deepscale.config import Config


@pytest.fixture
def config_file_path(tmp_path):
    # TODO: Update this to reflect a deepscale config to avoid confusion.
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
        dataset:
            name: wikitext2
            split: train
        training:
            batch_size: 16
            learning_rate: 0.001
            num_epochs: 10
        logging:
            interval: 100
    """

    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


class TestConfig:
    def test_config_can_be_created_from_yaml_file(self, config_file_path) -> None:
        config = Config.from_yaml(config_file_path)

        assert config.data is not None
        assert config.data["model"] is not None
        assert config.data["tokenizer"] is not None
        assert config.data["dataset"] is not None
        assert config.data["training"] is not None
        assert config.data["logging"] is not None

    def test_config_values_can_be_accessed_using_custom_dot_syntax(
        self, config_file_path
    ) -> None:
        config = Config.from_yaml(config_file_path)

        assert config["model.n_layers"] == 12
        assert config["tokenizer.name"] == "gpt2"
        assert config["dataset.split"] == "train"
        assert config["training.batch_size"] == 16
        assert config["logging.interval"] == 100

    def test_config_values_can_be_set_using_custom_dot_syntax(
        self, config_file_path
    ) -> None:
        config = Config.from_yaml(config_file_path)

        assert config["model.n_layers"] == 12

        config["model.n_layers"] = 16

        assert config["model.n_layers"] == 16

    def test_config_values_can_be_created_using_custom_dot_syntax(
        self, config_file_path
    ) -> None:
        config = Config.from_yaml(config_file_path)

        assert config.get("model.optimizer.learning_rate") is None

        config["model.optimizer.learning_rate"] = 0.001

        assert config["model.optimizer.learning_rate"] == 0.001


