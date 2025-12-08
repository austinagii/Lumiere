import pytest

from lumiere.config.config import Config


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
        assert config.model["name"] == "gpt2"
        assert config.model["vocab_size"] == 50257
        assert config.model["n_layers"] == 12
        assert config.model["n_heads"] == 12
        assert config.model["n_embd"] == 768

        assert config.tokenizer["name"] == "gpt2"
        assert config.tokenizer["vocab_size"] == 50257

        assert config.dataset["name"] == "wikitext2"
        assert config.dataset["split"] == "train"

        assert config.training["batch_size"] == 16
        assert config.training["learning_rate"] == 0.001
        assert config.training["num_epochs"] == 10
        assert config.logging["log_interval"] == 100

    def test_an_error_is_raised_if_the_config_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            Config.from_file("non_existent_file.yaml")
