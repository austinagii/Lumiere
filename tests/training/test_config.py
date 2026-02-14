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
        config = Config.from_file(file, override=True)
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

    def test_delitem_deletes_top_level_key(self):
        config = Config({"model": {"vocab_size": 1000}, "tokenizer": {"name": "gpt2"}}, override=True)

        del config["tokenizer"]

        assert config.get("tokenizer") is None
        assert config["model.vocab_size"] == 1000

    def test_delitem_deletes_nested_key_with_dot_notation(self):
        config = Config({"model": {"vocab_size": 1000, "layers": 12}}, override=True)

        del config["model.vocab_size"]

        assert config.get("model.vocab_size") is None
        assert config["model.layers"] == 12

    def test_delitem_deletes_deeply_nested_key(self):
        config = Config(
            {
                "model": {
                    "block": {
                        "feedforward": {
                            "type": "linear",
                            "d_ff": 2048
                        }
                    }
                }
            },
            override=True
        )

        del config["model.block.feedforward.type"]

        assert config.get("model.block.feedforward.type") is None
        assert config["model.block.feedforward.d_ff"] == 2048

    def test_delitem_raises_error_for_missing_key(self):
        config = Config({"model": {"vocab_size": 1000}}, override=True)

        with pytest.raises(KeyError, match="Field 'model.nonexistent' not found in config"):
            del config["model.nonexistent"]

    def test_delitem_raises_error_for_invalid_key_type(self):
        config = Config({"model": {"vocab_size": 1000}}, override=True)

        with pytest.raises(TypeError, match="Key must be a non-empty string"):
            del config[123]

    def test_setitem_creates_ancestor_if_missing(self):
        config = Config(
            {
                "context_size": 512,
                "embedding_size": 1024,
                "num_blocks": 6,
            },
            override=True
        )

        config["block.feedforward.type"] = "standard"

        assert config["block.feedforward.type"] == "standard"
        assert config["block"] == {"feedforward": {"type": "standard"}}
