from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfiguration:
    """Configuration class that loads and provides access to YAML configs."""

    config: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        """Get model configuration."""
        return self.config["model"]

    @property
    def tokenizer(self) -> dict[str, Any]:
        """Get tokenizer configuration."""
        return self.config["tokenizer"]

    @property
    def data(self) -> dict[str, Any]:
        """Get dataset configuration."""
        return self.config["data"]

    @property
    def training(self) -> dict[str, Any]:
        """Get training configuration."""
        return self.config["training"]

    @property
    def logging(self) -> dict[str, Any]:
        """Get logging configuration."""
        return self.config["logging"]

    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False)

    def __iter__(self):
        yield from self.config.items()

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Create a Config instance from a file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(config=config)
