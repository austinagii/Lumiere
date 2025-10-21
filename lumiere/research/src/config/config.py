from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Configuration class that loads and provides access to YAML configs."""

    config: Dict[str, Any]

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config["model"]

    @property
    def tokenizer(self) -> Dict[str, Any]:
        """Get tokenizer configuration."""
        return self.config["tokenizer"]

    @property
    def dataset(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config["dataset"]

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config["training"]

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config["logging"]

    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False)

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Create a Config instance from a file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(config=config)
