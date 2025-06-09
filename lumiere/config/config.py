import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class that loads and provides access to YAML configs."""
    
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        if config_path is not None:
            self.config_path = Path(config_path)
            self._config = self._load_config()
        elif config_dict is not None:
            self._config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, section: str, key: str = None, default=None):
        """Get configuration value by section and optionally by key."""
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        return cls(config_dict=config_dict)


class ModelConfig(Config):
    """Configuration class for model configuration."""
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        super().__init__(config_path, config_dict)

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False)


class TokenizerConfig(Config):
    """Configuration class for tokenizer configuration."""
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        super().__init__(config_path, config_dict)

    @property
    def tokenizer(self) -> Dict[str, Any]:
        """Get tokenizer configuration."""
        return self._config.get('tokenizer', {})
    
    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False)