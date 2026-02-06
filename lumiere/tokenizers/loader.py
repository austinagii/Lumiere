import contextlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumiere.tokenizers.base import Tokenizer


module_discovery_path = [
    "lumiere.data.datasets",
    "lumiere.tokenizers",
    "lumiere.model.components",
]


class TokenizerLoader:
    _registry: Mapping[str, type[Tokenizer]] = {}

    @classmethod
    def load(cls, spec: dict[str, Any]) -> Tokenizer:
        tokenizer_type = spec.get("type")
        if tokenizer_type is None:
            raise ValueError("Tokenizer specification missing required 'type' key.")

        tokenizers_dir = Path(__file__).parent
        module_files = tokenizers_dir.glob("*.py")
        module_file = None
        for file in module_files:
            if tokenizer_type in file.stem:
                module_file = file
                break

        if module_file is None:
            raise ValueError(f"Tokenizer '{tokenizer_type}' could not be found.")

        module_name = f"lumiere.tokenizers.{module_file.stem}"
        with contextlib.suppress(ImportError):
            importlib.import_module(module_name)

        # TODO: Still need to know what the class name is for initializer.
        # return _dataset_registry.get(dataset_name)

        tokenizer_cls = cls._registry.get(tokenizer_type)

        return tokenizer_cls(**spec)
