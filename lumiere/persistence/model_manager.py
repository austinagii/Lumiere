import logging
from typing import Tuple

import torch

from lumiere.config.config import TransformerConfig
from lumiere.data.tokenizer import Tokenizer
from lumiere.models import Transformer
from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.tokenizer_manager import TokenizerManager
from lumiere.training.checkpoint import CheckpointType


logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self,
        checkpoint_manager: CheckpointManager = None,
        tokenizer_manager: TokenizerManager = None,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer_manager = tokenizer_manager

    def load_model(
        self,
        model_name: str,
        checkpoint_name: str = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Transformer, TransformerConfig, Tokenizer]:
        """Load the model from local storage or blob storage.

        This function returns the model, fully constructed from the model config
        with it's weights loaded from either the final checkpoint of or the specified
        checkpoint. If the checkpoint is not found in local storage, it is synced from
        blob storage.

        Raises:
            - ConfigError: If the model config is not found.
            - PersistenceError: If the model cannot be loaded.
        """
        if checkpoint_name is None:
            checkpoint_name = str(CheckpointType.FINAL)

        checkpoint = self.checkpoint_manager.load_checkpoint(
            model_name, checkpoint_name, device
        )
        model_config = TransformerConfig.from_dict(checkpoint["model_config"])

        tokenizer = self.tokenizer_manager.load_tokenizer(
            model_config.model["tokenizer"]
        )

        # Load and initialize the model.
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            embedding_size=model_config.model["embedding_size"],
            context_size=model_config.model["context_size"],
            num_layers=model_config.model["num_layers"],
            num_heads=model_config.model["num_heads"],
            d_key=model_config.model["d_key"],
            d_value=model_config.model["d_value"],
            d_ff=model_config.model["d_ff"],
            dropout=model_config.model["dropout"],
        )

        # Initialize the model weights.
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict, strict=True)
        model.to(device)

        return model, model_config, tokenizer
