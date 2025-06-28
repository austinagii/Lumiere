import argparse
import logging
import os
import signal
import sys

import torch
import torch.nn.functional as F
from azure.storage.blob import BlobServiceClient

from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.model_manager import ModelManager
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.persistence.tokenizer_manager import TokenizerManager
from lumiere.utils import get_device


MODEL_CONFIG_DIR = "configs/models"
TOKENIZER_CONFIG_DIR = "configs/tokenizers"
CONFIG_FILE_EXTENSION = ".yaml"
MODEL_OUTPUT_DIR = "artifacts/models"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)


def signal_handler(sig, frame):
    """Handle inference interruption gracefully."""
    print()
    logger.info("Exiting inference...")
    sys.exit(0)


def main(
    model_name: str,
    checkpoint_name: str = None,
    model_manager: ModelManager = None,
    max_length: int = 200,
):
    signal.signal(signal.SIGINT, signal_handler)

    device = get_device()
    logger.info(f"Using device: {device}")

    model, model_config, tokenizer = model_manager.load_model(
        model_name, checkpoint_name, device
    )

    while True:
        text = input("User: ")

        tokens = tokenizer.encode(text).ids
        full_sequence = torch.tensor(tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(max_length):
                # Use only the last context_size tokens for model input
                if full_sequence.size(1) >= model.context_size:
                    model_input = full_sequence[:, -model.context_size :]
                else:
                    model_input = full_sequence

                logits, _ = model(model_input)
                probs = F.softmax(logits[0, -1], dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Append to the full sequence (never truncate this)
                full_sequence = torch.cat(
                    [full_sequence, torch.tensor([[next_token]]).to(device)], dim=1
                )

        output = tokenizer.decode(full_sequence[0].cpu().tolist())
        print(f"Model: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with the language model"
    )

    parser.add_argument(
        "model_name",
        default="transformer-tiny",
        help="Name of the model to run inference with",
    )
    parser.add_argument(
        "--checkpoint-name",
        dest="checkpoint_name",
        default=None,
        help="The checkpoint to run inference with",
    )
    parser.add_argument(
        "--max-length",
        dest="max_length",
        default=200,
        help="The maximum length of the generated text",
    )

    args = parser.parse_args()

    local_storage_client = LocalStorageClient()

    remote_storage_client = RemoteStorageClient(
        BlobServiceClient.from_connection_string(
            os.getenv("BLOB_STORAGE_CONNECTION_STRING")
        ),
        os.getenv("BLOB_STORAGE_CONTAINER_NAME"),
    )

    # Investigate using factory pattern.
    checkpoint_manager = CheckpointManager(
        remote_storage_client=remote_storage_client,
        local_storage_client=local_storage_client,
    )
    tokenizer_manager = TokenizerManager(
        remote_storage_client=remote_storage_client,
        local_storage_client=local_storage_client,
    )
    model_manager = ModelManager(checkpoint_manager, tokenizer_manager)

    main(args.model_name, args.checkpoint_name, model_manager, args.max_length)
