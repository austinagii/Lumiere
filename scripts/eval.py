import argparse
import logging
import os

import datasets
import torch
from azure.storage.blob import BlobServiceClient
from torch.nn import functional as F
from tqdm import tqdm

from lumiere.data.wikitext import to_batches
from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.model_manager import ModelManager
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.persistence.tokenizer_manager import TokenizerManager
from lumiere.utils import get_device


MODEL_CONFIG_DIR = "configs/models"
TOKENIZER_CONFIG_DIR = "configs/tokenizers"
CONFIG_FILE_EXTENSION = "yaml"

MODEL_OUTPUT_DIR = "artifacts/models"
MODEL_FILE_EXTENSION = "pth"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TEXT_COLUMN_NAME = "text"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# Disable Azure blob storage logging
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main(
    model_name: str,
    checkpoint_name: str = None,
    model_manager: ModelManager = None,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    model, model_config, tokenizer = model_manager.load_model(
        model_name, checkpoint_name, device
    )

    dataset = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    print(f"Loaded {len(dataset)} samples")

    batches = to_batches(
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=model_config.training["batch_size"],
        context_size=model_config.model["context_size"] + 1,
    )

    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    with tqdm(batches, desc="Test Evaluation:") as pbar:
        for batch in batches:
            num_batches += 1
            x, y = batch[:, :-1], batch[:, 1:]
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                perplexity = torch.exp(loss)
                total_loss += loss.item()
                total_perplexity += perplexity.item()
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "perplexity": f"{perplexity.item():.4f}",
                    }
                )

    avg_batch_loss = total_loss / num_batches
    avg_batch_perplexity = total_perplexity / num_batches
    print(
        f"Test set evaluation complete. "
        f"Avg loss: {avg_batch_loss:.4f} Avg perplexity: {avg_batch_perplexity:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")

    parser.add_argument(
        "model_name",
        default="transformer-tiny",
        help="Name of the model to evaluate",
    )
    parser.add_argument(
        "--checkpoint-name",
        dest="checkpoint_name",
        default=None,
        help="The checkpoint to evaluate",
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

    main(args.model_name, args.checkpoint_name, model_manager)
