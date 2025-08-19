import argparse
import logging
import os

import torch
from azure.storage.blob import BlobServiceClient
from torch.nn import functional as F
from tqdm import tqdm

from lumiere.config.config import Config
from lumiere.data.dataloader import get_data_loader
from lumiere.data.preprocessing import to_training_batches
from lumiere.data.tokenizer import SPECIAL_TOKENS
from lumiere.models.transformer import Transformer
from lumiere.persistence.checkpoint_manager import CheckpointManager
from lumiere.persistence.storage_client import LocalStorageClient, RemoteStorageClient
from lumiere.persistence.tokenizer_manager import TokenizerManager
from lumiere.utils import get_device
from lumiere.utils.run_finder import RunFinder


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
    run_id: str,
    checkpoint_name: str = None,
    checkpoint_manager: CheckpointManager = None,
    tokenizer_manager: TokenizerManager = None,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    # Load the checkpoint
    if checkpoint_name is None:
        raise ValueError("checkpoint_name is required")

    if checkpoint_manager is None:
        raise ValueError("A checkpoint manager is required to load from checkpoint")

    # Try to find the run name for the given run ID locally first
    run_name = RunFinder(
        BlobServiceClient.from_connection_string(
            os.getenv("BLOB_STORAGE_CONNECTION_STRING")
        )
    ).find_run(run_id)

    if run_name is None:
        raise ValueError(f"Run with ID '{run_id}' not found")

    checkpoint = checkpoint_manager.load_checkpoint(run_name, checkpoint_name, device)
    logger.info("Checkpoint loaded successfully")

    # Load the model config from checkpoint
    model_config = Config(checkpoint["model_config"])
    logger.info(f"Loaded model config: {model_config}")

    logger.info(f"Loading tokenizer for run '{run_name}'...")
    tokenizer = tokenizer_manager.load_tokenizer(run_name)
    logger.info("Tokenizer loaded successfully")

    # Initialize the model
    logger.info("Initializing model...")
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
        padding_id=SPECIAL_TOKENS["padding"].id,
    ).to(device)

    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded successfully")

    # Load the dataset
    logger.info("Loading the dataset...")
    dataloader = get_data_loader(
        dataset_name=model_config.dataset["name"],
        train_dataset_percentage=model_config.dataset["train_portion"],
        validation_dataset_percentage=model_config.dataset["validation_portion"],
    )
    logger.info("Dataset loaded successfully")

    # Use validation dataset for evaluation
    validation_batches = to_training_batches(
        corpus=dataloader.iter_validation(),
        tokenizer=tokenizer,
        context_size=model_config.model["context_size"] + 1,
        batch_size=model_config.training["batch_size"],
        pad_id=SPECIAL_TOKENS["padding"].id,
        sliding_window_size=model_config.dataset["sliding_window_size"],
    )

    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    with tqdm(validation_batches, desc="Validation Evaluation:") as pbar:
        for batch, padding_mask in pbar:
            num_batches += 1
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            padding_mask = padding_mask[:, :-1].to(device)

            with torch.no_grad():
                logits, _ = model(x, padding_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=SPECIAL_TOKENS["padding"].id,
                )
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
        f"Validation set evaluation complete. "
        f"Avg loss: {avg_batch_loss:.4f} Avg perplexity: {avg_batch_perplexity:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")

    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="The run ID to evaluate",
    )
    parser.add_argument(
        "--checkpoint-name",
        dest="checkpoint_name",
        default=None,
        help="The checkpoint to evaluate",
    )

    args = parser.parse_args()

    if args.run_id is None and args.checkpoint_name is None:
        raise ValueError("Either run_id or checkpoint_name is required")

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

    main(args.run_id, args.checkpoint_name, checkpoint_manager, tokenizer_manager)
