"""Evaluation script for Lumiére models."""

import argparse
import logging
import sys

import torch
from tqdm import tqdm

from lumiere import Loader, register_dependency
from lumiere.training import Checkpoint, RunManager, TrainingState
from lumiere.training.config import Config
from lumiere.training.loss import cross_entropy_loss
from lumiere.utils import get_device, register_signal_handlers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("testing.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

RUN_CONFIG_PATH = "lumiere.yaml"


def load_test_components(config: Config, checkpoint: Checkpoint, device: torch.device):
    """Load components for evaluation.

    Args:
        config: The configuration for the training run.
        checkpoint: The checkpoint to be loaded.
        device: The device to use for testing.

    Returns:
        Tuple of (model, dataloader, pipeline, tokenizer)
    """
    # fmt: off
    assert config.get("tokenizer") is not None, "Config missing required 'tokenizer' section"  # NOQA: E501
    assert config.get("data") is not None, "Config missing required 'data' section"
    assert config.get("pipeline") is not None, "Config missing required 'pipeline' section"  # NOQA: E501
    assert config.get("model") is not None, "Config missing required 'model' section"
    # fmt: on

    logger.info("Loading tokenizer from checkpoint...")
    tokenizer = Loader.tokenizer(config["tokenizer"], state=checkpoint["tokenizer"])
    register_dependency("tokenizer", tokenizer)
    logger.info(f"Tokenizer loaded successfully (vocab size: {tokenizer.vocab_size})")

    logger.info("Loading dataset...")
    dataloader = Loader.data(config["data"])
    logger.info(f"Dataloader initialized with {len(dataloader.datasets)} dataset(s)")

    logger.info("Loading pipeline...")
    pipeline = Loader.pipeline(config["pipeline"])
    logger.info("Pipeline loaded successfully")

    logger.info("Building model...")
    model = Loader.model(config["model"]).to(device)
    assert "model_state_dict" in checkpoint, (
        "Model state could not be found in checkpoint"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model loaded successfully - "
        f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
    )

    logger.info("All components loaded successfully from checkpoint")

    return model, dataloader, pipeline, tokenizer


def evaluate(model, dataloader, pipeline, device):
    """Evaluate the model on the test split.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader containing the test split.
        pipeline: The pipeline for processing data.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing evaluation metrics (avg_loss, perplexity, num_batches).
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    logger.info("Starting evaluation on test split...")

    with (
        torch.no_grad(),
        tqdm(
            pipeline.batches(dataloader["test"]),
            desc="Evaluating test split",
        ) as pbar,
    ):
        for samples, targets in pbar:
            outputs = model(*samples) if isinstance(samples, tuple) else model(samples)
            batch_loss = cross_entropy_loss(outputs, targets)
            total_loss += batch_loss
            num_batches += 1
            avg_loss = total_loss / num_batches
            avg_perplexity = torch.exp(avg_loss)

            pbar.set_postfix(
                {
                    "batch_loss": f"{batch_loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "avg_ppl": f"{avg_perplexity:.4f}",
                }
            )

    avg_loss = total_loss / num_batches
    avg_perplexity = torch.exp(avg_loss)

    return {
        "avg_loss": avg_loss.item(),
        "avg_perplexity": avg_perplexity.item(),
        "num_batches": num_batches,
    }


def test(
    run_id: str,
    checkpoint_tag: str,
):
    """Evaluate a model checkpoint on the test split.

    Args:
        run_id: The ID of the training run which produced the checkpoint.
        checkpoint_tag: The tag of the checkpoint to evaluate (e.g., 'best', 'final', 'epoch:0001').
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Initializing RunManager from {RUN_CONFIG_PATH}...")
    run_manager = RunManager.from_config_file(RUN_CONFIG_PATH)
    logger.info("RunManager initialized")

    if checkpoint_tag is None:
        checkpoint_tag = "best"
        logger.info(f"No checkpoint tag specified, defaulting to '{checkpoint_tag}'")

    logger.info(f"Loading run '{run_id}' from checkpoint '{checkpoint_tag}'...")
    config, checkpoint = run_manager.resume_run(run_id, checkpoint_tag, device=device)
    logger.info("Checkpoint loaded successfully")

    model, dataloader, pipeline, tokenizer = load_test_components(
        config, checkpoint, device
    )

    training_state = TrainingState.from_dict(checkpoint["training_state"])
    logger.info(
        f"Checkpoint is from epoch {training_state.current_epoch} "
        f"(global step {training_state.global_step})"
    )

    logger.info("=" * 60)
    logger.info(f"Evaluating run: {run_id}")
    logger.info(f"Checkpoint: {checkpoint_tag}")
    logger.info("=" * 60)

    metrics = evaluate(model, dataloader, pipeline, device)

    logger.info("=" * 60)
    logger.info("Test Evaluation Results:")
    logger.info(f"  Average Loss: {metrics['avg_loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['avg_perplexity']:.4f}")
    logger.info(f"  Batches Processed: {metrics['num_batches']}")
    logger.info("=" * 60)

    return metrics


def main():
    """Main entry point for the test script."""
    register_signal_handlers()

    parser = argparse.ArgumentParser(
        description="Evaluate a Lumière model checkpoint on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-id",
        dest="run_id",
        required=True,
        help="The run ID to load the checkpoint from",
    )

    parser.add_argument(
        "--checkpoint-tag",
        dest="checkpoint_tag",
        default=None,
        help="The checkpoint tag to evaluate (e.g., 'best', 'final', 'epoch:0001'). Defaults to 'best'",
    )

    args = parser.parse_args()

    try:
        test(
            run_id=args.run_id,
            checkpoint_tag=args.checkpoint_tag,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
