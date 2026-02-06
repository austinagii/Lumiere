import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import lumiere.deepscale as ds

from lumiere.config.config import Config
from lumiere.data.dataloader import get_data_loader
from lumiere.data.preprocessing import to_training_batches
from lumiere.tokenizers import SPECIAL_TOKENS, Tokenizer
from lumiere.models.transformer import Transformer
from lumiere.training.eval import evaluate
from lumiere.utils import get_device


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    num_layers: int
    num_attention_heads: int
    embedding_size: int
    context_size: int
    d_ff: int
    d_key: int
    dropout: float
    vocab_size: int
    num_params: int
    batch_size: int
    loss: float
    perplexity: float


def _load_model_card_template() -> str:
    """Load the model card template from disk."""
    project_base_dir = Path(__file__).parent.parent
    model_card_template_path = project_base_dir / "assets/model_card_template.md"
    return model_card_template_path.read_text()


def _save_model_card(model_card: str) -> None:
    """Save the model card to disk."""
    project_base_dir = Path(__file__).parent.parent
    model_card_path = project_base_dir / "MODEL_CARD.md"
    model_card_path.write_text(model_card)


def _get_model_info(run_id, checkpoint_tag):
    # Select the device to load the model onto.
    device = get_device()

    # Load the model checkpoint.
    logger.info(f"Loading model checkpoint '{run_id}:{checkpoint_tag}'...")
    run_config, checkpoint, run_manager = ds.resume_run(
        run_id, checkpoint_tag, device=device
    )
    model_config = Config(run_config)
    logger.info("Checkpoint loaded successfully\n")

    # Load the tokenizer.
    logger.info("Loading the tokenizer...")
    tokenizer_bytes = run_manager.load_artifact("tokenizer")
    if tokenizer_bytes is None:
        raise ValueError("Could not find tokenizer artifact.")

    tokenizer = Tokenizer.from_bytes(tokenizer_bytes, **model_config.tokenizer)
    logger.info("Tokenizer loaded successfully\n")

    # Initialize the model.
    logger.info("Initializing the model...")
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
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model initialized successfully.\n")

    # Load the dataset.
    logger.info("Evaluating the model on the test set...")
    dataloader = get_data_loader(model_config.dataset["name"])

    # Start the training loop.
    test_batches = to_training_batches(
        corpus=dataloader.iter_test(),
        tokenizer=tokenizer,
        context_size=model_config.model["context_size"] + 1,
        batch_size=model_config.training["batch_size"],
        pad_id=SPECIAL_TOKENS["padding"].id,
        sliding_window_size=model_config.dataset["sliding_window_size"],
    )

    eval_state = evaluate(
        model=model,
        data=test_batches,
        device=device,
    )

    logger.info("Model evaluation complete.")
    logger.info(
        f"Loss: {eval_state.avg_loss:.3f}, Perplexity: {eval_state.avg_perplexity:.3f}"
    )

    return ModelInfo(
        num_layers=model_config.model["num_layers"],
        num_attention_heads=model_config.model["num_heads"],
        embedding_size=model_config.model["embedding_size"],
        context_size=model_config.model["context_size"],
        d_ff=model_config.model["d_ff"],
        d_key=model_config.model["d_key"],
        dropout=model_config.model["dropout"],
        vocab_size=model_config.tokenizer["vocab_size"],
        num_params=sum((param.numel() for param in model.parameters())),
        batch_size=model_config.training["batch_size"],
        loss=eval_state.avg_loss,
        perplexity=eval_state.avg_perplexity,
    )


def generate_model_card(run_id, checkpoint_tag):
    model_info = _get_model_info(run_id, checkpoint_tag)
    model_card_template = _load_model_card_template()

    # breakpoint()
    model_card = model_card_template.format(
        run_id=run_id,
        checkpoint_tag=checkpoint_tag,
        num_layers=model_info.num_layers,
        num_attention_heads=model_info.num_attention_heads,
        embedding_size=model_info.embedding_size,
        context_size=model_info.context_size,
        d_ff=model_info.d_ff,
        d_key=model_info.d_key,
        dropout=model_info.dropout,
        vocab_size=model_info.vocab_size,
        num_params=model_info.num_params,
        batch_size=model_info.batch_size,
        loss=model_info.loss,
        perplexity=model_info.perplexity,
        generated_date=time.strftime("%b %d, %Y"),
    )

    _save_model_card(model_card)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new model card")

    parser.add_argument(
        "run_id",
        help="The run ID to use (required when loading from checkpoint)",
    )
    parser.add_argument(
        "checkpoint_tag",
        help="The checkpoint to resume training from",
    )
    args = parser.parse_args()

    if args.run_id is None or args.checkpoint_tag is None:
        raise ValueError("A checkpoint is required to generate a model card.")

    generate_model_card(args.run_id, args.checkpoint_tag)
