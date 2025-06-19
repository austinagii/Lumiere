import argparse
import logging

import torch
import torch.nn.functional as F

from lumiere.training.persistence import load_model
from lumiere.utils import get_device
from scripts import cli


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


def predict(model, tokenizer, text, device, max_length=200):
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

    return tokenizer.decode(full_sequence[0].cpu().tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with the language model"
    )
    parser.add_argument(
        "model", default="transformer_tiny", help="Name of the model config file"
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help='Name of the checkpoint to load (e.g., "best", "epoch:10")',
    )
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    checkpoint = cli.parse_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise RuntimeError("A likkle error")

    model, model_config, tokenizer = load_model(
        args.model, checkpoint=checkpoint, device=device
    )
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    while True:
        text = input("User: ")
        output = predict(model, tokenizer, text, device)
        print(f"Model: {output}")
