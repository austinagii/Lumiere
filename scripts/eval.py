import os
import argparse
from tqdm import tqdm

import datasets
import torch
from torch.nn import functional as F

from lumiere.config.config import ModelConfig
from lumiere.models.transformer import Transformer
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.training.persistence import load_checkpoint, load_tokenizer
from lumiere.utils import get_device
from lumiere.utils.data import to_batches

MODEL_CONFIG_DIR = "configs/models"
TOKENIZER_CONFIG_DIR = "configs/tokenizers"
CONFIG_FILE_EXTENSION = "yaml"

MODEL_OUTPUT_DIR = "artifacts/models"
MODEL_FILE_EXTENSION = "pth"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TEXT_COLUMN_NAME = "text"

def eval(model_name: str, checkpoint_name: str = None) -> None:
    if checkpoint_name:
        checkpoint = load_checkpoint(model_name, checkpoint_name)
        model_config = checkpoint['model_config']
    else:
        # Load the model config.
        model_config_path = f"{MODEL_CONFIG_DIR}/{model_name}.{CONFIG_FILE_EXTENSION}"
        model_config = ModelConfig(model_config_path)

    # Load the tokenizer.
    tokenizer = load_tokenizer(model_config.model['tokenizer'])

    # Load and initialize the model.
    device = get_device()
    print(f"Using device: {device}")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        embedding_size=model_config.model['embedding_size'],
        context_size=model_config.model['context_size'],
        num_layers=model_config.model['num_layers'],
        num_heads=model_config.model['num_heads'],
        d_key=model_config.model['d_key'],
        d_value=model_config.model['d_value'],
        d_ff=model_config.model['d_ff'],
        dropout=model_config.model['dropout']
    )

    if checkpoint_name:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_path = f"{MODEL_OUTPUT_DIR}/{model_name}.{MODEL_FILE_EXTENSION}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    dataset = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    print(f"Loaded {len(dataset)} samples")

    batches = to_batches(
        tokenizer=tokenizer, 
        dataset=dataset, 
        batch_size=model_config.training['batch_size'], 
        context_size=model_config.model['context_size']+1)
    
    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    with tqdm(batches, desc=f"Test Evaluation:") as pbar:
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
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'perplexity': f'{perplexity.item():.4f}'
                })

    avg_batch_loss = total_loss / num_batches
    avg_batch_perplexity = total_perplexity / num_batches
    print(f"Test set evaluation complete. Avg loss: {avg_batch_loss:.4f} Avg perplexity: {avg_batch_perplexity:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a language model')
    parser.add_argument('model', default='transformer_tiny', 
                        help='Name of the model config file')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to the checkpoint file')
    args = parser.parse_args()

    eval(args.model, args.checkpoint)


if __name__ == "__main__":
    main()