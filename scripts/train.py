import itertools
import logging
import os
import argparse
from pathlib import Path

import datasets
import torch

from lumiere.models.transformer import Transformer
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.training import schedulers
from lumiere.training.train import train
from lumiere.training.eval import evaluate
from lumiere.utils import get_device
from lumiere.config.config import TokenizerConfig, ModelConfig

MODEL_CONFIG_PATH_TEMPLATE = f"configs/models/{{}}.yaml"
TOKENIZER_CONFIG_PATH_TEMPLATE = f"configs/tokenizers/{{}}.yaml"

MODEL_OUTPUT_DIR = "artifacts/models"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_PORTION = 100
TEXT_COLUMN_NAME = "text"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_configs(model_name: str) -> tuple[ModelConfig, TokenizerConfig]:
    """Load the model and tokenizer configurations."""
    model_config_path = MODEL_CONFIG_PATH_TEMPLATE.format(model_name)
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config file not found: {model_config_path}")

    logger.info(f"Loading model config from '{model_config_path}'...")
    model_config = ModelConfig(model_config_path)
    logger.info(f"Model configuration loaded successfully")

    tokenizer_config_path = TOKENIZER_CONFIG_PATH_TEMPLATE.format(model_config.model['tokenizer'])
    if not os.path.exists(tokenizer_config_path):
        raise FileNotFoundError(
            f"Config file not found: {tokenizer_config_path}")

    logger.info(f"Loading tokenizer config from '{tokenizer_config_path}'...")
    tokenizer_config = TokenizerConfig(tokenizer_config_path)
    logger.info(f"Tokenizer configuration loaded successfully")

    return model_config, tokenizer_config

def load_datasets():
    """Load the training and validation datasets."""
    logger.info("Loading dataset...")
    train_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=f"train[:{DATASET_PORTION}%]")
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    validation_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=f"validation")
    logger.info(
        f"Validation dataset loaded: {len(validation_dataset)} samples")

    return train_dataset, validation_dataset

def main(model_name: str):
    model_config, tokenizer_config = load_configs(model_name)
    train_dataset, validation_dataset = load_datasets()
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Training tokenizer with configuration:\n{tokenizer_config}")
    tokenizer = Tokenizer().train(
        train_dataset,
        TEXT_COLUMN_NAME,
        tokenizer_config.tokenizer['batch_size'],
        tokenizer_config.tokenizer['vocab_size']
    )
    logger.info(f"Tokenizer trained successfully")

    logger.info(f"Training model with configuration:\n{model_config}")
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
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.training['learning_rate'],
        weight_decay=model_config.training['weight_decay']
    )
    scheduler = schedulers.cosine_annealing_lr_scheduler(
        optimizer,
        model_config.training['warmup_steps'],
        model_config.training['num_epochs']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    logger.info(f"Starting training for model '{model_name}'...")
    logger.info("--------------------------------")
    total_time_taken = 0.0
    best_loss = float('inf')
    best_perplexity = float('inf')
    patience_counter = 0
    patience = model_config.training['patience']
    max_epochs = model_config.training['num_epochs'] if model_config.training['num_epochs'] > 0 else float(
        'inf')

    # Start the training loop.
    for epoch in itertools.count(1):
        train_state = train(
            model=model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            current_epoch=epoch,
            max_epochs=max_epochs,
            batch_size=model_config.training['batch_size'],
            context_size=model_config.model['context_size'],
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=model_config.training['gradient_clip_norm'],
            device=device
        )
        logger.info(
            f"EPOCH {epoch:04d} - {"TRAINING":<10}: "
            f"Loss: {train_state.avg_loss:.4f}, "
            f"Perplexity: {train_state.avg_perplexity:.4f}, "
            f"LR: {train_state.current_lr:.2e}, "
            f"Time: {train_state.time_taken:.2f}s, "
        )

        eval_state = evaluate(
            model=model,
            tokenizer=tokenizer,
            validation_dataset=validation_dataset,
            batch_size=model_config.training['batch_size'],
            context_size=model_config.model['context_size'],
            device=device
        )
        logger.info(
            f"EPOCH {epoch:04d} - {"VALIDATION":<10}: "
            f"Loss: {eval_state.avg_loss:.4f}, "
            f"Perplexity: {eval_state.avg_perplexity:.4f}, "
            f"Time: {eval_state.time_taken:.2f}s, "
        )

        total_time_taken += (train_state.time_taken + eval_state.time_taken)

        # Capture performance improvements.
        if eval_state.avg_loss < best_loss:
            best_loss = eval_state.avg_loss
            best_perplexity = eval_state.avg_perplexity
            patience_counter = 0
            logger.info(
                f"New best validation loss: {best_loss:.4f}, "
                f"perplexity: {best_perplexity:.4f}")
        else:
            patience_counter += 1

        # Determine if the training should be stopped.
        if epoch > max_epochs:
            logger.info(f"Training completed after {epoch} epochs")
            break
        if patience_counter >= patience:
            logger.info(
                f"Training stopped after {patience} epochs without improvement")
            break

        logger.info("--------------------------------")

    logger.info(
        f"Total time taken: {total_time_taken:.2f}s, "
        f"Best validation loss: {best_loss:.4f}, "
        f"Best validation perplexity: {best_perplexity:.4f}")

    # Save the model.
    logger.info("Saving model to disk...")
    save_path = Path(MODEL_OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"{model_name}.pth")
    logger.info(f"Model saved to {save_path / f'{model_name}.pth'}")

    # Save the tokenizer.
    logger.info("Saving tokenizer to disk...")
    tokenizer_save_path = Path(TOKENIZER_OUTPUT_DIR)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_save_path /
                   f"{model_config.model['tokenizer']}.json"))
    logger.info(
        f"Tokenizer saved to {tokenizer_save_path / f'{model_config.model['tokenizer']}.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model')
    parser.add_argument('model', default='transformer-tiny',
                        help='Name of the model to train')
    args = parser.parse_args()

    main(args.model)
