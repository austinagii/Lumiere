import logging
import os
import time
import argparse
import sys
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lumiere.models.transformer import Transformer
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.utils import get_device
from lumiere.config.config import TokenizerConfig, ModelConfig
from lumiere.utils.data import to_batches

MODEL_CONFIG_DIR = "configs/models"
TOKENIZER_CONFIG_DIR = "configs/tokenizers"
CONFIG_FILE_EXTENSION = ".yaml"

MODEL_OUTPUT_DIR = "artifacts/models"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_PORTION = 1
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


def train(model_name: str):
    model_config_path = f"{MODEL_CONFIG_DIR}/{model_name}{CONFIG_FILE_EXTENSION}"
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config file not found: {model_config_path}")

    logger.info(f"Loading model config from '{model_config_path}'...")
    model_config = ModelConfig(model_config_path)
    logger.info(f"Model configuration loaded successfully")

    tokenizer_config_path = f"{TOKENIZER_CONFIG_DIR}/{model_config.model['tokenizer']}{CONFIG_FILE_EXTENSION}"
    if not os.path.exists(tokenizer_config_path):
        raise FileNotFoundError(
            f"Config file not found: {tokenizer_config_path}")

    logger.info(f"Loading tokenizer config from '{tokenizer_config_path}'...")
    tokenizer_config = TokenizerConfig(tokenizer_config_path)
    logger.info(f"Tokenizer configuration loaded successfully")

    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    train_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=f"train[:{DATASET_PORTION}%]")
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    validation_dataset = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG, split=f"validation")
    logger.info(f"Validation dataset loaded: {len(validation_dataset)} samples")

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

    def lr_lambda(step):
        warmup_steps = model_config.training['warmup_steps']
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            # Approximate steps
            progress = (step - warmup_steps) / \
                (model_config.training['num_epochs'] * 1000 - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    logger.info("Starting training...")
    start_time = time.time()
    best_loss = float('inf')
    best_perplexity = float('inf')
    patience_counter = 0
    patience = model_config.training['patience']
    max_epochs = model_config.training['num_epochs']
    current_epoch = 0

    while True:
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_perplexity = 0.0
        num_batches = 0

        batches = to_batches(tokenizer, train_dataset,
                             model_config.training['batch_size'], model_config.model['context_size']+1)

        with tqdm(batches, desc=f"Epoch {current_epoch+1}") as pbar:
            for batch in pbar:
                x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size), y.reshape(-1))
                perplexity = torch.exp(loss)
                epoch_perplexity += perplexity
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), model_config.training['gradient_clip_norm'])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1
                global_step += 1

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}',
                    'avg_perplexity': f'{epoch_perplexity/num_batches:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{grad_norm:.2f}'
                })

        # Evaluate performance on validation set.
        validation_loss = 0.0
        validation_perplexity = 0.0
        num_validation_batches = 0
        validation_batches = to_batches(tokenizer, validation_dataset,
                             model_config.training['batch_size'], model_config.model['context_size']+1)
        model.eval()
        with torch.no_grad():
            for validation_batch in validation_batches:
                num_validation_batches += 1
                x, y = validation_batch[:, :-1].to(device), validation_batch[:, 1:].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                validation_loss += loss.item()
                validation_perplexity += torch.exp(loss).item()

        model.train()

        # Log epoch statistics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_epoch_perplexity = epoch_perplexity / num_batches if num_batches > 0 else 0
        avg_validation_loss = validation_loss / num_validation_batches if num_validation_batches > 0 else 0
        avg_validation_perplexity = validation_perplexity / num_validation_batches if num_validation_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {current_epoch+1} - "
            f"Loss: {avg_epoch_loss:.4f}, "
            f"Perplexity: {avg_epoch_perplexity:.4f}, "
            f"LR: {current_lr:.2e}, "
            f"Time: {epoch_time:.2f}s, "
            f"Total: {total_time:.2f}s, "
            f"Validation Loss: {avg_validation_loss:.4f}, "
            f"Validation Perplexity: {avg_validation_perplexity:.4f}"
        )

        if avg_validation_perplexity < best_perplexity:
            best_perplexity = avg_validation_perplexity

        # Early stopping check
        if avg_validation_loss < best_loss:
            best_loss = avg_validation_loss
            patience_counter = 0
            logger.info(f"New best validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement")
                break

        current_epoch += 1
        if max_epochs > 0 and current_epoch >= max_epochs:
            logger.info(f"Training completed after {current_epoch} epochs")
            break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f}s\n")
    logger.info(f"Best validation loss: {best_loss:.4f}, Best validation perplexity: {best_perplexity:.4f}")

    # Save model and tokenizer
    logger.info("Saving model to disk...")
    save_path = Path(MODEL_OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / f"{model_name}.pth")
    logger.info(f"Model saved to {save_path / f'{model_name}.pth'}")

    logger.info("Saving tokenizer to disk...")
    tokenizer_save_path = Path(TOKENIZER_OUTPUT_DIR)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_save_path / f"{model_config.model['tokenizer']}.json"))
    logger.info(
        f"Tokenizer saved to {tokenizer_save_path / f'{model_config.model['tokenizer']}.json'}")

    logger.info("Training completed successfully\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model')
    parser.add_argument('model', default='base_small',
                        help='Name of the model config file')
    args = parser.parse_args()

    train(args.model)
