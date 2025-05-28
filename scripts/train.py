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

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_PORTION = 20
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


def train(tokenizer_config_path: str, model_config_path: str):
    if not os.path.exists(tokenizer_config_path):
        raise FileNotFoundError(
            f"Config file not found: {tokenizer_config_path}")

    logger.info(f"Loading tokenizer config from '{tokenizer_config_path}'...")
    tokenizer_config = TokenizerConfig(tokenizer_config_path)
    logger.info(f"Tokenizer configuration loaded successfully")

    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Config file not found: {model_config_path}")

    logger.info(f"Loading model config from '{model_config_path}'...")
    model_config = ModelConfig(model_config_path)
    logger.info(f"Model configuration loaded successfully")

    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    train_dataset = datasets.load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=f"train[:{DATASET_PORTION}%]")
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

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
    patience_counter = 0
    patience = model_config.training['patience']

    for epoch in range(model_config.training['num_epochs']):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        batches = to_batches(tokenizer, train_dataset,
                             model_config.training['batch_size'], model_config.model['context_size']+1)

        with tqdm(batches, desc=f"Epoch {epoch+1}/{model_config.training['num_epochs']}") as pbar:
            for batch in pbar:
                x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size), y.reshape(-1))
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
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{grad_norm:.2f}'
                })

        # Log epoch statistics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}/{model_config.training['num_epochs']} - "
            f"Loss: {avg_epoch_loss:.4f}, "
            f"LR: {current_lr:.2e}, "
            f"Time: {epoch_time:.2f}s, "
            f"Total: {total_time:.2f}s"
        )

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            logger.info(f"New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f}s\n")

    # Save model and tokenizer
    logger.info("Saving model to disk...")
    save_path = Path(model_config.model['output_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pth")
    logger.info(f"Model saved to {save_path / 'model.pth'}")

    logger.info("Saving tokenizer to disk...")
    tokenizer_save_path = Path(tokenizer_config.tokenizer['output_path'])
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_save_path / "tokenizer.json"))
    logger.info(f"Tokenizer saved to {tokenizer_save_path / 'tokenizer.json'}")

    logger.info("Training completed successfully\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model')
    parser.add_argument('--tokenizer_config', default='base_small',
                        help='Name of the tokenizer config file')
    parser.add_argument('--model_config', default='base_small',
                        help='Name of the model config file')
    args = parser.parse_args()

    tokenizer_config_path = f"configs/tokenizers/{args.tokenizer_config}.yaml"
    model_config_path = f"configs/models/{args.model_config}.yaml"
    train(tokenizer_config_path, model_config_path)
