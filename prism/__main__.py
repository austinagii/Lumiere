from prism.tokenizer import Tokenizer
from prism.model import Model
from prism.exceptions import TrainingError
from prism.data import to_batches

import datasets
import torch
from torch.nn import functional as F

import logging
import time
from tqdm import tqdm
import argparse

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TEXT_COLUMN_NAME = "text"
NUM_HEADS = 12
BATCH_SIZE = 16
CONTEXT_SIZE = 256
VOCAB_SIZE = 8_192
NUM_EPOCHS = 10
EMBEDDING_SIZE = 256

# Training hyperparameters
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRADIENT_CLIP_NORM = 1.0
WARMUP_STEPS = 1000
NUM_LAYERS = 6
DROPOUT = 0.1
DATASET_PORTION = 20


def train():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    try:
        logger.info("Loading dataset...")
        train_dataset = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG, split=f"train[:{DATASET_PORTION}%]")
        logger.info(f"Dataset loaded: {len(train_dataset)} samples")
    except Exception as e:
        raise TrainingError("An error occurred while loading the dataset", e) 

    logger.info("Training tokenizer...")
    tokenizer = Tokenizer().train(train_dataset, TEXT_COLUMN_NAME, BATCH_SIZE, VOCAB_SIZE)
    logger.info(f"Tokenizer trained with vocab size: {VOCAB_SIZE}")
    
    logger.info("Initializing model...")
    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        else:
            # Cosine annealing after warmup
            progress = (step - WARMUP_STEPS) / (NUM_EPOCHS * 1000 - WARMUP_STEPS)  # Approximate steps
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    logger.info("Starting training...")
    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    patience = 5  # Early stopping patience
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        batches = to_batches(tokenizer, train_dataset, BATCH_SIZE, CONTEXT_SIZE+1)
        
        with tqdm(batches, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
            for batch in pbar:
                x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                
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
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
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
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    torch.save(model.state_dict(), "model.pth")
    tokenizer.save("tokenizer.json")
    
    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f}s")
    logger.info("Model and tokenizer saved successfully")


def predict(text, max_length=200):
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model and tokenizer
    tokenizer = Tokenizer.load("tokenizer.json")
    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT)
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text).ids
    x = torch.tensor(tokens).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Generate predictions
    with torch.no_grad():
        for _ in range(max_length):
            if x.size(1) >= CONTEXT_SIZE:
                x = x[:, -CONTEXT_SIZE:]  # Keep only last CONTEXT_SIZE tokens
            logits = model(x)
            probs = F.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            # next_token = torch.argmax(logits[0, -1]).item()
            x = torch.cat([x, torch.tensor([[next_token]]).to(device)], dim=1)
            # Stop if we get a special token (simplified stopping condition)
            if next_token == 0:  # Stop if we get padding token
                break
    
    return tokenizer.decode(x[0].cpu().tolist())


def main():
    print("Starting Prism")
    parser = argparse.ArgumentParser(description='Train or run inference with the language model')
    parser.add_argument('mode', choices=['train', 'predict'], help='Whether to train the model or run inference')
    parser.add_argument('--text', default='Hello, how are you?', help='Text to predict from (only used in predict mode)')
    args = parser.parse_args()

    print(args)
    if args.mode == 'train':
        train()
    else:
        output = predict(args.text)
        print(output)


if __name__ == "__main__":
    main()