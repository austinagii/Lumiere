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

DATASET_NAME = "openwebtext"
TEXT_COLUMN_NAME = "text"
NUM_HEADS = 12
BATCH_SIZE = 64
CONTEXT_SIZE = 512
VOCAB_SIZE = 16_384
NUM_EPOCHS = 64
EMBEDDING_SIZE = 256


def main():
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
    
    try:
        logger.info("Loading dataset...")
        train_dataset = datasets.load_dataset(DATASET_NAME, split="train[:1%]")
        logger.info(f"Dataset loaded: {len(train_dataset)} samples")
    except Exception as e:
        raise TrainingError("An error occurred while loading the dataset", e) 

    logger.info("Training tokenizer...")
    tokenizer = Tokenizer().train(train_dataset, TEXT_COLUMN_NAME, BATCH_SIZE, VOCAB_SIZE)
    logger.info(f"Tokenizer trained with vocab size: {VOCAB_SIZE}")
    
    logger.info("Initializing model...")
    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        batches = to_batches(tokenizer, train_dataset, BATCH_SIZE, CONTEXT_SIZE+1)
        
        with tqdm(batches, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
            for batch in pbar:
                x, y = batch[:, :-1], batch[:, 1:]
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}'
                })
        
        # Log epoch statistics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Loss: {avg_epoch_loss:.4f}, "
            f"Time: {epoch_time:.2f}s, "
            f"Total: {total_time:.2f}s"
        )
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    torch.save(model.state_dict(), "model.pth")
    tokenizer.save("tokenizer.json")
    
    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f}s")
    logger.info("Model and tokenizer saved successfully")


def predict(text, max_length=50):
    # Load model and tokenizer
    tokenizer = Tokenizer.load("tokenizer.json")
    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text)
    x = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    
    # Generate predictions
    with torch.no_grad():
        for _ in range(max_length):
            if x.size(1) >= CONTEXT_SIZE:
                x = x[:, -CONTEXT_SIZE:]  # Keep only last CONTEXT_SIZE tokens
            logits = model(x)
            next_token = torch.argmax(logits[0, -1]).item()
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            if next_token == tokenizer.eos_token_id:  # Stop if end token
                break
    
    return tokenizer.decode(x[0].tolist())


main()