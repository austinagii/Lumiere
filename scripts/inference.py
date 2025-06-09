import argparse
import logging
import os

import torch
import torch.nn.functional as F

from lumiere.models.transformer import Transformer
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.utils import get_device
from lumiere.config.config import ModelConfig
from lumiere.training.persistence import load_checkpoint, load_tokenizer

MODEL_CONFIG_DIR = "configs/models"
TOKENIZER_CONFIG_DIR = "configs/tokenizers"
CONFIG_FILE_EXTENSION = ".yaml"
MODEL_OUTPUT_DIR = "artifacts/models"
TOKENIZER_OUTPUT_DIR = "artifacts/tokenizers"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Azure blob storage logging
logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)
logging.getLogger('azure.core').setLevel(logging.WARNING)

def predict(model, tokenizer, text, device, max_length=200):
    tokens = tokenizer.encode(text).ids
    full_sequence = torch.tensor(tokens).unsqueeze(0).to(device)  # Keep all tokens
    
    with torch.no_grad():
        for _ in range(max_length):
            # Use only the last context_size tokens for model input
            if full_sequence.size(1) >= model.context_size:
                model_input = full_sequence[:, -model.context_size:]
            else:
                model_input = full_sequence
                
            logits, _ = model(model_input)
            probs = F.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Append to the full sequence (never truncate this)
            full_sequence = torch.cat([full_sequence, torch.tensor([[next_token]]).to(device)], dim=1)
    
    return tokenizer.decode(full_sequence[0].cpu().tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with the language model')
    parser.add_argument('model', default='transformer_tiny', 
                        help='Name of the model config file')
    parser.add_argument('--checkpoint', default=None,
                        help='Name of the checkpoint to load (e.g., "best", "epoch_0010")')
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    if args.checkpoint:
        checkpoint = load_checkpoint(args.model, args.checkpoint, device)
        model_config = checkpoint['model_config']
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model_config_path = f"{MODEL_CONFIG_DIR}/{args.model}{CONFIG_FILE_EXTENSION}"
        model_config = ModelConfig(model_config_path)

    tokenizer = load_tokenizer(model_config.model['tokenizer'])

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
    

    if args.checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_path = f"{MODEL_OUTPUT_DIR}/{args.model}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    while True:
        text = input("User: ")
        output = predict(model, tokenizer, text, device)
        print(f"Model: {output}")