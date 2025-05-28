import argparse

import torch
import torch.nn.functional as F

from lumiere.core.model import Model
from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.utils import get_device


def predict(model_config, text, max_length=200):
    model = Model(**model_config)
    model.load_state_dict(torch.load(model_config.output_path), strict=True)

    device = get_device()
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    
    tokenizer = Tokenizer.load(model_config.tokenizer)
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


if __name__ == "__main__":
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