import torch
from torch import nn
from lumiere.components.embedding import Embedding
from lumiere.components.block import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        context_size: int,
        num_layers: int = 6,
        num_heads: int = 12,
        d_key: int = 64,
        d_value: int = 64,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self._context_size = context_size

        self.embedding = Embedding(vocab_size, context_size, embedding_size)
        
        # Create a stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                d_key=d_key,
                d_value=d_value,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_size)
        self.linear_out = nn.Linear(embedding_size, vocab_size, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        x = self.embedding(x)

        attention_weights = []
        # Pass through each transformer block
        for block in self.blocks:
            x, block_attention_weights = block(x)
            attention_weights.append(block_attention_weights)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        # Apply final normalization and linear output layer
        x = self.final_norm(x)
        x = self.linear_out(x)
        return x, attention_weights
    
    @property
    def context_size(self):
        return self.embedding.context_size
