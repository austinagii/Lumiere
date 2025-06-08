import torch
from torch import nn

from lumiere.components.attention import MultiHeadAttention
from lumiere.components.feedforward import FeedForward

class TransformerBlock(nn.Module):
    """A transformer block that combines attention and normalization.
    
    Args:
        embedding_size (int): The size of the embedding
        num_heads (int): The number of attention heads
        d_key (int): The size of the key
        d_value (int): The size of the value
        d_ff (int): The size of the feed-forward network
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_size=embedding_size,
            d_key=d_key,
            d_value=d_value,
            dropout=dropout
        )
        self.normalization_1 = nn.LayerNorm(embedding_size)
        self.feedforward = FeedForward(embedding_size, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.normalization_2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, context_size, embedding_size)
            
        Returns:
            Output tensor of shape (batch_size, context_size, embedding_size)
        """
        # Pre-normalization for better training stability
        x, attention_weights = self.attention(self.normalization_1(x))
        x = x + self.dropout(x)
        x = x + self.dropout(self.feedforward(self.normalization_2(x)))
        return x, attention_weights