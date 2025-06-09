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
        self.normalization_1 = nn.RMSNorm(embedding_size)
        self.feedforward = FeedForward(embedding_size, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.normalization_2 = nn.RMSNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, context_size, embedding_size)
            
        Returns:
            Output tensor of shape (batch_size, context_size, embedding_size)
        """
        norm_x1 = self.normalization_1(x)
        attention_values, attention_weights = self.attention(norm_x1)
        x = x + self.dropout(attention_values)
        
        norm_x2 = self.normalization_2(x)
        output = self.feedforward(norm_x2)
        x = x + self.dropout(output)
        return x, attention_weights