import torch
from torch import nn

from lumiere.utils import validation


class MultiHeadAttention(nn.Module):
    """Performs multi-head self attention on a batch of tokens.

    Args:
        num_heads (int): The number of attention heads
        embedding_size (int): The size of the embedding
        d_key (int): The size of the key
        d_value (int): The size of the value
        masked (bool): Whether to mask the attention scores
        dropout (float): Dropout probability for attention weights
    """
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        d_key: int,
        d_value: int,
        masked: bool = True,
        dropout: float = 0.0
    ) -> None:
        validation.validate_positive_integer(num_heads, "num_heads")
        validation.validate_positive_integer(embedding_size, "embedding_size")
        validation.validate_positive_integer(d_key, "d_key")
        validation.validate_positive_integer(d_value, "d_value")
        validation.validate_boolean(masked, "masked")
        validation.validate_positive_float_or_zero(dropout, "dropout")

        super().__init__()
        self._masked = masked
        self._d_key = d_key
        self._d_value = d_value
        self._num_heads = num_heads
        if dropout > 0.0:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = nn.Identity()

        self._q_proj = nn.Linear(embedding_size, d_key * num_heads) 
        self._k_proj = nn.Linear(embedding_size, d_key * num_heads)
        self._v_proj = nn.Linear(embedding_size, d_value * num_heads)
        self._o_proj = nn.Linear(d_value * num_heads, embedding_size) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates the query, key and value matrices for a given batch of 
        tokens.

        Args:
            x: A tensor of shape (batch_size, context_size, embedding_size)
            containing the token embeddings.

        Returns:
            A tuple containing:
            - A tensor of shape (batch_size, context_size, embedding_size)
              containing the input embeddings enriched with the attention values.
            - A tensor of shape (batch_size, num_heads, context_size, context_size)
              containing the attention weights.
        """
        if x.dim() != 3:
            raise ValueError("Input tensor must have shape (batch_size, context_size, embedding_size)")
        
        queries = self._q_proj(x)
        keys = self._k_proj(x) 
        values = self._v_proj(x)

        queries = split_heads(queries, self._num_heads, self._d_key)
        keys = split_heads(keys, self._num_heads, self._d_key)
        values = split_heads(values, self._num_heads, self._d_value)

        attention_scores = torch.matmul(queries, torch.transpose(keys, -2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(self._d_key, dtype=queries.dtype, device=queries.device))

        if self._masked:
            mask = torch.triu(torch.ones(queries.shape[2], queries.shape[2], dtype=torch.bool, device=queries.device), diagonal=1)
            scaled_attention_scores = scaled_attention_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -float('inf'))

        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        attention_weights = self._dropout(attention_weights)
        attention_values = torch.matmul(attention_weights, values)

        attention_values = concat_heads(attention_values)
        output = self._o_proj(attention_values)
        return output, attention_weights

    @property
    def masked(self):
        return self._masked

    @property
    def q_proj(self):
        return self._q_proj

    @property
    def k_proj(self):
        return self._k_proj

    @property
    def v_proj(self):
        return self._v_proj

    @property
    def o_proj(self):
        return self._o_proj
    
    
def split_heads(tensor: torch.Tensor, num_heads: int, num_features: int) -> torch.Tensor:
    """Splits the tensor into multiple attention heads and transposes the sequence and head dimensions.
    
    Args:
        tensor: Tensor to split of shape (batch_size, seq_len, num_heads * d_head)
        num_heads: Number of attention heads
        num_features: Number of features per attention head
        
    Returns:
        Tensor of shape (batch_size, num_heads, seq_len, d_head)
    """
    return tensor.view(tensor.shape[0], tensor.shape[1], num_heads, num_features).transpose(1, 2)

def concat_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenates the tensor from multiple attention heads and transposes the sequence and head dimensions.
    
    Args:
        tensor: Tensor to concatenate of shape (batch_size, num_heads, context_size, num_features)
        
    Returns:
        Tensor of shape (batch_size, context_size, num_heads * num_features)
    """ 
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)  