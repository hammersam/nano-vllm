import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """Cross-attention layer for multi-modal fusion between text and vision tokens."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        scale: Optional[float] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        self.head_dim = hidden_size // num_attention_heads
        self.scale = scale or (self.head_dim ** -0.5)
        
        assert self.head_dim * num_attention_heads == hidden_size
        
        # Query projection for text tokens
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Key and Value projections for vision tokens
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        text_hidden_states: torch.Tensor,
        vision_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            text_hidden_states: Text token hidden states [batch_size, text_seq_len, hidden_size]
            vision_hidden_states: Vision token hidden states [batch_size, vision_seq_len, hidden_size]
            attention_mask: Attention mask for text tokens [batch_size, text_seq_len, text_seq_len]
            vision_mask: Mask for vision tokens [batch_size, vision_seq_len]
            
        Returns:
            Updated text hidden states [batch_size, text_seq_len, hidden_size]
        """
        batch_size, text_seq_len, _ = text_hidden_states.shape
        vision_seq_len = vision_hidden_states.shape[1]
        
        # Project queries from text
        q = self.q_proj(text_hidden_states)
        
        # Project keys and values from vision
        k = self.k_proj(vision_hidden_states)
        v = self.v_proj(vision_hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, text_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, vision_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, vision_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply vision mask
        if vision_mask is not None:
            vision_mask = vision_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, vision_seq_len]
            scores = scores.masked_fill(~vision_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, text_seq_len, self.hidden_size)
        
        # Final projection and dropout
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output


class CrossModalAttention(nn.Module):
    """Complete cross-modal attention module with layer norm and residual connections."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.cross_attention = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        self.layer_norm_text = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm_vision = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(
        self,
        text_hidden_states: torch.Tensor,
        vision_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with residual connections and layer normalization.
        
        Args:
            text_hidden_states: Text token hidden states
            vision_hidden_states: Vision token hidden states
            attention_mask: Attention mask
            vision_mask: Vision token mask
            
        Returns:
            Updated text hidden states
        """
        # Layer normalization
        text_norm = self.layer_norm_text(text_hidden_states)
        vision_norm = self.layer_norm_vision(vision_hidden_states)
        
        # Cross-attention
        attn_output = self.cross_attention(
            text_norm,
            vision_norm,
            attention_mask=attention_mask,
            vision_mask=vision_mask
        )
        
        # Residual connection
        return text_hidden_states + attn_output


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention with additional features for multi-modal fusion."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        use_rotary_embeddings: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.use_rotary_embeddings = use_rotary_embeddings
        
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Rotary position embeddings
        if use_rotary_embeddings:
            from nanovllm.layers.rotary_embedding import RotaryEmbedding
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head cross-attention forward pass.
        
        Args:
            query_states: Query states [batch_size, q_len, hidden_size]
            key_states: Key states [batch_size, kv_len, hidden_size]
            value_states: Value states [batch_size, kv_len, hidden_size]
            attention_mask: Attention mask [batch_size, q_len, kv_len]
            position_ids: Position IDs for rotary embeddings
            
        Returns:
            Attention output [batch_size, q_len, hidden_size]
        """
        batch_size, q_len, _ = query_states.shape
        kv_len = key_states.shape[1]
        
        # Project to multi-head format
        q = self.q_proj(query_states)
        k = self.k_proj(key_states)
        v = self.v_proj(value_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary_embeddings and position_ids is not None:
            q, k = self.rotary_emb(q, k, position_ids)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.hidden_size)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output