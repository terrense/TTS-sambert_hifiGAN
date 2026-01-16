"""
BERT Encoder module for SAM-BERT Acoustic Model.

This module implements a multi-layer Transformer Encoder that processes
phoneme embeddings to extract context-aware representations.
"""

import torch
import torch.nn as nn
import os


class BERTEncoder(nn.Module):
    """
    BERT Encoder using multi-layer Transformer Encoder.
    
    Processes phoneme embeddings H0 [B, Tph, d] to produce
    context-aware representations Henc [B, Tph, d].
    
    Args:
        d_model: Hidden dimension size
        n_layers: Number of Transformer encoder layers
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout rate (default: 0.1)
    
    Shape:
        - Input: [B, Tph, d_model]
        - Output: [B, Tph, d_model]
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Check if debug mode is enabled
        self.debug_shapes = os.getenv("DEBUG_SHAPES", "0") == "1"
        
        # Create Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Use batch_first=True for [B, Tph, d] format
            norm_first=False
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        H0: torch.Tensor,
        mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through BERT Encoder.
        
        Args:
            H0: Input phoneme embeddings [B, Tph, d_model]
            mask: Attention mask [Tph, Tph] (optional)
            src_key_padding_mask: Padding mask [B, Tph] (optional)
                True indicates positions to ignore
        
        Returns:
            Henc: Encoded representations [B, Tph, d_model]
        """
        # Shape validation
        assert H0.dim() == 3, f"Expected 3D tensor, got {H0.dim()}D"
        assert H0.size(-1) == self.d_model, \
            f"Expected d_model={self.d_model}, got {H0.size(-1)}"
        
        if self.debug_shapes:
            print(f"[BERTEncoder] Input H0 shape: {H0.shape}")
        
        # Pass through Transformer Encoder
        Henc = self.encoder(
            H0,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        if self.debug_shapes:
            print(f"[BERTEncoder] Output Henc shape: {Henc.shape}")
        
        return Henc
    
    def get_config(self) -> dict:
        """
        Get configuration dictionary for the encoder.
        
        Returns:
            Dictionary containing encoder configuration
        """
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout
        }
