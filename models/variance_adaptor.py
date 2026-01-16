"""
Variance Adaptor module for SAM-BERT Acoustic Model.

This module contains predictors for duration, pitch, and energy,
as well as the Length Regulator for expanding phoneme-level features to frame-level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    Duration Predictor using Conv1d layers with ReLU and LayerNorm.
    
    Predicts log-duration for each phoneme to enable expansion to frame-level.
    
    Args:
        d_model (int): Input feature dimension
        n_layers (int): Number of convolutional layers (default: 2)
        kernel_size (int): Convolution kernel size (default: 3)
        dropout (float): Dropout rate (default: 0.1)
    
    Shape:
        - Input: [B, Tph, d_model]
        - Output: [B, Tph] log-duration predictions
    """
    
    def __init__(self, d_model, n_layers=2, kernel_size=3, dropout=0.1):
        super(DurationPredictor, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(n_layers):
            # Conv1d expects [B, C, T] format
            conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2  # Same padding
            )
            self.conv_layers.append(conv)
            
            # LayerNorm over feature dimension
            self.layer_norms.append(nn.LayerNorm(d_model))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
        
        # Final linear projection to scalar duration
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, Henc, mask=None):
        """
        Forward pass of Duration Predictor.
        
        Args:
            Henc (torch.FloatTensor): Encoder output [B, Tph, d_model]
            mask (torch.BoolTensor, optional): Padding mask [B, Tph]
                True for valid positions, False for padding
        
        Returns:
            log_dur_pred (torch.FloatTensor): Log-duration predictions [B, Tph]
        """
        # Shape logging
        print(f"[DurationPredictor] Input Henc shape: {Henc.shape}")
        
        # Transpose to [B, d_model, Tph] for Conv1d
        x = Henc.transpose(1, 2)  # [B, d_model, Tph]
        
        # Apply convolutional layers
        for i in range(self.n_layers):
            # Conv1d
            residual = x
            x = self.conv_layers[i](x)  # [B, d_model, Tph]
            
            # Transpose back for LayerNorm
            x = x.transpose(1, 2)  # [B, Tph, d_model]
            
            # ReLU activation
            x = F.relu(x)
            
            # LayerNorm
            x = self.layer_norms[i](x)
            
            # Dropout
            x = self.dropouts[i](x)
            
            # Residual connection (transpose residual back)
            x = x + residual.transpose(1, 2)
            
            # Transpose back for next conv layer
            x = x.transpose(1, 2)  # [B, d_model, Tph]
        
        # Transpose back to [B, Tph, d_model] for linear projection
        x = x.transpose(1, 2)  # [B, Tph, d_model]
        
        # Project to scalar duration
        log_dur_pred = self.linear(x).squeeze(-1)  # [B, Tph]
        
        # Apply mask if provided (set padding positions to large negative value)
        if mask is not None:
            log_dur_pred = log_dur_pred.masked_fill(~mask, -1e9)
        
        # Shape logging
        print(f"[DurationPredictor] Output log_dur_pred shape: {log_dur_pred.shape}")
        
        return log_dur_pred
