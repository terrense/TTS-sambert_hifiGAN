"""
PNCA AR-Decoder implementation for SAM-BERT TTS system.

This module implements an autoregressive decoder that generates mel-spectrograms
from the variance-adapted hidden representations. It supports both training mode
with teacher forcing and inference mode with autoregressive generation.
"""

import torch
import torch.nn as nn
import math


class PNCAARDecoder(nn.Module):
    """
    PNCA (Phoneme-level Non-Causal Attention) Autoregressive Decoder.
    
    This decoder uses a Transformer architecture to autoregressively generate
    mel-spectrograms. During training, it uses teacher forcing with shifted
    ground truth mel frames. During inference, it generates frames autoregressively.
    
    Architecture:
        - Prenet: Projects mel features to hidden dimension
        - Multi-layer Transformer Decoder: Processes mel features with attention to encoder output
        - Output projection: Projects hidden features back to mel dimension
    
    Shape Contract:
        Training:
            Input: Hvar [B, Tfrm, d_model], mel_gt [B, Tfrm, n_mels]
            Output: mel_pred [B, Tfrm, n_mels]
        
        Inference:
            Input: Hvar [B, Tfrm, d_model]
            Output: mel_pred [B, Tfrm, n_mels]
    """
    
    def __init__(self, d_model=256, n_mels=80, n_layers=6, n_heads=8, 
                 d_ff=2048, dropout=0.1, chunk_size=1):
        """
        Initialize the PNCA AR-Decoder.
        
        Args:
            d_model: Hidden dimension size
            n_mels: Number of mel-spectrogram bins
            n_layers: Number of Transformer decoder layers
            n_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
            chunk_size: Number of frames to generate per step in inference (for streaming)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_mels = n_mels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        
        # Prenet: Projects mel features to hidden dimension
        # Linear -> ReLU -> Dropout -> Linear
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Positional encoding for decoder input
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # Multi-layer Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Use batch_first=True for [B, T, D] format
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection: Projects hidden features back to mel dimension
        self.mel_proj = nn.Linear(d_model, n_mels)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, Hvar, mel_gt=None, max_len=None):
        """
        Forward pass of the AR-Decoder.
        
        Training mode (teacher forcing):
            Uses ground truth mel frames shifted right as decoder input.
            
        Inference mode (autoregressive):
            Generates mel frames autoregressively.
        
        Args:
            Hvar: Variance-adapted encoder output [B, Tfrm, d_model]
            mel_gt: Ground truth mel-spectrogram [B, Tfrm, n_mels] (training only)
            max_len: Maximum length for generation (inference only)
        
        Returns:
            mel_pred: Predicted mel-spectrogram [B, Tfrm, n_mels]
        """
        if self.training and mel_gt is not None:
            # Training mode: Teacher forcing
            return self._forward_teacher_forcing(Hvar, mel_gt)
        else:
            # Inference mode: Autoregressive generation
            return self._forward_autoregressive(Hvar, max_len)
    
    def _forward_teacher_forcing(self, Hvar, mel_gt):
        """
        Forward pass with teacher forcing for training.
        
        Args:
            Hvar: Encoder output [B, Tfrm, d_model]
            mel_gt: Ground truth mel [B, Tfrm, n_mels]
        
        Returns:
            mel_pred: Predicted mel [B, Tfrm, n_mels]
        """
        B, Tfrm, _ = Hvar.shape
        
        # Print shapes for debugging
        print(f"[PNCAARDecoder] Training mode - Input Hvar shape: {Hvar.shape}")
        print(f"[PNCAARDecoder] Training mode - Input mel_gt shape: {mel_gt.shape}")
        
        # Shift mel_gt right by prepending zeros (start token)
        # This ensures the decoder predicts frame t using frames 0 to t-1
        mel_shifted = self._shift_mel_right(mel_gt)
        print(f"[PNCAARDecoder] mel_shifted shape: {mel_shifted.shape}")
        
        # Pass through prenet
        tgt = self.prenet(mel_shifted)
        print(f"[PNCAARDecoder] After prenet shape: {tgt.shape}")
        
        # Add positional encoding
        tgt = self.pos_encoding(tgt)
        
        # Create causal mask to prevent attending to future frames
        tgt_mask = self._generate_square_subsequent_mask(Tfrm).to(Hvar.device)
        
        # Pass through Transformer decoder
        # memory = Hvar (encoder output)
        # tgt = decoder input (shifted mel)
        decoder_output = self.decoder(
            tgt=tgt,
            memory=Hvar,
            tgt_mask=tgt_mask
        )
        print(f"[PNCAARDecoder] After decoder shape: {decoder_output.shape}")
        
        # Project to mel dimension
        mel_pred = self.mel_proj(decoder_output)
        print(f"[PNCAARDecoder] Output mel_pred shape: {mel_pred.shape}")
        
        return mel_pred
    
    def _forward_autoregressive(self, Hvar, max_len=None):
        """
        Forward pass with chunk-based autoregressive generation for inference.
        
        Generates mel-spectrogram frames in chunks for efficient streaming inference.
        Each iteration generates chunk_size frames at a time by running the decoder
        multiple times within each chunk.
        
        Args:
            Hvar: Encoder output [B, Tfrm, d_model]
            max_len: Maximum length to generate (defaults to Hvar length)
        
        Returns:
            mel_pred: Generated mel [B, Tfrm, n_mels]
        """
        B, Tfrm, _ = Hvar.shape
        
        if max_len is None:
            max_len = Tfrm
        
        print(f"[PNCAARDecoder] Inference mode - Input Hvar shape: {Hvar.shape}")
        print(f"[PNCAARDecoder] Generating {max_len} frames autoregressively with chunk_size={self.chunk_size}")
        
        # Initialize with zeros (start token)
        mel_pred = torch.zeros(B, 1, self.n_mels, device=Hvar.device)
        print(f"[PNCAARDecoder] Initial mel_pred shape: {mel_pred.shape}")
        
        # Generate frames in chunks
        num_generated = 0
        chunk_count = 0
        
        while num_generated < max_len:
            # Determine how many frames to generate in this chunk
            frames_to_generate = min(self.chunk_size, max_len - num_generated)
            
            # Generate frames_to_generate frames one at a time within this chunk
            for _ in range(frames_to_generate):
                # Pass current sequence through prenet
                tgt = self.prenet(mel_pred)
                
                # Add positional encoding
                tgt = self.pos_encoding(tgt)
                
                # Create causal mask for current sequence length
                current_len = mel_pred.size(1)
                tgt_mask = self._generate_square_subsequent_mask(current_len).to(Hvar.device)
                
                # Pass through decoder
                decoder_output = self.decoder(
                    tgt=tgt,
                    memory=Hvar,
                    tgt_mask=tgt_mask
                )
                
                # Project to mel dimension and take the last frame
                next_mel = self.mel_proj(decoder_output[:, -1:, :])
                
                # Append to sequence
                mel_pred = torch.cat([mel_pred, next_mel], dim=1)
            
            print(f"[PNCAARDecoder] Chunk {chunk_count}: Generated {frames_to_generate} frames, current shape: {mel_pred.shape}")
            
            num_generated += frames_to_generate
            chunk_count += 1
        
        # Remove the initial zero frame
        mel_pred = mel_pred[:, 1:, :]
        
        print(f"[PNCAARDecoder] Final output mel_pred shape: {mel_pred.shape}")
        print(f"[PNCAARDecoder] Total chunks generated: {chunk_count}")
        
        return mel_pred
    
    def _shift_mel_right(self, mel):
        """
        Shift mel-spectrogram right by prepending zeros.
        
        This creates the decoder input for teacher forcing, where each position
        can only attend to previous positions.
        
        Args:
            mel: Ground truth mel [B, Tfrm, n_mels]
        
        Returns:
            mel_shifted: Shifted mel [B, Tfrm, n_mels]
        """
        B, Tfrm, n_mels = mel.shape
        
        # Create zero frame as start token
        start_token = torch.zeros(B, 1, n_mels, device=mel.device)
        
        # Concatenate and remove last frame
        mel_shifted = torch.cat([start_token, mel[:, :-1, :]], dim=1)
        
        return mel_shifted
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a causal mask for the decoder.
        
        The mask prevents positions from attending to subsequent positions.
        Positions are allowed to attend to themselves and previous positions.
        
        Args:
            sz: Sequence length
        
        Returns:
            mask: Causal mask [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    
    Adds sinusoidal positional information to the input embeddings.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Hidden dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [B, T, d_model]
        
        Returns:
            x: Input with positional encoding [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
