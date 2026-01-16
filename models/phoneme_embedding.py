"""
Phoneme Embedding Module for SAM-BERT Acoustic Model

This module converts discrete linguistic features (phoneme IDs, tone IDs, boundary IDs)
into continuous vector representations.
"""

import torch
import torch.nn as nn


class PhonemeEmbedding(nn.Module):
    """
    Phoneme Embedding layer that combines three types of linguistic features:
    - Phoneme IDs (ph_ids): The phoneme/character tokens
    - Tone IDs (tone_ids): Tonal information for each phoneme
    - Boundary IDs (boundary_ids): Prosodic boundary markers
    
    The three embeddings are summed to produce the initial hidden representation H0.
    """
    
    def __init__(self, vocab_size: int, tone_size: int, boundary_size: int, d_model: int):
        """
        Initialize the Phoneme Embedding module.
        
        Args:
            vocab_size: Size of the phoneme vocabulary
            tone_size: Number of tone categories
            boundary_size: Number of boundary categories
            d_model: Embedding dimension (hidden size)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.tone_size = tone_size
        self.boundary_size = boundary_size
        self.d_model = d_model
        
        # Three embedding layers for different linguistic features
        self.ph_emb = nn.Embedding(vocab_size, d_model)
        self.tone_emb = nn.Embedding(tone_size, d_model)
        self.boundary_emb = nn.Embedding(boundary_size, d_model)
        
    def forward(self, ph_ids: torch.LongTensor, tone_ids: torch.LongTensor, 
                boundary_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass: convert linguistic features to continuous embeddings.
        
        Args:
            ph_ids: Phoneme IDs [B, Tph]
            tone_ids: Tone IDs [B, Tph]
            boundary_ids: Boundary IDs [B, Tph]
            
        Returns:
            H0: Combined embedding [B, Tph, d_model]
        """
        # Shape logging for input tensors
        print(f"[PhonemeEmbedding] Input shapes:")
        print(f"  ph_ids: {ph_ids.shape}")
        print(f"  tone_ids: {tone_ids.shape}")
        print(f"  boundary_ids: {boundary_ids.shape}")
        
        # Embed each feature type
        ph_embedded = self.ph_emb(ph_ids)          # [B, Tph, d_model]
        tone_embedded = self.tone_emb(tone_ids)    # [B, Tph, d_model]
        boundary_embedded = self.boundary_emb(boundary_ids)  # [B, Tph, d_model]
        
        # Sum all embeddings to produce H0
        H0 = ph_embedded + tone_embedded + boundary_embedded  # [B, Tph, d_model]
        
        # Shape logging for output tensor
        print(f"[PhonemeEmbedding] Output shape:")
        print(f"  H0: {H0.shape}")
        
        return H0
