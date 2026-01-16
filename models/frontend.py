"""
Front-end text processing module for TTS system.

This module implements a simple character-to-token mapping (pseudo G2P) that converts
raw Chinese text into linguistic features (phoneme IDs, tone IDs, and boundary IDs).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


# Enable/disable shape logging via environment variable
DEBUG_SHAPES = os.getenv("DEBUG_SHAPES", "1") == "1"


@dataclass
class LinguisticFeature:
    """
    Linguistic features output from the Front-end module.
    
    Attributes:
        ph_ids: Phoneme IDs [B, Tph]
        tone_ids: Tone IDs [B, Tph]
        boundary_ids: Boundary IDs [B, Tph]
    """
    ph_ids: torch.LongTensor      # [B, Tph]
    tone_ids: torch.LongTensor     # [B, Tph]
    boundary_ids: torch.LongTensor # [B, Tph]


class FrontEnd(nn.Module):
    """
    Front-end text processing module that converts raw text to linguistic features.
    
    This is a simplified pseudo G2P implementation that maps each character to a token.
    In a production system, this would be replaced with proper text normalization (TN),
    word segmentation, polyphone disambiguation, and tone sandhi modules.
    
    Args:
        vocab_size: Size of the phoneme vocabulary
        tone_size: Number of tone categories
        boundary_size: Number of boundary categories
    """
    
    def __init__(self, vocab_size: int = 300, tone_size: int = 10, boundary_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.tone_size = tone_size
        self.boundary_size = boundary_size
        
        # Special tokens
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.BOS_ID = 2
        self.EOS_ID = 3
        
        # Simple character to ID mapping (pseudo G2P)
        # In a real system, this would be a proper G2P model
        self._build_char_mapping()
    
    def _build_char_mapping(self):
        """
        Build a simple character-to-ID mapping for demonstration.
        This is a placeholder that will be replaced with real G2P in production.
        """
        # For now, we'll use Unicode code points modulo vocab_size as a simple mapping
        # This ensures consistent mapping for the same character
        pass
    
    def _char_to_ph_id(self, char: str) -> int:
        """
        Convert a character to a phoneme ID.
        
        Args:
            char: Input character
            
        Returns:
            Phoneme ID (int)
        """
        if char == ' ':
            return self.PAD_ID
        
        # Simple mapping: use Unicode code point modulo vocab_size
        # Reserve first 4 IDs for special tokens
        char_id = (ord(char) % (self.vocab_size - 4)) + 4
        return char_id
    
    def _char_to_tone_id(self, char: str) -> int:
        """
        Convert a character to a tone ID.
        
        Args:
            char: Input character
            
        Returns:
            Tone ID (int)
        """
        # Pseudo tone assignment based on character
        # In a real system, this would come from proper tone analysis
        if char == ' ':
            return 0
        return (ord(char) % (self.tone_size - 1)) + 1
    
    def _char_to_boundary_id(self, idx: int, text_len: int) -> int:
        """
        Assign boundary ID based on position in text.
        
        Args:
            idx: Character index in text
            text_len: Total length of text
            
        Returns:
            Boundary ID (int)
        """
        # Simple boundary assignment:
        # 0: padding
        # 1: beginning of sentence
        # 2: middle of sentence
        # 3: end of sentence
        # 4: single character sentence
        
        if text_len == 1:
            return 4  # Single character
        elif idx == 0:
            return 1  # Beginning
        elif idx == text_len - 1:
            return 3  # End
        else:
            return 2  # Middle
    
    def text_to_sequence(self, text: str) -> tuple:
        """
        Convert text string to sequences of IDs.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (ph_ids, tone_ids, boundary_ids) as lists
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        if len(text) == 0:
            # Return empty sequences with BOS and EOS
            return [self.BOS_ID, self.EOS_ID], [0, 0], [1, 3]
        
        ph_ids = [self.BOS_ID]
        tone_ids = [0]  # BOS has no tone
        boundary_ids = [1]  # BOS is at beginning
        
        # Process each character
        for idx, char in enumerate(text):
            ph_id = self._char_to_ph_id(char)
            tone_id = self._char_to_tone_id(char)
            boundary_id = self._char_to_boundary_id(idx, len(text))
            
            ph_ids.append(ph_id)
            tone_ids.append(tone_id)
            boundary_ids.append(boundary_id)
        
        # Add EOS tokens
        ph_ids.append(self.EOS_ID)
        tone_ids.append(0)  # EOS has no tone
        boundary_ids.append(3)  # EOS is at end
        
        return ph_ids, tone_ids, boundary_ids
    
    def forward(self, text: str, batch_size: int = 1) -> LinguisticFeature:
        """
        Forward pass: convert text to linguistic features.
        
        Args:
            text: Input text string
            batch_size: Batch size (default: 1)
            
        Returns:
            LinguisticFeature with ph_ids, tone_ids, boundary_ids tensors
        """
        # Convert text to sequences
        ph_ids, tone_ids, boundary_ids = self.text_to_sequence(text)
        
        # Convert to tensors and add batch dimension
        ph_ids_tensor = torch.LongTensor(ph_ids).unsqueeze(0)  # [1, Tph]
        tone_ids_tensor = torch.LongTensor(tone_ids).unsqueeze(0)  # [1, Tph]
        boundary_ids_tensor = torch.LongTensor(boundary_ids).unsqueeze(0)  # [1, Tph]
        
        # Repeat for batch size if needed
        if batch_size > 1:
            ph_ids_tensor = ph_ids_tensor.repeat(batch_size, 1)
            tone_ids_tensor = tone_ids_tensor.repeat(batch_size, 1)
            boundary_ids_tensor = boundary_ids_tensor.repeat(batch_size, 1)
        
        # Shape logging
        if DEBUG_SHAPES:
            print(f"[FrontEnd] Input text: '{text}'")
            print(f"[FrontEnd] Output ph_ids shape: {ph_ids_tensor.shape}")
            print(f"[FrontEnd] Output tone_ids shape: {tone_ids_tensor.shape}")
            print(f"[FrontEnd] Output boundary_ids shape: {boundary_ids_tensor.shape}")
        
        return LinguisticFeature(
            ph_ids=ph_ids_tensor,
            tone_ids=tone_ids_tensor,
            boundary_ids=boundary_ids_tensor
        )
    
    def batch_forward(self, texts: List[str]) -> LinguisticFeature:
        """
        Process a batch of texts with padding.
        
        Args:
            texts: List of input text strings
            
        Returns:
            LinguisticFeature with batched and padded tensors
        """
        batch_size = len(texts)
        
        # Convert all texts to sequences
        all_ph_ids = []
        all_tone_ids = []
        all_boundary_ids = []
        
        for text in texts:
            ph_ids, tone_ids, boundary_ids = self.text_to_sequence(text)
            all_ph_ids.append(ph_ids)
            all_tone_ids.append(tone_ids)
            all_boundary_ids.append(boundary_ids)
        
        # Find max length for padding
        max_len = max(len(seq) for seq in all_ph_ids)
        
        # Pad sequences
        padded_ph_ids = []
        padded_tone_ids = []
        padded_boundary_ids = []
        
        for ph_ids, tone_ids, boundary_ids in zip(all_ph_ids, all_tone_ids, all_boundary_ids):
            pad_len = max_len - len(ph_ids)
            
            padded_ph_ids.append(ph_ids + [self.PAD_ID] * pad_len)
            padded_tone_ids.append(tone_ids + [0] * pad_len)
            padded_boundary_ids.append(boundary_ids + [0] * pad_len)
        
        # Convert to tensors
        ph_ids_tensor = torch.LongTensor(padded_ph_ids)  # [B, Tph]
        tone_ids_tensor = torch.LongTensor(padded_tone_ids)  # [B, Tph]
        boundary_ids_tensor = torch.LongTensor(padded_boundary_ids)  # [B, Tph]
        
        # Shape logging
        if DEBUG_SHAPES:
            print(f"[FrontEnd] Batch size: {batch_size}")
            print(f"[FrontEnd] Output ph_ids shape: {ph_ids_tensor.shape}")
            print(f"[FrontEnd] Output tone_ids shape: {tone_ids_tensor.shape}")
            print(f"[FrontEnd] Output boundary_ids shape: {boundary_ids_tensor.shape}")
        
        return LinguisticFeature(
            ph_ids=ph_ids_tensor,
            tone_ids=tone_ids_tensor,
            boundary_ids=boundary_ids_tensor
        )
