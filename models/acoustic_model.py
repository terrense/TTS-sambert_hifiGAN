"""
SAM-BERT Acoustic Model for Text-to-Speech Synthesis.

This module integrates all components of the SAM-BERT acoustic model:
- Phoneme Embedding: Converts linguistic features to continuous embeddings
- BERT Encoder: Extracts context-aware phoneme representations
- Variance Adaptor: Predicts and applies prosodic features (duration, pitch, energy)
- PNCA AR-Decoder: Autoregressively generates mel-spectrograms

The complete pipeline transforms linguistic features into mel-spectrograms:
    LinguisticFeature → H0 → Henc → Hvar → mel_pred
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from models.phoneme_embedding import PhonemeEmbedding
from models.bert_encoder import BERTEncoder
from models.variance_adaptor import VarianceAdaptor
from models.ar_decoder import PNCAARDecoder


class SAMBERTAcousticModel(nn.Module):
    """
    Complete SAM-BERT Acoustic Model.
    
    This model integrates all components of the SAM-BERT TTS system to transform
    linguistic features (phoneme IDs, tone IDs, boundary IDs) into mel-spectrograms.
    
    Architecture:
        1. Phoneme Embedding: Converts discrete linguistic features to continuous embeddings
        2. BERT Encoder: Applies multi-layer Transformer encoding for context modeling
        3. Variance Adaptor: Predicts duration, pitch, energy and expands to frame-level
        4. PNCA AR-Decoder: Autoregressively generates mel-spectrogram frames
    
    Training Mode:
        - Uses ground truth duration, pitch, energy for teacher forcing
        - Uses ground truth mel for decoder teacher forcing
        - Returns predictions for loss computation
    
    Inference Mode:
        - Uses predicted duration, pitch, energy
        - Autoregressively generates mel-spectrogram
        - Returns generated mel-spectrogram
    
    Args:
        vocab_size: Size of phoneme vocabulary
        tone_size: Number of tone categories
        boundary_size: Number of boundary categories
        d_model: Hidden dimension size
        encoder_layers: Number of BERT encoder layers
        encoder_heads: Number of attention heads in encoder
        encoder_ff_dim: Feed-forward dimension in encoder
        decoder_layers: Number of AR-decoder layers
        decoder_heads: Number of attention heads in decoder
        decoder_ff_dim: Feed-forward dimension in decoder
        n_mels: Number of mel-spectrogram bins
        dropout: Dropout rate
        pitch_bins: Number of pitch quantization bins
        pitch_min: Minimum pitch value in Hz
        pitch_max: Maximum pitch value in Hz
        energy_bins: Number of energy quantization bins
        energy_min: Minimum energy value
        energy_max: Maximum energy value
        chunk_size: Chunk size for streaming inference
    
    Shape Contract:
        Input:
            - ph_ids: [B, Tph] phoneme IDs
            - tone_ids: [B, Tph] tone IDs
            - boundary_ids: [B, Tph] boundary IDs
            - dur_gt: [B, Tph] ground truth duration (optional, training only)
            - pitch_gt: [B, Tfrm] ground truth pitch (optional, training only)
            - energy_gt: [B, Tfrm] ground truth energy (optional, training only)
            - mel_gt: [B, Tfrm, n_mels] ground truth mel (optional, training only)
        
        Output:
            - mel_pred: [B, Tfrm, n_mels] predicted mel-spectrogram
            - predictions: dict containing all intermediate predictions
    
    Example:
        >>> model = SAMBERTAcousticModel(
        ...     vocab_size=300, tone_size=10, boundary_size=5,
        ...     d_model=256, encoder_layers=6, encoder_heads=4
        ... )
        >>> ph_ids = torch.randint(0, 300, (2, 20))
        >>> tone_ids = torch.randint(0, 10, (2, 20))
        >>> boundary_ids = torch.randint(0, 5, (2, 20))
        >>> mel_pred, predictions = model(ph_ids, tone_ids, boundary_ids)
        >>> print(mel_pred.shape)  # [2, Tfrm, 80]
    
    References:
        - BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
        - FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)
        - SAM-BERT: Speech Acoustic Model with BERT
    """
    
    def __init__(
        self,
        # Vocabulary sizes
        vocab_size: int = 300,
        tone_size: int = 10,
        boundary_size: int = 5,
        # Model dimensions
        d_model: int = 256,
        # Encoder configuration
        encoder_layers: int = 6,
        encoder_heads: int = 4,
        encoder_ff_dim: int = 1024,
        # Decoder configuration
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        # Mel configuration
        n_mels: int = 80,
        # Regularization
        dropout: float = 0.1,
        # Variance Adaptor configuration
        pitch_bins: int = 256,
        pitch_min: float = 80.0,
        pitch_max: float = 600.0,
        energy_bins: int = 256,
        energy_min: float = 0.0,
        energy_max: float = 1.0,
        # Inference configuration
        chunk_size: int = 1
    ):
        """Initialize the SAM-BERT Acoustic Model."""
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.tone_size = tone_size
        self.boundary_size = boundary_size
        self.d_model = d_model
        self.n_mels = n_mels
        
        # ==================== Component 1: Phoneme Embedding ====================
        self.phoneme_embedding = PhonemeEmbedding(
            vocab_size=vocab_size,
            tone_size=tone_size,
            boundary_size=boundary_size,
            d_model=d_model
        )
        
        # ==================== Component 2: BERT Encoder ====================
        self.bert_encoder = BERTEncoder(
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            d_ff=encoder_ff_dim,
            dropout=dropout
        )
        
        # ==================== Component 3: Variance Adaptor ====================
        self.variance_adaptor = VarianceAdaptor(
            d_model=d_model,
            n_layers=2,  # Conv layers in predictors
            kernel_size=3,
            dropout=dropout,
            pitch_bins=pitch_bins,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            energy_bins=energy_bins,
            energy_min=energy_min,
            energy_max=energy_max
        )
        
        # ==================== Component 4: PNCA AR-Decoder ====================
        self.ar_decoder = PNCAARDecoder(
            d_model=d_model,
            n_mels=n_mels,
            n_layers=decoder_layers,
            n_heads=decoder_heads,
            d_ff=decoder_ff_dim,
            dropout=dropout,
            chunk_size=chunk_size
        )
    
    def forward(
        self,
        ph_ids: torch.LongTensor,
        tone_ids: torch.LongTensor,
        boundary_ids: torch.LongTensor,
        dur_gt: Optional[torch.LongTensor] = None,
        pitch_gt: Optional[torch.FloatTensor] = None,
        energy_gt: Optional[torch.FloatTensor] = None,
        mel_gt: Optional[torch.FloatTensor] = None,
        max_len: Optional[int] = None
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete SAM-BERT acoustic model.
        
        This method orchestrates the complete pipeline:
        1. Convert linguistic features to embeddings (H0)
        2. Apply BERT encoding for context modeling (Henc)
        3. Apply variance adaptation for prosody (Hvar)
        4. Generate mel-spectrogram autoregressively (mel_pred)
        
        Args:
            ph_ids: Phoneme IDs [B, Tph]
            tone_ids: Tone IDs [B, Tph]
            boundary_ids: Boundary IDs [B, Tph]
            dur_gt: Ground truth duration [B, Tph] (optional, training only)
            pitch_gt: Ground truth pitch [B, Tfrm] (optional, training only)
            energy_gt: Ground truth energy [B, Tfrm] (optional, training only)
            mel_gt: Ground truth mel [B, Tfrm, n_mels] (optional, training only)
            max_len: Maximum length for generation (optional, inference only)
        
        Returns:
            mel_pred: Predicted mel-spectrogram [B, Tfrm, n_mels]
            predictions: Dictionary containing all intermediate predictions:
                - 'log_dur_pred': [B, Tph] log-duration predictions
                - 'dur': [B, Tph] duration used for expansion
                - 'pitch_tok': [B, Tph] phoneme-level pitch predictions
                - 'pitch_frm': [B, Tfrm] frame-level pitch values
                - 'energy_tok': [B, Tph] phoneme-level energy predictions
                - 'energy_frm': [B, Tfrm] frame-level energy values
        
        Example:
            Training:
                >>> mel_pred, preds = model(
                ...     ph_ids, tone_ids, boundary_ids,
                ...     dur_gt=dur_gt, pitch_gt=pitch_gt,
                ...     energy_gt=energy_gt, mel_gt=mel_gt
                ... )
            
            Inference:
                >>> mel_pred, preds = model(ph_ids, tone_ids, boundary_ids)
        """
        print("=" * 80)
        print("[SAMBERTAcousticModel] Starting forward pass")
        print("=" * 80)
        
        # ==================== Stage 1: Phoneme Embedding ====================
        print("\n[SAMBERTAcousticModel] Stage 1: Phoneme Embedding")
        H0 = self.phoneme_embedding(ph_ids, tone_ids, boundary_ids)
        print(f"[SAMBERTAcousticModel] H0 shape: {H0.shape}")
        
        # ==================== Stage 2: BERT Encoder ====================
        print("\n[SAMBERTAcousticModel] Stage 2: BERT Encoder")
        Henc = self.bert_encoder(H0)
        print(f"[SAMBERTAcousticModel] Henc shape: {Henc.shape}")
        
        # ==================== Stage 3: Variance Adaptor ====================
        print("\n[SAMBERTAcousticModel] Stage 3: Variance Adaptor")
        Hvar, predictions = self.variance_adaptor(
            Henc,
            dur_gt=dur_gt,
            pitch_gt=pitch_gt,
            energy_gt=energy_gt
        )
        print(f"[SAMBERTAcousticModel] Hvar shape: {Hvar.shape}")
        
        # ==================== Stage 4: AR-Decoder ====================
        print("\n[SAMBERTAcousticModel] Stage 4: PNCA AR-Decoder")
        mel_pred = self.ar_decoder(Hvar, mel_gt=mel_gt, max_len=max_len)
        print(f"[SAMBERTAcousticModel] mel_pred shape: {mel_pred.shape}")
        
        print("\n" + "=" * 80)
        print("[SAMBERTAcousticModel] Forward pass complete")
        print("=" * 80 + "\n")
        
        return mel_pred, predictions
    
    def inference(
        self,
        ph_ids: torch.LongTensor,
        tone_ids: torch.LongTensor,
        boundary_ids: torch.LongTensor,
        max_len: Optional[int] = None
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        """
        Inference-only forward pass (convenience method).
        
        This is a convenience wrapper around forward() that explicitly sets
        the model to eval mode and doesn't require ground truth inputs.
        
        Args:
            ph_ids: Phoneme IDs [B, Tph]
            tone_ids: Tone IDs [B, Tph]
            boundary_ids: Boundary IDs [B, Tph]
            max_len: Maximum length for generation (optional)
        
        Returns:
            mel_pred: Predicted mel-spectrogram [B, Tfrm, n_mels]
            predictions: Dictionary containing all intermediate predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                ph_ids=ph_ids,
                tone_ids=tone_ids,
                boundary_ids=boundary_ids,
                max_len=max_len
            )
    
    def get_config(self) -> Dict:
        """
        Get model configuration as a dictionary.
        
        Returns:
            Dictionary containing all model configuration parameters
        """
        return {
            'vocab_size': self.vocab_size,
            'tone_size': self.tone_size,
            'boundary_size': self.boundary_size,
            'd_model': self.d_model,
            'n_mels': self.n_mels,
            'encoder_config': self.bert_encoder.get_config(),
        }
