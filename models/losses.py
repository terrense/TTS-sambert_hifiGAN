"""
Loss functions for SAM-BERT Acoustic Model.

This module implements loss functions for training the acoustic model:
- L_mel: L1 loss between predicted and ground truth mel-spectrograms
- L_dur: MSE loss between predicted and ground truth log-durations
- L_pitch: MSE loss between predicted and ground truth pitch (with masking)
- L_energy: MSE loss between predicted and ground truth energy

The total loss is a weighted combination of all component losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AcousticLoss(nn.Module):
    """
    Acoustic Loss for SAM-BERT model training.
    
    This class computes all loss components for training the acoustic model:
    1. Mel-spectrogram reconstruction loss (L1)
    2. Duration prediction loss (MSE in log-space)
    3. Pitch prediction loss (MSE with optional masking)
    4. Energy prediction loss (MSE)
    
    The total loss is computed as a weighted sum of all components:
        L_total = w_mel * L_mel + w_dur * L_dur + w_pitch * L_pitch + w_energy * L_energy
    
    Args:
        mel_weight (float): Weight for mel-spectrogram loss (default: 1.0)
        dur_weight (float): Weight for duration loss (default: 1.0)
        pitch_weight (float): Weight for pitch loss (default: 1.0)
        energy_weight (float): Weight for energy loss (default: 1.0)
    
    References:
        - FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)
        - SAM-BERT: Speech Acoustic Model with BERT
    """
    
    def __init__(
        self,
        mel_weight: float = 1.0,
        dur_weight: float = 1.0,
        pitch_weight: float = 1.0,
        energy_weight: float = 1.0
    ):
        """Initialize the AcousticLoss with configurable loss weights."""
        super().__init__()
        
        self.mel_weight = mel_weight
        self.dur_weight = dur_weight
        self.pitch_weight = pitch_weight
        self.energy_weight = energy_weight
    
    def compute_mel_loss(
        self,
        mel_pred: torch.FloatTensor,
        mel_gt: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute L1 loss between predicted and ground truth mel-spectrograms.
        
        L_mel = L1(mel_pred, mel_gt)
        
        This loss measures the reconstruction quality of the mel-spectrogram.
        L1 loss is preferred over L2 (MSE) for mel-spectrogram reconstruction
        as it tends to produce sharper and more natural-sounding results.
        
        Args:
            mel_pred (torch.FloatTensor): Predicted mel-spectrogram [B, Tfrm, n_mels]
            mel_gt (torch.FloatTensor): Ground truth mel-spectrogram [B, Tfrm, n_mels]
            mask (torch.BoolTensor, optional): Mask for valid frames [B, Tfrm]
                True for valid frames, False for padding
        
        Returns:
            torch.FloatTensor: Scalar L1 loss value
        
        Example:
            >>> loss_fn = AcousticLoss()
            >>> mel_pred = torch.randn(2, 100, 80)
            >>> mel_gt = torch.randn(2, 100, 80)
            >>> loss = loss_fn.compute_mel_loss(mel_pred, mel_gt)
        """
        # Compute L1 loss
        loss = F.l1_loss(mel_pred, mel_gt, reduction='none')  # [B, Tfrm, n_mels]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match mel dimensions [B, Tfrm] -> [B, Tfrm, 1]
            mask = mask.unsqueeze(-1)
            # Zero out loss for padded frames
            loss = loss * mask.float()
            # Average over valid frames only
            loss = loss.sum() / (mask.sum() * mel_pred.size(-1) + 1e-8)
        else:
            # Average over all elements
            loss = loss.mean()
        
        return loss
    
    def compute_duration_loss(
        self,
        log_dur_pred: torch.FloatTensor,
        dur_gt: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute MSE loss between predicted log-duration and ground truth log-duration.
        
        L_dur = MSE(log_dur_pred, log(dur_gt + 1))
        
        Duration prediction is performed in log-space because:
        1. Duration values have a long-tailed distribution
        2. Log-space makes optimization more stable
        3. Prevents negative duration predictions
        
        We add 1 to dur_gt before taking log to avoid log(0) for zero-duration phonemes.
        
        Args:
            log_dur_pred (torch.FloatTensor): Predicted log-duration [B, Tph]
            dur_gt (torch.LongTensor): Ground truth duration in frames [B, Tph]
            mask (torch.BoolTensor, optional): Mask for valid phonemes [B, Tph]
                True for valid phonemes, False for padding
        
        Returns:
            torch.FloatTensor: Scalar MSE loss value
        
        Example:
            >>> loss_fn = AcousticLoss()
            >>> log_dur_pred = torch.randn(2, 20)
            >>> dur_gt = torch.randint(1, 10, (2, 20))
            >>> loss = loss_fn.compute_duration_loss(log_dur_pred, dur_gt)
        """
        # Convert ground truth duration to log-space
        # Add 1 to avoid log(0) for zero-duration phonemes
        log_dur_gt = torch.log(dur_gt.float() + 1.0)  # [B, Tph]
        
        # Compute MSE loss
        loss = F.mse_loss(log_dur_pred, log_dur_gt, reduction='none')  # [B, Tph]
        
        # Apply mask if provided
        if mask is not None:
            # Zero out loss for padded phonemes
            loss = loss * mask.float()
            # Average over valid phonemes only
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            # Average over all elements
            loss = loss.mean()
        
        return loss
    
    def compute_pitch_loss(
        self,
        pitch_pred: torch.FloatTensor,
        pitch_gt: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute MSE loss between predicted and ground truth pitch with masking.
        
        L_pitch = MSE(pitch_pred, pitch_gt) with mask for unvoiced segments
        
        Pitch prediction is challenging because:
        1. Pitch is only defined for voiced segments (vowels, voiced consonants)
        2. Unvoiced segments (silence, unvoiced consonants) should not contribute to loss
        3. Masking is essential to avoid penalizing predictions on unvoiced segments
        
        The mask should mark voiced segments as True and unvoiced segments as False.
        
        Args:
            pitch_pred (torch.FloatTensor): Predicted pitch values [B, Tfrm]
                Can be phoneme-level [B, Tph] or frame-level [B, Tfrm]
            pitch_gt (torch.FloatTensor): Ground truth pitch values [B, Tfrm]
                Same shape as pitch_pred
            mask (torch.BoolTensor, optional): Mask for voiced segments [B, Tfrm]
                True for voiced segments, False for unvoiced/padding
                If not provided, all frames are considered valid
        
        Returns:
            torch.FloatTensor: Scalar MSE loss value
        
        Example:
            >>> loss_fn = AcousticLoss()
            >>> pitch_pred = torch.randn(2, 100)
            >>> pitch_gt = torch.randn(2, 100)
            >>> mask = torch.rand(2, 100) > 0.3  # 70% voiced
            >>> loss = loss_fn.compute_pitch_loss(pitch_pred, pitch_gt, mask)
        """
        # Compute MSE loss
        loss = F.mse_loss(pitch_pred, pitch_gt, reduction='none')  # [B, Tfrm] or [B, Tph]
        
        # Apply mask if provided
        if mask is not None:
            # Zero out loss for unvoiced/padded segments
            loss = loss * mask.float()
            # Average over valid (voiced) segments only
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            # Average over all elements
            loss = loss.mean()
        
        return loss
    
    def compute_energy_loss(
        self,
        energy_pred: torch.FloatTensor,
        energy_gt: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute MSE loss between predicted and ground truth energy.
        
        L_energy = MSE(energy_pred, energy_gt)
        
        Energy (also called amplitude or loudness) represents the intensity of speech.
        Unlike pitch, energy is defined for all segments including unvoiced ones,
        though masking can still be applied to exclude padding frames.
        
        Args:
            energy_pred (torch.FloatTensor): Predicted energy values [B, Tfrm]
                Can be phoneme-level [B, Tph] or frame-level [B, Tfrm]
            energy_gt (torch.FloatTensor): Ground truth energy values [B, Tfrm]
                Same shape as energy_pred
            mask (torch.BoolTensor, optional): Mask for valid frames [B, Tfrm]
                True for valid frames, False for padding
        
        Returns:
            torch.FloatTensor: Scalar MSE loss value
        
        Example:
            >>> loss_fn = AcousticLoss()
            >>> energy_pred = torch.randn(2, 100)
            >>> energy_gt = torch.randn(2, 100)
            >>> loss = loss_fn.compute_energy_loss(energy_pred, energy_gt)
        """
        # Compute MSE loss
        loss = F.mse_loss(energy_pred, energy_gt, reduction='none')  # [B, Tfrm] or [B, Tph]
        
        # Apply mask if provided
        if mask is not None:
            # Zero out loss for padded frames
            loss = loss * mask.float()
            # Average over valid frames only
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            # Average over all elements
            loss = loss.mean()
        
        return loss
    
    def forward(
        self,
        mel_pred: torch.FloatTensor,
        mel_gt: torch.FloatTensor,
        log_dur_pred: torch.FloatTensor,
        dur_gt: torch.LongTensor,
        pitch_pred: torch.FloatTensor,
        pitch_gt: torch.FloatTensor,
        energy_pred: torch.FloatTensor,
        energy_gt: torch.FloatTensor,
        mel_mask: Optional[torch.BoolTensor] = None,
        phoneme_mask: Optional[torch.BoolTensor] = None,
        pitch_mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Compute total acoustic loss and all component losses.
        
        This method computes all loss components and combines them into a total loss
        using the configured weights. It returns both the total loss (for backpropagation)
        and a dictionary of individual losses (for monitoring and logging).
        
        Args:
            mel_pred (torch.FloatTensor): Predicted mel-spectrogram [B, Tfrm, n_mels]
            mel_gt (torch.FloatTensor): Ground truth mel-spectrogram [B, Tfrm, n_mels]
            log_dur_pred (torch.FloatTensor): Predicted log-duration [B, Tph]
            dur_gt (torch.LongTensor): Ground truth duration [B, Tph]
            pitch_pred (torch.FloatTensor): Predicted pitch [B, Tfrm] or [B, Tph]
            pitch_gt (torch.FloatTensor): Ground truth pitch [B, Tfrm]
            energy_pred (torch.FloatTensor): Predicted energy [B, Tfrm] or [B, Tph]
            energy_gt (torch.FloatTensor): Ground truth energy [B, Tfrm]
            mel_mask (torch.BoolTensor, optional): Mask for mel frames [B, Tfrm]
            phoneme_mask (torch.BoolTensor, optional): Mask for phonemes [B, Tph]
            pitch_mask (torch.BoolTensor, optional): Mask for voiced segments [B, Tfrm]
        
        Returns:
            total_loss (torch.FloatTensor): Weighted sum of all losses (scalar)
            loss_dict (dict): Dictionary containing all individual losses:
                - 'total_loss': Total weighted loss
                - 'mel_loss': Mel-spectrogram reconstruction loss
                - 'dur_loss': Duration prediction loss
                - 'pitch_loss': Pitch prediction loss
                - 'energy_loss': Energy prediction loss
        
        Example:
            >>> loss_fn = AcousticLoss(mel_weight=1.0, dur_weight=0.1)
            >>> # ... get predictions from model ...
            >>> total_loss, loss_dict = loss_fn(
            ...     mel_pred, mel_gt, log_dur_pred, dur_gt,
            ...     pitch_pred, pitch_gt, energy_pred, energy_gt
            ... )
            >>> total_loss.backward()
            >>> print(f"Mel loss: {loss_dict['mel_loss']:.4f}")
        """
        # Compute individual loss components
        mel_loss = self.compute_mel_loss(mel_pred, mel_gt, mel_mask)
        dur_loss = self.compute_duration_loss(log_dur_pred, dur_gt, phoneme_mask)
        pitch_loss = self.compute_pitch_loss(pitch_pred, pitch_gt, pitch_mask)
        energy_loss = self.compute_energy_loss(energy_pred, energy_gt, mel_mask)
        
        # Compute weighted total loss
        total_loss = (
            self.mel_weight * mel_loss +
            self.dur_weight * dur_loss +
            self.pitch_weight * pitch_loss +
            self.energy_weight * energy_loss
        )
        
        # Prepare loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'mel_loss': mel_loss.item(),
            'dur_loss': dur_loss.item(),
            'pitch_loss': pitch_loss.item(),
            'energy_loss': energy_loss.item()
        }
        
        return total_loss, loss_dict
