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
from typing import Dict, Optional, Tuple, List


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


class VocoderLoss(nn.Module):
    """
    Vocoder Loss for HiFi-GAN training.
    
    This class computes all loss components for training the HiFi-GAN vocoder:
    1. Adversarial loss (least squares GAN loss)
    2. Feature matching loss between real and fake feature maps
    3. Multi-resolution STFT loss
    4. Mel reconstruction loss
    
    The generator is trained with:
        L_gen = L_adv + λ_fm * L_fm + λ_mel * L_mel
    
    The discriminator is trained with:
        L_disc = L_adv_real + L_adv_fake
    
    Args:
        feature_matching_weight (float): Weight for feature matching loss (default: 2.0)
        mel_weight (float): Weight for mel reconstruction loss (default: 45.0)
        use_mel_loss (bool): Whether to use mel reconstruction loss (default: True)
        stft_loss_weight (float): Weight for multi-resolution STFT loss (default: 1.0)
        mel_config (dict, optional): Mel-spectrogram configuration. If None, will load from config.yaml
    
    References:
        - HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
        - Least Squares GAN (LSGAN)
    """
    
    def __init__(
        self,
        feature_matching_weight: float = 2.0,
        mel_weight: float = 45.0,
        use_mel_loss: bool = True,
        stft_loss_weight: float = 1.0,
        mel_config: Optional[dict] = None
    ):
        """Initialize the VocoderLoss with configurable loss weights."""
        super().__init__()
        
        self.feature_matching_weight = feature_matching_weight
        self.mel_weight = mel_weight
        self.use_mel_loss = use_mel_loss
        self.stft_loss_weight = stft_loss_weight
        
        # Load mel configuration
        if mel_config is None:
            import yaml
            with open('configs/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            mel_config = config['audio']
        
        self.mel_config = mel_config
        
        # Create mel spectrogram transform using exact same parameters as audio_processing.py
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['n_fft'],
            hop_length=mel_config['hop_length'],
            win_length=mel_config['win_length'],
            n_mels=mel_config['n_mels'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            mel_scale=mel_config.get('mel_scale', 'slaney'),
            norm=mel_config.get('norm', 'slaney'),
            power=2.0  # Power spectrogram
        )
        
        # Store log base for mel conversion
        self.log_base = mel_config.get('log_base', 10.0)
        
        # Multi-resolution STFT parameters
        # Using multiple FFT sizes to capture different frequency resolutions
        self.stft_params = [
            {'n_fft': 1024, 'hop_length': 120, 'win_length': 600},
            {'n_fft': 2048, 'hop_length': 240, 'win_length': 1200},
            {'n_fft': 512, 'hop_length': 50, 'win_length': 240}
        ]
    
    def compute_discriminator_loss(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> torch.FloatTensor:
        """
        Compute discriminator adversarial loss using least squares loss.
        
        For LSGAN, the discriminator loss is:
            L_disc = E[(D(x_real) - 1)^2] + E[D(x_fake)^2]
        
        This encourages the discriminator to output 1 for real samples
        and 0 for fake samples.
        
        Args:
            disc_real_outputs (List[torch.Tensor]): List of discriminator outputs for real audio
                Each element is [B, 1, T'] or [B, 1, H, W] for period discriminators
            disc_fake_outputs (List[torch.Tensor]): List of discriminator outputs for fake audio
                Same structure as disc_real_outputs
        
        Returns:
            torch.FloatTensor: Scalar discriminator loss
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> # Get discriminator outputs from MSD and MPD
            >>> disc_real = [torch.randn(2, 1, 100) for _ in range(8)]
            >>> disc_fake = [torch.randn(2, 1, 100) for _ in range(8)]
            >>> loss = loss_fn.compute_discriminator_loss(disc_real, disc_fake)
        """
        loss = 0.0
        
        for dr, df in zip(disc_real_outputs, disc_fake_outputs):
            # Real loss: (D(x_real) - 1)^2
            real_loss = torch.mean((dr - 1.0) ** 2)
            
            # Fake loss: D(x_fake)^2
            fake_loss = torch.mean(df ** 2)
            
            loss += real_loss + fake_loss
        
        return loss
    
    def compute_generator_adversarial_loss(
        self,
        disc_fake_outputs: List[torch.Tensor]
    ) -> torch.FloatTensor:
        """
        Compute generator adversarial loss using least squares loss.
        
        For LSGAN, the generator loss is:
            L_gen_adv = E[(D(G(z)) - 1)^2]
        
        This encourages the generator to produce samples that the discriminator
        classifies as real (output close to 1).
        
        Args:
            disc_fake_outputs (List[torch.Tensor]): List of discriminator outputs for fake audio
                Each element is [B, 1, T'] or [B, 1, H, W] for period discriminators
        
        Returns:
            torch.FloatTensor: Scalar generator adversarial loss
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> disc_fake = [torch.randn(2, 1, 100) for _ in range(8)]
            >>> loss = loss_fn.compute_generator_adversarial_loss(disc_fake)
        """
        loss = 0.0
        
        for df in disc_fake_outputs:
            # Generator loss: (D(G(z)) - 1)^2
            loss += torch.mean((df - 1.0) ** 2)
        
        return loss
    
    def compute_feature_matching_loss(
        self,
        real_feature_maps: List[List[torch.Tensor]],
        fake_feature_maps: List[List[torch.Tensor]],
        return_per_discriminator: bool = False
    ) -> Tuple[torch.FloatTensor, Optional[List[float]]]:
        """
        Compute feature matching loss between real and fake feature maps.
        
        Feature matching loss encourages the generator to produce audio that has
        similar intermediate representations to real audio in the discriminator.
        
        L_fm = E[||D_i(x_real) - D_i(G(z))||_1]
        
        where D_i represents the i-th layer features of the discriminator.
        
        This implementation:
        1. Iterates over all discriminators (MSD + MPD)
        2. For each discriminator, iterates over all layers
        3. Computes L1 distance between real and fake feature maps
        4. Detaches real features to prevent backprop into discriminator
        5. Aggregates loss across all sub-discriminators using mean
        6. Optionally returns per-discriminator loss contributions for logging
        
        Args:
            real_feature_maps (List[List[torch.Tensor]]): Feature maps from real audio
                Outer list: one per discriminator (MSD + MPD)
                Inner list: one per layer in that discriminator
            fake_feature_maps (List[List[torch.Tensor]]): Feature maps from fake audio
                Same structure as real_feature_maps
            return_per_discriminator (bool): If True, return per-discriminator losses
                for logging purposes (default: False)
        
        Returns:
            loss (torch.FloatTensor): Scalar feature matching loss (averaged over all layers)
            per_disc_losses (Optional[List[float]]): Per-discriminator loss contributions
                Only returned if return_per_discriminator=True, otherwise None
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> # Each discriminator has multiple layers
            >>> real_fmaps = [[torch.randn(2, 128, 100) for _ in range(5)] for _ in range(8)]
            >>> fake_fmaps = [[torch.randn(2, 128, 100) for _ in range(5)] for _ in range(8)]
            >>> loss, per_disc = loss_fn.compute_feature_matching_loss(
            ...     real_fmaps, fake_fmaps, return_per_discriminator=True
            ... )
            >>> print(f"Total FM loss: {loss.item():.4f}")
            >>> for i, disc_loss in enumerate(per_disc):
            ...     print(f"Discriminator {i} FM loss: {disc_loss:.4f}")
        """
        total_loss = 0.0
        per_discriminator_losses = [] if return_per_discriminator else None
        
        # Iterate over discriminators
        for disc_idx, (real_fmap_list, fake_fmap_list) in enumerate(zip(real_feature_maps, fake_feature_maps)):
            disc_loss = 0.0
            
            # Iterate over layers within each discriminator
            for real_fmap, fake_fmap in zip(real_fmap_list, fake_fmap_list):
                # Compute L1 distance between real and fake features
                # Detach real features to prevent backprop into discriminator
                layer_loss = F.l1_loss(fake_fmap, real_fmap.detach())
                disc_loss += layer_loss
            
            # Average over layers in this discriminator
            disc_loss = disc_loss / len(real_fmap_list)
            total_loss += disc_loss
            
            # Store per-discriminator loss for logging
            if return_per_discriminator:
                per_discriminator_losses.append(disc_loss.item())
        
        # Average over all discriminators
        num_discriminators = len(real_feature_maps)
        total_loss = total_loss / num_discriminators
        
        return total_loss, per_discriminator_losses
    
    def compute_stft_loss(
        self,
        wav_real: torch.Tensor,
        wav_fake: torch.Tensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute multi-resolution STFT loss.
        
        STFT loss measures the difference between real and generated audio
        in the time-frequency domain using multiple STFT resolutions.
        
        For each STFT resolution:
            L_sc = ||log|STFT(x_real)| - log|STFT(x_fake)||_1  (spectral convergence)
            L_mag = ||log|STFT(x_real)| - log|STFT(x_fake)||_2  (log magnitude)
        
        Total STFT loss: L_stft = L_sc + L_mag
        
        Args:
            wav_real (torch.Tensor): Real waveform [B, 1, T]
            wav_fake (torch.Tensor): Generated waveform [B, 1, T]
        
        Returns:
            sc_loss (torch.FloatTensor): Spectral convergence loss
            mag_loss (torch.FloatTensor): Log magnitude loss
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> wav_real = torch.randn(2, 1, 22050)
            >>> wav_fake = torch.randn(2, 1, 22050)
            >>> sc_loss, mag_loss = loss_fn.compute_stft_loss(wav_real, wav_fake)
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        # Remove channel dimension for STFT computation
        wav_real = wav_real.squeeze(1)  # [B, T]
        wav_fake = wav_fake.squeeze(1)  # [B, T]
        
        for params in self.stft_params:
            n_fft = params['n_fft']
            hop_length = params['hop_length']
            win_length = params['win_length']
            
            # Create window
            window = torch.hann_window(win_length).to(wav_real.device)
            
            # Compute STFT for real audio
            stft_real = torch.stft(
                wav_real,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True
            )
            mag_real = torch.abs(stft_real)  # [B, F, T]
            
            # Compute STFT for fake audio
            stft_fake = torch.stft(
                wav_fake,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True
            )
            mag_fake = torch.abs(stft_fake)  # [B, F, T]
            
            # Spectral convergence loss (L1 in log-magnitude space)
            # Add small epsilon to avoid log(0)
            log_mag_real = torch.log(mag_real + 1e-5)
            log_mag_fake = torch.log(mag_fake + 1e-5)
            sc_loss += F.l1_loss(log_mag_fake, log_mag_real)
            
            # Log magnitude loss (L2 in log-magnitude space)
            mag_loss += F.mse_loss(log_mag_fake, log_mag_real)
        
        # Average over all STFT resolutions
        sc_loss = sc_loss / len(self.stft_params)
        mag_loss = mag_loss / len(self.stft_params)
        
        return sc_loss, mag_loss
    
    def mel_reconstruction_loss(
        self,
        wav_real: torch.Tensor,
        wav_fake: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Compute mel reconstruction loss between real and generated waveforms.
        
        This loss extracts mel-spectrograms from both real and generated waveforms
        using identical mel extraction parameters (from config.yaml), then computes
        the L1 distance between them.
        
        L_mel = L1(mel(wav_real), mel(wav_fake))
        
        This ensures that the generated waveform has similar spectral characteristics
        to the real waveform, which is crucial for high-quality audio synthesis.
        
        IMPORTANT: This method uses the exact same mel extraction parameters as
        data/audio_processing.py to ensure consistency across training and inference:
        - sample_rate, n_fft, win_length, hop_length
        - n_mels, fmin, fmax
        - mel_scale, norm, log_base
        
        Args:
            wav_real (torch.Tensor): Real waveform [B, 1, T_wav]
            wav_fake (torch.Tensor): Generated waveform [B, 1, T_wav]
        
        Returns:
            torch.FloatTensor: Scalar L1 mel reconstruction loss
        
        Shape Contract:
            Input: wav [B, 1, T_wav]
            Intermediate: mel [B, n_mels, T_mel] where T_mel = T_wav // hop_length + 1
            Output: scalar loss
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> wav_real = torch.randn(2, 1, 22050)
            >>> wav_fake = torch.randn(2, 1, 22050)
            >>> mel_loss = loss_fn.mel_reconstruction_loss(wav_real, wav_fake)
        
        References:
            - Requirement 14.3: Multi-resolution STFT loss
            - Requirement 15.3: Mel configuration consistency
        """
        # Validate input shapes
        assert wav_real.dim() == 3, f"Expected 3D tensor [B, 1, T], got {wav_real.dim()}D"
        assert wav_fake.dim() == 3, f"Expected 3D tensor [B, 1, T], got {wav_fake.dim()}D"
        assert wav_real.size(1) == 1, f"Expected mono audio [B, 1, T], got {wav_real.size(1)} channels"
        assert wav_fake.size(1) == 1, f"Expected mono audio [B, 1, T], got {wav_fake.size(1)} channels"
        
        # Remove channel dimension for mel extraction
        wav_real_mono = wav_real.squeeze(1)  # [B, T_wav]
        wav_fake_mono = wav_fake.squeeze(1)  # [B, T_wav]
        
        # Move mel_transform to the same device as input
        if self.mel_transform.mel_scale.fb.device != wav_real.device:
            self.mel_transform = self.mel_transform.to(wav_real.device)
        
        # Extract mel spectrogram from real waveform
        mel_real = self.mel_transform(wav_real_mono)  # [B, n_mels, T_mel]
        
        # Extract mel spectrogram from fake waveform
        mel_fake = self.mel_transform(wav_fake_mono)  # [B, n_mels, T_mel]
        
        # Validate mel shapes
        expected_T_mel = wav_real.size(2) // self.mel_config['hop_length'] + 1
        assert mel_real.size(1) == self.mel_config['n_mels'], \
            f"Expected {self.mel_config['n_mels']} mel bins, got {mel_real.size(1)}"
        
        # Convert to log scale using the same method as audio_processing.py
        epsilon = 1e-10
        
        if self.log_base == 10.0 or self.log_base == "10":
            # Log10 scale
            log_mel_real = torch.log10(mel_real + epsilon)
            log_mel_fake = torch.log10(mel_fake + epsilon)
        elif self.log_base == "e" or self.log_base == 2.718281828459045:
            # Natural log
            log_mel_real = torch.log(mel_real + epsilon)
            log_mel_fake = torch.log(mel_fake + epsilon)
        else:
            # Custom base: log_b(x) = log(x) / log(b)
            log_mel_real = torch.log(mel_real + epsilon) / torch.log(torch.tensor(self.log_base))
            log_mel_fake = torch.log(mel_fake + epsilon) / torch.log(torch.tensor(self.log_base))
        
        # Compute L1 loss between log-mel spectrograms
        mel_loss = F.l1_loss(log_mel_fake, log_mel_real)
        
        return mel_loss
    
    def forward_discriminator(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute discriminator loss.
        
        This method should be called when training the discriminator.
        The discriminator is trained to distinguish real from fake audio.
        
        Args:
            disc_real_outputs (List[torch.Tensor]): Discriminator outputs for real audio
            disc_fake_outputs (List[torch.Tensor]): Discriminator outputs for fake audio
        
        Returns:
            loss (torch.FloatTensor): Total discriminator loss
            loss_dict (dict): Dictionary containing loss components:
                - 'disc_loss': Total discriminator loss
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> disc_real = [torch.randn(2, 1, 100) for _ in range(8)]
            >>> disc_fake = [torch.randn(2, 1, 100) for _ in range(8)]
            >>> loss, loss_dict = loss_fn.forward_discriminator(disc_real, disc_fake)
            >>> loss.backward()
        """
        disc_loss = self.compute_discriminator_loss(disc_real_outputs, disc_fake_outputs)
        
        loss_dict = {
            'disc_loss': disc_loss.item()
        }
        
        return disc_loss, loss_dict
    
    def forward_generator(
        self,
        wav_real: torch.Tensor,
        wav_fake: torch.Tensor,
        disc_fake_outputs: List[torch.Tensor],
        real_feature_maps: List[List[torch.Tensor]],
        fake_feature_maps: List[List[torch.Tensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute generator loss.
        
        This method should be called when training the generator.
        The generator is trained with adversarial loss, feature matching loss,
        mel reconstruction loss, and optionally multi-resolution STFT loss.
        
        L_gen = L_adv + λ_fm * L_fm + λ_mel * L_mel + λ_stft * (L_sc + L_mag)
        
        Args:
            wav_real (torch.Tensor): Real waveform [B, 1, T]
            wav_fake (torch.Tensor): Generated waveform [B, 1, T]
            disc_fake_outputs (List[torch.Tensor]): Discriminator outputs for fake audio
            real_feature_maps (List[List[torch.Tensor]]): Feature maps from real audio
            fake_feature_maps (List[List[torch.Tensor]]): Feature maps from fake audio
        
        Returns:
            loss (torch.FloatTensor): Total generator loss
            loss_dict (dict): Dictionary containing loss components:
                - 'gen_loss': Total generator loss
                - 'gen_adv_loss': Adversarial loss
                - 'gen_fm_loss': Feature matching loss
                - 'gen_fm_loss_disc_N': Per-discriminator FM loss (N = 0, 1, 2, ...)
                - 'gen_mel_loss': Mel reconstruction loss (if use_mel_loss=True)
                - 'gen_sc_loss': Spectral convergence loss (if using STFT loss)
                - 'gen_mag_loss': Log magnitude loss (if using STFT loss)
        
        Example:
            >>> loss_fn = VocoderLoss()
            >>> wav_real = torch.randn(2, 1, 22050)
            >>> wav_fake = torch.randn(2, 1, 22050)
            >>> disc_fake = [torch.randn(2, 1, 100) for _ in range(8)]
            >>> real_fmaps = [[torch.randn(2, 128, 100) for _ in range(5)] for _ in range(8)]
            >>> fake_fmaps = [[torch.randn(2, 128, 100) for _ in range(5)] for _ in range(8)]
            >>> loss, loss_dict = loss_fn.forward_generator(
            ...     wav_real, wav_fake, disc_fake, real_fmaps, fake_fmaps
            ... )
            >>> loss.backward()
        """
        # Adversarial loss
        adv_loss = self.compute_generator_adversarial_loss(disc_fake_outputs)
        
        # Feature matching loss with per-discriminator logging
        fm_loss, per_disc_fm_losses = self.compute_feature_matching_loss(
            real_feature_maps, fake_feature_maps, return_per_discriminator=True
        )
        
        # Mel reconstruction loss
        if self.use_mel_loss:
            mel_loss = self.mel_reconstruction_loss(wav_real, wav_fake)
        else:
            mel_loss = torch.tensor(0.0, device=wav_real.device)
        
        # Multi-resolution STFT loss
        sc_loss, mag_loss = self.compute_stft_loss(wav_real, wav_fake)
        stft_loss = sc_loss + mag_loss
        
        # Total generator loss
        gen_loss = (
            adv_loss +
            self.feature_matching_weight * fm_loss +
            self.mel_weight * mel_loss +
            self.stft_loss_weight * stft_loss
        )
        
        # Prepare loss dictionary
        loss_dict = {
            'gen_loss': gen_loss.item(),
            'gen_adv_loss': adv_loss.item(),
            'gen_fm_loss': fm_loss.item(),
            'gen_mel_loss': mel_loss.item() if self.use_mel_loss else 0.0,
            'gen_sc_loss': sc_loss.item(),
            'gen_mag_loss': mag_loss.item(),
            'gen_stft_loss': stft_loss.item()
        }
        
        # Add per-discriminator feature matching losses for detailed logging
        if per_disc_fm_losses is not None:
            for i, disc_fm_loss in enumerate(per_disc_fm_losses):
                loss_dict[f'gen_fm_loss_disc_{i}'] = disc_fm_loss
        
        return gen_loss, loss_dict
