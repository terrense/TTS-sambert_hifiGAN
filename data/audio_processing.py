"""
Audio processing utilities for TTS system.

This module provides functions for extracting mel-spectrograms from audio waveforms
using torchaudio, with parameters loaded from configuration files.
"""

import os
import torch
import torchaudio
import yaml
from pathlib import Path
from typing import Union, Tuple, Optional


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def extract_mel(
    waveform: torch.Tensor,
    sample_rate: int = None,
    config: Optional[dict] = None,
    config_path: str = "configs/config.yaml"
) -> torch.Tensor:
    """
    Extract log-mel spectrogram from waveform using torchaudio.
    
    Args:
        waveform: Input waveform tensor [channels, time] or [time]
        sample_rate: Sample rate of the waveform (optional if config provided)
        config: Configuration dictionary (optional, will load from file if not provided)
        config_path: Path to config file if config not provided
        
    Returns:
        Log-mel spectrogram tensor [n_mels, T]
        
    Shape Contract:
        Input: waveform [channels, time] or [time]
        Output: mel [n_mels, T] where T = time // hop_length + 1
    """
    # Load config if not provided
    if config is None:
        config = load_config(config_path)
    
    audio_config = config['audio']
    debug_config = config.get('debug', {})
    print_shapes = debug_config.get('print_shapes', False)
    
    # Extract mel parameters from config
    target_sr = audio_config['sample_rate']
    n_fft = audio_config['n_fft']
    hop_length = audio_config['hop_length']
    win_length = audio_config['win_length']
    n_mels = audio_config['n_mels']
    fmin = audio_config['fmin']
    fmax = audio_config['fmax']
    mel_scale = audio_config.get('mel_scale', 'slaney')
    norm = audio_config.get('norm', 'slaney')
    log_base = audio_config.get('log_base', 10.0)
    
    # Handle waveform shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [time] -> [1, time]
    
    if print_shapes:
        print(f"[extract_mel] Input waveform shape: {waveform.shape}")
    
    # Resample if necessary
    if sample_rate is not None and sample_rate != target_sr:
        if print_shapes:
            print(f"[extract_mel] Resampling from {sample_rate}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
        if print_shapes:
            print(f"[extract_mel] Resampled waveform shape: {waveform.shape}")
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if print_shapes:
            print(f"[extract_mel] Converted to mono, shape: {waveform.shape}")
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        mel_scale=mel_scale,
        norm=norm,
        power=2.0  # Power spectrogram
    )
    
    # Extract mel spectrogram
    mel_spec = mel_transform(waveform)  # [channels, n_mels, T]
    
    # Remove channel dimension (we have mono)
    mel_spec = mel_spec.squeeze(0)  # [n_mels, T]
    
    if print_shapes:
        print(f"[extract_mel] Mel spectrogram shape (before log): {mel_spec.shape}")
    
    # Convert to log scale
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    
    if log_base == 10.0 or log_base == "10":
        # Log10 scale
        log_mel = torch.log10(mel_spec + epsilon)
    elif log_base == "e" or log_base == 2.718281828459045:
        # Natural log
        log_mel = torch.log(mel_spec + epsilon)
    else:
        # Custom base: log_b(x) = log(x) / log(b)
        log_mel = torch.log(mel_spec + epsilon) / torch.log(torch.tensor(log_base))
    
    if print_shapes:
        print(f"[extract_mel] Log-mel spectrogram shape: {log_mel.shape}")
        print(f"[extract_mel] Log-mel range: [{log_mel.min().item():.2f}, {log_mel.max().item():.2f}]")
    
    return log_mel


def extract_mel_from_file(
    audio_path: Union[str, Path],
    config: Optional[dict] = None,
    config_path: str = "configs/config.yaml"
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and extract log-mel spectrogram.
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary (optional)
        config_path: Path to config file if config not provided
        
    Returns:
        Tuple of (log-mel spectrogram [n_mels, T], sample_rate)
    """
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Extract mel spectrogram
    log_mel = extract_mel(waveform, sample_rate, config, config_path)
    
    return log_mel, sample_rate


def save_mel(
    mel: torch.Tensor,
    output_path: Union[str, Path]
) -> None:
    """
    Save mel spectrogram to file.
    
    Args:
        mel: Mel spectrogram tensor [n_mels, T]
        output_path: Path to save the mel spectrogram
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    import numpy as np
    mel_np = mel.cpu().numpy()
    np.save(output_path, mel_np)


def load_mel(mel_path: Union[str, Path]) -> torch.Tensor:
    """
    Load mel spectrogram from file.
    
    Args:
        mel_path: Path to mel spectrogram file
        
    Returns:
        Mel spectrogram tensor [n_mels, T]
    """
    import numpy as np
    mel_np = np.load(mel_path)
    mel = torch.from_numpy(mel_np).float()
    return mel
