"""
Tests for audio processing utilities.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.audio_processing import extract_mel, load_config


def test_extract_mel_basic():
    """Test basic mel extraction with synthetic waveform."""
    # Create synthetic waveform (1 second at 22050 Hz)
    sample_rate = 22050
    duration = 1.0
    waveform = torch.randn(1, int(sample_rate * duration))
    
    # Extract mel
    mel = extract_mel(waveform, sample_rate)
    
    # Verify shape
    assert mel.dim() == 2, f"Expected 2D tensor, got {mel.dim()}D"
    assert mel.size(0) == 80, f"Expected 80 mel bins, got {mel.size(0)}"
    
    # Verify it's log-mel (should be negative or small positive)
    assert mel.max() <= 10, f"Log-mel values too large: {mel.max()}"
    
    print(f"✓ Basic mel extraction test passed")
    print(f"  Mel shape: {mel.shape}")
    print(f"  Mel range: [{mel.min().item():.2f}, {mel.max().item():.2f}]")


def test_extract_mel_with_config():
    """Test mel extraction with config loading."""
    # Create synthetic waveform
    sample_rate = 22050
    waveform = torch.randn(1, 22050)
    
    # Load config
    config = load_config("configs/config.yaml")
    
    # Extract mel with config
    mel = extract_mel(waveform, sample_rate, config=config)
    
    # Verify shape matches config
    assert mel.size(0) == config['audio']['n_mels']
    
    print(f"✓ Config-based mel extraction test passed")
    print(f"  Mel shape: {mel.shape}")


def test_extract_mel_mono_input():
    """Test mel extraction with mono (1D) waveform input."""
    # Create 1D waveform
    waveform = torch.randn(22050)
    
    # Extract mel
    mel = extract_mel(waveform, sample_rate=22050)
    
    # Verify shape
    assert mel.dim() == 2
    assert mel.size(0) == 80
    
    print(f"✓ Mono input test passed")
    print(f"  Mel shape: {mel.shape}")


def test_extract_mel_shape_logging():
    """Test that shape logging works when enabled."""
    # Create synthetic waveform
    waveform = torch.randn(1, 22050)
    
    # Load config and enable shape logging
    config = load_config("configs/config.yaml")
    config['debug']['print_shapes'] = True
    
    print("\n--- Testing shape logging (should see debug output below) ---")
    mel = extract_mel(waveform, sample_rate=22050, config=config)
    print("--- End of shape logging test ---\n")
    
    assert mel.dim() == 2
    print(f"✓ Shape logging test passed")


def test_extract_mel_different_sample_rates():
    """Test mel extraction with resampling."""
    # Create waveform at 16kHz
    original_sr = 16000
    waveform = torch.randn(1, 16000)
    
    # Extract mel (should resample to 22050 Hz)
    mel = extract_mel(waveform, sample_rate=original_sr)
    
    # Verify shape
    assert mel.dim() == 2
    assert mel.size(0) == 80
    
    print(f"✓ Resampling test passed")
    print(f"  Mel shape: {mel.shape}")


if __name__ == "__main__":
    print("Running audio processing tests...\n")
    
    test_extract_mel_basic()
    test_extract_mel_with_config()
    test_extract_mel_mono_input()
    test_extract_mel_shape_logging()
    test_extract_mel_different_sample_rates()
    
    print("\n✓ All tests passed!")
