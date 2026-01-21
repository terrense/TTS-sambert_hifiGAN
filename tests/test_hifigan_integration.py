"""
Test for complete HiFiGAN model integration.

Tests the HiFiGAN class that integrates Generator, MSD, and MPD.
"""

import pytest
import torch
from models.hifigan import HiFiGAN


def test_hifigan_initialization():
    """Test HiFiGAN model initialization."""
    model = HiFiGAN(
        n_mels=80,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        debug_shapes=False
    )
    
    assert model.generator is not None
    assert model.msd is not None
    assert model.mpd is not None


def test_hifigan_forward_generation():
    """Test HiFiGAN forward pass for generation."""
    model = HiFiGAN(
        n_mels=80,
        upsample_rates=[8, 8, 2, 2],
        debug_shapes=False
    )
    model.eval()
    
    # Create random mel-spectrogram
    batch_size = 2
    n_mels = 80
    mel_frames = 100
    mel = torch.randn(batch_size, n_mels, mel_frames)
    
    # Generate waveform
    with torch.no_grad():
        wav = model(mel)
    
    # Verify output shape
    hop_length = 8 * 8 * 2 * 2  # Product of upsample_rates = 256
    expected_wav_length = mel_frames * hop_length
    
    assert wav.shape == (batch_size, 1, expected_wav_length)
    assert wav.dtype == torch.float32
    
    # Verify output is in valid range (tanh output)
    assert wav.min() >= -1.0
    assert wav.max() <= 1.0


def test_hifigan_generate_alias():
    """Test that generate() method works as alias for forward()."""
    model = HiFiGAN(n_mels=80, debug_shapes=False)
    model.eval()
    
    mel = torch.randn(1, 80, 50)
    
    with torch.no_grad():
        wav1 = model.forward(mel)
        wav2 = model.generate(mel)
    
    # Both methods should produce identical results
    assert torch.allclose(wav1, wav2)


def test_hifigan_discriminate():
    """Test HiFiGAN discriminate method."""
    model = HiFiGAN(
        n_mels=80,
        upsample_rates=[8, 8, 2, 2],
        debug_shapes=False
    )
    model.eval()
    
    # Create random waveforms
    batch_size = 2
    wav_length = 25600  # 256 * 100
    wav_real = torch.randn(batch_size, 1, wav_length)
    wav_fake = torch.randn(batch_size, 1, wav_length)
    
    # Discriminate
    with torch.no_grad():
        (msd_real_out, msd_real_feat,
         msd_fake_out, msd_fake_feat,
         mpd_real_out, mpd_real_feat,
         mpd_fake_out, mpd_fake_feat) = model.discriminate(wav_real, wav_fake)
    
    # Verify MSD outputs (3 discriminators)
    assert len(msd_real_out) == 3
    assert len(msd_fake_out) == 3
    assert len(msd_real_feat) == 3
    assert len(msd_fake_feat) == 3
    
    # Verify MPD outputs (5 discriminators)
    assert len(mpd_real_out) == 5
    assert len(mpd_fake_out) == 5
    assert len(mpd_real_feat) == 5
    assert len(mpd_fake_feat) == 5
    
    # Verify each discriminator produces outputs
    for i in range(3):
        assert msd_real_out[i].shape[0] == batch_size
        assert msd_fake_out[i].shape[0] == batch_size
        assert len(msd_real_feat[i]) > 0  # Has feature maps
        assert len(msd_fake_feat[i]) > 0
    
    for i in range(5):
        assert mpd_real_out[i].shape[0] == batch_size
        assert mpd_fake_out[i].shape[0] == batch_size
        assert len(mpd_real_feat[i]) > 0  # Has feature maps
        assert len(mpd_fake_feat[i]) > 0


def test_hifigan_end_to_end():
    """Test end-to-end generation and discrimination."""
    model = HiFiGAN(
        n_mels=80,
        upsample_rates=[8, 8, 2, 2],
        debug_shapes=False
    )
    model.eval()
    
    # Generate fake waveform from mel
    batch_size = 1
    mel = torch.randn(batch_size, 80, 100)
    
    with torch.no_grad():
        wav_fake = model(mel)
    
    # Create real waveform
    wav_real = torch.randn_like(wav_fake)
    
    # Discriminate both
    with torch.no_grad():
        results = model.discriminate(wav_real, wav_fake)
    
    # Verify we got all 8 outputs
    assert len(results) == 8


def test_hifigan_with_different_configs():
    """Test HiFiGAN with different configurations."""
    # Test with custom upsample rates
    model1 = HiFiGAN(
        n_mels=80,
        upsample_rates=[5, 5, 4, 2],
        upsample_kernel_sizes=[10, 10, 8, 4],
        debug_shapes=False
    )
    
    mel = torch.randn(1, 80, 50)
    with torch.no_grad():
        wav = model1(mel)
    
    # Verify basic shape properties (exact length may vary due to padding)
    assert wav.shape[0] == 1  # batch size
    assert wav.shape[1] == 1  # channels
    assert wav.shape[2] > 0   # has samples
    
    # Test with custom MPD periods
    model2 = HiFiGAN(
        n_mels=80,
        mpd_periods=[2, 3, 5],
        debug_shapes=False
    )
    
    wav_real = torch.randn(1, 1, 10000)
    wav_fake = torch.randn(1, 1, 10000)
    
    with torch.no_grad():
        results = model2.discriminate(wav_real, wav_fake)
    
    # Should have 3 MPD discriminators instead of 5
    assert len(results[4]) == 3  # mpd_real_outputs
    assert len(results[6]) == 3  # mpd_fake_outputs


def test_hifigan_shape_logging():
    """Test that shape logging works when enabled."""
    import io
    import sys
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    model = HiFiGAN(
        n_mels=80,
        upsample_rates=[8, 8, 2, 2],
        debug_shapes=True
    )
    model.eval()
    
    mel = torch.randn(1, 80, 10)
    
    with torch.no_grad():
        wav = model(mel)
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Check that shapes were logged
    output = captured_output.getvalue()
    assert "[HiFiGAN]" in output
    assert "shape" in output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
