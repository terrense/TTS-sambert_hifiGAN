"""
Tests for HiFi-GAN Generator
"""

import pytest
import torch
import yaml
from models.hifigan import HiFiGANGenerator


@pytest.fixture
def config():
    """Load configuration."""
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("configs/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    return config, model_config


def test_hifigan_generator_initialization(config):
    """Test HiFi-GAN Generator initialization."""
    audio_config, model_config = config
    vocoder_config = model_config["vocoder"]["generator"]
    
    generator = HiFiGANGenerator(
        n_mels=audio_config["audio"]["n_mels"],
        upsample_rates=vocoder_config["upsample_rates"],
        upsample_kernel_sizes=vocoder_config["upsample_kernel_sizes"],
        upsample_initial_channel=vocoder_config["upsample_initial_channel"],
        resblock_kernel_sizes=vocoder_config["resblock_kernel_sizes"],
        resblock_dilation_sizes=vocoder_config["resblock_dilation_sizes"]
    )
    
    assert generator is not None
    assert generator.n_mels == 80
    assert generator.num_upsamples == 4


def test_hifigan_generator_forward(config):
    """Test HiFi-GAN Generator forward pass."""
    audio_config, model_config = config
    vocoder_config = model_config["vocoder"]["generator"]
    
    generator = HiFiGANGenerator(
        n_mels=audio_config["audio"]["n_mels"],
        upsample_rates=vocoder_config["upsample_rates"],
        upsample_kernel_sizes=vocoder_config["upsample_kernel_sizes"],
        upsample_initial_channel=vocoder_config["upsample_initial_channel"],
        resblock_kernel_sizes=vocoder_config["resblock_kernel_sizes"],
        resblock_dilation_sizes=vocoder_config["resblock_dilation_sizes"],
        debug_shapes=True
    )
    
    # Test input
    batch_size = 2
    n_mels = audio_config["audio"]["n_mels"]
    Tfrm = 100
    hop_length = audio_config["audio"]["hop_length"]
    
    mel = torch.randn(batch_size, n_mels, Tfrm)
    
    # Forward pass
    wav = generator(mel)
    
    # Verify output shape
    expected_T_wav = Tfrm * hop_length
    assert wav.shape == (batch_size, 1, expected_T_wav), \
        f"Expected shape ({batch_size}, 1, {expected_T_wav}), got {wav.shape}"
    
    # Verify output range (tanh activation)
    assert wav.min() >= -1.0 and wav.max() <= 1.0, \
        f"Output should be in [-1, 1], got range [{wav.min()}, {wav.max()}]"


def test_hifigan_generator_different_lengths(config):
    """Test HiFi-GAN Generator with different input lengths."""
    audio_config, model_config = config
    vocoder_config = model_config["vocoder"]["generator"]
    
    generator = HiFiGANGenerator(
        n_mels=audio_config["audio"]["n_mels"],
        upsample_rates=vocoder_config["upsample_rates"],
        upsample_kernel_sizes=vocoder_config["upsample_kernel_sizes"],
        upsample_initial_channel=vocoder_config["upsample_initial_channel"],
        resblock_kernel_sizes=vocoder_config["resblock_kernel_sizes"],
        resblock_dilation_sizes=vocoder_config["resblock_dilation_sizes"]
    )
    
    hop_length = audio_config["audio"]["hop_length"]
    
    # Test with different lengths
    for Tfrm in [50, 100, 200]:
        mel = torch.randn(1, 80, Tfrm)
        wav = generator(mel)
        
        expected_T_wav = Tfrm * hop_length
        assert wav.shape == (1, 1, expected_T_wav), \
            f"For Tfrm={Tfrm}, expected T_wav={expected_T_wav}, got {wav.shape[2]}"


def test_hifigan_generator_batch_processing(config):
    """Test HiFi-GAN Generator with different batch sizes."""
    audio_config, model_config = config
    vocoder_config = model_config["vocoder"]["generator"]
    
    generator = HiFiGANGenerator(
        n_mels=audio_config["audio"]["n_mels"],
        upsample_rates=vocoder_config["upsample_rates"],
        upsample_kernel_sizes=vocoder_config["upsample_kernel_sizes"],
        upsample_initial_channel=vocoder_config["upsample_initial_channel"],
        resblock_kernel_sizes=vocoder_config["resblock_kernel_sizes"],
        resblock_dilation_sizes=vocoder_config["resblock_dilation_sizes"]
    )
    
    hop_length = audio_config["audio"]["hop_length"]
    Tfrm = 100
    
    # Test with different batch sizes
    for batch_size in [1, 4, 8]:
        mel = torch.randn(batch_size, 80, Tfrm)
        wav = generator(mel)
        
        expected_T_wav = Tfrm * hop_length
        assert wav.shape == (batch_size, 1, expected_T_wav), \
            f"For batch_size={batch_size}, expected shape ({batch_size}, 1, {expected_T_wav}), got {wav.shape}"


def test_hifigan_generator_upsampling_factor(config):
    """Test that upsampling rates multiply to hop_length."""
    audio_config, model_config = config
    vocoder_config = model_config["vocoder"]["generator"]
    
    upsample_rates = vocoder_config["upsample_rates"]
    hop_length = audio_config["audio"]["hop_length"]
    
    # Calculate product of upsample rates
    product = 1
    for rate in upsample_rates:
        product *= rate
    
    assert product == hop_length, \
        f"Product of upsample_rates ({product}) should equal hop_length ({hop_length})"


def test_hifigan_generator_gradient_flow(config):
    """Test that gradients flow through the generator."""
    audio_config, model_config = config
    vocoder_config = model_config["vocoder"]["generator"]
    
    generator = HiFiGANGenerator(
        n_mels=audio_config["audio"]["n_mels"],
        upsample_rates=vocoder_config["upsample_rates"],
        upsample_kernel_sizes=vocoder_config["upsample_kernel_sizes"],
        upsample_initial_channel=vocoder_config["upsample_initial_channel"],
        resblock_kernel_sizes=vocoder_config["resblock_kernel_sizes"],
        resblock_dilation_sizes=vocoder_config["resblock_dilation_sizes"]
    )
    
    mel = torch.randn(2, 80, 100, requires_grad=True)
    wav = generator(mel)
    
    # Compute loss and backward
    loss = wav.mean()
    loss.backward()
    
    # Check that gradients exist
    assert mel.grad is not None, "Gradients should flow back to input"
    assert not torch.isnan(mel.grad).any(), "Gradients should not contain NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
