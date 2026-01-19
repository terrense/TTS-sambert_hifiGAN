"""
Tests for HiFi-GAN Multi-Period Discriminator.

Tests verify:
1. PeriodDiscriminator output shapes with different periods
2. MultiPeriodDiscriminator output shapes for all periods
3. Feature maps are returned correctly
4. Waveform reshaping based on period works correctly
5. Padding for non-divisible lengths
"""

import torch
import pytest
from models.hifigan import PeriodDiscriminator, MultiPeriodDiscriminator


def test_period_discriminator_shapes():
    """Test single period discriminator output shapes."""
    periods = [2, 3, 5, 7, 11]
    batch_size = 2
    seq_len = 8192
    
    for period in periods:
        model = PeriodDiscriminator(period, use_spectral_norm=False, debug_shapes=False)
        x = torch.randn(batch_size, 1, seq_len)
        
        output, feature_maps = model(x)
        
        # Check output shape (should be 4D after 2D convolutions)
        assert output.dim() == 4, f"Period {period}: Expected 4D output, got {output.dim()}D"
        assert output.size(0) == batch_size, f"Period {period}: Expected batch size {batch_size}, got {output.size(0)}"
        assert output.size(1) == 1, f"Period {period}: Expected 1 channel output, got {output.size(1)}"
        
        # Check feature maps
        assert len(feature_maps) == 6, f"Period {period}: Expected 6 feature maps (5 conv + 1 post), got {len(feature_maps)}"
        
        # All feature maps should have batch dimension
        for i, fm in enumerate(feature_maps):
            assert fm.size(0) == batch_size, f"Period {period}, feature map {i}: Wrong batch size"
            assert fm.dim() == 4, f"Period {period}, feature map {i}: Expected 4D tensor"


def test_period_discriminator_padding():
    """Test that period discriminator handles non-divisible lengths correctly."""
    period = 5
    model = PeriodDiscriminator(period, use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    
    # Test with lengths that are and aren't divisible by period
    for seq_len in [1000, 1001, 1002, 1003, 1004]:
        x = torch.randn(batch_size, 1, seq_len)
        
        output, feature_maps = model(x)
        
        # Should work regardless of whether length is divisible by period
        assert output.dim() == 4, f"Length {seq_len}: Expected 4D output"
        assert output.size(0) == batch_size, f"Length {seq_len}: Wrong batch size"


def test_period_discriminator_reshape():
    """Test that waveform is correctly reshaped based on period."""
    period = 3
    model = PeriodDiscriminator(period, use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 1
    seq_len = 300  # Divisible by 3
    
    x = torch.randn(batch_size, 1, seq_len)
    
    output, feature_maps = model(x)
    
    # After reshape, the first feature map should have the period as one dimension
    # The reshape happens before the first conv, so we can't directly check it,
    # but we can verify the output is valid
    assert output.dim() == 4
    assert output.size(0) == batch_size


def test_multi_period_discriminator_shapes():
    """Test multi-period discriminator output shapes."""
    model = MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11], use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    seq_len = 8192  # Typical waveform length
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Should have 5 discriminators (5 periods)
    assert len(outputs) == 5, f"Expected 5 discriminator outputs, got {len(outputs)}"
    assert len(feature_maps_list) == 5, f"Expected 5 feature map lists, got {len(feature_maps_list)}"
    
    # Check each period
    for i, (output, feature_maps) in enumerate(zip(outputs, feature_maps_list)):
        # Check output shape
        assert output.dim() == 4, f"Period {i}: Expected 4D output, got {output.dim()}D"
        assert output.size(0) == batch_size, f"Period {i}: Wrong batch size"
        assert output.size(1) == 1, f"Period {i}: Expected 1 channel output"
        
        # Check feature maps
        assert len(feature_maps) == 6, f"Period {i}: Expected 6 feature maps, got {len(feature_maps)}"
        
        # All feature maps should have correct batch size
        for j, fm in enumerate(feature_maps):
            assert fm.size(0) == batch_size, f"Period {i}, feature map {j}: Wrong batch size"
            assert fm.dim() == 4, f"Period {i}, feature map {j}: Expected 4D tensor"


def test_multi_period_discriminator_different_periods():
    """Test that different periods produce different shaped outputs."""
    model = MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11], use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 1
    seq_len = 8192
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Get the shapes of first feature maps from each period
    first_feature_shapes = [fm_list[0].shape for fm_list in feature_maps_list]
    
    print(f"First feature map shapes for each period: {first_feature_shapes}")
    
    # Different periods should produce different shaped feature maps
    # The height dimension (dim 2) should vary based on period
    heights = [shape[2] for shape in first_feature_shapes]
    
    # Not all heights should be the same (different periods create different reshapes)
    # Note: Due to convolution and stride, the exact relationship is complex,
    # but we can verify they're not all identical
    assert len(set(heights)) > 1, f"Different periods should produce different feature map heights, got {heights}"


def test_multi_period_discriminator_with_spectral_norm():
    """Test multi-period discriminator with spectral normalization."""
    model = MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11], use_spectral_norm=True, debug_shapes=False)
    
    batch_size = 2
    seq_len = 4096
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Should work the same way with spectral norm
    assert len(outputs) == 5
    assert len(feature_maps_list) == 5
    
    for output in outputs:
        assert output.size(0) == batch_size
        assert output.size(1) == 1
        assert output.dim() == 4


def test_multi_period_discriminator_gradient_flow():
    """Test that gradients flow through the discriminator."""
    model = MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11], use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    seq_len = 4096
    
    x = torch.randn(batch_size, 1, seq_len, requires_grad=True)
    
    outputs, feature_maps_list = model(x)
    
    # Compute a simple loss
    loss = sum(output.mean() for output in outputs)
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_multi_period_discriminator_feature_maps_structure():
    """Test the structure of returned feature maps."""
    model = MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11], use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    seq_len = 4096
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Each period should have the same number of feature maps
    num_feature_maps = [len(fm_list) for fm_list in feature_maps_list]
    assert all(n == num_feature_maps[0] for n in num_feature_maps), \
        f"All periods should have same number of feature maps, got {num_feature_maps}"
    
    # Feature maps should have increasing channels
    for period_idx, fm_list in enumerate(feature_maps_list):
        channels = [fm.size(1) for fm in fm_list]
        print(f"Period {period_idx} feature map channels: {channels}")
        
        # Channels should increase: 32 -> 128 -> 512 -> 1024 -> 1024 -> 1
        expected_channels = [32, 128, 512, 1024, 1024, 1]
        assert channels == expected_channels, \
            f"Period {period_idx}: Expected channels {expected_channels}, got {channels}"


def test_multi_period_discriminator_custom_periods():
    """Test multi-period discriminator with custom periods."""
    custom_periods = [4, 8, 16]
    model = MultiPeriodDiscriminator(periods=custom_periods, use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    seq_len = 8192
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Should have 3 discriminators (3 custom periods)
    assert len(outputs) == 3, f"Expected 3 discriminator outputs, got {len(outputs)}"
    assert len(feature_maps_list) == 3, f"Expected 3 feature map lists, got {len(feature_maps_list)}"
    
    for output in outputs:
        assert output.size(0) == batch_size
        assert output.size(1) == 1
        assert output.dim() == 4


def test_period_discriminator_different_input_lengths():
    """Test period discriminator with various input lengths."""
    period = 5
    model = PeriodDiscriminator(period, use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    
    # Test with different lengths
    for seq_len in [1000, 2000, 4000, 8000, 16000]:
        x = torch.randn(batch_size, 1, seq_len)
        
        output, feature_maps = model(x)
        
        # Should work for all lengths
        assert output.dim() == 4, f"Length {seq_len}: Expected 4D output"
        assert output.size(0) == batch_size, f"Length {seq_len}: Wrong batch size"
        assert len(feature_maps) == 6, f"Length {seq_len}: Expected 6 feature maps"


if __name__ == "__main__":
    # Run tests with verbose output
    test_period_discriminator_shapes()
    print("✓ test_period_discriminator_shapes passed")
    
    test_period_discriminator_padding()
    print("✓ test_period_discriminator_padding passed")
    
    test_period_discriminator_reshape()
    print("✓ test_period_discriminator_reshape passed")
    
    test_multi_period_discriminator_shapes()
    print("✓ test_multi_period_discriminator_shapes passed")
    
    test_multi_period_discriminator_different_periods()
    print("✓ test_multi_period_discriminator_different_periods passed")
    
    test_multi_period_discriminator_with_spectral_norm()
    print("✓ test_multi_period_discriminator_with_spectral_norm passed")
    
    test_multi_period_discriminator_gradient_flow()
    print("✓ test_multi_period_discriminator_gradient_flow passed")
    
    test_multi_period_discriminator_feature_maps_structure()
    print("✓ test_multi_period_discriminator_feature_maps_structure passed")
    
    test_multi_period_discriminator_custom_periods()
    print("✓ test_multi_period_discriminator_custom_periods passed")
    
    test_period_discriminator_different_input_lengths()
    print("✓ test_period_discriminator_different_input_lengths passed")
    
    print("\nAll tests passed!")
