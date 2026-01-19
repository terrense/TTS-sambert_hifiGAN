"""
Tests for HiFi-GAN Multi-Scale Discriminator.

Tests verify:
1. ScaleDiscriminator output shapes
2. MultiScaleDiscriminator output shapes at different scales
3. Feature maps are returned correctly
4. Downsampling works as expected
"""

import torch
import pytest
from models.hifigan import ScaleDiscriminator, MultiScaleDiscriminator


def test_scale_discriminator_shapes():
    """Test single scale discriminator output shapes."""
    model = ScaleDiscriminator(use_spectral_norm=False, debug_shapes=False)
    
    # Test with different input lengths
    batch_size = 2
    for seq_len in [1000, 2000, 4000]:
        x = torch.randn(batch_size, 1, seq_len)
        
        output, feature_maps = model(x)
        
        # Check output shape
        assert output.dim() == 3, f"Expected 3D output, got {output.dim()}D"
        assert output.size(0) == batch_size, f"Expected batch size {batch_size}, got {output.size(0)}"
        assert output.size(1) == 1, f"Expected 1 channel output, got {output.size(1)}"
        
        # Check feature maps
        assert len(feature_maps) == 8, f"Expected 8 feature maps (7 conv + 1 post), got {len(feature_maps)}"
        
        # All feature maps should have batch dimension
        for i, fm in enumerate(feature_maps):
            assert fm.size(0) == batch_size, f"Feature map {i} has wrong batch size"


def test_multi_scale_discriminator_shapes():
    """Test multi-scale discriminator output shapes."""
    model = MultiScaleDiscriminator(use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    seq_len = 8192  # Typical waveform length
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Should have 3 discriminators (3 scales)
    assert len(outputs) == 3, f"Expected 3 discriminator outputs, got {len(outputs)}"
    assert len(feature_maps_list) == 3, f"Expected 3 feature map lists, got {len(feature_maps_list)}"
    
    # Check each scale
    for i, (output, feature_maps) in enumerate(zip(outputs, feature_maps_list)):
        # Check output shape
        assert output.dim() == 3, f"Scale {i}: Expected 3D output, got {output.dim()}D"
        assert output.size(0) == batch_size, f"Scale {i}: Wrong batch size"
        assert output.size(1) == 1, f"Scale {i}: Expected 1 channel output"
        
        # Check feature maps
        assert len(feature_maps) == 8, f"Scale {i}: Expected 8 feature maps, got {len(feature_maps)}"
        
        # All feature maps should have correct batch size
        for j, fm in enumerate(feature_maps):
            assert fm.size(0) == batch_size, f"Scale {i}, feature map {j}: Wrong batch size"


def test_multi_scale_discriminator_downsampling():
    """Test that different scales process different resolutions."""
    model = MultiScaleDiscriminator(use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 1
    seq_len = 8192
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # The temporal dimension should decrease with each scale due to downsampling
    # Scale 0: original (1x)
    # Scale 1: 2x downsampled
    # Scale 2: 4x downsampled
    
    # Get the first feature map from each scale (after first conv)
    first_feature_shapes = [fm_list[0].size(2) for fm_list in feature_maps_list]
    
    # Due to downsampling, scale 1 should have ~half the length of scale 0
    # and scale 2 should have ~quarter the length of scale 0
    # (approximately, due to padding effects)
    
    print(f"First feature map temporal dimensions: {first_feature_shapes}")
    
    # Scale 1 should be smaller than scale 0
    assert first_feature_shapes[1] < first_feature_shapes[0], \
        f"Scale 1 ({first_feature_shapes[1]}) should be smaller than scale 0 ({first_feature_shapes[0]})"
    
    # Scale 2 should be smaller than scale 1
    assert first_feature_shapes[2] < first_feature_shapes[1], \
        f"Scale 2 ({first_feature_shapes[2]}) should be smaller than scale 1 ({first_feature_shapes[1]})"


def test_multi_scale_discriminator_with_spectral_norm():
    """Test multi-scale discriminator with spectral normalization."""
    model = MultiScaleDiscriminator(use_spectral_norm=True, debug_shapes=False)
    
    batch_size = 2
    seq_len = 4096
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Should work the same way with spectral norm
    assert len(outputs) == 3
    assert len(feature_maps_list) == 3
    
    for output in outputs:
        assert output.size(0) == batch_size
        assert output.size(1) == 1


def test_multi_scale_discriminator_gradient_flow():
    """Test that gradients flow through the discriminator."""
    model = MultiScaleDiscriminator(use_spectral_norm=False, debug_shapes=False)
    
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


def test_multi_scale_discriminator_feature_maps_structure():
    """Test the structure of returned feature maps."""
    model = MultiScaleDiscriminator(use_spectral_norm=False, debug_shapes=False)
    
    batch_size = 2
    seq_len = 4096
    
    x = torch.randn(batch_size, 1, seq_len)
    
    outputs, feature_maps_list = model(x)
    
    # Each scale should have the same number of feature maps
    num_feature_maps = [len(fm_list) for fm_list in feature_maps_list]
    assert all(n == num_feature_maps[0] for n in num_feature_maps), \
        f"All scales should have same number of feature maps, got {num_feature_maps}"
    
    # Feature maps should have increasing channels (generally)
    for scale_idx, fm_list in enumerate(feature_maps_list):
        channels = [fm.size(1) for fm in fm_list]
        print(f"Scale {scale_idx} feature map channels: {channels}")
        
        # First few layers should increase in channels
        assert channels[1] >= channels[0] or channels[0] == 128, \
            f"Scale {scale_idx}: Channels should generally increase"


if __name__ == "__main__":
    # Run tests with verbose output
    test_scale_discriminator_shapes()
    print("✓ test_scale_discriminator_shapes passed")
    
    test_multi_scale_discriminator_shapes()
    print("✓ test_multi_scale_discriminator_shapes passed")
    
    test_multi_scale_discriminator_downsampling()
    print("✓ test_multi_scale_discriminator_downsampling passed")
    
    test_multi_scale_discriminator_with_spectral_norm()
    print("✓ test_multi_scale_discriminator_with_spectral_norm passed")
    
    test_multi_scale_discriminator_gradient_flow()
    print("✓ test_multi_scale_discriminator_gradient_flow passed")
    
    test_multi_scale_discriminator_feature_maps_structure()
    print("✓ test_multi_scale_discriminator_feature_maps_structure passed")
    
    print("\nAll tests passed!")
