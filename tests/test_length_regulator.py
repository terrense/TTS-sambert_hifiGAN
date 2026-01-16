"""
Tests for Length Regulator module.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.variance_adaptor import LengthRegulator


def test_length_regulator_shape():
    """Test that LengthRegulator outputs correct shape."""
    # Create model
    model = LengthRegulator()
    
    # Test input
    batch_size = 2
    Tph = 10
    d_model = 256
    
    Henc = torch.randn(batch_size, Tph, d_model)
    # Duration: each phoneme repeated 3 times
    dur = torch.ones(batch_size, Tph, dtype=torch.long) * 3
    
    # Forward pass
    with torch.no_grad():
        Hlr = model(Henc, dur)
    
    # Expected frame length: Tph * 3 = 30
    expected_Tfrm = Tph * 3
    
    # Check output shape
    assert Hlr.shape == (batch_size, expected_Tfrm, d_model), \
        f"Expected shape ({batch_size}, {expected_Tfrm}, {d_model}), got {Hlr.shape}"
    
    print(f"✓ Length Regulator output shape correct: {Hlr.shape}")


def test_length_regulator_variable_duration():
    """Test that LengthRegulator handles variable durations correctly."""
    model = LengthRegulator()
    
    batch_size = 2
    Tph = 5
    d_model = 128
    
    Henc = torch.randn(batch_size, Tph, d_model)
    # Variable durations: [1, 2, 3, 4, 5]
    dur = torch.tensor([[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]], dtype=torch.long)
    
    with torch.no_grad():
        Hlr = model(Henc, dur)
    
    # First batch item: 1+2+3+4+5 = 15 frames
    # Second batch item: 2+2+2+2+2 = 10 frames
    # Output should be padded to max length (15)
    expected_Tfrm = 15
    
    assert Hlr.shape == (batch_size, expected_Tfrm, d_model), \
        f"Expected shape ({batch_size}, {expected_Tfrm}, {d_model}), got {Hlr.shape}"
    
    print(f"✓ Length Regulator handles variable durations correctly")


def test_length_regulator_repeat_logic():
    """Test that LengthRegulator repeats features correctly."""
    model = LengthRegulator()
    
    batch_size = 1
    Tph = 3
    d_model = 4
    
    # Create simple input with identifiable values
    Henc = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]]])
    
    # Duration: repeat first phoneme 2 times, second 3 times, third 1 time
    dur = torch.tensor([[2, 3, 1]], dtype=torch.long)
    
    with torch.no_grad():
        Hlr = model(Henc, dur)
    
    # Expected output: [1,2,3,4] repeated 2 times, [5,6,7,8] repeated 3 times, [9,10,11,12] repeated 1 time
    # Total frames: 2 + 3 + 1 = 6
    assert Hlr.shape == (1, 6, 4)
    
    # Check that features are repeated correctly
    # First 2 frames should be [1,2,3,4]
    assert torch.allclose(Hlr[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(Hlr[0, 1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    
    # Next 3 frames should be [5,6,7,8]
    assert torch.allclose(Hlr[0, 2], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(Hlr[0, 3], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(Hlr[0, 4], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    
    # Last frame should be [9,10,11,12]
    assert torch.allclose(Hlr[0, 5], torch.tensor([9.0, 10.0, 11.0, 12.0]))
    
    print(f"✓ Length Regulator repeat logic works correctly")


def test_length_regulator_zero_duration():
    """Test that LengthRegulator handles zero durations correctly."""
    model = LengthRegulator()
    
    batch_size = 1
    Tph = 4
    d_model = 64
    
    Henc = torch.randn(batch_size, Tph, d_model)
    # Some phonemes have zero duration (should be skipped)
    dur = torch.tensor([[2, 0, 3, 0]], dtype=torch.long)
    
    with torch.no_grad():
        Hlr = model(Henc, dur)
    
    # Expected frames: 2 + 0 + 3 + 0 = 5
    assert Hlr.shape[1] == 5, f"Expected 5 frames, got {Hlr.shape[1]}"
    
    print(f"✓ Length Regulator handles zero durations correctly")


def test_length_regulator_different_batch_sizes():
    """Test that LengthRegulator works with different batch sizes."""
    model = LengthRegulator()
    
    Tph = 8
    d_model = 256
    
    for batch_size in [1, 4, 8]:
        Henc = torch.randn(batch_size, Tph, d_model)
        dur = torch.randint(1, 5, (batch_size, Tph), dtype=torch.long)
        
        with torch.no_grad():
            Hlr = model(Henc, dur)
        
        # Check batch dimension
        assert Hlr.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
        
        # Check feature dimension
        assert Hlr.shape[2] == d_model, \
            f"Feature dimension mismatch for batch_size={batch_size}"
    
    print(f"✓ Length Regulator works with different batch sizes")


def test_length_regulator_gradient_flow():
    """Test that gradients flow through the model."""
    model = LengthRegulator()
    
    batch_size = 2
    Tph = 5
    d_model = 128
    
    # Test input with gradient tracking
    Henc = torch.randn(batch_size, Tph, d_model, requires_grad=True)
    dur = torch.tensor([[2, 2, 2, 2, 2], [3, 3, 3, 3, 3]], dtype=torch.long)
    
    # Forward pass
    Hlr = model(Henc, dur)
    
    # Compute dummy loss
    loss = Hlr.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert Henc.grad is not None, "Gradients should flow to input"
    assert torch.any(Henc.grad != 0), "Gradients should be non-zero"
    
    print(f"✓ Gradients flow correctly through Length Regulator")


if __name__ == "__main__":
    print("Testing Length Regulator...")
    print()
    
    test_length_regulator_shape()
    test_length_regulator_variable_duration()
    test_length_regulator_repeat_logic()
    test_length_regulator_zero_duration()
    test_length_regulator_different_batch_sizes()
    test_length_regulator_gradient_flow()
    
    print()
    print("All Length Regulator tests passed! ✓")
