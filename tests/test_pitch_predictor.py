"""
Tests for Pitch Predictor module.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.variance_adaptor import PitchPredictor


def test_pitch_predictor_shape():
    """Test that PitchPredictor outputs correct shapes."""
    # Configuration
    d_model = 256
    n_bins = 256
    pitch_min = 80.0
    pitch_max = 600.0
    
    # Create model
    model = PitchPredictor(
        d_model=d_model,
        n_bins=n_bins,
        pitch_min=pitch_min,
        pitch_max=pitch_max
    )
    model.eval()
    
    # Test input
    batch_size = 2
    Tph = 20
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.randint(1, 10, (batch_size, Tph))
    
    # Calculate expected frame length
    Tfrm = dur.sum(dim=1).max().item()
    
    # Forward pass (inference mode)
    with torch.no_grad():
        pitch_tok, pitch_frm, Ep = model(Henc, dur)
    
    # Check output shapes
    assert pitch_tok.shape == (batch_size, Tph), \
        f"Expected pitch_tok shape ({batch_size}, {Tph}), got {pitch_tok.shape}"
    
    assert pitch_frm.shape == (batch_size, Tfrm), \
        f"Expected pitch_frm shape ({batch_size}, {Tfrm}), got {pitch_frm.shape}"
    
    assert Ep.shape == (batch_size, Tfrm, d_model), \
        f"Expected Ep shape ({batch_size}, {Tfrm}, {d_model}), got {Ep.shape}"
    
    print(f"✓ Pitch Predictor output shapes correct:")
    print(f"  - pitch_tok: {pitch_tok.shape}")
    print(f"  - pitch_frm: {pitch_frm.shape}")
    print(f"  - Ep: {Ep.shape}")


def test_pitch_predictor_with_ground_truth():
    """Test that PitchPredictor uses ground truth pitch during training."""
    d_model = 256
    batch_size = 2
    Tph = 20
    
    # Create model
    model = PitchPredictor(d_model=d_model)
    model.eval()
    
    # Test input
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.randint(1, 10, (batch_size, Tph))
    
    # Calculate frame length
    Tfrm = dur.sum(dim=1).max().item()
    
    # Ground truth pitch
    pitch_gt = torch.rand(batch_size, Tfrm) * 520 + 80  # Random pitch in [80, 600]
    
    # Forward pass with ground truth
    with torch.no_grad():
        pitch_tok, pitch_frm, Ep = model(Henc, dur, pitch_gt=pitch_gt)
    
    # Check shapes
    assert pitch_tok.shape == (batch_size, Tph)
    assert pitch_frm.shape == (batch_size, Tfrm)
    assert Ep.shape == (batch_size, Tfrm, d_model)
    
    print(f"✓ Pitch Predictor works with ground truth pitch")


def test_pitch_predictor_quantization():
    """Test that pitch quantization works correctly."""
    d_model = 256
    n_bins = 256
    pitch_min = 80.0
    pitch_max = 600.0
    
    model = PitchPredictor(
        d_model=d_model,
        n_bins=n_bins,
        pitch_min=pitch_min,
        pitch_max=pitch_max
    )
    
    # Test quantization with known values
    pitch_continuous = torch.tensor([80.0, 340.0, 600.0, 50.0, 700.0])
    pitch_bins = model.quantize_pitch(pitch_continuous)
    
    # Check that bins are in valid range
    assert torch.all(pitch_bins >= 0), "Pitch bins should be >= 0"
    assert torch.all(pitch_bins < n_bins), f"Pitch bins should be < {n_bins}"
    
    # Check boundary values
    assert pitch_bins[0] == 0, "Minimum pitch should map to bin 0"
    assert pitch_bins[2] == n_bins - 1, "Maximum pitch should map to last bin"
    
    # Check clamping
    assert pitch_bins[3] == 0, "Below-minimum pitch should clamp to bin 0"
    assert pitch_bins[4] == n_bins - 1, "Above-maximum pitch should clamp to last bin"
    
    print(f"✓ Pitch quantization works correctly")
    print(f"  - Pitch values: {pitch_continuous.tolist()}")
    print(f"  - Quantized bins: {pitch_bins.tolist()}")


def test_pitch_predictor_expansion():
    """Test that phoneme-level pitch is correctly expanded to frame-level."""
    d_model = 256
    batch_size = 1
    Tph = 5
    
    model = PitchPredictor(d_model=d_model)
    model.eval()
    
    # Simple test case
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.tensor([[2, 3, 1, 2, 2]])  # Total: 10 frames
    
    with torch.no_grad():
        pitch_tok, pitch_frm, Ep = model(Henc, dur)
    
    # Check that frame-level length matches duration sum
    expected_Tfrm = dur.sum().item()
    assert pitch_frm.shape[1] == expected_Tfrm, \
        f"Expected {expected_Tfrm} frames, got {pitch_frm.shape[1]}"
    
    print(f"✓ Pitch expansion works correctly")
    print(f"  - Phoneme-level: {pitch_tok.shape}")
    print(f"  - Frame-level: {pitch_frm.shape}")


def test_pitch_predictor_different_batch_sizes():
    """Test that PitchPredictor works with different batch sizes."""
    d_model = 256
    Tph = 15
    
    model = PitchPredictor(d_model=d_model)
    model.eval()
    
    for batch_size in [1, 4, 8]:
        Henc = torch.randn(batch_size, Tph, d_model)
        dur = torch.randint(1, 5, (batch_size, Tph))
        
        with torch.no_grad():
            pitch_tok, pitch_frm, Ep = model(Henc, dur)
        
        assert pitch_tok.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
        assert pitch_frm.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
        assert Ep.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
    
    print(f"✓ Pitch Predictor works with different batch sizes")


def test_pitch_predictor_gradient_flow():
    """Test that gradients flow through the model."""
    d_model = 256
    batch_size = 2
    Tph = 10
    
    model = PitchPredictor(d_model=d_model)
    model.train()
    
    # Test input
    Henc = torch.randn(batch_size, Tph, d_model, requires_grad=True)
    dur = torch.randint(1, 5, (batch_size, Tph))
    
    # Forward pass
    pitch_tok, pitch_frm, Ep = model(Henc, dur)
    
    # Compute dummy loss
    loss = pitch_tok.sum() + Ep.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert Henc.grad is not None, "Gradients should flow to input"
    assert torch.any(Henc.grad != 0), "Gradients should be non-zero"
    
    print(f"✓ Gradients flow correctly through Pitch Predictor")


def test_pitch_predictor_embedding_dimension():
    """Test that pitch embeddings have correct dimension."""
    d_model = 256
    batch_size = 2
    Tph = 10
    
    model = PitchPredictor(d_model=d_model)
    model.eval()
    
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.randint(1, 5, (batch_size, Tph))
    
    with torch.no_grad():
        _, _, Ep = model(Henc, dur)
    
    # Check that embedding dimension matches d_model
    assert Ep.shape[-1] == d_model, \
        f"Expected embedding dimension {d_model}, got {Ep.shape[-1]}"
    
    print(f"✓ Pitch embedding dimension correct: {Ep.shape[-1]}")


if __name__ == "__main__":
    print("Testing Pitch Predictor...")
    print()
    
    test_pitch_predictor_shape()
    print()
    test_pitch_predictor_with_ground_truth()
    print()
    test_pitch_predictor_quantization()
    print()
    test_pitch_predictor_expansion()
    print()
    test_pitch_predictor_different_batch_sizes()
    print()
    test_pitch_predictor_gradient_flow()
    print()
    test_pitch_predictor_embedding_dimension()
    print()
    
    print("All Pitch Predictor tests passed! ✓")
