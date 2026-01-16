"""
Tests for Energy Predictor module.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.variance_adaptor import EnergyPredictor


def test_energy_predictor_shape():
    """Test that EnergyPredictor outputs correct shapes."""
    # Configuration
    d_model = 256
    n_bins = 256
    energy_min = 0.0
    energy_max = 1.0
    
    # Create model
    model = EnergyPredictor(
        d_model=d_model,
        n_bins=n_bins,
        energy_min=energy_min,
        energy_max=energy_max
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
        energy_tok, energy_frm, Ee = model(Henc, dur)
    
    # Check output shapes
    assert energy_tok.shape == (batch_size, Tph), \
        f"Expected energy_tok shape ({batch_size}, {Tph}), got {energy_tok.shape}"
    
    assert energy_frm.shape == (batch_size, Tfrm), \
        f"Expected energy_frm shape ({batch_size}, {Tfrm}), got {energy_frm.shape}"
    
    assert Ee.shape == (batch_size, Tfrm, d_model), \
        f"Expected Ee shape ({batch_size}, {Tfrm}, {d_model}), got {Ee.shape}"
    
    print(f"✓ Energy Predictor output shapes correct:")
    print(f"  - energy_tok: {energy_tok.shape}")
    print(f"  - energy_frm: {energy_frm.shape}")
    print(f"  - Ee: {Ee.shape}")


def test_energy_predictor_with_ground_truth():
    """Test that EnergyPredictor uses ground truth energy during training."""
    d_model = 256
    batch_size = 2
    Tph = 20
    
    # Create model
    model = EnergyPredictor(d_model=d_model)
    model.eval()
    
    # Test input
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.randint(1, 10, (batch_size, Tph))
    
    # Calculate frame length
    Tfrm = dur.sum(dim=1).max().item()
    
    # Ground truth energy
    energy_gt = torch.rand(batch_size, Tfrm)  # Random energy in [0, 1]
    
    # Forward pass with ground truth
    with torch.no_grad():
        energy_tok, energy_frm, Ee = model(Henc, dur, energy_gt=energy_gt)
    
    # Check shapes
    assert energy_tok.shape == (batch_size, Tph)
    assert energy_frm.shape == (batch_size, Tfrm)
    assert Ee.shape == (batch_size, Tfrm, d_model)
    
    print(f"✓ Energy Predictor works with ground truth energy")


def test_energy_predictor_quantization():
    """Test that energy quantization works correctly."""
    d_model = 256
    n_bins = 256
    energy_min = 0.0
    energy_max = 1.0
    
    model = EnergyPredictor(
        d_model=d_model,
        n_bins=n_bins,
        energy_min=energy_min,
        energy_max=energy_max
    )
    
    # Test quantization with known values
    energy_continuous = torch.tensor([0.0, 0.5, 1.0, -0.1, 1.5])
    energy_bins = model.quantize_energy(energy_continuous)
    
    # Check that bins are in valid range
    assert torch.all(energy_bins >= 0), "Energy bins should be >= 0"
    assert torch.all(energy_bins < n_bins), f"Energy bins should be < {n_bins}"
    
    # Check boundary values
    assert energy_bins[0] == 0, "Minimum energy should map to bin 0"
    assert energy_bins[2] == n_bins - 1, "Maximum energy should map to last bin"
    
    # Check clamping
    assert energy_bins[3] == 0, "Below-minimum energy should clamp to bin 0"
    assert energy_bins[4] == n_bins - 1, "Above-maximum energy should clamp to last bin"
    
    print(f"✓ Energy quantization works correctly")
    print(f"  - Energy values: {energy_continuous.tolist()}")
    print(f"  - Quantized bins: {energy_bins.tolist()}")


def test_energy_predictor_expansion():
    """Test that phoneme-level energy is correctly expanded to frame-level."""
    d_model = 256
    batch_size = 1
    Tph = 5
    
    model = EnergyPredictor(d_model=d_model)
    model.eval()
    
    # Simple test case
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.tensor([[2, 3, 1, 2, 2]])  # Total: 10 frames
    
    with torch.no_grad():
        energy_tok, energy_frm, Ee = model(Henc, dur)
    
    # Check that frame-level length matches duration sum
    expected_Tfrm = dur.sum().item()
    assert energy_frm.shape[1] == expected_Tfrm, \
        f"Expected {expected_Tfrm} frames, got {energy_frm.shape[1]}"
    
    print(f"✓ Energy expansion works correctly")
    print(f"  - Phoneme-level: {energy_tok.shape}")
    print(f"  - Frame-level: {energy_frm.shape}")


def test_energy_predictor_different_batch_sizes():
    """Test that EnergyPredictor works with different batch sizes."""
    d_model = 256
    Tph = 15
    
    model = EnergyPredictor(d_model=d_model)
    model.eval()
    
    for batch_size in [1, 4, 8]:
        Henc = torch.randn(batch_size, Tph, d_model)
        dur = torch.randint(1, 5, (batch_size, Tph))
        
        with torch.no_grad():
            energy_tok, energy_frm, Ee = model(Henc, dur)
        
        assert energy_tok.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
        assert energy_frm.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
        assert Ee.shape[0] == batch_size, \
            f"Failed for batch_size={batch_size}"
    
    print(f"✓ Energy Predictor works with different batch sizes")


def test_energy_predictor_gradient_flow():
    """Test that gradients flow through the model."""
    d_model = 256
    batch_size = 2
    Tph = 10
    
    model = EnergyPredictor(d_model=d_model)
    model.train()
    
    # Test input
    Henc = torch.randn(batch_size, Tph, d_model, requires_grad=True)
    dur = torch.randint(1, 5, (batch_size, Tph))
    
    # Forward pass
    energy_tok, energy_frm, Ee = model(Henc, dur)
    
    # Compute dummy loss
    loss = energy_tok.sum() + Ee.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert Henc.grad is not None, "Gradients should flow to input"
    assert torch.any(Henc.grad != 0), "Gradients should be non-zero"
    
    print(f"✓ Gradients flow correctly through Energy Predictor")


def test_energy_predictor_embedding_dimension():
    """Test that energy embeddings have correct dimension."""
    d_model = 256
    batch_size = 2
    Tph = 10
    
    model = EnergyPredictor(d_model=d_model)
    model.eval()
    
    Henc = torch.randn(batch_size, Tph, d_model)
    dur = torch.randint(1, 5, (batch_size, Tph))
    
    with torch.no_grad():
        _, _, Ee = model(Henc, dur)
    
    # Check that embedding dimension matches d_model
    assert Ee.shape[-1] == d_model, \
        f"Expected embedding dimension {d_model}, got {Ee.shape[-1]}"
    
    print(f"✓ Energy embedding dimension correct: {Ee.shape[-1]}")


if __name__ == "__main__":
    print("Testing Energy Predictor...")
    print()
    
    test_energy_predictor_shape()
    print()
    test_energy_predictor_with_ground_truth()
    print()
    test_energy_predictor_quantization()
    print()
    test_energy_predictor_expansion()
    print()
    test_energy_predictor_different_batch_sizes()
    print()
    test_energy_predictor_gradient_flow()
    print()
    test_energy_predictor_embedding_dimension()
    print()
    
    print("All Energy Predictor tests passed! ✓")
