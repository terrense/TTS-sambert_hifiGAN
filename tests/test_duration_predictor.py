"""
Tests for Duration Predictor module.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.variance_adaptor import DurationPredictor


def test_duration_predictor_shape():
    """Test that DurationPredictor outputs correct shape."""
    # Configuration
    d_model = 256
    n_layers = 2
    kernel_size = 3
    dropout = 0.1
    
    # Create model
    model = DurationPredictor(
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=kernel_size,
        dropout=dropout
    )
    model.eval()
    
    # Test input
    batch_size = 2
    Tph = 20
    Henc = torch.randn(batch_size, Tph, d_model)
    
    # Forward pass
    with torch.no_grad():
        log_dur_pred = model(Henc)
    
    # Check output shape
    assert log_dur_pred.shape == (batch_size, Tph), \
        f"Expected shape ({batch_size}, {Tph}), got {log_dur_pred.shape}"
    
    print(f"✓ Duration Predictor output shape correct: {log_dur_pred.shape}")


def test_duration_predictor_with_mask():
    """Test that DurationPredictor handles masking correctly."""
    d_model = 256
    batch_size = 2
    Tph = 20
    
    # Create model
    model = DurationPredictor(d_model=d_model)
    model.eval()
    
    # Test input with mask
    Henc = torch.randn(batch_size, Tph, d_model)
    mask = torch.ones(batch_size, Tph, dtype=torch.bool)
    # Mask out last 5 positions
    mask[:, -5:] = False
    
    # Forward pass
    with torch.no_grad():
        log_dur_pred = model(Henc, mask=mask)
    
    # Check that masked positions have very negative values
    assert torch.all(log_dur_pred[:, -5:] < -1e8), \
        "Masked positions should have very negative values"
    
    # Check that unmasked positions have reasonable values
    assert torch.all(log_dur_pred[:, :-5] > -1e8), \
        "Unmasked positions should have reasonable values"
    
    print(f"✓ Duration Predictor masking works correctly")


def test_duration_predictor_different_batch_sizes():
    """Test that DurationPredictor works with different batch sizes."""
    d_model = 256
    Tph = 15
    
    model = DurationPredictor(d_model=d_model)
    model.eval()
    
    for batch_size in [1, 4, 8]:
        Henc = torch.randn(batch_size, Tph, d_model)
        
        with torch.no_grad():
            log_dur_pred = model(Henc)
        
        assert log_dur_pred.shape == (batch_size, Tph), \
            f"Failed for batch_size={batch_size}"
    
    print(f"✓ Duration Predictor works with different batch sizes")


def test_duration_predictor_gradient_flow():
    """Test that gradients flow through the model."""
    d_model = 256
    batch_size = 2
    Tph = 10
    
    model = DurationPredictor(d_model=d_model)
    model.train()
    
    # Test input
    Henc = torch.randn(batch_size, Tph, d_model, requires_grad=True)
    
    # Forward pass
    log_dur_pred = model(Henc)
    
    # Compute dummy loss
    loss = log_dur_pred.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert Henc.grad is not None, "Gradients should flow to input"
    assert torch.any(Henc.grad != 0), "Gradients should be non-zero"
    
    print(f"✓ Gradients flow correctly through Duration Predictor")


if __name__ == "__main__":
    print("Testing Duration Predictor...")
    print()
    
    test_duration_predictor_shape()
    test_duration_predictor_with_mask()
    test_duration_predictor_different_batch_sizes()
    test_duration_predictor_gradient_flow()
    
    print()
    print("All Duration Predictor tests passed! ✓")
