"""
Tests for PNCA AR-Decoder module.
"""

import torch
import pytest
from models.ar_decoder import PNCAARDecoder


def test_ar_decoder_initialization():
    """Test AR-Decoder initialization."""
    decoder = PNCAARDecoder(
        d_model=256,
        n_mels=80,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1
    )
    
    assert decoder.d_model == 256
    assert decoder.n_mels == 80
    assert decoder.n_layers == 6
    assert decoder.n_heads == 8


def test_ar_decoder_teacher_forcing():
    """Test AR-Decoder forward pass with teacher forcing (training mode)."""
    B, Tfrm, d_model, n_mels = 2, 100, 256, 80
    
    decoder = PNCAARDecoder(
        d_model=d_model,
        n_mels=n_mels,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        dropout=0.1
    )
    decoder.train()
    
    # Create random inputs
    Hvar = torch.randn(B, Tfrm, d_model)
    mel_gt = torch.randn(B, Tfrm, n_mels)
    
    # Forward pass
    mel_pred = decoder(Hvar, mel_gt=mel_gt)
    
    # Check output shape
    assert mel_pred.shape == (B, Tfrm, n_mels), \
        f"Expected shape {(B, Tfrm, n_mels)}, got {mel_pred.shape}"
    
    # Check that output is not all zeros
    assert not torch.allclose(mel_pred, torch.zeros_like(mel_pred)), \
        "Output should not be all zeros"


def test_ar_decoder_inference():
    """Test AR-Decoder autoregressive generation (inference mode)."""
    B, Tfrm, d_model, n_mels = 2, 50, 256, 80
    
    decoder = PNCAARDecoder(
        d_model=d_model,
        n_mels=n_mels,
        n_layers=2,  # Use fewer layers for faster test
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        chunk_size=1  # Single frame generation
    )
    decoder.eval()
    
    # Create random input
    Hvar = torch.randn(B, Tfrm, d_model)
    
    # Forward pass without mel_gt (inference mode)
    with torch.no_grad():
        mel_pred = decoder(Hvar, mel_gt=None, max_len=Tfrm)
    
    # Check output shape
    assert mel_pred.shape == (B, Tfrm, n_mels), \
        f"Expected shape {(B, Tfrm, n_mels)}, got {mel_pred.shape}"


def test_ar_decoder_chunk_based_inference():
    """Test AR-Decoder chunk-based autoregressive generation."""
    B, Tfrm, d_model, n_mels = 2, 50, 256, 80
    chunk_size = 5
    
    decoder = PNCAARDecoder(
        d_model=d_model,
        n_mels=n_mels,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        chunk_size=chunk_size
    )
    decoder.eval()
    
    # Create random input
    Hvar = torch.randn(B, Tfrm, d_model)
    
    # Forward pass with chunk-based generation
    with torch.no_grad():
        mel_pred = decoder(Hvar, mel_gt=None, max_len=Tfrm)
    
    # Check output shape
    assert mel_pred.shape == (B, Tfrm, n_mels), \
        f"Expected shape {(B, Tfrm, n_mels)}, got {mel_pred.shape}"
    
    # Check that output is not all zeros
    assert not torch.allclose(mel_pred, torch.zeros_like(mel_pred)), \
        "Output should not be all zeros"


def test_ar_decoder_max_len_parameter():
    """Test AR-Decoder with different max_len values."""
    B, Tfrm, d_model, n_mels = 2, 100, 256, 80
    
    decoder = PNCAARDecoder(
        d_model=d_model,
        n_mels=n_mels,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        chunk_size=10
    )
    decoder.eval()
    
    # Create random input
    Hvar = torch.randn(B, Tfrm, d_model)
    
    # Test with different max_len values
    for max_len in [30, 50, 75]:
        with torch.no_grad():
            mel_pred = decoder(Hvar, mel_gt=None, max_len=max_len)
        
        assert mel_pred.shape == (B, max_len, n_mels), \
            f"Expected shape {(B, max_len, n_mels)}, got {mel_pred.shape}"


def test_ar_decoder_chunk_sizes():
    """Test AR-Decoder with various chunk sizes."""
    B, Tfrm, d_model, n_mels = 2, 60, 256, 80
    
    for chunk_size in [1, 3, 5, 10, 20]:
        decoder = PNCAARDecoder(
            d_model=d_model,
            n_mels=n_mels,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            chunk_size=chunk_size
        )
        decoder.eval()
        
        Hvar = torch.randn(B, Tfrm, d_model)
        
        with torch.no_grad():
            mel_pred = decoder(Hvar, mel_gt=None, max_len=Tfrm)
        
        assert mel_pred.shape == (B, Tfrm, n_mels), \
            f"Failed for chunk_size={chunk_size}: expected {(B, Tfrm, n_mels)}, got {mel_pred.shape}"


def test_ar_decoder_shift_mel_right():
    """Test mel shifting for teacher forcing."""
    B, Tfrm, n_mels = 2, 10, 80
    
    decoder = PNCAARDecoder(d_model=256, n_mels=n_mels)
    
    # Create test mel with identifiable values
    mel = torch.arange(Tfrm).unsqueeze(0).unsqueeze(-1).repeat(B, 1, n_mels).float()
    
    # Shift right
    mel_shifted = decoder._shift_mel_right(mel)
    
    # Check shape
    assert mel_shifted.shape == mel.shape
    
    # Check that first frame is zeros
    assert torch.allclose(mel_shifted[:, 0, :], torch.zeros(B, n_mels))
    
    # Check that subsequent frames are shifted
    assert torch.allclose(mel_shifted[:, 1, :], mel[:, 0, :])
    assert torch.allclose(mel_shifted[:, 2, :], mel[:, 1, :])


def test_ar_decoder_causal_mask():
    """Test causal mask generation."""
    decoder = PNCAARDecoder(d_model=256, n_mels=80)
    
    # Generate mask for sequence length 5
    mask = decoder._generate_square_subsequent_mask(5)
    
    # Check shape
    assert mask.shape == (5, 5)
    
    # Check that mask is upper triangular (excluding diagonal)
    # True values indicate positions that should be masked
    expected = torch.tensor([
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
        [False, False, False, False, True],
        [False, False, False, False, False]
    ])
    
    assert torch.equal(mask, expected), \
        "Causal mask should be upper triangular"


def test_ar_decoder_different_batch_sizes():
    """Test AR-Decoder with different batch sizes."""
    d_model, n_mels = 256, 80
    
    decoder = PNCAARDecoder(
        d_model=d_model,
        n_mels=n_mels,
        n_layers=2,
        n_heads=4,
        d_ff=512
    )
    decoder.train()
    
    for B in [1, 2, 4, 8]:
        Tfrm = 50
        Hvar = torch.randn(B, Tfrm, d_model)
        mel_gt = torch.randn(B, Tfrm, n_mels)
        
        mel_pred = decoder(Hvar, mel_gt=mel_gt)
        
        assert mel_pred.shape == (B, Tfrm, n_mels), \
            f"Failed for batch size {B}"


def test_ar_decoder_gradient_flow():
    """Test that gradients flow through the decoder."""
    B, Tfrm, d_model, n_mels = 2, 50, 256, 80
    
    decoder = PNCAARDecoder(
        d_model=d_model,
        n_mels=n_mels,
        n_layers=2,
        n_heads=4,
        d_ff=512
    )
    decoder.train()
    
    Hvar = torch.randn(B, Tfrm, d_model, requires_grad=True)
    mel_gt = torch.randn(B, Tfrm, n_mels)
    
    # Forward pass
    mel_pred = decoder(Hvar, mel_gt=mel_gt)
    
    # Compute loss
    loss = mel_pred.mean()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert Hvar.grad is not None, "Gradients should flow to input"
    assert not torch.allclose(Hvar.grad, torch.zeros_like(Hvar.grad)), \
        "Gradients should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
